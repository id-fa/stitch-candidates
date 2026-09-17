// recon.js - panorama_recon.py の WebGPU 移植（位置合わせ部）
// 粗探索は FFT ではなく階層的総当たり NCC（1/16 → 1/4 解像度）。精密化は Gauss-Newton（GPU 上で完結）。

import { IMG } from "./shaders_img.js";
import { ALIGN } from "./shaders_align.js";
import { solveGlobal, solveDense, solveArrowhead, solvePositions, rot2, mulRot, subpix, median, mean,
         mat3mul, mat3inv, mat3norm, applyH, simMatrix, normMatrix, localScaleOfH, orthonormalize, focalsFromHomography,
         solveGlobalH, solveRotations } from "./solve.js";

export const DEFAULTS = {
  model: "scale", pairs: [1, 2, 4], coarseScale: 0.25, coarseTol: 2.0, minOverlap: 0.15,
  scaleMax: 0.06, scaleStep: 0.004, fineScale: 1.0, gnIters: 15, ignoreRects: [], textRects: [],
  staticMask: false, staticSpan: 6, staticDiff: 0.03, staticGrad: 0.08, staticDilate: 7, staticHalo: 12, staticClose: 3, textHalo: 4,
  canvasScale: "auto", band: 64, inlierTol: 0.06, sharpTop: 0.3, resTol: 1.25, anchorFrame: -1, anchorWindow: 2, stackBudgetMB: 128, levelCacheMB: 768,
  exposureProfile: true, exposureMinScore: 0.2, exposureLocal: 6, feather: 0,
  // 実写向け（app.js の「実写に特化」で既定値が切り替わる）
  projection: "planar", cylMinSpan: 35, focal: 0, lensK1: "off", quality: "off", qualityThr: 0.75, qualityWindow: 8, qualityMin: 3,
  sharpSigma: 0, blendDetect: "off", blendThr: 0.4, exposureRadial: false, motionReject: false,
};
// 周辺減光プロファイルの節点（正規化半径 r: 中心 0、隅 1）
export const RAD_KNOTS = [0.0, 0.25, 0.45, 0.6, 0.72, 0.82, 0.9, 0.96, 1.0];

// 露出補正プロファイルの節点（正規化座標 0..1）。プレイヤーのグラデーション等は端で急に変わるので端に密に置く
export const EXPO_KNOTS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0];

const odd = (n) => { n = Math.round(n); return n % 2 === 1 ? n : n + 1; };
const now = () => performance.now() / 1000;
const MODEL_ID = { translation: 0, scale: 1, similarity: 2 };

export class Reconstructor {
  /** store: FrameStore（フレームの RGBA8 GPUBuffer を供給。ストリーミング時は必要なフレームだけを常駐させる） */
  constructor(gpu, W, H, store, args, log, progress) {
    const n = store.n;
    this.gpu = gpu; this.W = W; this.H = H; this.n = n;
    this.a = { ...DEFAULTS, ...args };
    this.log = log; this.progress = progress || (() => {});
    this.store = store;
    this.sizes = store.sizes;                    // 各フレームの有効サイズ [w, h]（パディング領域は除外）
    this.padded = this.sizes.some(([w, h]) => w !== W || h !== H);
    this.exposure = null;                        // 露出補正 {gain, offset, profile}（estimateExposure で設定）
    this.abort = false;
    this._dbgFail = 0;
    const cs = this.a.coarseScale;
    this.hc = Math.max(16, Math.round(H * cs)); this.wc = Math.max(16, Math.round(W * cs));
    this.h16 = Math.max(16, Math.round(this.hc / 4)); this.w16 = Math.max(16, Math.round(this.wc / 4));
    this.sx = this.wc / W; this.sy = this.hc / H;
    this.Hq = Math.max(1, Math.floor(H / 4)); this.Wq = Math.max(1, Math.floor(W / 4));
    this.C4 = new Array(n); this.C16 = new Array(n);
    this.overlayRatio = new Float32Array(n);
    this.levelCache = new Map();   // k -> {levels: {lv: {buf, hs, ws}}, bytes, sig}
    this.levelBytes = 0;
    this.k = {};
    for (const [name, def] of Object.entries({ ...IMG, ...ALIGN })) this.k[name] = gpu.kernel(name, def.code, def.fields, def.bindings, def.wg);
    // スクラッチ
    const full = W * H * 4;
    this.f = [0, 1, 2, 3, 4, 5].map((i) => gpu.buf(full, `scratch_f${i}`));
    const cfull = this.hc * this.wc * 4;
    this.c = [0, 1, 2].map((i) => gpu.buf(cfull, `scratch_c${i}`));
    this.s16 = [0, 1, 2].map((i) => gpu.buf(this.h16 * this.w16 * 4, `scratch_s${i}`));
    this.sharpq = gpu.buf(n * this.Hq * this.Wq * 4, "sharpq");
    this.ratioBuf = gpu.buf(n * 4, "ratio");
    this.surf = gpu.buf(4 * 1024 * 1024 * 4, "ncc_surf");
    this.resBuf = gpu.buf(4096 * 8 * 4, "ncc_res");
    this.Bs = gpu.buf(Math.max(cfull * 2 * 2, W * H * 8), "resampled");
    this.NB = 2048;
    this.hist = gpu.buf((this.NB + 1) * 4, "hist");
    this.NWG = 1024;
    this.partials = gpu.buf(this.NWG * 60 * 4, "partials");   // gn_accum 20 / gnh_accum 60 floats per workgroup
    this.frameTier = null;     // フレーム品質の段階（0 良, 1 ボケ/ブレ, 2 ブレンド）。detectBlendFrames / estimateQuality で設定
    this.state = gpu.buf(256 * 16 * 4, "gn_state");
  }

  _check() { if (this.abort) throw new Error("中断されました"); }

  // ------------------------------------------------------------ 前処理
  async preprocess() {
    const a = this.a, g = this.gpu, K = this.k, W = this.W, H = this.H, n = this.n, st = this.store;
    const t0 = now();
    const [fA, fB, fC, fD, fE, fF] = this.f;
    const [cA, cB, cC] = this.c;
    const [sA, sB, sC] = this.s16;
    const r4 = (odd(Math.max(3, 0.08 * Math.min(this.hc, this.wc))) - 1) / 2;
    const r16 = (odd(Math.max(3, 0.08 * Math.min(this.h16, this.w16))) - 1) / 2;
    const rects = new Array(32).fill(0);
    const nrect = Math.min(8, a.ignoreRects.length);
    for (let i = 0; i < nrect; i++) for (let c = 0; c < 4; c++) rects[i * 4 + c] = a.ignoreRects[i][c];
    const span = a.staticSpan;
    const trects = new Array(32).fill(0);
    const ntext = Math.min(8, a.textRects.length);
    for (let i = 0; i < ntext; i++) for (let c = 0; c < 4; c++) trects[i * 4 + c] = a.textRects[i][c];
    const doMask = (a.staticMask && n > 1) || ntext > 0;
    const nclose = doMask ? Math.max(0, Math.min(3, a.staticClose | 0)) : 0;
    const packMasks = !st.allResident;   // 破棄したフレームを読み直す時にマスク（alpha）を復元するため
    // フレームは k の順に処理する。フレーム k の処理に必要なのは k±span（静止検出）と k±nclose（クロージング）だけなので、
    // ストリーミング時も窓のぶんだけ常駐していれば足りる。複数フレームを集めてからカーネルを積む箇所は、
    // 集めている間に破棄されないよう pin する

    // 段階 1: 静止オーバーレイ（+ テキスト矩形内の勾配ベース文字マスク）→ フレーム k の alpha に書く
    //   fA: 静止エッジ, fD: flat（静止 / 矩形内）→ 密度 (5x5) > 0.3 を二値化 → 膨張 r（fC）と膨張 rh（fE）
    const stage1 = async (k) => {
      const fr = await st.get(k);
      st.pin(k);                     // マスク確定（段階 3）まで破棄させない
      let useDens = 0;
      if (doMask) {
        // 比較相手: k±span のうち範囲内のもの（いずれかと一致すれば静止）
        let cands = [k - span, k + span].filter((j) => j >= 0 && j < n && j !== k);
        if (!cands.length) cands = [Math.max(0, Math.min(n - 1, k + (k === 0 ? span : -span)))];
        cands = cands.slice(0, 4);
        const diff = a.staticMask && n > 1 ? a.staticDiff : -1.0;   // 静止検出を無効化する場合は diff を負にする
        const cf = [], newly = [];   // 新たに pin したものだけ後で外す（段階 3 待ちで pin 中のフレームと重なることがある）
        for (const j of cands) { cf.push(await st.get(j)); if (!st.pinned.has(j)) { st.pin(j); newly.push(j); } }
        const cf4 = [0, 1, 2, 3].map((q) => cf[Math.min(q, cf.length - 1)]);
        const vc = [0, 1, 2, 3].flatMap((q) => this.sizes[cands[Math.min(q, cands.length - 1)]]);
        K.static_detect.run2d({ W, H, diff, grad: a.staticGrad, ncand: cands.length, ntext, trects, vk: [...this.sizes[k], 0, 0], vc },
          [fr, cf4[0], cf4[1], cf4[2], cf4[3], fA, fD], W, H);
        for (const j of newly) st.unpin(j);
        K.box_blur.run2d({ W, H, r: 2, axis: 0, sub: 0 }, [fA, fA, fB], W, H);
        K.box_blur.run2d({ W, H, r: 2, axis: 1, sub: 0 }, [fB, fA, fC], W, H);
        K.threshold.run({ n: W * H, thr: 0.3 }, [fC, fA], Math.ceil(W * H / 256));
        const rd = Math.floor(a.staticDilate / 2), rh = Math.max(rd, Math.floor(a.staticHalo));
        K.box_blur.run2d({ W, H, r: rd, axis: 0, sub: 0 }, [fA, fA, fB], W, H);
        K.box_blur.run2d({ W, H, r: rd, axis: 1, sub: 0 }, [fB, fA, fC], W, H);
        K.box_blur.run2d({ W, H, r: rh, axis: 0, sub: 0 }, [fA, fA, fB], W, H);
        K.box_blur.run2d({ W, H, r: rh, axis: 1, sub: 0 }, [fB, fA, fE], W, H);
        const rt = Math.max(rd, Math.floor(a.textHalo));
        K.box_blur.run2d({ W, H, r: rt, axis: 0, sub: 0 }, [fA, fA, fB], W, H);
        K.box_blur.run2d({ W, H, r: rt, axis: 1, sub: 0 }, [fB, fA, fF], W, H);
        useDens = 1;
      }
      K.finalize_mask.run2d({ W, H, use_dens: useDens, nrect, vw: this.sizes[k][0], vh: this.sizes[k][1], rects }, [fC, fE, fF, fD, fr], W, H);
    };
    // 段階 2: マスクの時間方向クロージング（j=1..nclose の前後両方でマスクされていれば埋める）。
    //   k-j は確定済み（読み直しても pack 済みマスクが復元される）、k+j は段階 1 済みで pin 中
    const closeK = async (k) => {
      const pairs = [], held = [];
      for (let j = 1; j <= nclose; j++) {
        if (k - j < 0 || k + j >= n) continue;
        const pa = await st.get(k - j);
        if (!st.pinned.has(k - j)) { st.pin(k - j); held.push(k - j); }
        const pb = await st.get(k + j);
        pairs.push([pa, pb]);
      }
      if (pairs.length) {
        const bufs = [st.peek(k)];
        for (let q = 0; q < 3; q++) { const pr = pairs[Math.min(q, pairs.length - 1)]; bufs.push(pr[0], pr[1]); }
        K.mask_close.run({ n: W * H, npair: pairs.length }, bufs, Math.ceil(W * H / 256));
      }
      for (const j of held) st.unpin(j);
    };
    // 段階 3: グレー / 鮮明度 / 1/4, 1/16 レベル（マスク確定後）。ストリーミング時はマスクをパックしてから固定解除
    const finalizeK = (k) => {
      const fr = st.peek(k);
      K.gray_area.run2d({ W, H, w: W, h: H }, [fr, fA], W, H);
      if (a.sharpSigma > 0) {
        // 先に軽くぼかしてノイズ / 粒子のラプラシアン応答を抑える（実写）
        K.gauss_f32.run2d({ W, H, sigma: a.sharpSigma, axis: 0 }, [fA, fC], W, H);
        K.gauss_f32.run2d({ W, H, sigma: a.sharpSigma, axis: 1 }, [fC, fB], W, H);
        K.laplacian_abs.run2d({ W, H }, [fB, fC], W, H);
        K.box_blur.run2d({ W, H, r: 7, axis: 0, sub: 0 }, [fC, fC, fB], W, H);
        K.box_blur.run2d({ W, H, r: 7, axis: 1, sub: 0 }, [fB, fB, fC], W, H);
        K.area_f32.run2d({ W, H, w: this.Wq, h: this.Hq, src_off: 0, dst_off: k * this.Hq * this.Wq }, [fC, this.sharpq], this.Wq, this.Hq);
      } else {
      K.laplacian_abs.run2d({ W, H }, [fA, fB], W, H);
      K.box_blur.run2d({ W, H, r: 7, axis: 0, sub: 0 }, [fB, fB, fC], W, H);
      K.box_blur.run2d({ W, H, r: 7, axis: 1, sub: 0 }, [fC, fC, fB], W, H);
      K.area_f32.run2d({ W, H, w: this.Wq, h: this.Hq, src_off: 0, dst_off: k * this.Hq * this.Wq }, [fB, this.sharpq], this.Wq, this.Hq);
      }
      // 1/4 レベル
      K.area_f32.run2d({ W, H, w: this.wc, h: this.hc, src_off: 0, dst_off: 0 }, [fA, cA], this.wc, this.hc);
      K.box_blur.run2d({ W: this.wc, H: this.hc, r: r4, axis: 0, sub: 0 }, [cA, cA, cB], this.wc, this.hc);
      K.box_blur.run2d({ W: this.wc, H: this.hc, r: r4, axis: 1, sub: 1 }, [cB, cA, cC], this.wc, this.hc);
      K.alpha_f32.run({ n: W * H }, [fr, fB], Math.ceil(W * H / 256));
      K.area_f32.run2d({ W, H, w: this.wc, h: this.hc, src_off: 0, dst_off: 0 }, [fB, cB], this.wc, this.hc);
      K.sum_f32.run({ n: this.wc * this.hc, out_off: k }, [cB, this.ratioBuf], 1);
      this.C4[k] = g.buf(this.wc * this.hc * 8, `C4_${k}`);
      K.pack2.run({ n: this.wc * this.hc }, [cC, cB, this.C4[k]], Math.ceil(this.wc * this.hc / 256));
      // 1/16 レベル
      K.area_f32.run2d({ W: this.wc, H: this.hc, w: this.w16, h: this.h16, src_off: 0, dst_off: 0 }, [cA, sA], this.w16, this.h16);
      K.box_blur.run2d({ W: this.w16, H: this.h16, r: r16, axis: 0, sub: 0 }, [sA, sA, sB], this.w16, this.h16);
      K.box_blur.run2d({ W: this.w16, H: this.h16, r: r16, axis: 1, sub: 1 }, [sB, sA, sC], this.w16, this.h16);
      K.area_f32.run2d({ W, H, w: this.w16, h: this.h16, src_off: 0, dst_off: 0 }, [fB, sB], this.w16, this.h16);
      this.C16[k] = g.buf(this.w16 * this.h16 * 8, `C16_${k}`);
      K.pack2.run({ n: this.w16 * this.h16 }, [sC, sB, this.C16[k]], Math.ceil(this.w16 * this.h16 / 256));
      if (packMasks) st.packMask(k);
      st.unpin(k);
    };
    // 段階 1 を k の順に進め、nclose 遅れで段階 2 → 3 を行う（k-j は確定済み、k+j は段階 1 済み）
    const lag = nclose;
    for (let k = 0; k < n + lag; k++) {
      if (k < n) await stage1(k);
      const kk = k - lag;
      if (kk >= 0) { if (nclose > 0) await closeK(kk); finalizeK(kk); }
      g.submit();
      if (k % 8 === 7 || k === n + lag - 1) { await g.done(); this.progress("preprocess", (k + 1) / (n + lag)); this._check(); }
    }
    const rat = new Float32Array(await g.read(this.ratioBuf, n * 4));
    for (let k = 0; k < n; k++) {
      const vf = (this.sizes[k][0] * this.sizes[k][1]) / (W * H);   // 比率は有効領域内で
      this.overlayRatio[k] = Math.max(0, 1.0 - rat[k] / (this.wc * this.hc * vf));
    }
    const mr = mean(this.overlayRatio);
    this.log(`[preprocess] ${n} frames ${W}x${H}${this.padded ? " (padded, sizes differ)" : ""}, coarse ${this.wc}x${this.hc} / ${this.w16}x${this.h16}, ` +
             `overlay mask mean ratio=${mr.toFixed(3)}, ${(now() - t0).toFixed(1)}s` + (st.allResident ? "" : ` (streaming, ${st.stats()})`));
    if (mr > 0.5) this.log("  警告: オーバーレイマスクが 50% を超えています。static-span を大きくするか static-diff を小さくしてください");
    this.progress("preprocess", 1);
  }

  // ------------------------------------------------------------ フル解像度レベル（LRU）
  _levelSpec() {
    const a = this.a;
    if (a.model === "translation") return { lvs: [1.0], sigma: 0.0 };
    const s = new Set([1.0]);
    for (const lv of [0.25, 0.5, 1.0]) if (lv <= a.fineScale + 1e-9) s.add(lv);
    s.delete(1.0); s.add(a.fineScale);
    return { lvs: [...s].sort((x, y) => x - y), sigma: 1.0 };
  }

  async _levels(k) {
    const e = this.levelCache.get(k);
    if (e) { this.levelCache.delete(k); this.levelCache.set(k, e); return e.levels; }
    const K = this.k, g = this.gpu, W = this.W, H = this.H;
    const [fA, fB, fC, fD, fE, fF] = this.f;
    const { lvs, sigma } = this._levelSpec();
    const rF = (odd(Math.max(3, 0.08 * Math.min(H, W))) - 1) / 2;
    const fr = await this.store.get(k);
    K.gray_area.run2d({ W, H, w: W, h: H }, [fr, fA], W, H);
    K.box_blur.run2d({ W, H, r: rF, axis: 0, sub: 0 }, [fA, fA, fB], W, H);
    K.box_blur.run2d({ W, H, r: rF, axis: 1, sub: 1 }, [fB, fA, fC], W, H);   // fC = highpass
    K.alpha_f32.run({ n: W * H }, [fr, fB], Math.ceil(W * H / 256)); // fB = clean
    const levels = {};
    let bytes = 0;
    for (const lv of lvs) {
      let hs = H, ws = W, hp = fC, cl = fB;
      if (lv < 0.999) {
        hs = Math.max(8, Math.round(H * lv)); ws = Math.max(8, Math.round(W * lv));
        K.area_f32.run2d({ W, H, w: ws, h: hs, src_off: 0, dst_off: 0 }, [fC, fD], ws, hs);
        K.area_f32.run2d({ W, H, w: ws, h: hs, src_off: 0, dst_off: 0 }, [fB, fE], ws, hs);
        hp = fD; cl = fE;
      }
      if (sigma > 0.05) {
        K.gauss_f32.run2d({ W: ws, H: hs, sigma, axis: 0 }, [hp, fF], ws, hs);
        K.gauss_f32.run2d({ W: ws, H: hs, sigma, axis: 1 }, [fF, fA], ws, hs);
        hp = fA;
      }
      const buf = g.buf(hs * ws * 8, `L${k}_${lv}`);
      K.pack2.run({ n: hs * ws }, [hp, cl, buf], Math.ceil(hs * ws / 256));
      levels[lv] = { buf, hs, ws };
      bytes += hs * ws * 8;
    }
    this.levelCache.set(k, { levels, bytes });
    this.levelBytes += bytes;
    return levels;
  }

  /** 送信済みの仕事が完了した後にだけ呼ぶ（バッファ破棄のため） */
  _evictLevels(keep) {
    const budget = this.a.levelCacheMB * 1048576;
    for (const [k, e] of this.levelCache) {
      if (this.levelBytes <= budget) break;
      if (keep.has(k)) continue;
      for (const lv of Object.keys(e.levels)) this.gpu.free(e.levels[lv].buf);
      this.levelBytes -= e.bytes;
      this.levelCache.delete(k);
    }
  }

  // ------------------------------------------------------------ 粗探索（階層的総当たり NCC）
  /**
   * jobs: [{i, j, level: 16|4, scales: [s,...], win: null | {dx, dy, r}, nmin}]
   * 戻り値: jobs と同じ順で [{scales:[{s_eff, dx, dy, score, l, r, u, d}]}]
   */
  async _searchBatch(jobs) {
    const K = this.k, g = this.gpu;
    let slot = 0;
    const meta = [];
    for (const jb of jobs) {
      const src = jb.level === 16 ? this.C16 : this.C4;
      const h = jb.level === 16 ? this.h16 : this.hc, w = jb.level === 16 ? this.w16 : this.wc;
      const A = src[jb.i], B = src[jb.j];
      const ms = [];
      for (const s of jb.scales) {
        const hs = Math.max(8, Math.round(h * s)), ws = Math.max(8, Math.round(w * s));
        K.resample2.run2d({ W: w, H: h, w: ws, h: hs }, [B, this.Bs], ws, hs);
        let dx0, dy0, nx, ny;
        if (jb.win) { dx0 = jb.win.dx - jb.win.r; dy0 = jb.win.dy - jb.win.r; nx = ny = 2 * jb.win.r + 1; }
        else { dx0 = -(ws - 1); dy0 = -(hs - 1); nx = w + ws - 1; ny = h + hs - 1; }
        if (nx * ny > 4 * 1024 * 1024) throw new Error("NCC surface too large");
        K.ncc_search.run({ hA: h, wA: w, hB: hs, wB: ws, dx0, dy0, nx, ny, nmin: jb.nmin, out_off: 0 }, [A, this.Bs, this.surf], nx, ny);
        K.argmax.run({ in_off: 0, n: nx * ny, nx, out_off: slot * 8 }, [this.surf, this.resBuf], 1);
        ms.push({ slot, s_eff: hs / h, dx0, dy0 });
        slot++;
        if (slot >= 4096) throw new Error("too many search slots in a batch");
      }
      meta.push(ms);
    }
    const res = new Float32Array(await g.read(this.resBuf, slot * 8 * 4));
    return meta.map((ms) => ({
      scales: ms.map((m) => {
        const o = m.slot * 8;
        return { s_eff: m.s_eff, score: res[o], dx: m.dx0 + res[o + 6], dy: m.dy0 + res[o + 7],
                 l: res[o + 2], r: res[o + 3], u: res[o + 4], d: res[o + 5] };
      }),
    }));
  }

  /**
   * 1/16 レベルの全シフト探索で、スケールごとのサーフェスを読み戻して局所ピーク上位 M 個を返す。
   * jobs: [{i, j, scales: [s,...], nmin}] → [[{s_eff, dx, dy, score}, ...] (score 降順、最大 M 個)]
   * 1/16 では自己相似な内容（ジーンズ、平坦な壁）で最大値が誤ることがあり、±12 px の 1/4 窓では回復できないため、
   * 複数の候補を 1/4 で評価して選ぶ（Python 版の 1/4 FFT 全探索に相当する頑健性を得る）
   */
  async _search16Peaks(jobs, M) {
    const K = this.k, g = this.gpu;
    const h = this.h16, w = this.w16;
    let slot = 0;
    const meta = [];
    for (const jb of jobs) {
      const A = this.C16[jb.i], B = this.C16[jb.j];
      const ms = [];
      for (const s of jb.scales) {
        const hs = Math.max(8, Math.round(h * s)), ws = Math.max(8, Math.round(w * s));
        const nx = w + ws - 1, ny = h + hs - 1;
        if (nx * ny > 4 * 1024 * 1024) throw new Error("NCC surface too large");
        K.resample2.run2d({ W: w, H: h, w: ws, h: hs }, [B, this.Bs], ws, hs);
        K.ncc_search.run({ hA: h, wA: w, hB: hs, wB: ws, dx0: -(ws - 1), dy0: -(hs - 1), nx, ny, nmin: jb.nmin, out_off: 0 }, [A, this.Bs, this.surf], nx, ny);
        // GPU 上で argmax → その周囲 ±3 を潰す → argmax … を M 回（読み戻しはピークごとに 8 floats）
        for (let t = 0; t < M; t++) {
          K.argmax.run({ in_off: 0, n: nx * ny, nx, out_off: slot * 8 }, [this.surf, this.resBuf], 1);
          if (t + 1 < M) K.suppress.run({ in_off: 0, nx, ny, res_off: slot * 8, r: 3 }, [this.resBuf, this.surf], 1);
          ms.push({ slot, s_eff: hs / h, dx0: -(ws - 1), dy0: -(hs - 1) });
          slot++;
          if (slot >= 4096) throw new Error("too many search slots in a batch");
        }
      }
      meta.push(ms);
    }
    const res = new Float32Array(await g.read(this.resBuf, slot * 8 * 4));
    // スコア降順に並べ、近い位置（3 px 以内、スケール違いも含む）は 1 つにまとめて上位 M 個
    return meta.map((ms) => {
      const peaks = ms.map((m) => {
        const o = m.slot * 8;
        return { s_eff: m.s_eff, score: res[o], dx: m.dx0 + res[o + 6], dy: m.dy0 + res[o + 7] };
      }).filter((p) => p.score > -1.5);
      peaks.sort((a, b) => b.score - a.score);
      const sel = [];
      for (const p of peaks) {
        if (sel.length >= M) break;
        if (sel.some((s) => Math.abs(s.dx - p.dx) <= 3 && Math.abs(s.dy - p.dy) <= 3)) continue;
        sel.push(p);
      }
      return sel;
    });
  }

  /** 粗推定: 戻り値 pc (m×4: s, th, tx, ty フル解像度単位), sc (m) */
  async coarse(pairs) {
    const a = this.a, n = this.n, m = pairs.length;
    const pc = new Float64Array(m * 4), sc = new Float64Array(m);
    const scaleModel = a.model !== "translation";
    const step16 = Math.max(a.scaleStep, 0.02);
    const K16 = Math.round(a.scaleMax / step16);
    const nref = Math.ceil(step16 / a.scaleStep / 2) + 1;
    const r4 = 12;
    const NPEAK = 4;   // 1/16 レベルで 1/4 に持ち上げる候補ピーク数
    const nmin16 = a.minOverlap * this.h16 * this.w16, nmin4 = a.minOverlap * this.hc * this.wc;
    const chain = new Float64Array(n);
    const minOff = Math.min(...a.pairs);
    const groups = [[], []];
    for (let r = 0; r < m; r++) (pairs[r][1] - pairs[r][0] === minOff ? groups[0] : groups[1]).push(r);
    const BATCH = 24;
    let done = 0;
    for (const grp of groups) {
      for (let b0 = 0; b0 < grp.length; b0 += BATCH) {
        const rs = grp.slice(b0, b0 + BATCH);
        // レベル 1/16: 全シフト探索の局所ピーク上位 M 個（自己相似な内容での誤ピーク対策）
        const jobs16 = rs.map((r) => {
          const [i, j] = pairs[r];
          let scales = [1.0];
          if (scaleModel) {
            const full = j - i === minOff;
            const c0 = full ? 0 : chain[j] - chain[i];
            const ks = full ? [...Array(2 * K16 + 1).keys()].map((q) => q - K16) : [-1, 0, 1];
            scales = ks.map((q) => Math.exp(c0 + q * step16));
          }
          return { i, j, scales, nmin: nmin16 };
        });
        const peaks16 = await this._search16Peaks(jobs16, NPEAK);
        // レベル 1/4: 各候補の周辺（±r4 px、スケールは候補値の近傍）を再探索し、最良の候補を採る
        const jobs4 = [], owner = [];
        rs.forEach((r, q) => {
          const [i, j] = pairs[r];
          const cands = peaks16[q].length ? peaks16[q] : [{ s_eff: 1.0, dx: 0, dy: 0, score: -2 }];
          for (const c of cands) {
            let ls = Math.log(c.s_eff);
            const f = this.wc / this.w16;
            const scales = scaleModel ? [...Array(2 * nref + 1).keys()].map((q2) => Math.exp(ls + (q2 - nref) * a.scaleStep)) : [1.0];
            jobs4.push({ i, j, level: 4, scales, win: { dx: Math.round(c.dx * f), dy: Math.round(c.dy * f), r: r4 }, nmin: nmin4 });
            owner.push(q);
          }
        });
        const out4all = await this._searchBatch(jobs4);
        const out4 = rs.map(() => null);
        const dbg = typeof window !== "undefined" && window.__dbgCoarse;
        out4all.forEach((o, t) => {
          const q = owner[t];
          let best = 0;
          for (let u = 1; u < o.scales.length; u++) if (o.scales[u].score > o.scales[best].score) best = u;
          if (dbg) {
            const c16 = peaks16[q][jobs4.slice(0, t).filter((jb) => jb.i === jobs4[t].i && jb.j === jobs4[t].j).length] || {};
            this.log(`  [coarse dbg] (${jobs4[t].i},${jobs4[t].j}) 1/16 peak s=${(c16.s_eff || 1).toFixed(3)} d=(${c16.dx},${c16.dy}) score=${(c16.score ?? -2).toFixed(3)}` +
                     ` → 1/4 s=${o.scales[best].s_eff.toFixed(3)} d=(${o.scales[best].dx},${o.scales[best].dy}) score=${o.scales[best].score.toFixed(3)}`);
          }
          if (!out4[q] || o.scales[best].score > out4[q].scales[out4[q].best].score) out4[q] = { scales: o.scales, best };
        });
        rs.forEach((r, q) => {
          const [i, j] = pairs[r];
          const cand = out4[q].scales;
          const best = out4[q].best;
          const c = cand[best];
          let ls = Math.log(c.s_eff);
          if (best > 0 && best < cand.length - 1) {
            const st = Math.log(cand[best + 1].s_eff) - Math.log(cand[best].s_eff);
            ls += subpix(cand[best - 1].score, c.score, cand[best + 1].score) * st;
          }
          const sdx = (c.l > -1.5 && c.r > -1.5) ? subpix(c.l, c.score, c.r) : 0;
          const sdy = (c.u > -1.5 && c.d > -1.5) ? subpix(c.u, c.score, c.d) : 0;
          const s_ref = Math.exp(ls);
          const dx = c.dx + sdx + 0.5 * (c.s_eff - 1.0), dy = c.dy + sdy + 0.5 * (c.s_eff - 1.0);
          const corr = 0.5 * (s_ref - 1.0) * (1.0 - 1.0 / this.sx);
          pc[r * 4] = scaleModel ? s_ref : 1.0; pc[r * 4 + 1] = 0;
          pc[r * 4 + 2] = dx / this.sx + (scaleModel ? corr : 0);
          pc[r * 4 + 3] = dy / this.sy + (scaleModel ? corr : 0);
          sc[r] = c.score;
          if (scaleModel && j - i === minOff) chain[j] = chain[i] + Math.log(s_ref);
        });
        done += rs.length;
        this.progress("coarse", done / m);
        this._check();
      }
    }
    return { pc, sc };
  }

  // ------------------------------------------------------------ 精密推定（平行移動: フル解像度窓探索）
  async fineTranslation(jobs) {
    // jobs: [{i, j, dx, dy, r}] → [{ok, dx, dy, score}]
    const out = new Array(jobs.length);
    const BATCH = 8;
    for (let b0 = 0; b0 < jobs.length; b0 += BATCH) {
      const sub = jobs.slice(b0, b0 + BATCH);
      const keep = new Set(); sub.forEach((jb) => { keep.add(jb.i); keep.add(jb.j); });
      this._evictLevels(keep);
      const sjobs = []; const idx = [];
      for (let q = 0; q < sub.length; q++) {
        const jb = sub[q];
        const H = this.H, W = this.W;
        const by0 = Math.max(0, -jb.dy), by1 = Math.min(H, H - jb.dy), bx0 = Math.max(0, -jb.dx), bx1 = Math.min(W, W - jb.dx);
        if (by1 - by0 < 4 * jb.r + 16 || bx1 - bx0 < 4 * jb.r + 16) { out[b0 + q] = { ok: false }; continue; }
        const Li = (await this._levels(jb.i))[1.0], Lj = (await this._levels(jb.j))[1.0];
        sjobs.push({ A: Li.buf, B: Lj.buf, dx: jb.dx, dy: jb.dy, r: jb.r, nmin: 0.1 * (by1 - by0) * (bx1 - bx0) });
        idx.push(b0 + q);
      }
      const K = this.k, g = this.gpu;
      sjobs.forEach((sj, slot) => {
        const nx = 2 * sj.r + 1;
        K.ncc_search.run({ hA: this.H, wA: this.W, hB: this.H, wB: this.W, dx0: sj.dx - sj.r, dy0: sj.dy - sj.r, nx, ny: nx, nmin: sj.nmin, out_off: 0 },
          [sj.A, sj.B, this.surf], nx, nx);
        K.argmax.run({ in_off: 0, n: nx * nx, nx, out_off: slot * 8 }, [this.surf, this.resBuf], 1);
      });
      if (sjobs.length) {
        const res = new Float32Array(await g.read(this.resBuf, sjobs.length * 8 * 4));
        sjobs.forEach((sj, slot) => {
          const o = slot * 8;
          const score = res[o];
          const sdx = (res[o + 2] > -1.5 && res[o + 3] > -1.5) ? subpix(res[o + 2], score, res[o + 3]) : 0;
          const sdy = (res[o + 4] > -1.5 && res[o + 5] > -1.5) ? subpix(res[o + 4], score, res[o + 5]) : 0;
          out[idx[slot]] = { ok: score > -1.5, dx: sj.dx - sj.r + res[o + 6] + sdx, dy: sj.dy - sj.r + res[o + 7] + sdy, score };
        });
      }
      this.progress("fine", Math.min(1, (b0 + sub.length) / jobs.length));
      this._check();
    }
    return out;
  }

  // ------------------------------------------------------------ 精密推定（Gauss-Newton）
  async gnBatch(jobs) {
    // jobs: [{i, j, p0: [ls, th, tx, ty]}] → [{ok, p: [ls, th, tx, ty], score, nValid}]
    const a = this.a, K = this.k, g = this.gpu;
    const out = new Array(jobs.length);
    const { lvs } = this._levelSpec();
    const BATCH = 8;
    const model = MODEL_ID[a.model];
    for (let b0 = 0; b0 < jobs.length; b0 += BATCH) {
      const sub = jobs.slice(b0, b0 + BATCH);
      const keep = new Set(); sub.forEach((jb) => { keep.add(jb.i); keep.add(jb.j); });
      this._evictLevels(keep);
      const st = new Float32Array(sub.length * 16);
      sub.forEach((jb, q) => { st.set(jb.p0, q * 16); });
      // 各スロット: (ls, th, tx, ty, score, n_valid, fail, conv)
      g.upload(this.state, st);
      for (let slot = 0; slot < sub.length; slot++) {
        const jb = sub[slot];
        const Li = await this._levels(jb.i), Lj = await this._levels(jb.j);
        for (const lv of lvs) {
          const A = Li[lv], B = Lj[lv];
          const npx = A.hs * A.ws;
          const nwg = Math.max(1, Math.min(this.NWG, Math.ceil(npx / (128 * 8))));
          const prm = { hs: A.hs, ws: A.ws, hb: B.hs, wb: B.ws, lv, slot, nbins: this.NB, rmax: 1.0, model, nwg, last: 0 };
          const bufs = [A.buf, B.buf, this.hist, this.partials, this.state];
          for (let it = 0; it < a.gnIters; it++) {
            prm.last = it === a.gnIters - 1 ? 1 : 0;
            g.clear(this.hist);
            K.gn_resid.run(prm, bufs, nwg);
            K.gn_accum.run(prm, bufs, nwg);
            K.gn_solve.run(prm, bufs, 1);
          }
        }
      }
      const res = new Float32Array(await g.read(this.state, sub.length * 16 * 4));
      sub.forEach((_jb, q) => {
        const o = q * 16;
        const ok = res[o + 6] < 0.5 && Number.isFinite(res[o]) && Number.isFinite(res[o + 2]) && Number.isFinite(res[o + 3]);
        out[b0 + q] = { ok, p: [res[o], res[o + 1], res[o + 2], res[o + 3]], score: res[o + 4], nValid: res[o + 5] };
        if (!ok && this._dbgFail < 3) { this._dbgFail++; this.log(`  [gn debug] pair (${sub[q].i},${sub[q].j}) p0=${sub[q].p0.map((v) => v.toFixed(3))} state=${Array.from(res.slice(o, o + 8)).map((v) => v.toFixed(3))}`); }
      });
      this.progress("fine", Math.min(1, (b0 + sub.length) / jobs.length));
      this._check();
    }
    return out;
  }

  /** ジョブ {i, j, ...} をフレーム順（i, j 昇順）に並べ替えて実行し、元の順で返す（ストリーミング時の読み直しを減らす） */
  async _inFrameOrder(jobs, fn) {
    const order = jobs.map((_, r) => r).sort((p, q) => jobs[p].i - jobs[q].i || jobs[p].j - jobs[q].j);
    const out = await fn(order.map((r) => jobs[r]));
    const res = new Array(jobs.length);
    order.forEach((r, q) => { res[r] = out[q]; });
    return res;
  }

  /** 粗探索データ（1/4, 1/16 レベル）を解放する。位置合わせ完了後に呼ぶ */
  freeCoarse() {
    const g = this.gpu;
    for (const arr of [this.C4, this.C16]) for (let k = 0; k < this.n; k++) if (arr[k]) { g.free(arr[k]); arr[k] = null; }
    for (const e of this.levelCache.values()) for (const lv of Object.keys(e.levels)) g.free(e.levels[lv].buf);
    this.levelCache.clear(); this.levelBytes = 0;
  }

  // ------------------------------------------------------------ 精密推定（ホモグラフィ + 任意で放射歪み k1、Gauss-Newton）
  async gnhBatch(jobs, useK1 = false, levelsOverride = null) {
    // jobs: [{i, j, G0: Float64Array(9) (正規化座標 x_j = G x_i), k1?}] → [{ok, G: Float64Array(9), k1, score, nValid}]
    const a = this.a, K = this.k, g = this.gpu;
    const out = new Array(jobs.length);
    const lvs = levelsOverride || this._levelSpec().lvs;
    const BATCH = 8;
    const fFull = 0.5 * Math.max(this.W, this.H), cxF = 0.5 * (this.W - 1), cyF = 0.5 * (this.H - 1);
    const kapScale = (fFull / Math.hypot(cxF, cyF)) ** 2;
    for (let b0 = 0; b0 < jobs.length; b0 += BATCH) {
      const sub = jobs.slice(b0, b0 + BATCH);
      const keep = new Set(); sub.forEach((jb) => { keep.add(jb.i); keep.add(jb.j); });
      this._evictLevels(keep);
      const st = new Float32Array(sub.length * 16);
      sub.forEach((jb, q) => { for (let t = 0; t < 8; t++) st[q * 16 + t] = jb.G0[t] / jb.G0[8]; st[q * 16 + 12] = jb.k1 || 0; });
      g.upload(this.state, st);
      for (let slot = 0; slot < sub.length; slot++) {
        const jb = sub[slot];
        const Li = await this._levels(jb.i), Lj = await this._levels(jb.j);
        for (const lv of lvs) {
          const A = Li[lv], B = Lj[lv];
          const npx = A.hs * A.ws;
          const nwg = Math.max(1, Math.min(this.NWG, Math.ceil(npx / (64 * 16))));
          const prm = { hs: A.hs, ws: A.ws, hb: B.hs, wb: B.ws, lv, slot, nbins: this.NB, rmax: 1.0, nwg, last: 0, use_k1: useK1 ? 1 : 0,
                        f_full: fFull, cx_full: cxF, cy_full: cyF, kap_scale: kapScale };
          const bufs = [A.buf, B.buf, this.hist, this.partials, this.state];
          for (let it = 0; it < a.gnIters; it++) {
            prm.last = it === a.gnIters - 1 ? 1 : 0;
            g.clear(this.hist);
            K.gnh_resid.run(prm, bufs, nwg);
            K.gnh_accum.run(prm, bufs, nwg);
            K.gnh_solve.run(prm, bufs, 1);
          }
        }
      }
      const res = new Float32Array(await g.read(this.state, sub.length * 16 * 4));
      sub.forEach((_jb, q) => {
        const o = q * 16;
        const G = Float64Array.from(res.subarray(o, o + 8)); const Gm = new Float64Array(9); Gm.set(G); Gm[8] = 1;
        const ok = res[o + 10] < 0.5 && Gm.every(Number.isFinite);
        out[b0 + q] = { ok, G: Gm, k1: res[o + 12], score: res[o + 8], nValid: res[o + 9] };
        if (!ok && this._dbgFail < 3) { this._dbgFail++; this.log(`  [gnh debug] pair (${sub[q].i},${sub[q].j}) state=${Array.from(res.slice(o, o + 13)).map((v) => v.toFixed(3))}`); }
      });
      this.progress("fine", Math.min(1, (b0 + sub.length) / jobs.length));
      this._check();
    }
    return out;
  }

  /** 粗推定 (s, th, tx, ty)（x_i = s R x_j + t、画素座標）→ 正規化座標の G（x_j = G x_i） */
  _simToG(s, th, tx, ty) {
    const N = normMatrix(this.W, this.H);
    return mat3mul(mat3mul(N, mat3inv(simMatrix(s, th, tx, ty))), mat3inv(N));
  }
  /** 画素座標のペアホモグラフィ → 概略の (s, th, tx, ty)（CSV / ログ用） */
  _hToPf(Hpix) {
    return [localScaleOfH(Hpix, 0.5 * (this.W - 1), 0.5 * (this.H - 1)), Math.atan2(Hpix[3], Hpix[0]), Hpix[2], Hpix[5]];
  }

  // ------------------------------------------------------------ 実写向け: ブレンドフレーム / 相対鮮明度 / レンズ歪み
  /**
   * ブレンドフレーム（フレームレート変換で前後 2 フレームが位置合わせなしに混ざった二重像）を検出し frameTier に 2 を立てる。
   * 粗レベルハイパスで hp_k ≈ α hp_{k-1} + (1-α) hp_{k+1} を当てはめ、残差 ρ = |hp_k - fit| / (|a-b|/2) < blendThr なら判定
   * （通常のフレームは ρ ≈ 1、複製は α ≈ 0/1 で除外）。位置合わせの前に呼ぶ（align はそのペアの重みを落とす）
   */
  async detectBlendFrames() {
    const a = this.a, K = this.k, g = this.gpu, n = this.n;
    this.blendRho = new Float64Array(n).fill(NaN);
    this.blendAlpha = new Float64Array(n).fill(NaN);
    this.frameTier = new Int32Array(n);
    if (n < 3) return;
    const npx = this.wc * this.hc;
    const nwg = Math.max(1, Math.min(256, Math.ceil(npx / (256 * 8))));
    const outb = g.buf(nwg * 10 * 4, "moments");
    for (let k = 1; k < n - 1; k++) {
      K.moments3.run({ n: npx, nwg }, [this.C4[k - 1], this.C4[k], this.C4[k + 1], outb], nwg);
      const part = new Float32Array(await g.read(outb, nwg * 10 * 4));
      const m = new Float64Array(10);
      for (let w = 0; w < nwg; w++) for (let q = 0; q < 10; q++) m[q] += part[w * 10 + q];
      const [cnt, sa, sb, sc, sab, sac, sbc, saa, sbb, scc] = m;
      if (cnt < 0.1 * npx) continue;
      // d = a - b, e = c - b: Σd² = Σaa - 2Σab + Σbb, Σed = Σac - Σbc - Σab + Σbb, Σe² = Σcc - 2Σbc + Σbb
      const sdd = saa - 2 * sab + sbb, sed = sac - sbc - sab + sbb, see = scc - 2 * sbc + sbb;
      if (sdd < 1e-6 || Math.sqrt(sdd / cnt) < 0.004) continue;
      const al = sed / sdd;
      const res2 = Math.max(0, see - 2 * al * sed + al * al * sdd);
      const rho = Math.sqrt(res2 / cnt) / (0.5 * Math.sqrt(sdd / cnt));
      this.blendRho[k] = rho; this.blendAlpha[k] = al;
      if (rho < a.blendThr && al > 0.15 && al < 0.85) this.frameTier[k] = 2;
    }
    g.free(outb);
    const idx = [];
    for (let k = 0; k < n; k++) if (this.frameTier[k] === 2) idx.push(k);
    this.log(`[blend] ブレンドフレーム（ρ < ${a.blendThr}）${idx.length} frames` + (idx.length ? `: ${idx.slice(0, 20).join(",")}（位置合わせの重みを下げ、合成から外します）` : ""));
  }

  /**
   * フレームごとの相対鮮明度: 位置合わせ済みペアの重なりで鮮明度マップ（1/4）の平均の比 log(s_j / s_i) を観測し、
   * log 鮮明度 q_k を IRLS で解く。近傍 ±qualityWindow の中央値に対する比が qualityThr 未満なら frameTier = 1（ブレンド = 2 は保持）
   */
  async estimateQuality(al) {
    const a = this.a, K = this.k, g = this.gpu, n = this.n;
    const t0 = now();
    if (!this.frameTier) this.frameTier = new Int32Array(n);
    this.qualityLog = new Float64Array(n);
    this.qualityRatio = new Float64Array(n).fill(1);
    const doQ = a.quality === "on";
    let nObs = 0;
    if (doQ) {
      // C4 は align 後に解放されているので、ここでは鮮明度マップと C4 valid が必要 → align が freeCoarse を遅らせる（keepCoarse）
      const nwg = Math.max(1, Math.min(256, Math.ceil(this.Wq * this.Hq / (256 * 8))));
      const outb = g.buf(nwg * 4 * 4, "quality");
      const cx = 0.5 * (this.W - 1), cy = 0.5 * (this.H - 1);
      const obsPairs = [], dObs = [], wObs = [];
      for (let r = 0; r < al.pairs.length; r++) {
        if (al.sc[r] < a.exposureMinScore) continue;
        const [i, j] = al.pairs[r];
        const M = al.pm[r];                                     // x_i = M x_j
        if (Math.abs(Math.log(Math.max(1e-6, localScaleOfH(M, cx, cy)))) > 0.05) continue;
        const Mi = mat3inv(M);                                  // x_j = Mi x_i
        K.quality_pair.run({ Wq: this.Wq, Hq: this.Hq, W: this.W, H: this.H, wc: this.wc, hc: this.hc,
                             off_i: i * this.Hq * this.Wq, off_j: j * this.Hq * this.Wq, nwg,
                             m0: [Mi[0], Mi[1], Mi[2], 0], m1: [Mi[3], Mi[4], Mi[5], 0], m2: [Mi[6], Mi[7], Mi[8], 0] },
                           [this.sharpq, this.C4[i], this.C4[j], outb], nwg);
        const part = new Float32Array(await g.read(outb, nwg * 4 * 4));
        let cnt = 0, si = 0, sj = 0;
        for (let w = 0; w < nwg; w++) { cnt += part[w * 4]; si += part[w * 4 + 1]; sj += part[w * 4 + 2]; }
        if (cnt < 0.05 * this.Wq * this.Hq || si <= 1e-6 * cnt || sj <= 1e-6 * cnt) continue;
        obsPairs.push([i, j]); dObs.push(Math.log(sj / cnt) - Math.log(si / cnt)); wObs.push(Math.sqrt(cnt / (this.Wq * this.Hq)));
        if (r % 32 === 31) this._check();
      }
      g.free(outb);
      nObs = obsPairs.length;
      if (nObs >= n - 1) {
        const sol = solvePositions(n, obsPairs, Float64Array.from(dObs), Float64Array.from(wObs), 1, 5, 0.15);
        const med = median(sol.pos);
        for (let k = 0; k < n; k++) this.qualityLog[k] = sol.pos[k] - med;
        const Wn = Math.max(1, a.qualityWindow | 0);
        for (let k = 0; k < n; k++) {
          const lo = Math.max(0, k - Wn), hi = Math.min(n, k + Wn + 1);
          this.qualityRatio[k] = Math.exp(this.qualityLog[k] - median(this.qualityLog.slice(lo, hi)));
          if (this.qualityRatio[k] < a.qualityThr && this.frameTier[k] === 0) this.frameTier[k] = 1;
        }
      } else {
        this.log(`[quality] 鮮明度の比較ができるペアが少ない（${nObs} 件）ため、ボケ / ブレの判定は行いません`);
      }
    }
    const t1 = [], t2 = [];
    for (let k = 0; k < n; k++) { if (this.frameTier[k] === 1) t1.push(k); else if (this.frameTier[k] === 2) t2.push(k); }
    let msg = "[quality]";
    if (doQ) msg += ` 相対鮮明度: ${nObs} 観測, 比 ${Math.min(...this.qualityRatio).toFixed(2)}..${Math.max(...this.qualityRatio).toFixed(2)}, ` +
                    `ボケ / ブレ判定（< ${a.qualityThr}）${t1.length} frames` + (t1.length ? `: ${t1.slice(0, 20).join(",")}` : "");
    if (a.blendDetect === "on") msg += `; ブレンドフレーム ${t2.length} frames` + (t2.length ? `: ${t2.slice(0, 20).join(",")}` : "");
    this.log(`${msg}, ${(now() - t0).toFixed(1)}s`);
    if (t1.length + t2.length) this.log(`  段階の高いフレームは、良いフレームだけで ${a.qualityMin} 標本以上ある画素では合成に使いません（足りない画素では使います）`);
  }

  /** quality.csv */
  qualityCsv() {
    if (!this.frameTier) return "";
    let s = "frame,log_sharp,ratio_to_local,blend_rho,blend_alpha,tier\n";
    const f = (v, d) => (v === undefined || Number.isNaN(v) ? "" : v.toFixed(d));
    for (let k = 0; k < this.n; k++) {
      s += `${k},${f(this.qualityLog ? this.qualityLog[k] : 0, 4)},${f(this.qualityRatio ? this.qualityRatio[k] : 1, 3)},` +
           `${f(this.blendRho ? this.blendRho[k] : NaN, 3)},${f(this.blendAlpha ? this.blendAlpha[k] : NaN, 3)},${this.frameTier[k]}\n`;
    }
    return s;
  }

  /**
   * 放射歪み k1 の推定: 間隔の大きいペアを数組選び、ホモグラフィと k1 を同時に GN で解いて（1/4 → 1/2）、スコアの高いペアの
   * k1 の中央値を採る。ばらつきが大きい / 小さすぎるときは 0（Python 版 search_lens_k1 と同じ）
   */
  async searchLensK1() {
    const a = this.a, n = this.n;
    const t0 = now();
    const koff = Math.max(...a.pairs);
    if (n <= koff) return 0;
    const nsel = Math.min(8, n - koff);
    const sel = [...new Set(Array.from({ length: nsel }, (_, q) => Math.round(q * (n - 1 - koff) / Math.max(1, nsel - 1))))].map((s) => [s, s + koff]);
    const { pc, sc } = await this.coarse(sel);
    const jobs = [], meta = [];
    sel.forEach(([i, j], r) => {
      if (sc[r] < 0.2) return;
      jobs.push({ i, j, G0: this._simToG(pc[r * 4], pc[r * 4 + 1], pc[r * 4 + 2], pc[r * 4 + 3]), k1: 0 });
      meta.push({ i, j, motion: Math.hypot(pc[r * 4 + 2], pc[r * 4 + 3]) });
    });
    const lvs = [0.25, 0.5].filter((lv) => lv <= Math.max(0.5, a.fineScale) + 1e-9);
    const res = await this._inFrameOrder(jobs, (js) => this.gnhBatch(js, true, lvs));
    const ests = [];
    res.forEach((rs, q) => { if (rs.ok && rs.score >= 0.3) ests.push({ k1: rs.k1, pair: [meta[q].i, meta[q].j] }); });
    if (ests.length < 3) { this.log(`[lens] k1 を推定できるペアが少ない（${ests.length} 件）ため歪み補正は行いません, ${(now() - t0).toFixed(1)}s`); return 0; }
    const ks = Float64Array.from(ests.map((e) => e.k1)).sort();
    const k1 = median(ks), q1 = ks[Math.floor(0.25 * (ks.length - 1))], q3 = ks[Math.ceil(0.75 * (ks.length - 1))];
    this.log(`[lens] k1 の同時推定: ${ests.length} ペア, median=${k1.toFixed(4)} (四分位 ${q1.toFixed(4)}..${q3.toFixed(4)}), ` +
             ests.map((e) => `(${e.pair[0]},${e.pair[1]})=${e.k1.toFixed(3)}`).join(" ") + `, ${(now() - t0).toFixed(1)}s`);
    if (q3 - q1 > Math.max(0.02, 0.5 * Math.abs(k1))) { this.log("[lens] ペア間のばらつきが大きいため歪み補正は行いません（k1 = 0）"); return 0; }
    if (Math.abs(k1) < 0.015) { this.log("[lens] 歪みは無視できる大きさです（k1 = 0）"); return 0; }
    return k1;
  }

  /** 位置の定まらないフレーム（ブレンド: ペアの重みが全て 1e-3）の値を前後の良いフレームの平均で置き換える（3x3 は正規化 / 直交化） */
  static interpBadFrames(bad, arrays, stride = 1, kind = "") {
    const n = bad.length;
    const good = [];
    for (let k = 0; k < n; k++) if (!bad[k]) good.push(k);
    if (!good.length) return;
    for (let k = 0; k < n; k++) {
      if (!bad[k]) continue;
      const prev = good.filter((g) => g < k).pop(), next = good.find((g) => g > k);
      for (const arr of arrays) {
        const isMat = Array.isArray(arr) && arr[0] instanceof Float64Array && arr[0].length === 9;
        if (isMat) {
          const a = arr[prev ?? next], b = arr[next ?? prev];
          const m = Float64Array.from(a, (v, q) => 0.5 * (v + b[q]));
          arr[k] = kind === "rot" ? orthonormalize(m) : mat3norm(m);
        } else {
          for (let c = 0; c < stride; c++) {
            const a = arr[(prev ?? next) * stride + c], b = arr[(next ?? prev) * stride + c];
            arr[k * stride + c] = 0.5 * (a + b);
          }
        }
      }
    }
  }

  /**
   * ホモグラフィモデルの投影面を決める（Python 版 _choose_projection と同じ）。cylindrical / auto では焦点距離を推定し
   * 回転平均で各フレームの回転を解いて円筒投影にする。auto: パン角が cylMinSpan 度を超えたときだけ
   */
  _chooseProjection(al, pairs, pm, sf, badFrame) {
    const a = this.a, n = this.n, W = this.W, H = this.H;
    if (a.projection === "planar") return al;
    const goodPair = pairs.map(([i, j]) => !(badFrame[i] || badFrame[j]));
    const rnH = al.rn.filter((_, r) => goodPair[r]);
    const cx = 0.5 * (W - 1), cy = 0.5 * (H - 1);
    const C = Float64Array.from([1, 0, -cx, 0, 1, -cy, 0, 0, 1]), Ci = mat3inv(C);
    const fs = [];
    for (let r = 0; r < pairs.length; r++) {
      if (sf[r] < 0.3) continue;
      const [f1, f0] = focalsFromHomography(mat3mul(mat3mul(C, pm[r]), Ci));
      const lo = 0.2 * Math.max(W, H), hi = 20 * Math.max(W, H);
      if (f1 !== null && f0 !== null && f1 > lo && f1 < hi && f0 > lo && f0 < hi) fs.push(Math.sqrt(f0 * f1));
    }
    let f;
    if (a.focal > 0) { f = a.focal; this.log(`[projection] 焦点距離 ${f.toFixed(1)} px（指定）`); }
    else if (fs.length >= Math.max(3, Math.floor(n / 8))) {
      const s = Float64Array.from(fs).sort();
      f = median(s);
      this.log(`[projection] 焦点距離の推定: ${f.toFixed(1)} px（${fs.length} ペア、四分位 ${s[Math.floor(0.25 * (s.length - 1))].toFixed(0)}-${s[Math.ceil(0.75 * (s.length - 1))].toFixed(0)}、` +
               `水平画角 ${(2 * Math.atan(0.5 * W / f) * 180 / Math.PI).toFixed(1)}°）`);
    } else { this.log(`[projection] 焦点距離を推定できるペアが少ない（${fs.length} 件）ため平面投影を使います`); return al; }
    const Km = Float64Array.from([f, 0, cx, 0, f, cy, 0, 0, 1]), Ki = mat3inv(Km);
    const Rp = pairs.map((_, r) => orthonormalize(mat3mul(mat3mul(Ki, pm[r]), Km)));
    const w = Float64Array.from(sf, (s) => Math.min(1, Math.max(0.05, s)) ** 2);
    const { R, rn } = solveRotations(n, pairs, Rp, w);
    if (badFrame.some(Boolean)) Reconstructor.interpBadFrames(badFrame, [R], 1, "rot");
    const rnPx = Float64Array.from(rn.filter((_, r) => goodPair[r]), (v) => v * f);
    const yaw = [], pitch = [];
    for (let k = 0; k < n; k++) { const ax = [R[k][2], R[k][5], R[k][8]]; yaw.push(Math.atan2(ax[0], ax[2]) * 180 / Math.PI); pitch.push(Math.asin(Math.max(-1, Math.min(1, ax[1]))) * 180 / Math.PI); }
    const span = Math.max(Math.max(...yaw) - Math.min(...yaw), Math.max(...pitch) - Math.min(...pitch));
    this.log(`[projection] 回転モデルの残差 median=${median(rnPx).toFixed(2)}px max=${Math.max(...rnPx).toFixed(2)}px（ホモグラフィ median=${median(rnH).toFixed(2)}px）、` +
             `パン角 yaw ${Math.min(...yaw).toFixed(1)}..${Math.max(...yaw).toFixed(1)}° pitch ${Math.min(...pitch).toFixed(1)}..${Math.max(...pitch).toFixed(1)}°`);
    const parallax = median(rnPx) > Math.max(1.0, 3.0 * median(rnH));
    if (parallax) this.log("  注意: 回転だけでは説明できない残差があります（カメラの平行移動 = 視差）");
    if (a.projection === "auto") {
      if (span < a.cylMinSpan) { this.log(`[projection] パン角 ${span.toFixed(1)}° < ${a.cylMinSpan}° なので平面投影のままにします`); return al; }
      if (parallax && span < 2 * a.cylMinSpan) { this.log("[projection] 視差があるため平面投影のままにします（projection = cylindrical で強制できます）"); return al; }
    }
    this.log(`[projection] 円筒投影（半径 = 焦点距離 ${f.toFixed(1)} px）`);
    const T = new Float64Array(n * 2), TH = new Float64Array(n), S = new Float64Array(n).fill(1);
    const toCanvas = (k, x, y) => {
      const d = [Ki[0] * x + Ki[1] * y + Ki[2], Ki[3] * x + Ki[4] * y + Ki[5], 1];
      const Rk = R[k];
      const dw = [Rk[0] * d[0] + Rk[1] * d[1] + Rk[2] * d[2], Rk[3] * d[0] + Rk[4] * d[1] + Rk[5] * d[2], Rk[6] * d[0] + Rk[7] * d[1] + Rk[8] * d[2]];
      return [f * Math.atan2(dw[0], dw[2]), f * dw[1] / Math.hypot(dw[0], dw[2])];
    };
    for (let k = 0; k < n; k++) { const c = toCanvas(k, cx, cy); T[k * 2] = c[0]; T[k * 2 + 1] = c[1]; TH[k] = Math.atan2(R[k][3], R[k][0]); }
    for (let k = n - 1; k >= 0; k--) { T[k * 2] -= T[0]; T[k * 2 + 1] -= T[1]; }
    return { ...al, kind: "cyl", S, TH, T, R, K: Km, fc: f, focal: f, yaw, pitch };
  }

  // ------------------------------------------------------------ 位置合わせ全体
  async align() {
    const a = this.a, n = this.n;
    const pairs = [];
    for (const k of a.pairs) for (let i = 0; i < n - k; i++) pairs.push([i, i + k]);
    const m = pairs.length;
    const model = a.model;
    const homog = model === "homography";
    let t0 = now();
    const { pc, sc } = await this.coarse(pairs);
    const wc = new Float64Array(m);
    for (let r = 0; r < m; r++) wc[r] = Math.min(1, Math.max(0.05, sc[r])) ** 2;
    // ブレンドフレーム（二重像）を含むペアは位置が定まらないので重みを落とす（k=2 のペアで連鎖はつながる）
    const badFrame = this.frameTier ? Array.from(this.frameTier, (t) => t === 2) : new Array(n).fill(false);
    const pairBad = pairs.map(([i, j]) => badFrame[i] || badFrame[j]);
    for (let r = 0; r < m; r++) if (pairBad[r]) wc[r] *= 1e-3;
    // 離れたペア（k > 最小間隔）の粗推定が隣接ペアの連鎖と矛盾していれば重みを落とす（Python 版と同じ）。
    // 重なりの無いペアでも滑らかな輪郭が偶然合って高い NCC が出ることがあり（静止画列で顕著）、そのままグローバル解に
    // 入れると正しい隣接ペアまで「不整合」扱いになって精密化の初期値が壊れる
    const kMin = Math.min(...a.pairs);
    // 連鎖 = 最小間隔のペア + ブレンドフレームを跨ぐ 2 倍間隔のペア（ブレンドフレームのペアは重みが無いので、跨がないと連鎖が切れる）
    const chain = [];
    for (let r = 0; r < m; r++) {
      const off = pairs[r][1] - pairs[r][0];
      if (off === kMin || (off === 2 * kMin && !pairBad[r] && badFrame[pairs[r][0] + kMin])) chain.push(r);
    }
    const chainSet = new Set(chain);
    let nFarBad = 0;
    if (chain.length < m && chain.length >= n - 1) {
      const pcC = new Float64Array(chain.length * 4), wcC = new Float64Array(chain.length);
      chain.forEach((r, q) => { pcC.set(pc.subarray(r * 4, r * 4 + 4), q * 4); wcC[q] = wc[r]; });
      const gh = solveGlobal(n, chain.map((r) => pairs[r]), pcC, wcC, model);
      for (let r = 0; r < m; r++) {
        if (chainSet.has(r) || pairBad[r]) continue;
        const [i, j] = pairs[r];
        const dp = mulRot(rot2(gh.TH[i]), [pc[r * 4 + 2], pc[r * 4 + 3]]).map((v) => gh.S[i] * v);
        const err = Math.hypot(dp[0] - (gh.T[j * 2] - gh.T[i * 2]), dp[1] - (gh.T[j * 2 + 1] - gh.T[i * 2 + 1]));
        const tolFar = Math.max(32, 8 * (j - i) / this.sx);
        if (err > tolFar) { wc[r] *= 1e-3; nFarBad++; }
      }
    }
    const gc = solveGlobal(n, pairs, pc, wc, model);
    const tol = a.coarseTol / this.sx;
    let nBad = 0;
    for (let r = 0; r < m; r++) if (gc.rn[r] > tol) nBad++;
    this.log(`[coarse] ${m} pairs, model=${model}, score mean=${mean(sc).toFixed(3)}, ` +
             `residual median=${(median(gc.rn) * this.sx).toFixed(2)}px (coarse), inconsistent pairs=${nBad}` +
             (nFarBad ? `, far pairs inconsistent with chain=${nFarBad}` : "") + `, ${(now() - t0).toFixed(1)}s`);

    t0 = now();
    const pf = new Float64Array(m * 4), sf = new Float64Array(m);
    const ok = new Array(m).fill(true);
    const Hn = new Array(m).fill(null);          // homography: ペア変換 x_i = Hn x_j（正規化座標）
    const N = normMatrix(this.W, this.H), Ni = mat3inv(N);
    const fFull = 0.5 * Math.max(this.W, this.H);
    if (model === "translation") {
      const rSmall = Math.ceil(1.0 / a.coarseScale) + 2, rBig = Math.max(8, Math.ceil(3.0 / a.coarseScale));
      const jobs = pairs.map(([i, j], r) => {
        let d0, rr;
        if (gc.rn[r] <= tol) { d0 = [pc[r * 4 + 2], pc[r * 4 + 3]]; rr = rSmall; }
        else { d0 = [gc.T[j * 2] - gc.T[i * 2], gc.T[j * 2 + 1] - gc.T[i * 2 + 1]]; rr = rBig; }
        return { i, j, dx: Math.round(d0[0]), dy: Math.round(d0[1]), r: rr };
      });
      const res = await this._inFrameOrder(jobs, (js) => this.fineTranslation(js));
      res.forEach((rs, r) => {
        if (!rs.ok) { ok[r] = false; return; }
        pf[r * 4] = 1; pf[r * 4 + 1] = 0; pf[r * 4 + 2] = rs.dx; pf[r * 4 + 3] = rs.dy; sf[r] = rs.score;
      });
    } else {
      const p0s = pairs.map(([i, j], r) => {
        if (gc.rn[r] <= tol) return [Math.log(pc[r * 4]), pc[r * 4 + 1], pc[r * 4 + 2], pc[r * 4 + 3]];
        const relS = gc.S[j] / gc.S[i], relTh = gc.TH[j] - gc.TH[i];
        const dT = [gc.T[j * 2] - gc.T[i * 2], gc.T[j * 2 + 1] - gc.T[i * 2 + 1]];
        const relT = mulRot(rot2(-gc.TH[i]), dT).map((v) => v / gc.S[i]);
        return [Math.log(relS), relTh, relT[0], relT[1]];
      });
      if (homog) {
        const jobs = pairs.map(([i, j], r) => ({ i, j, G0: this._simToG(Math.exp(p0s[r][0]), p0s[r][1], p0s[r][2], p0s[r][3]) }));
        const res = await this._inFrameOrder(jobs, (js) => this.gnhBatch(js));
        res.forEach((rs, r) => {
          if (!rs.ok) { ok[r] = false; return; }
          Hn[r] = mat3norm(mat3inv(rs.G));
          pf.set(this._hToPf(mat3mul(mat3mul(Ni, Hn[r]), N)), r * 4); sf[r] = rs.score;
        });
      } else {
        const jobs = pairs.map(([i, j], r) => ({ i, j, p0: p0s[r] }));
        const res = await this._inFrameOrder(jobs, (js) => this.gnBatch(js));
        res.forEach((rs, r) => {
          if (!rs.ok) { ok[r] = false; return; }
          pf[r * 4] = Math.exp(rs.p[0]); pf[r * 4 + 1] = rs.p[1]; pf[r * 4 + 2] = rs.p[2]; pf[r * 4 + 3] = rs.p[3]; sf[r] = rs.score;
        });
      }
    }
    const pairsOk = [], pfOk = [], sfOk = [], HnOk = [], badOk = [];
    for (let r = 0; r < m; r++) if (ok[r]) { pairsOk.push(pairs[r]); pfOk.push(pf.subarray(r * 4, r * 4 + 4)); sfOk.push(sf[r]); HnOk.push(Hn[r]); badOk.push(pairBad[r]); }
    const mo = pairsOk.length;
    if (mo < n - 1) this.log(`  警告: 有効ペアが少なすぎます (${mo} / ${m})`);
    if (mo === 0) throw new Error("有効なペアがありません");
    const PF = new Float64Array(mo * 4); pfOk.forEach((p, r) => PF.set(p, r * 4));
    const SF = Float64Array.from(sfOk);
    const weights = () => SF.map((s, r) => Math.min(1, Math.max(0.05, s)) ** 2 * (badOk[r] ? 1e-3 : 1));
    let sol, Hk = null;
    const solveAll = () => {
      if (homog) {
        const gh = solveGlobalH(n, pairsOk, HnOk, weights(), this.W, this.H);
        Hk = gh.Hk;
        const S = new Float64Array(n), TH = new Float64Array(n), T = new Float64Array(n * 2);
        const cx = 0.5 * (this.W - 1), cy = 0.5 * (this.H - 1);
        for (let k = 0; k < n; k++) {
          const Hp = mat3mul(mat3mul(Ni, Hk[k]), N);
          S[k] = localScaleOfH(Hp, cx, cy); TH[k] = Math.atan2(Hp[3], Hp[0]);
          const c = applyH(Hp, cx, cy); T[k * 2] = c[0] - cx; T[k * 2 + 1] = c[1] - cy;
        }
        return { S, TH, T, rn: Float64Array.from(gh.rn, (v) => v * fFull) };
      }
      return solveGlobal(n, pairsOk, PF, weights(), model);
    };
    sol = solveAll();
    if (model !== "translation") {
      const redo = [];
      for (let k = 0; k < mo; k++) if (sol.rn[k] > 1.5 && !badOk[k]) redo.push(k);
      let nFixed = 0;
      if (redo.length) {
        if (homog) {
          const jobs = redo.map((k) => { const [i, j] = pairsOk[k]; return { i, j, G0: mat3mul(mat3inv(Hk[j]), Hk[i]) }; });
          const res = await this._inFrameOrder(jobs, (js) => this.gnhBatch(js));
          const ctr = [[0, 0], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]];
          const dist = (Ha, Hb) => { let s = 0; for (const [x, y] of ctr) { const p = applyH(Ha, x, y), q = applyH(Hb, x, y); s += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2; } return Math.sqrt(s); };
          res.forEach((rs, q) => {
            if (!rs.ok) return;
            const k = redo[q]; const [i, j] = pairsOk[k];
            const cand = mat3norm(mat3inv(rs.G));
            const pred = mat3mul(mat3inv(Hk[i]), Hk[j]);
            if (dist(cand, pred) < dist(HnOk[k], pred) && rs.score >= SF[k] - 0.05) { HnOk[k] = cand; PF.set(this._hToPf(mat3mul(mat3mul(Ni, cand), N)), k * 4); SF[k] = rs.score; nFixed++; }
          });
        } else {
          const jobs = redo.map((k) => {
            const [i, j] = pairsOk[k];
            const dT = [sol.T[j * 2] - sol.T[i * 2], sol.T[j * 2 + 1] - sol.T[i * 2 + 1]];
            const relT = mulRot(rot2(-sol.TH[i]), dT).map((v) => v / sol.S[i]);
            return { i, j, p0: [Math.log(sol.S[j] / sol.S[i]), sol.TH[j] - sol.TH[i], relT[0], relT[1]] };
          });
          const res = await this._inFrameOrder(jobs, (js) => this.gnBatch(js));
          res.forEach((rs, q) => {
            if (!rs.ok) return;
            const k = redo[q]; const [i, j] = pairsOk[k];
            const cand = [Math.exp(rs.p[0]), rs.p[1], rs.p[2], rs.p[3]];
            const dT = [sol.T[j * 2] - sol.T[i * 2], sol.T[j * 2 + 1] - sol.T[i * 2 + 1]];
            const R = rot2(sol.TH[i]);
            const dOld = Math.hypot(...mulRot(R, [PF[k * 4 + 2], PF[k * 4 + 3]]).map((v, c) => sol.S[i] * v - dT[c]));
            const dNew = Math.hypot(...mulRot(R, [cand[2], cand[3]]).map((v, c) => sol.S[i] * v - dT[c]));
            if (dNew < dOld && rs.score >= SF[k] - 0.05) { PF.set(cand, k * 4); SF[k] = rs.score; nFixed++; }
          });
        }
        sol = solveAll();
        this.log(`  2パス目: 残差 1.5px 超 ${redo.length} ペアを再精密化、${nFixed} 件更新`);
      }
    }
    if (homog) {
      // 位置合わせに失敗したフレーム（そのフレームの全ペアの残差の中央値が 20 px 超）は合成から外し、位置は前後から補間する
      const wild = new Array(n).fill(false);
      for (let k = 0; k < n; k++) {
        const rk = [];
        for (let q = 0; q < mo; q++) if ((pairsOk[q][0] === k || pairsOk[q][1] === k) && !badOk[q]) rk.push(sol.rn[q]);
        if (rk.length && median(rk) > 20) wild[k] = true;
      }
      if (wild.some(Boolean)) {
        const idx = wild.map((w, k) => (w ? k : -1)).filter((k) => k >= 0);
        this.log(`  位置合わせ失敗フレーム（ペア残差の中央値 > 20 px）${idx.length} frames: ${idx.slice(0, 20).join(",")}（合成から外し、位置は前後から補間）`);
        if (!this.frameTier) this.frameTier = new Int32Array(n);
        for (const k of idx) { badFrame[k] = true; this.frameTier[k] = 2; }
        for (let q = 0; q < mo; q++) badOk[q] = badFrame[pairsOk[q][0]] || badFrame[pairsOk[q][1]];
      }
    }
    if (badFrame.some(Boolean)) {
      if (homog) {
        if (badFrame[0] && !badFrame.every(Boolean)) {
          // 基準（単位行列）のフレーム 0 が使えないので、最初の良いフレームを基準にし直す
          const ref = badFrame.findIndex((b) => !b);
          const HrefInv = mat3inv(Hk[ref]);
          for (let k = 0; k < n; k++) Hk[k] = mat3norm(mat3mul(HrefInv, Hk[k]));
          this.log(`  基準フレームを ${ref} に変更（フレーム 0 は位置合わせ失敗 / ブレンド）`);
        }
        Reconstructor.interpBadFrames(badFrame, [Hk]);
        const cx = 0.5 * (this.W - 1), cy = 0.5 * (this.H - 1);
        for (let k = 0; k < n; k++) {
          if (!(badFrame[k] || badFrame[0])) continue;
          const Hp = mat3mul(mat3mul(Ni, Hk[k]), N);
          sol.S[k] = localScaleOfH(Hp, cx, cy); sol.TH[k] = Math.atan2(Hp[3], Hp[0]);
          const c = applyH(Hp, cx, cy); sol.T[k * 2] = c[0] - cx; sol.T[k * 2 + 1] = c[1] - cy;
        }
      } else {
        Reconstructor.interpBadFrames(badFrame, [sol.S, sol.TH]); Reconstructor.interpBadFrames(badFrame, [sol.T], 2);
      }
    }
    const rn = sol.rn;
    const rnGood = rn.filter((_, k) => !badOk[k]);
    this.log(`[fine]   ${mo} pairs, score mean=${mean(SF).toFixed(3)}, residual median=${median(rnGood).toFixed(2)}px, ` +
             `max=${Math.max(...rnGood).toFixed(2)}px, ${(now() - t0).toFixed(1)}s`);
    const sus = [];
    for (let k = 0; k < mo; k++) if (rn[k] > 1.5 && !badOk[k]) sus.push(k);
    if (sus.length) {
      this.log(`  残差 1.5px 超のペア: ${sus.length} 件（重みを下げて解決済み）`);
      for (const k of sus.slice(0, 10)) this.log(`    (${pairsOk[k][0]},${pairsOk[k][1]}) residual=${rn[k].toFixed(2)}px score=${SF[k].toFixed(3)}`);
    }
    if (n > 1) {
      const dx = [], dy = [], nm = [];
      for (let k = 1; k < n; k++) { const a1 = sol.T[k * 2] - sol.T[(k - 1) * 2], b1 = sol.T[k * 2 + 1] - sol.T[(k - 1) * 2 + 1]; dx.push(a1); dy.push(b1); nm.push(Math.hypot(a1, b1)); }
      this.log(`  1フレームあたりの移動: dx mean=${mean(dx).toFixed(2)} dy mean=${mean(dy).toFixed(2)} (min ${Math.min(...nm).toFixed(2)} / max ${Math.max(...nm).toFixed(2)} px)`);
    }
    if (model !== "translation") {
      this.log(`  スケール: min=${Math.min(...sol.S).toFixed(4)} max=${Math.max(...sol.S).toFixed(4)} (frame0=1), ` +
               `回転: max |θ|=${(Math.max(...sol.TH.map(Math.abs)) * 180 / Math.PI).toFixed(3)}°`);
    }
    // ペア変換の 3x3 行列（画素座標、x_i = pm x_j）: 露出補正・品質推定で使う
    const pm = pairsOk.map((_, k) => (homog ? mat3mul(mat3mul(Ni, HnOk[k]), N) : simMatrix(PF[k * 4], PF[k * 4 + 1], PF[k * 4 + 2], PF[k * 4 + 3])));
    // 品質推定は C4 と鮮明度マップを使うので、必要なときは粗探索データの解放を遅らせる
    if (!(a.quality === "on")) this.freeCoarse();
    if (!this.store.allResident) this.log(`  streaming: ${this.store.stats()}`);
    let al = { S: sol.S, TH: sol.TH, T: sol.T, pairs: pairsOk, pp: PF, sc: SF, rn, pm, kind: "sim" };
    if (homog) {
      const Hpix = Hk.map((Hk_) => mat3mul(mat3mul(Ni, Hk_), N));
      al = { ...al, kind: "homography", Hm: Hpix };
      al = this._chooseProjection(al, pairsOk, pm, SF, badFrame);
    }
    return al;
  }

  // ------------------------------------------------------------ 露出補正（Python 版 estimate_exposure と同じモデル）
  /**
   * フレーム間の明るさの違い（露出・フェード・プレイヤーの周辺グラデーション等）を重なりから推定する。
   * モデル: I_k(x, y) = g_k L(X) + a_k + f(y/h_k) + g(x/w_k)
   *   g_k, a_k: フレームごとのゲインとオフセット（チャンネル別）、f, g: 全フレーム共通の加算プロファイル（折れ線、節点は端に密）
   * ペア (i, j) の重なりを 32 px ブロックに分け、クリーン画素のブロック平均の差 I_i - I_j を観測として全パラメータを
   * 最小二乗（IRLS, Cauchy）で解く。Σ a_k = 0, Σ (g_k - 1) = 0 で全体の明るさを保つ。合成時に (I - a_k - f - g) / g_k。
   * 結果は this.exposure = {gain: Float64Array(n*3), offset: Float64Array(n*3) (0..1 単位), profile: null | {fy, gx: Float64Array(NK*3)}}
   */
  async estimateExposure(al) {
    const a = this.a, g = this.gpu, K = this.k, W = this.W, H = this.H, n = this.n, st = this.store;
    const t0 = now();
    const cell = 8, blk = 4;
    const h8 = Math.max(2, Math.floor(H / cell)), w8 = Math.max(2, Math.floor(W / cell));
    const knots = a.exposureProfile ? EXPO_KNOTS : [];
    const NK = knots.length;
    // セル統計（クリーン画素の RGB 平均とクリーン率）を GPU で計算して読み戻す
    const cb = g.buf(h8 * w8 * 16, "cells");
    const M = new Array(n);
    for (let k = 0; k < n; k++) {
      const fr = await st.get(k);
      K.cell_rgba.run2d({ W, H, w: w8, h: h8, cell }, [fr, cb], w8, h8);
      M[k] = new Float32Array(await g.read(cb, h8 * w8 * 16));
      if (k % 16 === 15) { this.progress("exposure", 0.5 * k / n); this._check(); }
    }
    g.free(cb);
    // セル画像の双一次サンプル（align_corners=True: セル座標 0..w8-1）
    const smp = new Float64Array(4);
    const sample = (Mk, u, v) => {
      const x0 = Math.min(w8 - 2, Math.max(0, Math.floor(u))), y0 = Math.min(h8 - 2, Math.max(0, Math.floor(v)));
      const fx = Math.min(1, Math.max(0, u - x0)), fy = Math.min(1, Math.max(0, v - y0));
      const o00 = (y0 * w8 + x0) * 4, o10 = o00 + 4, o01 = o00 + w8 * 4, o11 = o01 + 4;
      for (let c = 0; c < 4; c++) {
        smp[c] = Mk[o00 + c] * (1 - fx) * (1 - fy) + Mk[o10 + c] * fx * (1 - fy) + Mk[o01 + c] * (1 - fx) * fy + Mk[o11 + c] * fx * fy;
      }
      return smp;
    };
    const hb = Math.floor(h8 / blk), wb = Math.floor(w8 / blk);
    const obs = { i: [], j: [], mi: [], mj: [], xi: [], yi: [], xj: [], yj: [], w: [] };
    let nPairs = 0;
    const acc = new Float64Array(hb * wb * 12);
    for (let r = 0; r < al.pairs.length; r++) {
      if (al.sc[r] < a.exposureMinScore) continue;
      const [i, j] = al.pairs[r];
      const s = al.pp[r * 4], th = al.pp[r * 4 + 1], tx = al.pp[r * 4 + 2], ty = al.pp[r * 4 + 3];
      const c = Math.cos(th), sn = Math.sin(th);
      const homog = a.model === "homography";
      const Minv = homog ? mat3inv(al.pm[r]) : null;   // x_j = Minv x_i
      const Mi = M[i], Mj = M[j];
      const [wi, hi] = this.sizes[i], [wj, hj] = this.sizes[j];
      acc.fill(0);
      for (let cy = 0; cy < hb * blk; cy++) {
        for (let cx = 0; cx < wb * blk; cx++) {
          const oi = (cy * w8 + cx) * 4;
          if (Mi[oi + 3] <= 0.5) continue;
          const Xi = (cx + 0.5) * cell - 0.5, Yi = (cy + 0.5) * cell - 0.5;
          let Xj, Yj;
          if (homog) {
            const z = Minv[6] * Xi + Minv[7] * Yi + Minv[8];
            if (z <= 1e-9) continue;
            Xj = (Minv[0] * Xi + Minv[1] * Yi + Minv[2]) / z; Yj = (Minv[3] * Xi + Minv[4] * Yi + Minv[5]) / z;
          } else {
            // x_i = s R x_j + t  →  x_j = R^T (x_i - t) / s
            const u = (Xi - tx) / s, v = (Yi - ty) / s;
            Xj = c * u + sn * v; Yj = -sn * u + c * v;
          }
          const uj = (Xj + 0.5) / cell - 0.5, vj = (Yj + 0.5) / cell - 0.5;
          if (uj < 0 || uj > w8 - 1 || vj < 0 || vj > h8 - 1) continue;
          const sj = sample(Mj, uj, vj);
          if (sj[3] <= 0.5) continue;
          const b = (Math.floor(cy / blk) * wb + Math.floor(cx / blk)) * 12;
          acc[b] += 1;
          acc[b + 1] += Mi[oi]; acc[b + 2] += Mi[oi + 1]; acc[b + 3] += Mi[oi + 2];
          acc[b + 4] += sj[0]; acc[b + 5] += sj[1]; acc[b + 6] += sj[2];
          acc[b + 7] += Xi; acc[b + 8] += Yi; acc[b + 9] += Xj; acc[b + 10] += Yj;
        }
      }
      let used = false;
      for (let b = 0; b < hb * wb; b++) {
        const cnt = acc[b * 12], wf = cnt / (blk * blk);
        if (wf < 0.7) continue;
        used = true;
        const o = b * 12;
        obs.i.push(i); obs.j.push(j); obs.w.push(wf);
        obs.mi.push([acc[o + 1] / cnt, acc[o + 2] / cnt, acc[o + 3] / cnt]);
        obs.mj.push([acc[o + 4] / cnt, acc[o + 5] / cnt, acc[o + 6] / cnt]);
        obs.xi.push((acc[o + 7] / cnt + 0.5) / wi); obs.yi.push((acc[o + 8] / cnt + 0.5) / hi);
        obs.xj.push((acc[o + 9] / cnt + 0.5) / wj); obs.yj.push((acc[o + 10] / cnt + 0.5) / hj);
      }
      if (used) nPairs++;
    }
    const N = obs.i.length;
    if (N === 0) { this.log("[exposure] 重なりの観測が得られないため露出補正を行いません"); return; }
    // 局所オフセット場: フレームごとの双一次格子（長辺 exposureLocal セル）。フレーム数が多いと未知数が増えすぎるので上限あり
    let Gx = 0, Gy = 0;
    const Gl = a.exposureLocal | 0;
    if (Gl > 0) {
      if (n > 100) this.log(`[exposure] フレーム数 ${n} > 100 のため局所オフセット場は使いません（local 0 と同じ）`);
      else if (W >= H) { Gx = Gl; Gy = Math.max(1, Math.round(Gl * H / W)); }
      else { Gy = Gl; Gx = Math.max(1, Math.round(Gl * W / H)); }
    }
    const NG = Gx > 0 ? (Gx + 1) * (Gy + 1) : 0;
    // 放射状（周辺減光）項: 全フレーム共通の乗算プロファイル v(r)、I = g_k (1 + v(r)) L + ...。r は中心 0、隅 1。中心の節点 v(0) = 0 で固定
    const rknots = a.exposureRadial ? RAD_KNOTS : [];
    const KR = rknots.length;
    const baseR = 2 * n + 2 * NK + n * NG;
    const rhat = (xn, yn) => {
      const rr = Math.min(1, Math.max(0, Math.hypot((xn - 0.5) * 2, (yn - 0.5) * 2) / Math.SQRT2));
      let q = 0;
      for (let i = 1; i + 1 < KR; i++) if (rr >= rknots[i]) q = i;
      return [q, Math.min(1, Math.max(0, (rr - rknots[q]) / (rknots[q + 1] - rknots[q])))];
    };
    const gridBasis = (xn, yn) => {   // → [idx0..3, w0..3]
      const u = Math.min(1, Math.max(0, xn)) * Gx, v = Math.min(1, Math.max(0, yn)) * Gy;
      const q0 = Math.min(Gx - 1, Math.floor(u)), r0 = Math.min(Gy - 1, Math.floor(v));
      const fu = Math.min(1, Math.max(0, u - q0)), fv = Math.min(1, Math.max(0, v - r0));
      const b = r0 * (Gx + 1) + q0;
      return [[b, b + 1, b + Gx + 1, b + Gx + 2], [(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]];
    };
    // 未知数 [δg(n), a(n), f(NK), g(NK), F_0(NG) .. F_{n-1}(NG)]。観測ごとの疎な行（最大 20 要素）
    const P = baseR + KR, Q = 4 + (NK ? 8 : 0) + (NG ? 8 : 0) + (KR ? 4 : 0);
    const cols = new Int32Array(N * Q), vals = new Float64Array(N * Q);
    const radHat = new Float64Array(N * 4);   // 放射項の基底（チャンネル別に L̄ を掛けて vals に入れる）
    const hat = (t) => {
      t = Math.min(1, Math.max(0, t));
      let q = 0;
      for (let i = 1; i + 1 < NK; i++) if (t >= knots[i]) q = i;
      const f = Math.min(1, Math.max(0, (t - knots[q]) / (knots[q + 1] - knots[q])));
      return [q, f];
    };
    for (let o = 0; o < N; o++) {
      const b = o * Q, i = obs.i[o], j = obs.j[o];
      cols[b] = i; cols[b + 1] = j; cols[b + 2] = n + i; cols[b + 3] = n + j;
      vals[b + 2] = 1; vals[b + 3] = -1;                       // vals[b], vals[b+1] = ±L̄（チャンネル別に設定）
      if (NK) {
        const [qi, fi] = hat(obs.yi[o]), [qj, fj] = hat(obs.yj[o]), [pi, gi] = hat(obs.xi[o]), [pj, gj] = hat(obs.xj[o]);
        cols[b + 4] = 2 * n + qi; vals[b + 4] = 1 - fi; cols[b + 5] = 2 * n + qi + 1; vals[b + 5] = fi;
        cols[b + 6] = 2 * n + qj; vals[b + 6] = -(1 - fj); cols[b + 7] = 2 * n + qj + 1; vals[b + 7] = -fj;
        cols[b + 8] = 2 * n + NK + pi; vals[b + 8] = 1 - gi; cols[b + 9] = 2 * n + NK + pi + 1; vals[b + 9] = gi;
        cols[b + 10] = 2 * n + NK + pj; vals[b + 10] = -(1 - gj); cols[b + 11] = 2 * n + NK + pj + 1; vals[b + 11] = -gj;
      }
      if (NG) {
        const b2 = b + (NK ? 12 : 4), baseF = 2 * n + 2 * NK;
        const [ii, wi] = gridBasis(obs.xi[o], obs.yi[o]), [ij, wj] = gridBasis(obs.xj[o], obs.yj[o]);
        for (let q = 0; q < 4; q++) {
          cols[b2 + q] = baseF + i * NG + ii[q]; vals[b2 + q] = wi[q];
          cols[b2 + 4 + q] = baseF + j * NG + ij[q]; vals[b2 + 4 + q] = -wj[q];
        }
      }
      if (KR) {
        const b3 = b + Q - 4;
        const [qi, fi] = rhat(obs.xi[o], obs.yi[o]), [qj, fj] = rhat(obs.xj[o], obs.yj[o]);
        cols[b3] = baseR + qi; cols[b3 + 1] = baseR + qi + 1; cols[b3 + 2] = baseR + qj; cols[b3 + 3] = baseR + qj + 1;
        radHat[o * 4] = 1 - fi; radHat[o * 4 + 1] = fi; radHat[o * 4 + 2] = -(1 - fj); radHat[o * 4 + 3] = -fj;
      }
    }
    // 正則化: ゲインは 1 へ、プロファイルは 0 と滑らかさへ（弱く）、Σ δg = Σ a = 0（ゲージ）
    const reg = new Float64Array(P);
    // オフセットのリッジはプロファイルより強く（フレーム固定の縦ランプはゲージ不定なので共通プロファイル側に寄せる。Python 版と同じ）
    for (let k = 0; k < n; k++) { reg[k] = 1e-2 * N / n; reg[n + k] = 1e-2 * N / n; }
    const fixed = new Float64Array(P * P);
    for (const [lo, hi] of [[0, n], [n, 2 * n]]) for (let p = lo; p < hi; p++) for (let q = lo; q < hi; q++) fixed[p * P + q] += N;
    if (KR) {
      for (let q = baseR; q < P; q++) reg[q] = 1e-4 * N / KR;
      reg[baseR] += N;                                            // v(0) = 0（ゲインと縮退するため）
      for (let t = 0; t < KR - 2; t++) {
        const h1 = rknots[t + 1] - rknots[t], h2 = rknots[t + 2] - rknots[t + 1], sc = 0.5 * (h1 + h2);
        const row = [[baseR + t, sc / h1], [baseR + t + 1, -sc * (1 / h1 + 1 / h2)], [baseR + t + 2, sc / h2]];
        for (const [p, vp] of row) for (const [q, vq] of row) fixed[p * P + q] += vp * vq * 1e-2 * N / KR;
      }
    }
    if (NK) {
      for (let q = 2 * n; q < 2 * n + 2 * NK; q++) reg[q] = 1e-4 * N / NK;
      for (const base of [2 * n, 2 * n + NK]) {
        for (let t = 0; t < NK - 2; t++) {
          const h1 = knots[t + 1] - knots[t], h2 = knots[t + 2] - knots[t + 1], sc = 0.5 * (h1 + h2);
          const row = [[base + t, sc / h1], [base + t + 1, -sc * (1 / h1 + 1 / h2)], [base + t + 2, sc / h2]];
          for (const [p, vp] of row) for (const [q, vq] of row) fixed[p * P + q] += vp * vq * 1e-2 * N / NK;
        }
      }
    }
    if (NG) {
      // 局所場: 0 へのリッジ（観測の無い節点は補正しない）と隣接節点の差の平滑化（Python 版と同じ）
      const baseF = 2 * n + 2 * NK, lamL = 3e-2 * N / n;
      for (let q = baseF; q < P; q++) reg[q] = 1e-3 * N / n;
      for (let k = 0; k < n; k++) {
        const b0 = baseF + k * NG;
        for (let r = 0; r <= Gy; r++) for (let q = 0; q <= Gx; q++) {
          const p = b0 + r * (Gx + 1) + q;
          for (const pn of [q < Gx ? p + 1 : -1, r < Gy ? p + Gx + 1 : -1]) {
            if (pn < 0) continue;
            fixed[p * P + p] += lamL; fixed[pn * P + pn] += lamL; fixed[p * P + pn] -= lamL; fixed[pn * P + p] -= lamL;
          }
        }
      }
    }
    for (let p = 0; p < P; p++) fixed[p * P + p] += reg[p];
    // 未知数を [局所場 F（帯）, 共通項（ゲイン / オフセット / プロファイル / 放射）] の順に並べ替え、矢じり型の消去で解く
    // （密行列の O(P³) は 100 フレーム × 6x3 格子で 3 分かかった）
    const nF = n * NG, baseF0 = 2 * n + 2 * NK;
    const perm = new Int32Array(P);
    for (let q = 0; q < P; q++) perm[q] = q < baseF0 ? nF + q : (q < baseR ? q - baseF0 : nF + baseF0 + (q - baseR));
    const maxOff = Math.max(...al.pairs.map(([i, j]) => j - i));
    const bwF = NG * (maxOff + 1);
    const fixedP = new Float64Array(P * P);
    for (let p2 = 0; p2 < P; p2++) for (let q = 0; q < P; q++) fixedP[perm[p2] * P + perm[q]] = fixed[p2 * P + q];
    const colsP = new Int32Array(N * Q);
    for (let q = 0; q < N * Q; q++) colsP[q] = perm[cols[q]];
    const cthr = 0.03;                                            // Cauchy スケール（0..1 単位 ≈ 8 階調）
    const theta = new Float64Array(P * 3);
    const res = new Float64Array(N * 3), d = new Float64Array(N * 3);
    for (let ch = 0; ch < 3; ch++) {
      const wt = Float64Array.from(obs.w);
      for (let o = 0; o < N; o++) {
        const lb = 0.5 * (obs.mi[o][ch] + obs.mj[o][ch]);
        vals[o * Q] = lb; vals[o * Q + 1] = -lb;
        if (KR) for (let q = 0; q < 4; q++) vals[o * Q + Q - 4 + q] = radHat[o * 4 + q] * lb;
        d[o * 3 + ch] = obs.mi[o][ch] - obs.mj[o][ch];
      }
      let th = null;
      for (let it = 0; it < 5; it++) {
        const AtA = Float64Array.from(fixedP), Atb = new Float64Array(P);
        for (let o = 0; o < N; o++) {
          const b = o * Q, w = wt[o], dc = d[o * 3 + ch];
          for (let p = 0; p < Q; p++) {
            const cp = colsP[b + p], vp = vals[b + p] * w;
            if (vp === 0) continue;
            Atb[cp] += vp * dc;
            for (let q = 0; q < Q; q++) AtA[cp * P + colsP[b + q]] += vp * vals[b + q];
          }
        }
        const xP = nF > 0 ? solveArrowhead(P, nF, bwF, AtA, Atb) : solveDense(P, AtA, Atb, 1);
        th = new Float64Array(P);
        for (let q = 0; q < P; q++) th[q] = xP[perm[q]];
        for (let o = 0; o < N; o++) {
          const b = o * Q;
          let r = -d[o * 3 + ch];
          for (let p = 0; p < Q; p++) r += vals[b + p] * th[cols[b + p]];
          res[o * 3 + ch] = r;
          wt[o] = obs.w[o] / (1 + (r / cthr) ** 2);
        }
      }
      for (let p = 0; p < P; p++) theta[p * 3 + ch] = th[p];
    }
    const gain = new Float64Array(n * 3), offset = new Float64Array(n * 3);
    for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) { gain[k * 3 + c] = 1 + theta[k * 3 + c]; offset[k * 3 + c] = theta[(n + k) * 3 + c]; }
    let profile = null;
    if (NK) {
      profile = { fy: theta.slice(2 * n * 3, (2 * n + NK) * 3), gx: theta.slice((2 * n + NK) * 3, (2 * n + 2 * NK) * 3) };
    }
    const local = NG ? theta.slice((2 * n + 2 * NK) * 3, baseR * 3) : null;   // n × NG × 3（0..1 単位）
    const radial = KR ? theta.slice(baseR * 3, P * 3) : null;                    // KR × 3（比）
    this.exposure = { gain, offset, profile, knots, local, localGrid: [Gx, Gy], radial, rknots };
    const absmean = (arr) => { const out = new Float64Array(N); for (let o = 0; o < N; o++) out[o] = (Math.abs(arr[o * 3]) + Math.abs(arr[o * 3 + 1]) + Math.abs(arr[o * 3 + 2])) / 3; return out; };
    const mad0 = median(absmean(d)) * 255, mad1 = median(absmean(res)) * 255;
    let msg = `[exposure] ${nPairs} pairs, ${N} blocks, 重なりの差（中央値）${mad0.toFixed(2)} → ${mad1.toFixed(2)} 階調, ` +
              `gain ${Math.min(...gain).toFixed(3)}-${Math.max(...gain).toFixed(3)}, offset ${(Math.min(...offset) * 255).toFixed(1)}..${(Math.max(...offset) * 255).toFixed(1)} 階調`;
    if (profile) {
      const lum = (arr) => Array.from({ length: NK }, (_, q) => (arr[q * 3] + arr[q * 3 + 1] + arr[q * 3 + 2]) / 3 * 255);
      const py = lum(profile.fy), px = lum(profile.gx);
      msg += `, profile y ${Math.min(...py).toFixed(1)}..${Math.max(...py).toFixed(1)} / x ${Math.min(...px).toFixed(1)}..${Math.max(...px).toFixed(1)} 階調`;
    }
    if (radial) msg += `, vignette (corner gain) ${(1 + (radial[(KR - 1) * 3] + radial[(KR - 1) * 3 + 1] + radial[(KR - 1) * 3 + 2]) / 3).toFixed(3)}`;
    if (local) {
      let lmin = Infinity, lmax = -Infinity;
      for (let q = 0; q < n * NG; q++) { const v = (local[q * 3] + local[q * 3 + 1] + local[q * 3 + 2]) / 3 * 255; lmin = Math.min(lmin, v); lmax = Math.max(lmax, v); }
      msg += `, local field (${Gx}x${Gy} cells) ${lmin.toFixed(1)}..${lmax.toFixed(1)} 階調`;
    }
    this.log(`${msg}, ${(now() - t0).toFixed(1)}s`);
    this.progress("exposure", 1);
  }

  /** 露出補正の CSV（exposure.csv と同じ形式） */
  exposureCsv() {
    const e = this.exposure;
    if (!e) return "";
    let s = "frame,gain_r,gain_g,gain_b,offset_r,offset_g,offset_b\n";
    for (let k = 0; k < this.n; k++) {
      s += `${k},${[0, 1, 2].map((c) => e.gain[k * 3 + c].toFixed(4)).join(",")},${[0, 1, 2].map((c) => (e.offset[k * 3 + c] * 255).toFixed(2)).join(",")}\n`;
    }
    if (e.profile) {
      s += "\nknot,fy_r,fy_g,fy_b,gx_r,gx_g,gx_b\n";
      for (let q = 0; q < e.knots.length; q++) {
        s += `${e.knots[q].toFixed(2)},${[0, 1, 2].map((c) => (e.profile.fy[q * 3 + c] * 255).toFixed(2)).join(",")},` +
             `${[0, 1, 2].map((c) => (e.profile.gx[q * 3 + c] * 255).toFixed(2)).join(",")}\n`;
      }
    }
    if (e.radial) {
      s += "\nradial_knot,vignette_r,vignette_g,vignette_b\n";
      for (let q = 0; q < e.rknots.length; q++) s += `${e.rknots[q].toFixed(2)},${[0, 1, 2].map((c) => (1 + e.radial[q * 3 + c]).toFixed(4)).join(",")}\n`;
    }
    if (e.local) {
      const [Gx, Gy] = e.localGrid, NG = (Gx + 1) * (Gy + 1);
      s += "\nlocal_frame,knot_x,knot_y,r,g,b\n";
      for (let k = 0; k < this.n; k++) for (let r = 0; r <= Gy; r++) for (let q = 0; q <= Gx; q++) {
        const o = (k * NG + r * (Gx + 1) + q) * 3;
        s += `${k},${q},${r},${[0, 1, 2].map((c) => (e.local[o + c] * 255).toFixed(2)).join(",")}\n`;
      }
    }
    return s;
  }

  /** フレーム k の RGBA（alpha = クリーンフラグ）を読み戻す */
  async readFrame(k) { return this.store.readRGBA(k); }

  destroy(keepStore = false) {
    const g = this.gpu;
    if (!keepStore) this.store.destroy();
    for (const b of [...this.C4, ...this.C16, ...this.f, ...this.c, ...this.s16,
                     this.sharpq, this.ratioBuf, this.surf, this.resBuf, this.Bs, this.hist, this.partials, this.state]) g.free(b);
    for (const e of this.levelCache.values()) for (const lv of Object.keys(e.levels)) g.free(e.levels[lv].buf);
    this.levelCache.clear();
  }
}
