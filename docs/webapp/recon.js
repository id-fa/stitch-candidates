// recon.js - panorama_recon.py の WebGPU 移植（位置合わせ部）
// 粗探索は FFT ではなく階層的総当たり NCC（1/16 → 1/4 解像度）。精密化は Gauss-Newton（GPU 上で完結）。

import { IMG } from "./shaders_img.js";
import { ALIGN } from "./shaders_align.js";
import { solveGlobal, solveDense, rot2, mulRot, subpix, median, mean } from "./solve.js";

export const DEFAULTS = {
  model: "scale", pairs: [1, 2, 4], coarseScale: 0.25, coarseTol: 2.0, minOverlap: 0.15,
  scaleMax: 0.06, scaleStep: 0.004, fineScale: 1.0, gnIters: 15, ignoreRects: [], textRects: [],
  staticMask: true, staticSpan: 6, staticDiff: 0.03, staticGrad: 0.08, staticDilate: 7, staticHalo: 12, staticClose: 3, textHalo: 4,
  canvasScale: "auto", band: 64, inlierTol: 0.06, sharpTop: 0.3, resTol: 1.25, anchorFrame: -1, anchorWindow: 2, stackBudgetMB: 128, levelCacheMB: 768,
  exposureProfile: true, exposureMinScore: 0.2, exposureLocal: 6, feather: 0,
};

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
    this.partials = gpu.buf(this.NWG * 20 * 4, "partials");
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
      K.laplacian_abs.run2d({ W, H }, [fA, fB], W, H);
      K.box_blur.run2d({ W, H, r: 7, axis: 0, sub: 0 }, [fB, fB, fC], W, H);
      K.box_blur.run2d({ W, H, r: 7, axis: 1, sub: 0 }, [fC, fC, fB], W, H);
      K.area_f32.run2d({ W, H, w: this.Wq, h: this.Hq, src_off: 0, dst_off: k * this.Hq * this.Wq }, [fB, this.sharpq], this.Wq, this.Hq);
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

  // ------------------------------------------------------------ 位置合わせ全体
  async align() {
    const a = this.a, n = this.n;
    const pairs = [];
    for (const k of a.pairs) for (let i = 0; i < n - k; i++) pairs.push([i, i + k]);
    const m = pairs.length;
    const model = a.model;
    let t0 = now();
    const { pc, sc } = await this.coarse(pairs);
    const wc = new Float64Array(m);
    for (let r = 0; r < m; r++) wc[r] = Math.min(1, Math.max(0.05, sc[r])) ** 2;
    // 離れたペア（k > 最小間隔）の粗推定が隣接ペアの連鎖と矛盾していれば重みを落とす（Python 版と同じ）。
    // 重なりの無いペアでも滑らかな輪郭が偶然合って高い NCC が出ることがあり（静止画列で顕著）、そのままグローバル解に
    // 入れると正しい隣接ペアまで「不整合」扱いになって精密化の初期値が壊れる
    const kMin = Math.min(...a.pairs);
    const chain = [];
    for (let r = 0; r < m; r++) if (pairs[r][1] - pairs[r][0] === kMin) chain.push(r);
    let nFarBad = 0;
    if (chain.length < m && chain.length >= n - 1) {
      const pcC = new Float64Array(chain.length * 4), wcC = new Float64Array(chain.length);
      chain.forEach((r, q) => { pcC.set(pc.subarray(r * 4, r * 4 + 4), q * 4); wcC[q] = wc[r]; });
      const gh = solveGlobal(n, chain.map((r) => pairs[r]), pcC, wcC, model);
      for (let r = 0; r < m; r++) {
        if (pairs[r][1] - pairs[r][0] === kMin) continue;
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
      const jobs = pairs.map(([i, j], r) => {
        let p0;
        if (gc.rn[r] <= tol) p0 = [Math.log(pc[r * 4]), pc[r * 4 + 1], pc[r * 4 + 2], pc[r * 4 + 3]];
        else {
          const relS = gc.S[j] / gc.S[i], relTh = gc.TH[j] - gc.TH[i];
          const dT = [gc.T[j * 2] - gc.T[i * 2], gc.T[j * 2 + 1] - gc.T[i * 2 + 1]];
          const relT = mulRot(rot2(-gc.TH[i]), dT).map((v) => v / gc.S[i]);
          p0 = [Math.log(relS), relTh, relT[0], relT[1]];
        }
        return { i, j, p0 };
      });
      const res = await this._inFrameOrder(jobs, (js) => this.gnBatch(js));
      res.forEach((rs, r) => {
        if (!rs.ok) { ok[r] = false; return; }
        pf[r * 4] = Math.exp(rs.p[0]); pf[r * 4 + 1] = rs.p[1]; pf[r * 4 + 2] = rs.p[2]; pf[r * 4 + 3] = rs.p[3]; sf[r] = rs.score;
      });
    }
    let pairsOk = [], pfOk = [], sfOk = [];
    for (let r = 0; r < m; r++) if (ok[r]) { pairsOk.push(pairs[r]); pfOk.push(pf.subarray(r * 4, r * 4 + 4)); sfOk.push(sf[r]); }
    const mo = pairsOk.length;
    if (mo < n - 1) this.log(`  警告: 有効ペアが少なすぎます (${mo} / ${m})`);
    if (mo === 0) throw new Error("有効なペアがありません");
    const PF = new Float64Array(mo * 4); pfOk.forEach((p, r) => PF.set(p, r * 4));
    const SF = Float64Array.from(sfOk);
    const wf = SF.map((s) => Math.min(1, Math.max(0.05, s)) ** 2);
    let sol = solveGlobal(n, pairsOk, PF, wf, model);
    if (model !== "translation") {
      const redo = [];
      for (let k = 0; k < mo; k++) if (sol.rn[k] > 1.5) redo.push(k);
      let nFixed = 0;
      if (redo.length) {
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
        const wf2 = SF.map((s) => Math.min(1, Math.max(0.05, s)) ** 2);
        sol = solveGlobal(n, pairsOk, PF, wf2, model);
        this.log(`  2パス目: 残差 1.5px 超 ${redo.length} ペアを再精密化、${nFixed} 件更新`);
      }
    }
    const rn = sol.rn;
    this.log(`[fine]   ${mo} pairs, score mean=${mean(SF).toFixed(3)}, residual median=${median(rn).toFixed(2)}px, ` +
             `max=${Math.max(...rn).toFixed(2)}px, ${(now() - t0).toFixed(1)}s`);
    const sus = [];
    for (let k = 0; k < mo; k++) if (rn[k] > 1.5) sus.push(k);
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
    this.freeCoarse();
    if (!this.store.allResident) this.log(`  streaming: ${this.store.stats()}`);
    return { S: sol.S, TH: sol.TH, T: sol.T, pairs: pairsOk, pp: PF, sc: SF, rn };
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
      const Mi = M[i], Mj = M[j];
      const [wi, hi] = this.sizes[i], [wj, hj] = this.sizes[j];
      acc.fill(0);
      for (let cy = 0; cy < hb * blk; cy++) {
        for (let cx = 0; cx < wb * blk; cx++) {
          const oi = (cy * w8 + cx) * 4;
          if (Mi[oi + 3] <= 0.5) continue;
          const Xi = (cx + 0.5) * cell - 0.5, Yi = (cy + 0.5) * cell - 0.5;
          // x_i = s R x_j + t  →  x_j = R^T (x_i - t) / s
          const u = (Xi - tx) / s, v = (Yi - ty) / s;
          const Xj = c * u + sn * v, Yj = -sn * u + c * v;
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
    const gridBasis = (xn, yn) => {   // → [idx0..3, w0..3]
      const u = Math.min(1, Math.max(0, xn)) * Gx, v = Math.min(1, Math.max(0, yn)) * Gy;
      const q0 = Math.min(Gx - 1, Math.floor(u)), r0 = Math.min(Gy - 1, Math.floor(v));
      const fu = Math.min(1, Math.max(0, u - q0)), fv = Math.min(1, Math.max(0, v - r0));
      const b = r0 * (Gx + 1) + q0;
      return [[b, b + 1, b + Gx + 1, b + Gx + 2], [(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]];
    };
    // 未知数 [δg(n), a(n), f(NK), g(NK), F_0(NG) .. F_{n-1}(NG)]。観測ごとの疎な行（最大 20 要素）
    const P = 2 * n + 2 * NK + n * NG, Q = 4 + (NK ? 8 : 0) + (NG ? 8 : 0);
    const cols = new Int32Array(N * Q), vals = new Float64Array(N * Q);
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
    }
    // 正則化: ゲインは 1 へ、プロファイルは 0 と滑らかさへ（弱く）、Σ δg = Σ a = 0（ゲージ）
    const reg = new Float64Array(P);
    // オフセットのリッジはプロファイルより強く（フレーム固定の縦ランプはゲージ不定なので共通プロファイル側に寄せる。Python 版と同じ）
    for (let k = 0; k < n; k++) { reg[k] = 1e-2 * N / n; reg[n + k] = 1e-2 * N / n; }
    const fixed = new Float64Array(P * P);
    for (const [lo, hi] of [[0, n], [n, 2 * n]]) for (let p = lo; p < hi; p++) for (let q = lo; q < hi; q++) fixed[p * P + q] += N;
    if (NK) {
      for (let q = 2 * n; q < P; q++) reg[q] = 1e-4 * N / NK;
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
    const cthr = 0.03;                                            // Cauchy スケール（0..1 単位 ≈ 8 階調）
    const theta = new Float64Array(P * 3);
    const res = new Float64Array(N * 3), d = new Float64Array(N * 3);
    for (let ch = 0; ch < 3; ch++) {
      const wt = Float64Array.from(obs.w);
      for (let o = 0; o < N; o++) {
        const lb = 0.5 * (obs.mi[o][ch] + obs.mj[o][ch]);
        vals[o * Q] = lb; vals[o * Q + 1] = -lb;
        d[o * 3 + ch] = obs.mi[o][ch] - obs.mj[o][ch];
      }
      let th = null;
      for (let it = 0; it < 5; it++) {
        const AtA = Float64Array.from(fixed), Atb = new Float64Array(P);
        for (let o = 0; o < N; o++) {
          const b = o * Q, w = wt[o], dc = d[o * 3 + ch];
          for (let p = 0; p < Q; p++) {
            const cp = cols[b + p], vp = vals[b + p] * w;
            if (vp === 0) continue;
            Atb[cp] += vp * dc;
            for (let q = 0; q < Q; q++) AtA[cp * P + cols[b + q]] += vp * vals[b + q];
          }
        }
        th = solveDense(P, AtA, Atb, 1);
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
    const local = NG ? theta.slice((2 * n + 2 * NK) * 3, P * 3) : null;   // n × NG × 3（0..1 単位）
    this.exposure = { gain, offset, profile, knots, local, localGrid: [Gx, Gy] };
    const absmean = (arr) => { const out = new Float64Array(N); for (let o = 0; o < N; o++) out[o] = (Math.abs(arr[o * 3]) + Math.abs(arr[o * 3 + 1]) + Math.abs(arr[o * 3 + 2])) / 3; return out; };
    const mad0 = median(absmean(d)) * 255, mad1 = median(absmean(res)) * 255;
    let msg = `[exposure] ${nPairs} pairs, ${N} blocks, 重なりの差（中央値）${mad0.toFixed(2)} → ${mad1.toFixed(2)} 階調, ` +
              `gain ${Math.min(...gain).toFixed(3)}-${Math.max(...gain).toFixed(3)}, offset ${(Math.min(...offset) * 255).toFixed(1)}..${(Math.max(...offset) * 255).toFixed(1)} 階調`;
    if (profile) {
      const lum = (arr) => Array.from({ length: NK }, (_, q) => (arr[q * 3] + arr[q * 3 + 1] + arr[q * 3 + 2]) / 3 * 255);
      const py = lum(profile.fy), px = lum(profile.gx);
      msg += `, profile y ${Math.min(...py).toFixed(1)}..${Math.max(...py).toFixed(1)} / x ${Math.min(...px).toFixed(1)}..${Math.max(...px).toFixed(1)} 階調`;
    }
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

  destroy() {
    const g = this.gpu;
    this.store.destroy();
    for (const b of [...this.C4, ...this.C16, ...this.f, ...this.c, ...this.s16,
                     this.sharpq, this.ratioBuf, this.surf, this.resBuf, this.Bs, this.hist, this.partials, this.state]) g.free(b);
    for (const e of this.levelCache.values()) for (const lv of Object.keys(e.levels)) g.free(e.levels[lv].buf);
    this.levelCache.clear();
  }
}
