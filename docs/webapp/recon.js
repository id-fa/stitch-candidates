// recon.js - panorama_recon.py の WebGPU 移植（位置合わせ部）
// 粗探索は FFT ではなく階層的総当たり NCC（1/16 → 1/4 解像度）。精密化は Gauss-Newton（GPU 上で完結）。

import { IMG } from "./shaders_img.js";
import { ALIGN } from "./shaders_align.js";
import { solveGlobal, rot2, mulRot, subpix, median, mean } from "./solve.js";

export const DEFAULTS = {
  model: "scale", pairs: [1, 2, 4], coarseScale: 0.25, coarseTol: 2.0, minOverlap: 0.15,
  scaleMax: 0.06, scaleStep: 0.004, fineScale: 1.0, gnIters: 15, ignoreRects: [], textRects: [],
  staticMask: true, staticSpan: 6, staticDiff: 0.03, staticGrad: 0.08, staticDilate: 7, staticHalo: 12, staticClose: 3, textHalo: 4,
  canvasScale: "auto", band: 64, inlierTol: 0.06, sharpTop: 0.3, stackBudgetMB: 128, levelCacheMB: 768,
};

const odd = (n) => { n = Math.round(n); return n % 2 === 1 ? n : n + 1; };
const now = () => performance.now() / 1000;
const MODEL_ID = { translation: 0, scale: 1, similarity: 2 };

export class Reconstructor {
  /** frames: フレームごとの RGBA8 GPUBuffer（W*H*4 バイト） */
  constructor(gpu, W, H, frames, args, log, progress) {
    const n = frames.length;
    this.gpu = gpu; this.W = W; this.H = H; this.n = n;
    this.a = { ...DEFAULTS, ...args };
    this.log = log; this.progress = progress || (() => {});
    this.frames = frames;
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
    const a = this.a, g = this.gpu, K = this.k, W = this.W, H = this.H, n = this.n;
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
    for (let k = 0; k < n; k++) {
      const fr = this.frames[k];
      // 静止オーバーレイ（+ テキスト矩形内の勾配ベース文字マスク）
      //   fA: 静止エッジ, fD: flat（静止 / 矩形内）→ 密度 (5x5) > 0.3 を二値化 → 膨張 r（fC）と膨張 rh（fE）
      let useDens = 0;
      if ((a.staticMask && n > 1) || ntext > 0) {
        // 比較相手: k±span のうち範囲内のもの（いずれかと一致すれば静止）
        let cands = [k - span, k + span].filter((j) => j >= 0 && j < n && j !== k);
        if (!cands.length) cands = [Math.max(0, Math.min(n - 1, k + (k === 0 ? span : -span)))];
        cands = cands.slice(0, 4);
        const diff = a.staticMask && n > 1 ? a.staticDiff : -1.0;   // 静止検出を無効化する場合は diff を負にする
        const cf = [0, 1, 2, 3].map((q) => this.frames[cands[Math.min(q, cands.length - 1)]]);
        K.static_detect.run2d({ W, H, diff, grad: a.staticGrad, ncand: cands.length, ntext, trects },
          [fr, cf[0], cf[1], cf[2], cf[3], fA, fD], W, H);
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
      K.finalize_mask.run2d({ W, H, use_dens: useDens, nrect, rects }, [fC, fE, fF, fD, fr], W, H);
      g.submit();
      if (k % 8 === 7) { await g.done(); this.progress("preprocess", 0.5 * (k + 1) / n); this._check(); }
    }
    // マスクの時間方向クロージング（j=1..staticClose の前後両方でマスクされていれば埋める）
    const nclose = Math.max(0, Math.min(3, a.staticClose | 0));
    if (nclose > 0 && (a.staticMask || ntext > 0)) {
      for (let k = 0; k < n; k++) {
        const pairs = [];
        for (let j = 1; j <= nclose; j++) if (k - j >= 0 && k + j < n) pairs.push([this.frames[k - j], this.frames[k + j]]);
        if (!pairs.length) continue;
        const bufs = [this.frames[k]];
        for (let q = 0; q < 3; q++) { const pr = pairs[Math.min(q, pairs.length - 1)]; bufs.push(pr[0], pr[1]); }
        K.mask_close.run({ n: W * H, npair: pairs.length }, bufs, Math.ceil(W * H / 256));
      }
      g.submit();
    }
    for (let k = 0; k < n; k++) {
      const fr = this.frames[k];
      // グレー、鮮明度
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
      g.submit();
      if (k % 8 === 7) { await g.done(); this.progress("preprocess", 0.5 + 0.5 * (k + 1) / n); this._check(); }
    }
    const rat = new Float32Array(await g.read(this.ratioBuf, n * 4));
    for (let k = 0; k < n; k++) this.overlayRatio[k] = 1.0 - rat[k] / (this.wc * this.hc);
    const mr = mean(this.overlayRatio);
    this.log(`[preprocess] ${n} frames ${W}x${H}, coarse ${this.wc}x${this.hc} / ${this.w16}x${this.h16}, ` +
             `overlay mask mean ratio=${mr.toFixed(3)}, ${(now() - t0).toFixed(1)}s`);
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

  _levels(k) {
    const e = this.levelCache.get(k);
    if (e) { this.levelCache.delete(k); this.levelCache.set(k, e); return e.levels; }
    const K = this.k, g = this.gpu, W = this.W, H = this.H;
    const [fA, fB, fC, fD, fE, fF] = this.f;
    const { lvs, sigma } = this._levelSpec();
    const rF = (odd(Math.max(3, 0.08 * Math.min(H, W))) - 1) / 2;
    K.gray_area.run2d({ W, H, w: W, h: H }, [this.frames[k], fA], W, H);
    K.box_blur.run2d({ W, H, r: rF, axis: 0, sub: 0 }, [fA, fA, fB], W, H);
    K.box_blur.run2d({ W, H, r: rF, axis: 1, sub: 1 }, [fB, fA, fC], W, H);   // fC = highpass
    K.alpha_f32.run({ n: W * H }, [this.frames[k], fB], Math.ceil(W * H / 256)); // fB = clean
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

  /** 粗推定: 戻り値 pc (m×4: s, th, tx, ty フル解像度単位), sc (m) */
  async coarse(pairs) {
    const a = this.a, n = this.n, m = pairs.length;
    const pc = new Float64Array(m * 4), sc = new Float64Array(m);
    const scaleModel = a.model !== "translation";
    const step16 = Math.max(a.scaleStep, 0.02);
    const K16 = Math.round(a.scaleMax / step16);
    const nref = Math.ceil(step16 / a.scaleStep / 2) + 1;
    const r4 = 12;
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
        // レベル 1/16
        const jobs16 = rs.map((r) => {
          const [i, j] = pairs[r];
          let scales = [1.0];
          if (scaleModel) {
            const full = j - i === minOff;
            const c0 = full ? 0 : chain[j] - chain[i];
            const ks = full ? [...Array(2 * K16 + 1).keys()].map((q) => q - K16) : [-1, 0, 1];
            scales = ks.map((q) => Math.exp(c0 + q * step16));
          }
          return { i, j, level: 16, scales, win: null, nmin: nmin16 };
        });
        const out16 = await this._searchBatch(jobs16);
        // レベル 1/4
        const jobs4 = rs.map((r, q) => {
          const [i, j] = pairs[r];
          const cand = out16[q].scales;
          let best = 0;
          for (let t = 1; t < cand.length; t++) if (cand[t].score > cand[best].score) best = t;
          let ls = Math.log(cand[best].s_eff);
          if (best > 0 && best < cand.length - 1) {
            const st = Math.log(cand[best + 1].s_eff) - Math.log(cand[best].s_eff);
            ls += subpix(cand[best - 1].score, cand[best].score, cand[best + 1].score) * st;
          }
          const f = this.wc / this.w16;
          const scales = scaleModel ? [...Array(2 * nref + 1).keys()].map((q2) => Math.exp(ls + (q2 - nref) * a.scaleStep)) : [1.0];
          return { i, j, level: 4, scales, win: { dx: Math.round(cand[best].dx * f), dy: Math.round(cand[best].dy * f), r: r4 }, nmin: nmin4 };
        });
        const out4 = await this._searchBatch(jobs4);
        rs.forEach((r, q) => {
          const [i, j] = pairs[r];
          const cand = out4[q].scales;
          let best = 0;
          for (let t = 1; t < cand.length; t++) if (cand[t].score > cand[best].score) best = t;
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
      sub.forEach((jb, q) => {
        const H = this.H, W = this.W;
        const by0 = Math.max(0, -jb.dy), by1 = Math.min(H, H - jb.dy), bx0 = Math.max(0, -jb.dx), bx1 = Math.min(W, W - jb.dx);
        if (by1 - by0 < 4 * jb.r + 16 || bx1 - bx0 < 4 * jb.r + 16) { out[b0 + q] = { ok: false }; return; }
        const Li = this._levels(jb.i)[1.0], Lj = this._levels(jb.j)[1.0];
        sjobs.push({ A: Li.buf, B: Lj.buf, dx: jb.dx, dy: jb.dy, r: jb.r, nmin: 0.1 * (by1 - by0) * (bx1 - bx0) });
        idx.push(b0 + q);
      });
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
        const Li = this._levels(jb.i), Lj = this._levels(jb.j);
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
    const gc = solveGlobal(n, pairs, pc, wc, model);
    const tol = a.coarseTol / this.sx;
    let nBad = 0;
    for (let r = 0; r < m; r++) if (gc.rn[r] > tol) nBad++;
    this.log(`[coarse] ${m} pairs, model=${model}, score mean=${mean(sc).toFixed(3)}, ` +
             `residual median=${(median(gc.rn) * this.sx).toFixed(2)}px (coarse), inconsistent pairs=${nBad}, ${(now() - t0).toFixed(1)}s`);

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
      const res = await this.fineTranslation(jobs);
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
      const res = await this.gnBatch(jobs);
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
        const res = await this.gnBatch(jobs);
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
    return { S: sol.S, TH: sol.TH, T: sol.T, pairs: pairsOk, pp: PF, sc: SF, rn };
  }

  /** フレーム k の RGBA（alpha = クリーンフラグ）を読み戻す */
  async readFrame(k) { return new Uint8ClampedArray(await this.gpu.read(this.frames[k], this.W * this.H * 4)); }

  destroy() {
    const g = this.gpu;
    for (const b of [...this.frames, ...this.C4, ...this.C16, ...this.f, ...this.c, ...this.s16,
                     this.sharpq, this.ratioBuf, this.surf, this.resBuf, this.Bs, this.hist, this.partials, this.state]) g.free(b);
    for (const e of this.levelCache.values()) for (const lv of Object.keys(e.levels)) g.free(e.levels[lv].buf);
    this.levelCache.clear();
  }
}
