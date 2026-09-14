// render.js - 整列済みフレームの合成（時間方向中央値 / インライア平均 / 鮮明度上位平均 / 被覆）

import { RENDER } from "./shaders_render.js";

const now = () => performance.now() / 1000;

function corners(S, TH, T, k, H, W) {
  const c = Math.cos(TH[k]), s = Math.sin(TH[k]);
  const pts = [[-0.5, -0.5], [W - 0.5, -0.5], [-0.5, H - 0.5], [W - 0.5, H - 0.5]];
  return pts.map(([x, y]) => [S[k] * (c * x - s * y) + T[k * 2], S[k] * (s * x + c * y) + T[k * 2 + 1]]);
}

/**
 * rc: Reconstructor, al: {S, TH, T}
 * 戻り値 {Wc, Hc, median, mean, sharp: Uint8ClampedArray(RGBA), coverage: Uint32Array}
 */
export async function render(rc, al) {
  const a = rc.a, g = rc.gpu, H = rc.H, W = rc.W, n = rc.n;
  const K = {};
  for (const [name, def] of Object.entries(RENDER)) K[name] = g.kernel(name, def.code, def.fields, def.bindings, def.wg);
  const S = Float64Array.from(al.S), TH = Float64Array.from(al.TH), T = Float64Array.from(al.T);
  const gsc = a.canvasScale === "auto" ? 1.0 / Math.min(...S) : Number(a.canvasScale);
  for (let k = 0; k < n; k++) { S[k] *= gsc; T[k * 2] *= gsc; T[k * 2 + 1] *= gsc; }
  let minX = Infinity, minY = Infinity;
  for (let k = 0; k < n; k++) for (const [x, y] of corners(S, TH, T, k, H, W)) { minX = Math.min(minX, x); minY = Math.min(minY, y); }
  const ox = Math.floor(minX + 0.5), oy = Math.floor(minY + 0.5);
  for (let k = 0; k < n; k++) { T[k * 2] -= ox; T[k * 2 + 1] -= oy; }
  const boxes = [];
  let maxX = -Infinity, maxY = -Infinity;
  for (let k = 0; k < n; k++) {
    const cs = corners(S, TH, T, k, H, W);
    const b = { x0: Math.min(...cs.map((p) => p[0])), x1: Math.max(...cs.map((p) => p[0])),
                y0: Math.min(...cs.map((p) => p[1])), y1: Math.max(...cs.map((p) => p[1])) };
    boxes.push(b); maxX = Math.max(maxX, b.x1); maxY = Math.max(maxY, b.y1);
  }
  const Wc = Math.ceil(maxX + 0.5), Hc = Math.ceil(maxY + 0.5);
  const outBytes = Wc * Hc * 4;
  const lim = g.device.limits.maxStorageBufferBindingSize;
  if (outBytes > lim) throw new Error(`キャンバス ${Wc}x${Hc} が大きすぎます（${(outBytes / 1048576).toFixed(0)}MB > ${(lim / 1048576).toFixed(0)}MB）。canvas scale を下げてください`);
  // 帯域高さ: サンプルスタック (n × bh × Wc × 8B) が予算に収まるように
  const budget = Math.min(a.stackBudgetMB * 1048576, lim);
  const bh = Math.max(4, Math.min(a.band, Math.floor(budget / (n * Wc * 8))));
  // 拡大率レベル: 1/12 オクターブ刻み（レベル 0 = 拡大率 1/4）。res_lv は許容する差（255 = 無効）
  const magLevel = (s) => Math.max(0, Math.min(63, Math.round((Math.log2(s) + 2) * 12)));
  const resLv = a.resTol > 0 ? Math.round(Math.log2(a.resTol) * 12) : 255;
  const anchor = Number.isFinite(a.anchorFrame) && a.anchorFrame >= 0 ? [Math.max(0, a.anchorFrame - a.anchorWindow), Math.min(n - 1, a.anchorFrame + a.anchorWindow)] : null;
  rc.log(`[render] canvas ${Wc}x${Hc}, canvas scale=${gsc.toFixed(4)}, band=${bh}, res tol=${a.resTol}` + (anchor ? `, anchor frames ${anchor[0]}-${anchor[1]}` : ""));
  g.reserve(n * bh * Wc * 8 + outBytes * 4 + W * H * 16 * 2, `render (canvas ${Wc}x${Hc})`);
  const t0 = now();
  const stack = g.buf(n * bh * Wc * 8, "stack");
  const outMed = g.buf(outBytes, "out_med"), outMean = g.buf(outBytes, "out_mean"), outSharp = g.buf(outBytes, "out_sharp"), outCov = g.buf(outBytes, "out_cov");
  const tmp1 = g.buf(W * H * 16, "blur_tmp1"), tmp2 = g.buf(W * H * 16, "blur_tmp2");
  let nb = 0, nbTotal = Math.ceil(Hc / bh);
  for (let y0 = 0; y0 < Hc; y0 += bh) {
    const y1 = Math.min(Hc, y0 + bh);
    const ks = [];
    for (let k = 0; k < n; k++) if (boxes[k].y0 < y1 && boxes[k].y1 > y0) ks.push(k);
    if (ks.length) {
      ks.forEach((k, slot) => {
        const s = S[k], c = Math.cos(TH[k]), sn = Math.sin(TH[k]), Tx = T[k * 2], Ty = T[k * 2 + 1];
        let useBlur = 0;
        if (s < 0.9) {
          const sigma = 0.5 * Math.sqrt(1.0 / (s * s) - 1.0);
          const r = Math.min(12, Math.max(1, Math.ceil(3 * sigma)));
          // 帯域が参照するフレーム行の範囲
          let fy0 = Infinity, fy1 = -Infinity;
          for (const [X, Y] of [[0, y0], [Wc, y0], [0, y1], [Wc, y1]]) {
            const u = (X - Tx) / s, v = (Y - Ty) / s;
            const fy = -sn * u + c * v;
            fy0 = Math.min(fy0, fy); fy1 = Math.max(fy1, fy);
          }
          const r0 = Math.max(0, Math.floor(fy0) - 1), r1 = Math.min(H, Math.ceil(fy1) + 2);
          if (r1 > r0) {
            const e0 = Math.max(0, r0 - r), e1 = Math.min(H, r1 + r);
            K.gauss_rgba.run2d({ W, H, sigma, axis: 0, row0: e0, row1: e1 }, [rc.frames[k], tmp1, tmp1], W, e1 - e0);
            K.gauss_rgba.run2d({ W, H, sigma, axis: 1, row0: r0, row1: r1 }, [rc.frames[k], tmp1, tmp2], W, r1 - r0);
            useBlur = 1;
          }
        }
        K.warp.run2d({ Wc, y0, bh, W, H, Wq: rc.Wq, Hq: rc.Hq, slot, sharp_off: k * rc.Hq * rc.Wq, use_blur: useBlur, s, c, sn, Tx, Ty, mag_lv: magLevel(s) },
          [rc.frames[k], tmp2, rc.sharpq, stack], Wc, y1 - y0);
      });
      // アンカーフレームのスロット範囲（ks は昇順なので連続）。無ければ lo > hi
      let ancLo = 1, ancHi = 0;
      if (anchor) {
        const idx = ks.map((k, i) => (k >= anchor[0] && k <= anchor[1] ? i : -1)).filter((i) => i >= 0);
        if (idx.length) { ancLo = idx[0]; ancHi = idx[idx.length - 1]; }
      }
      K.median.run2d({ Wc, bh, K: ks.length, y0, tol: a.inlierTol * 255.0, sharp_top: a.sharpTop, res_lv: resLv, anc_lo: ancLo, anc_hi: ancHi },
        [stack, outMed, outMean, outSharp, outCov], Wc, y1 - y0);
    }
    nb++;
    if (nb % 4 === 0) { await g.done(); rc.progress("render", nb / nbTotal); }
  }
  const med = new Uint8ClampedArray(await g.read(outMed, outBytes));
  const mn = new Uint8ClampedArray(await g.read(outMean, outBytes));
  const sh = new Uint8ClampedArray(await g.read(outSharp, outBytes));
  const covRaw = new Uint32Array(await g.read(outCov, outBytes));
  const cov = new Uint32Array(covRaw.length), covG = new Uint32Array(covRaw.length);
  let holes = 0, fallback = 0;
  for (let i = 0; i < covRaw.length; i++) {
    cov[i] = covRaw[i] & 0xffff; covG[i] = covRaw[i] >>> 16;
    if (covG[i] === 0) holes++; else if (cov[i] === 0) fallback++;
  }
  for (const b of [stack, outMed, outMean, outSharp, outCov, tmp1, tmp2]) g.free(b);
  rc.log(`[render] done ${(now() - t0).toFixed(1)}s, uncovered px=${holes}, fallback (no clean sample) px=${fallback}`);
  rc.progress("render", 1);
  return { Wc, Hc, median: med, mean: mn, sharp: sh, coverage: cov, coverageGeom: covG, canvasScale: gsc };
}
