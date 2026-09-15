// render.js - 整列済みフレームの合成（時間方向中央値 / インライア平均 / 鮮明度上位平均 / 被覆）
//
// キャンバスを帯（動きが縦ならフル幅の行帯、横ならフル高さの列帯）に分け、帯ごとに、それを覆うフレームだけを
// 順に投影してサンプルスタックを作り、画素ごとの統計を取る。フレームは FrameStore から 1 枚ずつ取得するので
// 同時に常駐させる必要はなく、ストリーミング時は帯の順に動画から読み直される（隣り合う帯はほぼ同じフレーム集合を
// 使うので、キャッシュが「1 帯を覆うフレーム数」以上あれば各フレームの読み直しは 1 回で済む）。

import { RENDER } from "./shaders_render.js";
import { fmtMB } from "./gpu.js";

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
  const a = rc.a, g = rc.gpu, H = rc.H, W = rc.W, n = rc.n, st = rc.store;
  const K = {};
  for (const [name, def] of Object.entries(RENDER)) K[name] = g.kernel(name, def.code, def.fields, def.bindings, def.wg);
  const S = Float64Array.from(al.S), TH = Float64Array.from(al.TH), T = Float64Array.from(al.T);
  const sizes = rc.sizes;   // 各フレームの有効サイズ [w, h]（パディングはキャンバスに含めない）
  const gsc = a.canvasScale === "auto" ? 1.0 / Math.min(...S) : Number(a.canvasScale);
  for (let k = 0; k < n; k++) { S[k] *= gsc; T[k * 2] *= gsc; T[k * 2 + 1] *= gsc; }
  let minX = Infinity, minY = Infinity;
  for (let k = 0; k < n; k++) for (const [x, y] of corners(S, TH, T, k, sizes[k][1], sizes[k][0])) { minX = Math.min(minX, x); minY = Math.min(minY, y); }
  const ox = Math.floor(minX + 0.5), oy = Math.floor(minY + 0.5);
  for (let k = 0; k < n; k++) { T[k * 2] -= ox; T[k * 2 + 1] -= oy; }
  const boxes = [];
  let maxX = -Infinity, maxY = -Infinity;
  for (let k = 0; k < n; k++) {
    const cs = corners(S, TH, T, k, sizes[k][1], sizes[k][0]);
    const b = { x0: Math.min(...cs.map((p) => p[0])), x1: Math.max(...cs.map((p) => p[0])),
                y0: Math.min(...cs.map((p) => p[1])), y1: Math.max(...cs.map((p) => p[1])) };
    boxes.push(b); maxX = Math.max(maxX, b.x1); maxY = Math.max(maxY, b.y1);
  }
  const Wc = Math.ceil(maxX + 0.5), Hc = Math.ceil(maxY + 0.5);
  const outBytes = Wc * Hc * 4;
  const lim = g.device.limits.maxStorageBufferBindingSize;
  if (outBytes > lim) throw new Error(`キャンバス ${Wc}x${Hc} が大きすぎます（${(outBytes / 1048576).toFixed(0)}MB > ${(lim / 1048576).toFixed(0)}MB）。canvas scale を下げてください`);
  // 拡大率レベル: 1/12 オクターブ刻み（レベル 0 = 拡大率 1/4）。res_lv は許容する差（255 = 無効）
  const magLevel = (s) => Math.max(0, Math.min(63, Math.round((Math.log2(s) + 2) * 12)));
  const resLv = a.resTol > 0 ? Math.round(Math.log2(a.resTol) * 12) : 255;
  const anchor = Number.isFinite(a.anchorFrame) && a.anchorFrame >= 0 ? [Math.max(0, a.anchorFrame - a.anchorWindow), Math.min(n - 1, a.anchorFrame + a.anchorWindow)] : null;
  const inAnchor = (k) => anchor !== null && k >= anchor[0] && k <= anchor[1];

  // 帯の向き: キャンバスがフレームより横に伸びていれば列帯、そうでなければ行帯
  const horiz = (Wc - W) > (Hc - H);
  const L = horiz ? Hc : Wc;        // 帯の長辺
  const span = horiz ? Wc : Hc;     // 帯を並べる方向の長さ
  const lo = (k) => (horiz ? boxes[k].x0 : boxes[k].y0), hi = (k) => (horiz ? boxes[k].x1 : boxes[k].y1);
  // 1 画素幅の線を覆うフレーム数の最大（区間の重なりの最大値）
  const ev = [];
  for (let k = 0; k < n; k++) ev.push([lo(k), 1], [hi(k), -1]);
  ev.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  let cur = 0, maxCover = 0;
  for (const [, d] of ev) { cur += d; maxCover = Math.max(maxCover, cur); }
  // 帯幅: スタック (K × bw × L × 8B) が予算に収まるように。K は帯を覆うフレーム数の最大
  const budget = Math.min(a.stackBudgetMB * 1048576, lim);
  const tilesFor = (bw, use) => {
    const tiles = [];
    for (let t0 = 0; t0 < span; t0 += bw) {
      const t1 = Math.min(span, t0 + bw);
      const ks = [];
      for (let k = 0; k < n; k++) if (use(k) && lo(k) < t1 && hi(k) > t0) ks.push(k);
      tiles.push({ t0, t1, ks });
    }
    return tiles;
  };
  const pick = (use) => {
    let bw = Math.max(4, a.band | 0);
    let tiles = tilesFor(bw, use);
    let Kmax = Math.max(1, ...tiles.map((t) => t.ks.length));
    if (Kmax * bw * L * 12 > budget) {
      bw = Math.max(4, Math.floor(budget / (Kmax * L * 12)));
      tiles = tilesFor(bw, use);
      Kmax = Math.max(1, ...tiles.map((t) => t.ks.length));
    }
    return { bw, tiles, Kmax };
  };
  let sel = pick(() => true);
  const needOf = (s) => s.Kmax * s.bw * L * 12 + outBytes * 5 + W * H * 16 * 2;
  // フレームキャッシュ: 合成で必要なバッファ（スタック、出力 4 面、ぼかし作業）を引いた残りに合わせる
  let cacheFrames = st.maxResident;
  if (g.budgetBytes > 0) {
    const avail = Math.floor((g.budgetBytes - (g.allocBytes - st.residentBytes) - needOf(sel)) / st.frameBytes);
    if (avail < 2) throw new Error(`合成に必要なメモリ（${fmtMB(needOf(sel))}）を確保できません。memory limit を上げるか stack budget / canvas scale を下げてください`);
    cacheFrames = Math.min(cacheFrames, avail);
  }
  // ストリーミングでキャッシュが帯の被覆数より小さいと帯ごとに全フレームを読み直すことになるので、フレームを等間隔に間引く
  let stride = 1;
  if (cacheFrames < n) {
    const cap = Math.max(4, cacheFrames - 4);
    if (maxCover > cap) stride = Math.ceil(maxCover / cap);
  }
  if (stride > 1) {
    sel = pick((k) => k % stride === 0 || k === n - 1 || inAnchor(k));   // 端のフレームはキャンバス端の被覆に必要
    rc.log(`  警告: フレームキャッシュ（${cacheFrames} 枚）が帯を覆うフレーム数（最大 ${maxCover}）より少ないため、合成には ${stride} フレームごとに 1 枚を使います` +
           `（標本数 1/${stride}）。memory limit を上げるか fps を下げると回避できます`);
  }
  if (cacheFrames < st.maxResident) { st.setMaxResident(cacheFrames); rc.log(`[render] フレームキャッシュを ${cacheFrames} 枚に縮小`); }
  const { bw, tiles, Kmax } = sel;
  rc.log(`[render] canvas ${Wc}x${Hc}, canvas scale=${gsc.toFixed(4)}, ${horiz ? "column" : "row"} bands of ${bw}px (${tiles.length}), ` +
         `max ${Kmax} frames/band, res tol=${a.resTol}` + (anchor ? `, anchor frames ${anchor[0]}-${anchor[1]}` : ""));
  const need = needOf(sel);
  g.reserve(need, `render (canvas ${Wc}x${Hc})`);
  const t0 = now();
  const stack = g.buf(Kmax * bw * L * 12, "stack");   // 1 標本 3 ワード（rgb+flags, 鮮明度, フェザー重み）
  const outMed = g.buf(outBytes, "out_med"), outMean = g.buf(outBytes, "out_mean"), outSharp = g.buf(outBytes, "out_sharp"), outCov = g.buf(outBytes, "out_cov");
  const outBlend = g.buf(outBytes, "out_blend");
  const tmp1 = g.buf(W * H * 16, "blur_tmp1"), tmp2 = g.buf(W * H * 16, "blur_tmp2");
  // 露出補正: プロファイル（節点 + f(y) + g(x)）と局所場（フレームごとの格子）を 1 つのバッファ（0..255 単位）で渡し、
  // フレームごとのゲイン / オフセットは uniform
  const ex = rc.exposure;
  let expoBuf = g.zero, nk = 0, gxg = 0, gyg = 0, NG = 0, locBase = 0;
  if (ex && (ex.profile || ex.local)) {
    nk = ex.profile ? ex.knots.length : 0;
    if (ex.local) { [gxg, gyg] = ex.localGrid; NG = (gxg + 1) * (gyg + 1); }
    locBase = nk + 2 * nk * 3;
    const arr = new Float32Array(locBase + n * NG * 3);
    for (let q = 0; q < nk; q++) arr[q] = ex.knots[q];
    for (let q = 0; q < nk * 3; q++) { arr[nk + q] = ex.profile.fy[q] * 255; arr[nk + nk * 3 + q] = ex.profile.gx[q] * 255; }
    for (let q = 0; q < n * NG * 3; q++) arr[locBase + q] = ex.local[q] * 255;
    expoBuf = g.buf(arr.byteLength, "expo");
    g.upload(expoBuf, arr);
  }
  const feather = Math.max(0, a.feather || 0);
  if (ex) rc.log(`[render] 露出補正を適用（ゲイン / オフセット${ex.profile ? " + 周辺プロファイル" : ""}${ex.local ? ` + 局所場 ${gxg}x${gyg}` : ""}）` +
                 (feather > 0 ? `、フェザー合成 ${feather.toFixed(0)} px` : ""));
  let nb = 0;
  for (const tile of tiles) {
    const x0 = horiz ? tile.t0 : 0, y0 = horiz ? 0 : tile.t0;
    const tw = horiz ? tile.t1 - tile.t0 : Wc, th = horiz ? Hc : tile.t1 - tile.t0;
    const ks = tile.ks;
    if (ks.length) {
      for (let slot = 0; slot < ks.length; slot++) {
        const k = ks[slot];
        const fr = await st.get(k);
        const s = S[k], c = Math.cos(TH[k]), sn = Math.sin(TH[k]), Tx = T[k * 2], Ty = T[k * 2 + 1];
        let useBlur = 0;
        if (s < 0.9) {
          const sigma = 0.5 * Math.sqrt(1.0 / (s * s) - 1.0);
          const r = Math.min(12, Math.max(1, Math.ceil(3 * sigma)));
          // タイルが参照するフレーム行の範囲
          let fy0 = Infinity, fy1 = -Infinity;
          for (const [X, Y] of [[x0, y0], [x0 + tw, y0], [x0, y0 + th], [x0 + tw, y0 + th]]) {
            const u = (X - Tx) / s, v = (Y - Ty) / s;
            const fy = -sn * u + c * v;
            fy0 = Math.min(fy0, fy); fy1 = Math.max(fy1, fy);
          }
          const r0 = Math.max(0, Math.floor(fy0) - 1), r1 = Math.min(H, Math.ceil(fy1) + 2);
          if (r1 > r0) {
            const e0 = Math.max(0, r0 - r), e1 = Math.min(H, r1 + r);
            K.gauss_rgba.run2d({ W, H, sigma, axis: 0, row0: e0, row1: e1 }, [fr, tmp1, tmp1], W, e1 - e0);
            K.gauss_rgba.run2d({ W, H, sigma, axis: 1, row0: r0, row1: r1 }, [fr, tmp1, tmp2], W, r1 - r0);
            useBlur = 1;
          }
        }
        const gain = ex ? [ex.gain[k * 3], ex.gain[k * 3 + 1], ex.gain[k * 3 + 2], 1] : [1, 1, 1, 1];
        const off = ex ? [ex.offset[k * 3] * 255, ex.offset[k * 3 + 1] * 255, ex.offset[k * 3 + 2] * 255, 0] : [0, 0, 0, 0];
        K.warp.run2d({ x0, y0, tw, th, W, H, Wq: rc.Wq, Hq: rc.Hq, slot, sharp_off: k * rc.Hq * rc.Wq, use_blur: useBlur, s, c, sn, Tx, Ty, mag_lv: magLevel(s),
                       vw: sizes[k][0], vh: sizes[k][1], use_expo: ex ? 1 : 0, nk, loc_off: locBase + k * NG * 3, gx: gxg, gy: gyg, feather, gain, off },
          [fr, tmp2, rc.sharpq, stack, expoBuf], tw, th);
      }
      // アンカーフレームのスロット範囲（ks は昇順なので連続）。無ければ lo > hi
      let ancLo = 1, ancHi = 0;
      if (anchor) {
        const idx = ks.map((k, i) => (inAnchor(k) ? i : -1)).filter((i) => i >= 0);
        if (idx.length) { ancLo = idx[0]; ancHi = idx[idx.length - 1]; }
      }
      K.median.run2d({ Wc, x0, y0, tw, th, K: ks.length, tol: a.inlierTol * 255.0, sharp_top: a.sharpTop, res_lv: resLv, anc_lo: ancLo, anc_hi: ancHi },
        [stack, outMed, outMean, outSharp, outCov, outBlend], tw, th);
    }
    nb++;
    if (nb % 4 === 0) { await g.done(); rc.progress("render", nb / tiles.length); rc._check(); }
  }
  const med = new Uint8ClampedArray(await g.read(outMed, outBytes));
  const mn = new Uint8ClampedArray(await g.read(outMean, outBytes));
  const sh = new Uint8ClampedArray(await g.read(outSharp, outBytes));
  const bl = feather > 0 ? new Uint8ClampedArray(await g.read(outBlend, outBytes)) : null;
  const covRaw = new Uint32Array(await g.read(outCov, outBytes));
  const cov = new Uint32Array(covRaw.length), covG = new Uint32Array(covRaw.length);
  let holes = 0, fallback = 0;
  for (let i = 0; i < covRaw.length; i++) {
    cov[i] = covRaw[i] & 0xffff; covG[i] = covRaw[i] >>> 16;
    if (covG[i] === 0) holes++; else if (cov[i] === 0) fallback++;
  }
  for (const b of [stack, outMed, outMean, outSharp, outCov, outBlend, tmp1, tmp2]) g.free(b);
  if (expoBuf !== g.zero) g.free(expoBuf);
  rc.log(`[render] done ${(now() - t0).toFixed(1)}s, uncovered px=${holes}, fallback (no clean sample) px=${fallback}` +
         (st.allResident ? "" : ` (streaming, ${st.stats()})`));
  rc.progress("render", 1);
  return { Wc, Hc, median: med, mean: mn, sharp: sh, blend: bl, coverage: cov, coverageGeom: covG, canvasScale: gsc };
}
