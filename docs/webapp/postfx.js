// postfx.js - 合成後の後処理（CPU）
// fillHoles: クリーンな標本が 1 つも無かった画素（フォールバック領域）を周囲から埋める
//   mode "inpaint": push-pull 補間（粗いピラミッドから滑らかに埋める）
//   mode "blur":    穴領域だけをぼかした値で置き換える（文字はぼやけて残る）

/**
 * rgba: Uint8ClampedArray (W*H*4)、破壊的に更新
 * fill: Uint8Array (W*H) 1 = 埋める画素、0 = そのまま
 * 戻り値: 埋めた画素数
 */
export function fillHoles(rgba, fill, W, H, mode = "inpaint", blurRadius = 8) {
  let n = 0;
  for (let i = 0; i < fill.length; i++) if (fill[i]) n++;
  if (n === 0) return 0;
  if (mode === "blur") blurFill(rgba, fill, W, H, blurRadius);
  else pushPullFill(rgba, fill, W, H);
  return n;
}

function pushPullFill(rgba, fill, W, H) {
  // レベル 0: 値 (float RGB) と重み (1 = 既知)
  let vals = [new Float32Array(W * H * 3)], wts = [new Float32Array(W * H)], dims = [[W, H]];
  const v0 = vals[0], w0 = wts[0];
  for (let i = 0; i < W * H; i++) {
    const known = fill[i] ? 0 : 1;
    w0[i] = known;
    v0[i * 3] = rgba[i * 4] * known; v0[i * 3 + 1] = rgba[i * 4 + 1] * known; v0[i * 3 + 2] = rgba[i * 4 + 2] * known;
  }
  // push: 重み付き平均で縮小
  while (dims[dims.length - 1][0] > 1 || dims[dims.length - 1][1] > 1) {
    const [w, h] = dims[dims.length - 1];
    const w2 = Math.max(1, Math.ceil(w / 2)), h2 = Math.max(1, Math.ceil(h / 2));
    const sv = vals[vals.length - 1], sw = wts[wts.length - 1];
    const dv = new Float32Array(w2 * h2 * 3), dw = new Float32Array(w2 * h2);
    for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
      let r = 0, g = 0, b = 0, ws = 0;
      for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
        const sx = Math.min(w - 1, 2 * x + dx), sy = Math.min(h - 1, 2 * y + dy);
        const si = sy * w + sx, wt = sw[si];
        r += sv[si * 3]; g += sv[si * 3 + 1]; b += sv[si * 3 + 2]; ws += wt;
      }
      const di = y * w2 + x;
      if (ws > 0) { dv[di * 3] = r / ws; dv[di * 3 + 1] = g / ws; dv[di * 3 + 2] = b / ws; }
      dw[di] = Math.min(1, ws / 4);
      // 値は「重み込み」で保持（次の push で Σ w v になるように）
      dv[di * 3] *= dw[di]; dv[di * 3 + 1] *= dw[di]; dv[di * 3 + 2] *= dw[di];
    }
    vals.push(dv); wts.push(dw); dims.push([w2, h2]);
  }
  // 各レベルの値を正規化（v / w）
  for (let l = 0; l < vals.length; l++) {
    const v = vals[l], w = wts[l];
    for (let i = 0; i < w.length; i++) if (w[i] > 0) { v[i * 3] /= w[i]; v[i * 3 + 1] /= w[i]; v[i * 3 + 2] /= w[i]; }
  }
  // pull: 粗いレベルから双一次で持ち上げ、重みの足りない画素に混ぜる
  for (let l = vals.length - 2; l >= 0; l--) {
    const [w, h] = dims[l], [wc, hc] = dims[l + 1];
    const v = vals[l], wt = wts[l], cv = vals[l + 1];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (wt[i] >= 1) continue;
      const fx = Math.min(wc - 1, Math.max(0, (x + 0.5) / 2 - 0.5)), fy = Math.min(hc - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
      const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(wc - 1, x0 + 1), y1 = Math.min(hc - 1, y0 + 1);
      const ax = fx - x0, ay = fy - y0;
      const c00 = (y0 * wc + x0) * 3, c10 = (y0 * wc + x1) * 3, c01 = (y1 * wc + x0) * 3, c11 = (y1 * wc + x1) * 3;
      for (let c = 0; c < 3; c++) {
        const up = cv[c00 + c] * (1 - ax) * (1 - ay) + cv[c10 + c] * ax * (1 - ay) + cv[c01 + c] * (1 - ax) * ay + cv[c11 + c] * ax * ay;
        v[i * 3 + c] = wt[i] * v[i * 3 + c] + (1 - wt[i]) * up;
      }
      wt[i] = 1;
    }
  }
  const v = vals[0];
  for (let i = 0; i < W * H; i++) {
    if (!fill[i]) continue;
    rgba[i * 4] = v[i * 3]; rgba[i * 4 + 1] = v[i * 3 + 1]; rgba[i * 4 + 2] = v[i * 3 + 2]; rgba[i * 4 + 3] = 255;
  }
}

function blurFill(rgba, fill, W, H, r) {
  // 穴の周辺だけ分離型ボックスぼかし（2 回）を計算して穴に書き込む
  const src = new Float32Array(W * H * 3);
  for (let i = 0; i < W * H; i++) { src[i * 3] = rgba[i * 4]; src[i * 3 + 1] = rgba[i * 4 + 1]; src[i * 3 + 2] = rgba[i * 4 + 2]; }
  let cur = src;
  for (let pass = 0; pass < 2; pass++) {
    const tmp = new Float32Array(W * H * 3), out = new Float32Array(W * H * 3);
    for (let y = 0; y < H; y++) {
      // スライディングウィンドウ
      let sr = 0, sg = 0, sb = 0, sn = 0;
      for (let x = 0; x <= Math.min(W - 1, r); x++) { const i = (y * W + x) * 3; sr += cur[i]; sg += cur[i + 1]; sb += cur[i + 2]; sn++; }
      for (let x = 0; x < W; x++) {
        const o = (y * W + x) * 3;
        tmp[o] = sr / sn; tmp[o + 1] = sg / sn; tmp[o + 2] = sb / sn;
        const xa = x + r + 1, xr = x - r;
        if (xa < W) { const i = (y * W + xa) * 3; sr += cur[i]; sg += cur[i + 1]; sb += cur[i + 2]; sn++; }
        if (xr >= 0) { const i = (y * W + xr) * 3; sr -= cur[i]; sg -= cur[i + 1]; sb -= cur[i + 2]; sn--; }
      }
    }
    for (let x = 0; x < W; x++) {
      let sr = 0, sg = 0, sb = 0, sn = 0;
      for (let y = 0; y <= Math.min(H - 1, r); y++) { const i = (y * W + x) * 3; sr += tmp[i]; sg += tmp[i + 1]; sb += tmp[i + 2]; sn++; }
      for (let y = 0; y < H; y++) {
        const o = (y * W + x) * 3;
        out[o] = sr / sn; out[o + 1] = sg / sn; out[o + 2] = sb / sn;
        const ya = y + r + 1, yr = y - r;
        if (ya < H) { const i = (ya * W + x) * 3; sr += tmp[i]; sg += tmp[i + 1]; sb += tmp[i + 2]; sn++; }
        if (yr >= 0) { const i = (yr * W + x) * 3; sr -= tmp[i]; sg -= tmp[i + 1]; sb -= tmp[i + 2]; sn--; }
      }
    }
    cur = out;
  }
  for (let i = 0; i < W * H; i++) {
    if (!fill[i]) continue;
    rgba[i * 4] = cur[i * 3]; rgba[i * 4 + 1] = cur[i * 3 + 1]; rgba[i * 4 + 2] = cur[i * 3 + 2]; rgba[i * 4 + 3] = 255;
  }
}
