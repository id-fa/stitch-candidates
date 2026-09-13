// solve.js - CPU 側の小規模線形代数（グローバル最小二乗、IRLS）

/** 密行列 A (n×n, Float64Array 行優先) X = B (n×D) をガウス消去（部分ピボット）で解く。A, B は破壊される */
export function solveDense(n, A, B, D) {
  for (let c = 0; c < n; c++) {
    let piv = c, pv = Math.abs(A[c * n + c]);
    for (let r = c + 1; r < n; r++) { const v = Math.abs(A[r * n + c]); if (v > pv) { pv = v; piv = r; } }
    if (pv < 1e-300) throw new Error("singular system in solveDense");
    if (piv !== c) {
      for (let k = 0; k < n; k++) { const t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; }
      for (let k = 0; k < D; k++) { const t = B[c * D + k]; B[c * D + k] = B[piv * D + k]; B[piv * D + k] = t; }
    }
    const inv = 1.0 / A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] * inv;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      for (let k = 0; k < D; k++) B[r * D + k] -= f * B[c * D + k];
    }
  }
  const X = new Float64Array(n * D);
  for (let r = n - 1; r >= 0; r--) {
    for (let k = 0; k < D; k++) {
      let s = B[r * D + k];
      for (let c = r + 1; c < n; c++) s -= A[r * n + c] * X[c * D + k];
      X[r * D + k] = s / A[r * n + r];
    }
  }
  return X;
}

/**
 * p_j - p_i = d_ij を重み w で最小二乗（p は D 次元）。IRLS (Cauchy) で外れ値を減衰。
 * pairs: [[i,j],...], d: Float64Array(m*D), w: Float64Array(m)
 * 戻り値 {pos: Float64Array(n*D), rn: Float64Array(m), wt: Float64Array(m)}
 */
export function solvePositions(n, pairs, d, w, D, iters = 5, c = 2.0) {
  const m = pairs.length;
  let wt = Float64Array.from(w);
  let pos = new Float64Array(n * D);
  const rn = new Float64Array(m);
  for (let it = 0; it < iters; it++) {
    const AtA = new Float64Array(n * n);
    const Atb = new Float64Array(n * D);
    for (let r = 0; r < m; r++) {
      const [i, j] = pairs[r];
      const w2 = wt[r] * wt[r];
      AtA[i * n + i] += w2; AtA[j * n + j] += w2; AtA[i * n + j] -= w2; AtA[j * n + i] -= w2;
      for (let k = 0; k < D; k++) { const v = w2 * d[r * D + k]; Atb[j * D + k] += v; Atb[i * D + k] -= v; }
    }
    AtA[0] += m * m; // p_0 = 0 のアンカー
    for (let i = 0; i < n; i++) AtA[i * n + i] += 1e-9;
    pos = solveDense(n, AtA, Atb, D);
    for (let r = 0; r < m; r++) {
      const [i, j] = pairs[r];
      let s = 0;
      for (let k = 0; k < D; k++) { const e = pos[j * D + k] - pos[i * D + k] - d[r * D + k]; s += e * e; }
      rn[r] = Math.sqrt(s);
      wt[r] = w[r] / (1.0 + (rn[r] / c) ** 2);
    }
  }
  return { pos, rn, wt };
}

export function rot2(theta) {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [[c, -s], [s, c]];
}
export function mulRot(R, v) { return [R[0][0] * v[0] + R[0][1] * v[1], R[1][0] * v[0] + R[1][1] * v[1]]; }

/**
 * ペア変換 x_i = s_ij R(th_ij) x_j + t_ij からフレームごとの (S, TH, T) を解く。
 * pp: Float64Array(m*4) = (s, th, tx, ty), w: Float64Array(m)
 */
export function solveGlobal(n, pairs, pp, w, model) {
  const m = pairs.length;
  let S = new Float64Array(n).fill(1), TH = new Float64Array(n);
  if (model !== "translation") {
    const ls = new Float64Array(m);
    for (let r = 0; r < m; r++) ls[r] = Math.log(pp[r * 4]);
    const rs = solvePositions(n, pairs, ls, w, 1, 5, 0.01);
    for (let k = 0; k < n; k++) S[k] = Math.exp(rs.pos[k]);
    if (model === "similarity") {
      const th = new Float64Array(m);
      for (let r = 0; r < m; r++) th[r] = pp[r * 4 + 1];
      TH = solvePositions(n, pairs, th, w, 1, 5, 0.005).pos;
    }
  }
  const d = new Float64Array(m * 2);
  for (let r = 0; r < m; r++) {
    const i = pairs[r][0];
    const v = mulRot(rot2(TH[i]), [pp[r * 4 + 2], pp[r * 4 + 3]]);
    d[r * 2] = S[i] * v[0]; d[r * 2 + 1] = S[i] * v[1];
  }
  const rt = solvePositions(n, pairs, d, w, 2, 5, 2.0);
  return { S, TH, T: rt.pos, rn: rt.rn };
}

export function subpix(cm, c0, cp) {
  const den = cm - 2.0 * c0 + cp;
  if (den >= 0 || Math.abs(den) < 1e-12) return 0.0;
  return Math.max(-0.5, Math.min(0.5, 0.5 * (cm - cp) / den));
}

export function median(arr) {
  const a = Float64Array.from(arr).sort();
  if (a.length === 0) return 0;
  const h = a.length >> 1;
  return a.length % 2 ? a[h] : 0.5 * (a[h - 1] + a[h]);
}
export function mean(arr) { let s = 0; for (const v of arr) s += v; return arr.length ? s / arr.length : 0; }
