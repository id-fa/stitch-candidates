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

// ------------------------------------------------------------ 3x3 行列（行優先 Float64Array(9)）とホモグラフィ用の道具
export const I3 = () => Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
export function mat3mul(A, B) {
  const C = new Float64Array(9);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c];
  return C;
}
export function mat3inv(A) {
  const [a, b, c, d, e, f, g, h, i] = A;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-300) throw new Error("singular 3x3");
  const s = 1 / det;
  return Float64Array.from([(e * i - f * h) * s, (c * h - b * i) * s, (b * f - c * e) * s,
                            (f * g - d * i) * s, (a * i - c * g) * s, (c * d - a * f) * s,
                            (d * h - e * g) * s, (b * g - a * h) * s, (a * e - b * d) * s]);
}
export function mat3T(A) { return Float64Array.from([A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]); }
export function mat3norm(A) { const s = A[8]; return Float64Array.from(A, (v) => v / s); }
/** 同次変換を点 (x, y) に適用（透視除算） */
export function applyH(H, x, y) {
  const z = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / z, (H[3] * x + H[4] * y + H[5]) / z];
}
/** x_i = s R(th) x_j + t の 3x3 */
export function simMatrix(s, th, tx, ty) {
  const c = Math.cos(th), sn = Math.sin(th);
  return Float64Array.from([s * c, -s * sn, tx, s * sn, s * c, ty, 0, 0, 1]);
}
/** 画素座標（画素中心）→ 正規化座標（中心 0、半長辺 1） */
export function normMatrix(W, H) {
  const f = 0.5 * Math.max(W, H), cx = 0.5 * (W - 1), cy = 0.5 * (H - 1);
  return Float64Array.from([1 / f, 0, -cx / f, 0, 1 / f, -cy / f, 0, 0, 1]);
}
/** ホモグラフィの点 (x, y) における局所倍率 sqrt(|det J|) */
export function localScaleOfH(H, x, y) {
  const z = H[6] * x + H[7] * y + H[8];
  const qx = H[0] * x + H[1] * y + H[2], qy = H[3] * x + H[4] * y + H[5];
  const j00 = (H[0] * z - qx * H[6]) / (z * z), j01 = (H[1] * z - qx * H[7]) / (z * z);
  const j10 = (H[3] * z - qy * H[6]) / (z * z), j11 = (H[4] * z - qy * H[7]) / (z * z);
  return Math.sqrt(Math.abs(j00 * j11 - j01 * j10));
}
export function rotFromRodrigues(v) {
  const th = Math.hypot(v[0], v[1], v[2]);
  if (th < 1e-12) return I3();
  const k = [v[0] / th, v[1] / th, v[2] / th];
  const Kx = Float64Array.from([0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0]);
  const K2 = mat3mul(Kx, Kx);
  const R = I3();
  const s = Math.sin(th), c1 = 1 - Math.cos(th);
  for (let q = 0; q < 9; q++) R[q] += s * Kx[q] + c1 * K2[q];
  return R;
}
export function rodriguesFromRot(R) {
  const c = Math.max(-1, Math.min(1, 0.5 * (R[0] + R[4] + R[8] - 1)));
  const th = Math.acos(c);
  if (th < 1e-9) return [0, 0, 0];
  const s = 2 * Math.sin(th);
  return [(R[7] - R[5]) / s * th, (R[2] - R[6]) / s * th, (R[3] - R[1]) / s * th];
}
/** 回転行列に近い行列を直交化（極分解の反復 R ← R (3I - RᵀR)/2） */
export function orthonormalize(R0) {
  let R = Float64Array.from(R0);
  for (let it = 0; it < 8; it++) {
    const RtR = mat3mul(mat3T(R), R);
    const M = I3();
    for (let q = 0; q < 9; q++) M[q] = 1.5 * M[q] - 0.5 * RtR[q];
    R = mat3mul(R, M);
  }
  return R;
}
/**
 * 主点原点の画素座標でのペアホモグラフィ x_i = H x_j から焦点距離の候補 [f_j, f_i]（Szeliski & Shum、OpenCV focalsFromHomography と同じ式）。
 * 推定できない成分は null
 */
export function focalsFromHomography(Hm) {
  const h = mat3norm(Hm);
  let f1 = null, f0 = null;
  let d1 = h[6] * h[7], d2 = (h[7] - h[6]) * (h[7] + h[6]);
  let v1 = Math.abs(d1) > 1e-12 ? -(h[0] * h[1] + h[3] * h[4]) / d1 : null;
  let v2 = Math.abs(d2) > 1e-12 ? (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2 : null;
  if (v1 !== null && v2 !== null && v1 < v2) [v1, v2] = [v2, v1];
  if (v1 !== null && v1 > 0 && v2 !== null && v2 > 0) f1 = Math.sqrt(Math.abs(d1) > Math.abs(d2) ? v1 : v2);
  else if (v1 !== null && v1 > 0) f1 = Math.sqrt(v1);
  d1 = h[0] * h[3] + h[1] * h[4]; d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
  v1 = Math.abs(d1) > 1e-12 ? -h[2] * h[5] / d1 : null;
  v2 = Math.abs(d2) > 1e-12 ? (h[5] * h[5] - h[2] * h[2]) / d2 : null;
  if (v1 !== null && v2 !== null && v1 < v2) [v1, v2] = [v2, v1];
  if (v1 !== null && v1 > 0 && v2 !== null && v2 > 0) f0 = Math.sqrt(Math.abs(d1) > Math.abs(d2) ? v1 : v2);
  else if (v1 !== null && v1 > 0) f0 = Math.sqrt(v1);
  return [f1, f0];
}

/** 対称帯行列（半帯幅 bw、A[r*P + c] 密格納だが |r-c| <= bw のみ非零）の Ax = b をガウス消去（ピボットなし）で解く。A, b は破壊される */
export function solveBanded(P, bw, A, b) {
  for (let c = 0; c < P; c++) {
    const piv = A[c * P + c];
    if (Math.abs(piv) < 1e-300) throw new Error("singular banded system");
    const rEnd = Math.min(P, c + bw + 1);
    for (let r = c + 1; r < rEnd; r++) {
      const f = A[r * P + c] / piv;
      if (f === 0) continue;
      for (let k = c; k < rEnd; k++) A[r * P + k] -= f * A[c * P + k];
      b[r] -= f * b[c];
    }
  }
  const x = new Float64Array(P);
  for (let r = P - 1; r >= 0; r--) {
    let s = b[r];
    const cEnd = Math.min(P, r + bw + 1);
    for (let c = r + 1; c < cEnd; c++) s -= A[r * P + c] * x[c];
    x[r] = s / A[r * P + r];
  }
  return x;
}

/**
 * ペアホモグラフィ x_i = Hp[r] x_j（正規化座標）から各フレームの H_k（→ フレーム 0 の座標）を解く（Python 版 _solve_global_h と同じ）。
 * 残差はフレーム j の 4 隅 + 中心を H_i Hp と H_j で写した差。IRLS (Cauchy)、数値微分の Gauss-Newton、帯行列で解く。
 * pairs: [[i,j]], Hp: Float64Array[] (m), w: Float64Array(m), W, H: フレームサイズ
 * 戻り値 {Hk: Float64Array[] (n), rn: Float64Array(m) (正規化座標)}
 */
export function solveGlobalH(n, pairs, Hp, w, W, H, iters = 12) {
  const m = pairs.length;
  const Hk = Array.from({ length: n }, () => I3());
  const kMin = Math.min(...pairs.map(([i, j]) => j - i));
  const chain = new Map();
  pairs.forEach(([i, j], r) => { if (j - i === kMin && (!chain.has(j) || w[r] > chain.get(j)[0])) chain.set(j, [w[r], Hp[r]]); });
  for (let j = 1; j < n; j++) Hk[j] = chain.has(j) ? mat3mul(Hk[j - kMin], chain.get(j)[1]) : Float64Array.from(Hk[j - 1]);
  const N = normMatrix(W, H);
  const pts = [[-0.5, -0.5], [W - 0.5, -0.5], [-0.5, H - 0.5], [W - 0.5, H - 0.5], [0.5 * (W - 1), 0.5 * (H - 1)]].map(([x, y]) => applyH(N, x, y));
  const f = 0.5 * Math.max(W, H), cC = 2.0 / f;
  const P = 8 * (n - 1);
  const maxOff = Math.max(...pairs.map(([i, j]) => j - i));
  const bw = 8 * maxOff + 7;
  const proj = (M, out) => { for (let q = 0; q < 5; q++) { const [x, y] = applyH(M, pts[q][0], pts[q][1]); out[2 * q] = x; out[2 * q + 1] = y; } };
  const plus = (Hb, q, eps) => { const M = Float64Array.from(Hb); M[q] += eps; return M; };
  const eps = 1e-6;
  const base = new Float64Array(10), tmp = new Float64Array(10);
  const rn = new Float64Array(m);
  // 残差はフレーム i の座標で測る（H_i^-1 H_j とペア推定 Hp の差）: フレーム 0 の平面座標では長いパンの遠いフレームで発散する
  const target = Hp.map((H) => { const t = new Float64Array(10); proj(H, t); return t; });
  for (let it = 0; it < iters; it++) {
    const res = new Float64Array(m * 10), J = new Float64Array(m * 160);
    for (let r = 0; r < m; r++) {
      const [i, j] = pairs[r];
      const HiInv = mat3inv(Hk[i]);
      proj(mat3mul(HiInv, Hk[j]), base);
      let s2 = 0;
      for (let q = 0; q < 10; q++) { res[r * 10 + q] = base[q] - target[r][q]; s2 += res[r * 10 + q] ** 2; }
      rn[r] = Math.sqrt(s2 / 5);
      if (i > 0) for (let q = 0; q < 8; q++) {
        proj(mat3mul(mat3inv(plus(Hk[i], q, eps)), Hk[j]), tmp);
        for (let t = 0; t < 10; t++) J[(r * 10 + t) * 16 + q] = (tmp[t] - base[t]) / eps;
      }
      if (j > 0) for (let q = 0; q < 8; q++) {
        proj(mat3mul(HiInv, plus(Hk[j], q, eps)), tmp);
        for (let t = 0; t < 10; t++) J[(r * 10 + t) * 16 + 8 + q] = (tmp[t] - base[t]) / eps;
      }
    }
    const A = new Float64Array(P * P), bvec = new Float64Array(P);
    for (let r = 0; r < m; r++) {
      const [i, j] = pairs[r];
      const cIt = cC * Math.max(1, 32 / 2 ** it);   // Cauchy 尺度は大きい値から絞る（連鎖初期値の外れ値に固着しないため）
      const wt = w[r] / (1 + (rn[r] / cIt) ** 2);
      const cols = [];
      if (i > 0) cols.push([8 * (i - 1), 0]);
      if (j > 0) cols.push([8 * (j - 1), 8]);
      for (const [ca, oa] of cols) for (const [cb, ob] of cols) {
        for (let p = 0; p < 8; p++) for (let q = 0; q < 8; q++) {
          let s = 0;
          for (let t = 0; t < 10; t++) s += J[(r * 10 + t) * 16 + oa + p] * J[(r * 10 + t) * 16 + ob + q];
          A[(ca + p) * P + cb + q] += wt * s;
        }
      }
      for (const [ca, oa] of cols) for (let p = 0; p < 8; p++) {
        let s = 0;
        for (let t = 0; t < 10; t++) s += J[(r * 10 + t) * 16 + oa + p] * res[r * 10 + t];
        bvec[ca + p] -= wt * s;
      }
    }
    for (let p = 0; p < P; p++) A[p * P + p] += A[p * P + p] * 1e-4 + 1e-10;
    let delta;
    try { delta = solveBanded(P, bw, A, bvec); } catch (e) { break; }
    if (!delta.every(Number.isFinite)) break;
    let mx = 0;
    for (let k = 1; k < n; k++) {
      for (let q = 0; q < 8; q++) { Hk[k][q] += delta[8 * (k - 1) + q]; mx = Math.max(mx, Math.abs(delta[8 * (k - 1) + q])); }
      Hk[k] = mat3norm(Hk[k]);
    }
    if (mx < 1e-7) break;
  }
  for (let r = 0; r < m; r++) {
    const [i, j] = pairs[r];
    proj(mat3mul(mat3inv(Hk[i]), Hk[j]), base);
    let s2 = 0;
    for (let q = 0; q < 10; q++) s2 += (base[q] - target[r][q]) ** 2;
    rn[r] = Math.sqrt(s2 / 5);
  }
  return { Hk, rn };
}

/**
 * ペア回転 R_ij（光線 d_i = R_ij d_j）からフレームごとの R_k（d_world = R_k d_k）を解く（回転平均、Python 版 _solve_rotations と同じ）
 * 戻り値 {R: Float64Array[] (n), rn: Float64Array(m) (rad)}
 */
export function solveRotations(n, pairs, Rp, w, iters = 8) {
  const R = Array.from({ length: n }, () => I3());
  const kMin = Math.min(...pairs.map(([i, j]) => j - i));
  const chain = new Map();
  pairs.forEach(([i, j], r) => { if (j - i === kMin && (!chain.has(j) || w[r] > chain.get(j)[0])) chain.set(j, [w[r], Rp[r]]); });
  for (let j = 1; j < n; j++) R[j] = chain.has(j) ? mat3mul(R[j - kMin], chain.get(j)[1]) : Float64Array.from(R[j - 1]);
  const m = pairs.length;
  const rn = new Float64Array(m);
  const d = new Float64Array(m * 3);
  for (let it = 0; it < iters; it++) {
    let mx = 0;
    for (let r = 0; r < m; r++) {
      const [i, j] = pairs[r];
      const e = rodriguesFromRot(mat3mul(mat3mul(R[i], Rp[r]), mat3T(R[j])));
      d[r * 3] = e[0]; d[r * 3 + 1] = e[1]; d[r * 3 + 2] = e[2];
      rn[r] = Math.hypot(e[0], e[1], e[2]); mx = Math.max(mx, rn[r]);
    }
    const sol = solvePositions(n, pairs, d, w, 3, 3, 0.2 * Math.PI / 180);
    for (let k = 0; k < n; k++) {
      const v = [sol.pos[k * 3] - sol.pos[0], sol.pos[k * 3 + 1] - sol.pos[1], sol.pos[k * 3 + 2] - sol.pos[2]];
      R[k] = orthonormalize(mat3mul(rotFromRodrigues(v), R[k]));
    }
    if (mx < 1e-7) break;
  }
  for (let r = 0; r < m; r++) {
    const [i, j] = pairs[r];
    const e = rodriguesFromRot(mat3mul(mat3mul(R[i], Rp[r]), mat3T(R[j])));
    rn[r] = Math.hypot(e[0], e[1], e[2]);
  }
  return { R, rn };
}

/**
 * 矢じり型の対称行列 A (P×P, 密格納) x = b を解く: 先頭 Pb 列は半帯幅 bw の帯行列、残り Pg = P - Pb 列は密（全行と結合）。
 * 帯部分を列順に消去し（各列で帯内の行と密ブロックの行だけ更新）、最後に密ブロックを消去する。A, b は破壊される。
 * 露出補正の局所オフセット場（フレームごとの格子 = 帯）+ 共通項（ゲイン / オフセット / プロファイル = 密）向け
 */
export function solveArrowhead(P, Pb, bw, A, b) {
  const gStart = Pb;
  for (let c = 0; c < Pb; c++) {
    const piv = A[c * P + c];
    if (Math.abs(piv) < 1e-300) throw new Error("singular arrowhead system");
    const bandEnd = Math.min(Pb, c + bw + 1);
    // 更新対象の行: 帯内 (c+1..bandEnd) と密ブロック (gStart..P)
    const rows = [];
    for (let r = c + 1; r < bandEnd; r++) rows.push(r);
    for (let r = gStart; r < P; r++) rows.push(r);
    for (const r of rows) {
      const f = A[r * P + c] / piv;
      if (f === 0) continue;
      for (let k = c; k < bandEnd; k++) A[r * P + k] -= f * A[c * P + k];
      for (let k = gStart; k < P; k++) A[r * P + k] -= f * A[c * P + k];
      b[r] -= f * b[c];
    }
  }
  for (let c = gStart; c < P; c++) {
    const piv = A[c * P + c];
    if (Math.abs(piv) < 1e-300) throw new Error("singular arrowhead system (dense block)");
    for (let r = c + 1; r < P; r++) {
      const f = A[r * P + c] / piv;
      if (f === 0) continue;
      for (let k = c; k < P; k++) A[r * P + k] -= f * A[c * P + k];
      b[r] -= f * b[c];
    }
  }
  const x = new Float64Array(P);
  for (let r = P - 1; r >= 0; r--) {
    let sum = b[r];
    if (r < Pb) {
      const bandEnd = Math.min(Pb, r + bw + 1);
      for (let c = r + 1; c < bandEnd; c++) sum -= A[r * P + c] * x[c];
      for (let c = gStart; c < P; c++) sum -= A[r * P + c] * x[c];
    } else {
      for (let c = r + 1; c < P; c++) sum -= A[r * P + c] * x[c];
    }
    x[r] = sum / A[r * P + r];
  }
  return x;
}
