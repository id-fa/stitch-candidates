// shaders_align.js - 位置合わせカーネル
// - ncc_search: 全シフト総当たりのマスク付き NCC（1 シフト = 1 ワークグループ）
// - argmax: NCC サーフェスの最大値と近傍値
// - gn_resid / gn_accum / gn_solve: Gauss-Newton 直接法（Huber 重み、ヒストグラム MAD）
//   状態は state[slot*16 + ...] = (ls, th, tx, ty, score, n_valid, fail, conv)

export const ALIGN = {};

// A(x+d) と B(x) のマスク付き NCC を全シフト d について計算（Padfield の式を直接和で評価）
// 出力 out[sy*nx + sx]、d = (dx0+sx, dy0+sy)。有効画素数 N < nmin は -2
ALIGN.ncc_search = {
  fields: [["hA", "u32"], ["wA", "u32"], ["hB", "u32"], ["wB", "u32"], ["dx0", "i32"], ["dy0", "i32"],
           ["nx", "u32"], ["ny", "u32"], ["nmin", "f32"], ["out_off", "u32"]],
  bindings: ["r", "r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { hA: u32, wA: u32, hB: u32, wB: u32, dx0: i32, dy0: i32, nx: u32, ny: u32, nmin: f32, out_off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> outb: array<f32>;
var<workgroup> red: array<f32, 1536>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let sx = wid.x; let sy = wid.y;
  let dx = p.dx0 + i32(sx); let dy = p.dy0 + i32(sy);
  let wA = i32(p.wA); let hA = i32(p.hA); let wB = i32(p.wB); let hB = i32(p.hB);
  let xs = max(0, -dx); let xe = min(wB, wA - dx);
  let ys = max(0, -dy); let ye = min(hB, hA - dy);
  let ow = xe - xs; let oh = ye - ys;
  var N = 0.0; var Sf = 0.0; var Sm = 0.0; var Sff = 0.0; var Smm = 0.0; var Sfm = 0.0;
  if (ow > 0 && oh > 0) {
    let n = u32(ow * oh);
    for (var t = l.x; t < n; t += 256u) {
      let x = xs + i32(t % u32(ow)); let y = ys + i32(t / u32(ow));
      let a = A[(y + dy) * wA + x + dx];
      let b = B[y * wB + x];
      let m = a.y * b.y;
      N += m; Sf += a.x * m; Sm += b.x * m; Sff += a.x * a.x * m; Smm += b.x * b.x * m; Sfm += a.x * b.x * m;
    }
  }
  red[l.x] = N; red[256u + l.x] = Sf; red[512u + l.x] = Sm; red[768u + l.x] = Sff; red[1024u + l.x] = Smm; red[1280u + l.x] = Sfm;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (l.x < st) {
      for (var q = 0u; q < 6u; q++) { red[q * 256u + l.x] += red[q * 256u + l.x + st]; }
    }
    workgroupBarrier();
  }
  if (l.x == 0u) {
    let Nn = red[0]; let sf = red[256]; let sm = red[512]; let sff = red[768]; let smm = red[1024]; let sfm = red[1280];
    var v = -2.0;
    if (Nn >= p.nmin && Nn >= 1.0) {
      let num = sfm - sf * sm / Nn;
      let vf = max(sff - sf * sf / Nn, 0.0);
      let vm = max(smm - sm * sm / Nn, 0.0);
      v = num / sqrt(vf * vm + 1e-12);
    }
    outb[p.out_off + sy * p.nx + sx] = v;
  }
}`,
};

// サーフェス src[in_off .. in_off+n) の最大値。dst[out_off..+8] = (score, idx, left, right, up, down, ix, iy)
ALIGN.argmax = {
  fields: [["in_off", "u32"], ["n", "u32"], ["nx", "u32"], ["out_off", "u32"]],
  bindings: ["r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { in_off: u32, n: u32, nx: u32, out_off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var best = -1e30; var bidx = 0u;
  for (var i = l.x; i < p.n; i += 256u) {
    let v = src[p.in_off + i];
    if (v > best) { best = v; bidx = i; }
  }
  bv[l.x] = best; bi[l.x] = bidx;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (l.x < st) {
      let o = l.x + st;
      if (bv[o] > bv[l.x] || (bv[o] == bv[l.x] && bi[o] < bi[l.x])) { bv[l.x] = bv[o]; bi[l.x] = bi[o]; }
    }
    workgroupBarrier();
  }
  if (l.x == 0u) {
    let idx = bi[0]; let ix = idx % p.nx; let iy = idx / p.nx; let ny = p.n / p.nx;
    dst[p.out_off + 0u] = bv[0];
    dst[p.out_off + 1u] = f32(idx);
    dst[p.out_off + 2u] = select(-2.0, src[p.in_off + idx - 1u], ix > 0u);
    dst[p.out_off + 3u] = select(-2.0, src[p.in_off + idx + 1u], ix + 1u < p.nx);
    dst[p.out_off + 4u] = select(-2.0, src[p.in_off + idx - p.nx], iy > 0u);
    dst[p.out_off + 5u] = select(-2.0, src[p.in_off + idx + p.nx], iy + 1u < ny);
    dst[p.out_off + 6u] = f32(ix);
    dst[p.out_off + 7u] = f32(iy);
  }
}`,
};

// argmax の結果 (ix, iy) = res[res_off+6, +7] の周囲 ±r をサーフェス上で -2 にする（次の argmax で別のピークを得るため）
ALIGN.suppress = {
  fields: [["in_off", "u32"], ["nx", "u32"], ["ny", "u32"], ["res_off", "u32"], ["r", "i32"]],
  bindings: ["r", "rw"], wg: [64, 1, 1],
  code: /* wgsl */ `
struct P { in_off: u32, nx: u32, ny: u32, res_off: u32, r: i32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> res: array<f32>;
@group(0) @binding(2) var<storage, read_write> surf: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  let ix = i32(res[p.res_off + 6u]); let iy = i32(res[p.res_off + 7u]);
  let d = 2 * p.r + 1;
  let n = u32(d * d);
  for (var t = l.x; t < n; t += 64u) {
    let x = ix + i32(t % u32(d)) - p.r; let y = iy + i32(t / u32(d)) - p.r;
    if (x >= 0 && x < i32(p.nx) && y >= 0 && y < i32(p.ny)) { surf[p.in_off + u32(y) * p.nx + u32(x)] = -2.0; }
  }
}`,
};

const GN_COMMON = /* wgsl */ `
struct P { hs: u32, ws: u32, hb: u32, wb: u32, lv: f32, slot: u32, nbins: u32, rmax: f32, model: u32, nwg: u32, last: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;
@group(0) @binding(5) var<storage, read_write> state: array<f32>;

// B を (Wx, Wy) で双一次サンプル（ゼロパディング）。戻り値 (hp, valid, gx, gy)
fn sampB(Wx: f32, Wy: f32) -> vec4<f32> {
  let wb = i32(p.wb); let hb = i32(p.hb);
  let x0f = floor(Wx); let y0f = floor(Wy);
  let fx = Wx - x0f; let fy = Wy - y0f;
  let x0 = i32(x0f); let y0 = i32(y0f);
  var acc = vec4<f32>(0.0);
  for (var t = 0; t < 4; t++) {
    let xi = x0 + (t & 1); let yi = y0 + (t >> 1);
    let w = select(1.0 - fx, fx, (t & 1) == 1) * select(1.0 - fy, fy, (t >> 1) == 1);
    if (xi >= 0 && xi < wb && yi >= 0 && yi < hb && w > 0.0) {
      let c = B[yi * wb + xi];
      let gx = 0.5 * (B[yi * wb + min(xi + 1, wb - 1)].x - B[yi * wb + max(xi - 1, 0)].x);
      let gy = 0.5 * (B[min(yi + 1, hb - 1) * wb + xi].x - B[max(yi - 1, 0) * wb + xi].x);
      acc += w * vec4<f32>(c.x, c.y, gx, gy);
    }
  }
  return acc;
}
struct Pose { s: f32, c: f32, sn: f32, tlx: f32, tly: f32 }
fn pose_of(base: u32) -> Pose {
  let ls = state[base]; let th = state[base + 1u]; let tx = state[base + 2u]; let ty = state[base + 3u];
  let s = exp(ls); let c = cos(th); let sn = sin(th);
  let off = 0.5 * p.lv - 0.5;
  let tlx = p.lv * tx + off + (1.0 - p.lv) * s * 0.5 * (c - sn);
  let tly = p.lv * ty + off + (1.0 - p.lv) * s * 0.5 * (sn + c);
  return Pose(s, c, sn, tlx, tly);
}
`;

// パス 1: 有効画素の |r| ヒストグラム（MAD 用）と有効画素数 hist[nbins]
ALIGN.gn_resid = {
  fields: [["hs", "u32"], ["ws", "u32"], ["hb", "u32"], ["wb", "u32"], ["lv", "f32"], ["slot", "u32"], ["nbins", "u32"],
           ["rmax", "f32"], ["model", "u32"], ["nwg", "u32"], ["last", "u32"]],
  bindings: ["r", "r", "rw", "rw", "rw"], wg: [128, 1, 1],
  code: GN_COMMON + /* wgsl */ `
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let base = p.slot * 16u;
  if (state[base + 6u] > 0.5 || state[base + 7u] > 0.5) { return; }
  let q = pose_of(base);
  let n = p.hs * p.ws;
  let stride = p.nwg * 128u;
  for (var i = g.x; i < n; i += stride) {
    let x = f32(i % p.ws); let y = f32(i / p.ws);
    let u = (x - q.tlx) / q.s; let v = (y - q.tly) / q.s;
    let Wx = q.c * u + q.sn * v; let Wy = -q.sn * u + q.c * v;
    if (Wx < 0.0 || Wx > f32(p.wb - 1u) || Wy < 0.0 || Wy > f32(p.hb - 1u)) { continue; }
    let a = A[i];
    if (a.y <= 0.5) { continue; }
    let sb = sampB(Wx, Wy);
    if (sb.y <= 0.5) { continue; }
    let r = a.x - sb.x;
    let bin = min(p.nbins - 1u, u32(abs(r) * f32(p.nbins) / p.rmax));
    atomicAdd(&hist[bin], 1u);
    atomicAdd(&hist[p.nbins], 1u);
  }
}`,
};

// パス 2: Huber 重み付き正規方程式の部分和 partials[wg*20 + q]
//   q: H の上三角 10 個 (00,01,02,03,11,12,13,22,23,33), g 4 個, スコア和 5 個 (a, b, aa, bb, ab), 個数 1
ALIGN.gn_accum = {
  fields: ALIGN.gn_resid.fields,
  bindings: ["r", "r", "rw", "rw", "rw"], wg: [128, 1, 1],
  code: GN_COMMON + /* wgsl */ `
var<workgroup> red: array<f32, 2560>;
var<workgroup> cth_s: f32;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let base = p.slot * 16u;
  // 注意: 短絡評価の a || b をバリア前に置くと一部環境で誤コンパイルされるため max で判定
  let skip = max(state[base + 6u], state[base + 7u]) > 0.5;
  if (l.x == 0u) {
    let cnt = atomicLoad(&hist[p.nbins]);
    let tgt = select(0u, (cnt - 1u) / 2u, cnt > 0u);
    var cum = 0u; var b = 0u;
    for (; b < p.nbins; b++) { cum += atomicLoad(&hist[b]); if (cum > tgt) { break; } }
    let med = (f32(min(b, p.nbins - 1u)) + 0.5) * p.rmax / f32(p.nbins);
    cth_s = 1.345 * (med * 1.4826 + 1e-6);
  }
  workgroupBarrier();
  let cth = cth_s;
  var acc: array<f32, 20>;
  for (var q = 0u; q < 20u; q++) { acc[q] = 0.0; }
  if (!skip) {
    let po = pose_of(base);
    let n = p.hs * p.ws;
    let stride = p.nwg * 128u;
    for (var i = g.x; i < n; i += stride) {
      let x = f32(i % p.ws); let y = f32(i / p.ws);
      let u = (x - po.tlx) / po.s; let v = (y - po.tly) / po.s;
      let Wx = po.c * u + po.sn * v; let Wy = -po.sn * u + po.c * v;
      if (Wx < 0.0 || Wx > f32(p.wb - 1u) || Wy < 0.0 || Wy > f32(p.hb - 1u)) { continue; }
      let a = A[i];
      if (a.y <= 0.5) { continue; }
      let sb = sampB(Wx, Wy);
      if (sb.y <= 0.5) { continue; }
      let r = a.x - sb.x;
      let Bx = sb.z; let By = sb.w;
      var J: array<f32, 4>;
      J[0] = -(Bx * Wx + By * Wy);
      J[1] = Bx * Wy - By * Wx;
      J[2] = -(Bx * po.c - By * po.sn) / po.s;
      J[3] = -(Bx * po.sn + By * po.c) / po.s;
      let ar = abs(r);
      let w = select(cth / max(ar, 1e-12), 1.0, ar <= cth);
      var q = 0u;
      for (var i1 = 0u; i1 < 4u; i1++) {
        for (var i2 = i1; i2 < 4u; i2++) { acc[q] += w * J[i1] * J[i2]; q++; }
      }
      for (var i1 = 0u; i1 < 4u; i1++) { acc[10u + i1] += w * J[i1] * r; }
      acc[14] += a.x; acc[15] += sb.x; acc[16] += a.x * a.x; acc[17] += sb.x * sb.x; acc[18] += a.x * sb.x;
      acc[19] += 1.0;
    }
  }
  for (var q = 0u; q < 20u; q++) { red[l.x * 20u + q] = acc[q]; }
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) {
    if (l.x < st) { for (var q = 0u; q < 20u; q++) { red[l.x * 20u + q] += red[(l.x + st) * 20u + q]; } }
    workgroupBarrier();
  }
  if (l.x == 0u) { for (var q = 0u; q < 20u; q++) { partials[wid.x * 20u + q] = red[q]; } }
}`,
};

// パス 3: 部分和を合計し、正規方程式を解いて状態を更新（1 ワークグループ）
ALIGN.gn_solve = {
  fields: ALIGN.gn_resid.fields,
  bindings: ["r", "r", "rw", "rw", "rw"], wg: [128, 1, 1],
  code: GN_COMMON + /* wgsl */ `
var<workgroup> red: array<f32, 2560>;
fn hidx(i1: u32, i2: u32) -> u32 {
  let a = min(i1, i2); let b = max(i1, i2);
  // 上三角の線形インデックス
  return a * 4u - (a * (a - 1u)) / 2u + (b - a);
}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var acc: array<f32, 20>;
  for (var q = 0u; q < 20u; q++) { acc[q] = 0.0; }
  for (var w = l.x; w < p.nwg; w += 128u) { for (var q = 0u; q < 20u; q++) { acc[q] += partials[w * 20u + q]; } }
  for (var q = 0u; q < 20u; q++) { red[l.x * 20u + q] = acc[q]; }
  workgroupBarrier();
  for (var st = 64u; st > 0u; st >>= 1u) {
    if (l.x < st) { for (var q = 0u; q < 20u; q++) { red[l.x * 20u + q] += red[(l.x + st) * 20u + q]; } }
    workgroupBarrier();
  }
  if (l.x != 0u) { return; }
  let base = p.slot * 16u;
  if (state[base + 6u] > 0.5) { return; }
  if (state[base + 7u] > 0.5) { if (p.last == 1u) { state[base + 7u] = 0.0; } return; }
  let n = red[19];
  if (n < 0.02 * f32(p.hs * p.ws)) { state[base + 6u] = 1.0; return; }
  // スコア（有効領域の中心化 NCC）
  let sa = red[14]; let sb = red[15]; let saa = red[16]; let sbb = red[17]; let sab = red[18];
  let num = sab - sa * sb / n;
  let den = sqrt(max(saa - sa * sa / n, 0.0) * max(sbb - sb * sb / n, 0.0)) + 1e-12;
  state[base + 4u] = num / den;
  state[base + 5u] = n;
  // 列の選択
  var cols: array<u32, 4>;
  var P = 4u;
  if (p.model == 0u) { P = 2u; cols[0] = 2u; cols[1] = 3u; }
  else if (p.model == 1u) { P = 3u; cols[0] = 0u; cols[1] = 2u; cols[2] = 3u; }
  else { cols[0] = 0u; cols[1] = 1u; cols[2] = 2u; cols[3] = 3u; }
  var M: array<f32, 20>; // 4x5 拡大行列
  var tr = 0.0;
  for (var a = 0u; a < P; a++) { tr += red[hidx(cols[a], cols[a])]; }
  let damp = 1e-6 * tr / f32(P);
  for (var a = 0u; a < P; a++) {
    for (var b = 0u; b < P; b++) { M[a * 5u + b] = red[hidx(cols[a], cols[b])] + select(0.0, damp, a == b); }
    M[a * 5u + 4u] = red[10u + cols[a]];
  }
  // ガウス消去（部分ピボット）
  var ok = true;
  for (var c = 0u; c < P; c++) {
    var piv = c; var pv = abs(M[c * 5u + c]);
    for (var r = c + 1u; r < P; r++) { if (abs(M[r * 5u + c]) > pv) { pv = abs(M[r * 5u + c]); piv = r; } }
    if (pv < 1e-30) { ok = false; break; }
    if (piv != c) { for (var k = 0u; k < 5u; k++) { let t = M[c * 5u + k]; M[c * 5u + k] = M[piv * 5u + k]; M[piv * 5u + k] = t; } }
    for (var r = 0u; r < P; r++) {
      if (r == c) { continue; }
      let f = M[r * 5u + c] / M[c * 5u + c];
      for (var k = c; k < 5u; k++) { M[r * 5u + k] -= f * M[c * 5u + k]; }
    }
  }
  var upd = vec4<f32>(0.0);
  if (ok) {
    for (var a = 0u; a < P; a++) {
      let d = M[a * 5u + 4u] / M[a * 5u + a];
      if (d != d || abs(d) > 1e30) { ok = false; }
      upd[cols[a]] = d;
    }
  }
  if (!ok) { state[base + 6u] = 1.0; return; }
  // 更新: レベル座標の平行移動に加算し、新しい s, th でフル解像度座標へ戻す
  let q = pose_of(base);
  let ls = state[base] + upd.x;
  let th = state[base + 1u] + upd.y;
  let tlx = q.tlx + upd.z;
  let tly = q.tly + upd.w;
  let s = exp(ls); let c = cos(th); let sn = sin(th);
  let off = 0.5 * p.lv - 0.5;
  state[base] = ls;
  state[base + 1u] = th;
  state[base + 2u] = (tlx - off - (1.0 - p.lv) * s * 0.5 * (c - sn)) / p.lv;
  state[base + 3u] = (tly - off - (1.0 - p.lv) * s * 0.5 * (sn + c)) / p.lv;
  if (abs(upd.x) < 1e-6 && abs(upd.y) < 1e-6 && abs(upd.z) < 2e-3 && abs(upd.w) < 2e-3 && p.last == 0u) {
    state[base + 7u] = 1.0;
  }
}`,
};

// ------------------------------------------------------------ ホモグラフィ（8 自由度）+ 任意で放射歪み k1 の Gauss-Newton
// 状態 state[slot*16 + ...] = (g0..g7, score, n_valid, fail, conv, k1, -, -, -)。G は正規化座標で x_j = π(G x_i)。
// 正規化座標: xn = (x - cx_lv) / f_lv、f_lv = lv f_full（全レベルで同じ座標）。use_k1=1 なら A の画素を歪み除去 → G → 歪み付与して B をサンプル
// （A を再標本化しないので、補正画像のぼけで残差が減る方向に k1 が引かれない）。k1 の微分は数値微分（eps）
const GNH_COMMON = /* wgsl */ `
struct P { hs: u32, ws: u32, hb: u32, wb: u32, lv: f32, slot: u32, nbins: u32, rmax: f32, nwg: u32, last: u32, use_k1: u32,
           f_full: f32, cx_full: f32, cy_full: f32, kap_scale: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> B: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> partials: array<f32>;
@group(0) @binding(5) var<storage, read_write> state: array<f32>;

fn sampB(Wx: f32, Wy: f32) -> vec4<f32> {
  let wb = i32(p.wb); let hb = i32(p.hb);
  let x0f = floor(Wx); let y0f = floor(Wy);
  let fx = Wx - x0f; let fy = Wy - y0f;
  let x0 = i32(x0f); let y0 = i32(y0f);
  var acc = vec4<f32>(0.0);
  for (var t = 0; t < 4; t++) {
    let xi = x0 + (t & 1); let yi = y0 + (t >> 1);
    let w = select(1.0 - fx, fx, (t & 1) == 1) * select(1.0 - fy, fy, (t >> 1) == 1);
    if (xi >= 0 && xi < wb && yi >= 0 && yi < hb && w > 0.0) {
      let c = B[yi * wb + xi];
      let gx = 0.5 * (B[yi * wb + min(xi + 1, wb - 1)].x - B[yi * wb + max(xi - 1, 0)].x);
      let gy = 0.5 * (B[min(yi + 1, hb - 1) * wb + xi].x - B[max(yi - 1, 0) * wb + xi].x);
      acc += w * vec4<f32>(c.x, c.y, gx, gy);
    }
  }
  return acc;
}
// A のレベル画素 (x, y) → 正規化 (xu, yu)（歪み除去後）→ G → (un, vn), Z → B のレベル画素 (Wx, Wy)（歪み付与後）
struct Map { xu: f32, yu: f32, un: f32, vn: f32, Z: f32, Wx: f32, Wy: f32 }
fn map_px(x: f32, y: f32, base: u32, k1: f32) -> Map {
  let f_lv = p.lv * p.f_full;
  let cx_lv = p.lv * (p.cx_full + 0.5) - 0.5; let cy_lv = p.lv * (p.cy_full + 0.5) - 0.5;
  let xdn = (x - cx_lv) / f_lv; let ydn = (y - cy_lv) / f_lv;
  var xu = xdn; var yu = ydn;
  let kap = k1 * p.kap_scale;
  if (p.use_k1 == 1u) {
    for (var t = 0; t < 6; t++) { let r2 = xu * xu + yu * yu; xu = xdn / (1.0 + kap * r2); yu = ydn / (1.0 + kap * r2); }
  }
  let g0 = state[base]; let g1 = state[base + 1u]; let g2 = state[base + 2u];
  let g3 = state[base + 3u]; let g4 = state[base + 4u]; let g5 = state[base + 5u];
  let g6 = state[base + 6u]; let g7 = state[base + 7u];
  var Z = g6 * xu + g7 * yu + 1.0;
  let Zs = select(Z, 1e-6, abs(Z) < 1e-6);
  let un = (g0 * xu + g1 * yu + g2) / Zs;
  let vn = (g3 * xu + g4 * yu + g5) / Zs;
  var fct = 1.0;
  if (p.use_k1 == 1u) { fct = 1.0 + kap * (un * un + vn * vn); }
  return Map(xu, yu, un, vn, Zs, un * fct * f_lv + cx_lv, vn * fct * f_lv + cy_lv);
}
fn inside_b(mp: Map) -> bool {
  return mp.Wx >= 0.0 && mp.Wx <= f32(p.wb - 1u) && mp.Wy >= 0.0 && mp.Wy <= f32(p.hb - 1u) && mp.Z > 0.05;
}
`;

ALIGN.gnh_resid = {
  fields: [["hs", "u32"], ["ws", "u32"], ["hb", "u32"], ["wb", "u32"], ["lv", "f32"], ["slot", "u32"], ["nbins", "u32"],
           ["rmax", "f32"], ["nwg", "u32"], ["last", "u32"], ["use_k1", "u32"], ["f_full", "f32"], ["cx_full", "f32"], ["cy_full", "f32"], ["kap_scale", "f32"]],
  bindings: ["r", "r", "rw", "rw", "rw"], wg: [128, 1, 1],
  code: GNH_COMMON + /* wgsl */ `
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let base = p.slot * 16u;
  if (state[base + 10u] > 0.5 || state[base + 11u] > 0.5) { return; }
  let k1 = state[base + 12u];
  let n = p.hs * p.ws;
  let stride = p.nwg * 128u;
  for (var i = g.x; i < n; i += stride) {
    let x = f32(i % p.ws); let y = f32(i / p.ws);
    let mp = map_px(x, y, base, k1);
    if (!inside_b(mp)) { continue; }
    let a = A[i];
    if (a.y <= 0.5) { continue; }
    let sb = sampB(mp.Wx, mp.Wy);
    if (sb.y <= 0.5) { continue; }
    let r = a.x - sb.x;
    let bin = min(p.nbins - 1u, u32(abs(r) * f32(p.nbins) / p.rmax));
    atomicAdd(&hist[bin], 1u);
    atomicAdd(&hist[p.nbins], 1u);
  }
}`,
};

// 部分和 partials[wg*60 + q]: q = 9x9 上三角 45 個, g 9 個, スコア和 5 個 (a, b, aa, bb, ab), 個数 1
ALIGN.gnh_accum = {
  fields: ALIGN.gnh_resid.fields,
  bindings: ["r", "r", "rw", "rw", "rw"], wg: [64, 1, 1],
  code: GNH_COMMON + /* wgsl */ `
var<workgroup> red: array<f32, 3840>;
var<workgroup> cth_s: f32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let base = p.slot * 16u;
  let skip = max(state[base + 10u], state[base + 11u]) > 0.5;
  if (l.x == 0u) {
    let cnt = atomicLoad(&hist[p.nbins]);
    let tgt = select(0u, (cnt - 1u) / 2u, cnt > 0u);
    var cum = 0u; var b = 0u;
    for (; b < p.nbins; b++) { cum += atomicLoad(&hist[b]); if (cum > tgt) { break; } }
    let med = (f32(min(b, p.nbins - 1u)) + 0.5) * p.rmax / f32(p.nbins);
    cth_s = 1.345 * (med * 1.4826 + 1e-6);
  }
  workgroupBarrier();
  let cth = cth_s;
  var acc: array<f32, 60>;
  for (var q = 0u; q < 60u; q++) { acc[q] = 0.0; }
  if (!skip) {
    let k1 = state[base + 12u];
    let f_lv = p.lv * p.f_full;
    let eps_k = 2e-3;
    let n = p.hs * p.ws;
    let stride = p.nwg * 64u;
    let NP = select(8u, 9u, p.use_k1 == 1u);
    for (var i = g.x; i < n; i += stride) {
      let x = f32(i % p.ws); let y = f32(i / p.ws);
      let mp = map_px(x, y, base, k1);
      if (!inside_b(mp)) { continue; }
      let a = A[i];
      if (a.y <= 0.5) { continue; }
      let sb = sampB(mp.Wx, mp.Wy);
      if (sb.y <= 0.5) { continue; }
      let r = a.x - sb.x;
      let bx = sb.z * f_lv; let by = sb.w * f_lv;
      let iz = 1.0 / mp.Z;
      var J: array<f32, 9>;
      J[0] = bx * mp.xu * iz; J[1] = bx * mp.yu * iz; J[2] = bx * iz;
      J[3] = by * mp.xu * iz; J[4] = by * mp.yu * iz; J[5] = by * iz;
      let t = bx * mp.un + by * mp.vn;
      J[6] = -t * mp.xu * iz; J[7] = -t * mp.yu * iz;
      J[8] = 0.0;
      if (p.use_k1 == 1u) {
        let mp2 = map_px(x, y, base, k1 + eps_k);
        let sb2 = sampB(mp2.Wx, mp2.Wy);
        let r2 = a.x - sb2.x;
        J[8] = -(r2 - r) / eps_k;
      }
      let ar = abs(r);
      let w = select(cth / max(ar, 1e-12), 1.0, ar <= cth);
      var q = 0u;
      for (var i1 = 0u; i1 < NP; i1++) {
        for (var i2 = i1; i2 < 9u; i2++) { if (i2 < NP) { acc[i1 * 9u - (i1 * (i1 + 1u)) / 2u + i2] += w * J[i1] * J[i2]; } }
      }
      for (var i1 = 0u; i1 < NP; i1++) { acc[45u + i1] += w * J[i1] * r; }
      acc[54] += a.x; acc[55] += sb.x; acc[56] += a.x * a.x; acc[57] += sb.x * sb.x; acc[58] += a.x * sb.x;
      acc[59] += 1.0;
    }
  }
  for (var q = 0u; q < 60u; q++) { red[l.x * 60u + q] = acc[q]; }
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (l.x < st) { for (var q = 0u; q < 60u; q++) { red[l.x * 60u + q] += red[(l.x + st) * 60u + q]; } }
    workgroupBarrier();
  }
  if (l.x == 0u) { for (var q = 0u; q < 60u; q++) { partials[wid.x * 60u + q] = red[q]; } }
}`,
};

ALIGN.gnh_solve = {
  fields: ALIGN.gnh_resid.fields,
  bindings: ["r", "r", "rw", "rw", "rw"], wg: [64, 1, 1],
  code: GNH_COMMON + /* wgsl */ `
var<workgroup> red: array<f32, 3840>;
fn hidx(i1: u32, i2: u32) -> u32 {
  let a = min(i1, i2); let b = max(i1, i2);
  return a * 9u - (a * (a + 1u)) / 2u + b;
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var acc: array<f32, 60>;
  for (var q = 0u; q < 60u; q++) { acc[q] = 0.0; }
  for (var w = l.x; w < p.nwg; w += 64u) { for (var q = 0u; q < 60u; q++) { acc[q] += partials[w * 60u + q]; } }
  for (var q = 0u; q < 60u; q++) { red[l.x * 60u + q] = acc[q]; }
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (l.x < st) { for (var q = 0u; q < 60u; q++) { red[l.x * 60u + q] += red[(l.x + st) * 60u + q]; } }
    workgroupBarrier();
  }
  if (l.x != 0u) { return; }
  let base = p.slot * 16u;
  if (state[base + 10u] > 0.5) { return; }
  if (state[base + 11u] > 0.5) { if (p.last == 1u) { state[base + 11u] = 0.0; } return; }
  let n = red[59];
  if (n < 0.02 * f32(p.hs * p.ws)) { state[base + 10u] = 1.0; return; }
  let sa = red[54]; let sb = red[55]; let saa = red[56]; let sbb = red[57]; let sab = red[58];
  let num = sab - sa * sb / n;
  let den = sqrt(max(saa - sa * sa / n, 0.0) * max(sbb - sb * sb / n, 0.0)) + 1e-12;
  state[base + 8u] = num / den;
  state[base + 9u] = n;
  let NP = select(8u, 9u, p.use_k1 == 1u);
  var M: array<f32, 90>; // 9x10 拡大行列
  for (var a = 0u; a < NP; a++) {
    let d = red[hidx(a, a)];
    for (var b = 0u; b < NP; b++) { M[a * 10u + b] = red[hidx(a, b)] + select(0.0, d * 1e-4 + 1e-12, a == b); }
    M[a * 10u + 9u] = red[45u + a];
  }
  var ok = true;
  for (var c = 0u; c < NP; c++) {
    var piv = c; var pv = abs(M[c * 10u + c]);
    for (var r = c + 1u; r < NP; r++) { if (abs(M[r * 10u + c]) > pv) { pv = abs(M[r * 10u + c]); piv = r; } }
    if (pv < 1e-30) { ok = false; break; }
    if (piv != c) { for (var k = 0u; k < 10u; k++) { let t = M[c * 10u + k]; M[c * 10u + k] = M[piv * 10u + k]; M[piv * 10u + k] = t; } }
    for (var r = 0u; r < NP; r++) {
      if (r == c) { continue; }
      let f = M[r * 10u + c] / M[c * 10u + c];
      for (var k = c; k < 10u; k++) { M[r * 10u + k] -= f * M[c * 10u + k]; }
    }
  }
  var upd: array<f32, 9>;
  var mx = 0.0;
  if (ok) {
    for (var a = 0u; a < NP; a++) {
      let d = M[a * 10u + 9u] / M[a * 10u + a];
      if (d != d || abs(d) > 1e30) { ok = false; }
      upd[a] = d;
      if (a < 8u) { mx = max(mx, abs(d)); }
    }
  }
  if (!ok) { state[base + 10u] = 1.0; return; }
  for (var a = 0u; a < 8u; a++) { state[base + a] += upd[a]; }
  var conv = mx < 3e-6;
  if (p.use_k1 == 1u) {
    state[base + 12u] = clamp(state[base + 12u] + upd[8], -0.6, 0.6);
    conv = conv && abs(upd[8]) < 1e-5;
  }
  if (conv && p.last == 0u) { state[base + 11u] = 1.0; }
}`,
};
