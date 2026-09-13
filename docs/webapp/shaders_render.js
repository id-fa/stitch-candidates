// shaders_render.js - 合成カーネル
// - gauss_rgba: フレームの行範囲だけをガウスぼかし（縮小時のプリフィルタ）
// - warp: フレームをキャンバス帯域に相似変換で投影し、サンプルスタックに書く
//     word0 = r | g<<8 | b<<16 | flags<<24 (bit0: 幾何的有効, bit1: クリーン), word1 = 鮮明度 f32 のビット
// - median: ピクセルごとに時間方向中央値 / インライア平均 / 鮮明度上位平均 / 被覆数

export const RENDER = {};

RENDER.gauss_rgba = {
  fields: [["W", "u32"], ["H", "u32"], ["sigma", "f32"], ["axis", "u32"], ["row0", "u32"], ["row1", "u32"]],
  bindings: ["r", "r", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { W: u32, H: u32, sigma: f32, axis: u32, row0: u32, row1: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read> tmp: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
fn rgb_of(c: u32) -> vec4<f32> {
  return vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), 0.0);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = i32(g.x); let y = i32(p.row0) + i32(g.y);
  let W = i32(p.W); let H = i32(p.H);
  if (x >= W || y >= i32(p.row1) || y >= H) { return; }
  let r = min(12, max(1, i32(ceil(3.0 * p.sigma))));
  var s = vec4<f32>(0.0); var n = 0.0;
  for (var t = -r; t <= r; t++) {
    let w = exp(-0.5 * (f32(t) / p.sigma) * (f32(t) / p.sigma));
    var v: vec4<f32>;
    if (p.axis == 0u) { v = rgb_of(frame[y * W + clamp(x + t, 0, W - 1)]); }
    else { v = tmp[clamp(y + t, 0, H - 1) * W + x]; }
    s += w * v; n += w;
  }
  dst[y * W + x] = s / n;
}`,
};

RENDER.warp = {
  fields: [["Wc", "u32"], ["y0", "u32"], ["bh", "u32"], ["W", "u32"], ["H", "u32"], ["Wq", "u32"], ["Hq", "u32"],
           ["slot", "u32"], ["sharp_off", "u32"], ["use_blur", "u32"],
           ["s", "f32"], ["c", "f32"], ["sn", "f32"], ["Tx", "f32"], ["Ty", "f32"]],
  bindings: ["r", "r", "r", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { Wc: u32, y0: u32, bh: u32, W: u32, H: u32, Wq: u32, Hq: u32, slot: u32, sharp_off: u32, use_blur: u32,
           s: f32, c: f32, sn: f32, Tx: f32, Ty: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> sharpq: array<f32>;
@group(0) @binding(4) var<storage, read_write> stack: array<u32>;
fn rgba_of(c: u32) -> vec4<f32> {
  return vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), f32(c >> 24u));
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let yy = g.y;
  if (x >= p.Wc || yy >= p.bh) { return; }
  let o = ((p.slot * p.bh + yy) * p.Wc + x) * 2u;
  let X = f32(x); let Y = f32(p.y0 + yy);
  let u = (X - p.Tx) / p.s; let v = (Y - p.Ty) / p.s;
  let fx = p.c * u + p.sn * v; let fy = -p.sn * u + p.c * v;
  let W = i32(p.W); let H = i32(p.H);
  let eps = 1e-3;
  if (fx < -eps || fx > f32(W - 1) + eps || fy < -eps || fy > f32(H - 1) + eps) {
    stack[o] = 0u; stack[o + 1u] = 0u; return;
  }
  let x0f = floor(fx); let y0f = floor(fy);
  let ax = fx - x0f; let ay = fy - y0f;
  let x0 = clamp(i32(x0f), 0, W - 1); let x1 = clamp(i32(x0f) + 1, 0, W - 1);
  let y0 = clamp(i32(y0f), 0, H - 1); let y1 = clamp(i32(y0f) + 1, 0, H - 1);
  let w00 = (1.0 - ax) * (1.0 - ay); let w10 = ax * (1.0 - ay); let w01 = (1.0 - ax) * ay; let w11 = ax * ay;
  let f00 = rgba_of(frame[y0 * W + x0]); let f10 = rgba_of(frame[y0 * W + x1]);
  let f01 = rgba_of(frame[y1 * W + x0]); let f11 = rgba_of(frame[y1 * W + x1]);
  var col = f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11;
  if (p.use_blur == 1u) {
    let b = blurred[y0 * W + x0] * w00 + blurred[y0 * W + x1] * w10 + blurred[y1 * W + x0] * w01 + blurred[y1 * W + x1] * w11;
    col = vec4<f32>(b.x, b.y, b.z, col.w);
  }
  let clean = col.w > 127.5;
  // 鮮明度（1/4 マップ、align_corners=False の拡大に相当）
  let qx = clamp((fx + 0.5) * f32(p.Wq) / f32(p.W) - 0.5, 0.0, f32(p.Wq - 1u));
  let qy = clamp((fy + 0.5) * f32(p.Hq) / f32(p.H) - 0.5, 0.0, f32(p.Hq - 1u));
  let qx0 = i32(floor(qx)); let qy0 = i32(floor(qy));
  let bx = qx - f32(qx0); let by = qy - f32(qy0);
  let qx1 = min(qx0 + 1, i32(p.Wq) - 1); let qy1 = min(qy0 + 1, i32(p.Hq) - 1);
  let Wq = i32(p.Wq);
  let sh = sharpq[p.sharp_off + u32(qy0 * Wq + qx0)] * (1.0 - bx) * (1.0 - by) + sharpq[p.sharp_off + u32(qy0 * Wq + qx1)] * bx * (1.0 - by)
         + sharpq[p.sharp_off + u32(qy1 * Wq + qx0)] * (1.0 - bx) * by + sharpq[p.sharp_off + u32(qy1 * Wq + qx1)] * bx * by;
  let r = u32(clamp(col.x + 0.5, 0.0, 255.0)); let gg = u32(clamp(col.y + 0.5, 0.0, 255.0)); let b = u32(clamp(col.z + 0.5, 0.0, 255.0));
  let flags = 1u | select(0u, 2u, clean);
  stack[o] = r | (gg << 8u) | (b << 16u) | (flags << 24u);
  stack[o + 1u] = bitcast<u32>(max(sh, 0.0));
}`,
};

// 時間方向の統計。基数選択（ニブル単位）で中央値と分位点を求める
RENDER.median = {
  fields: [["Wc", "u32"], ["bh", "u32"], ["K", "u32"], ["y0", "u32"], ["tol", "f32"], ["sharp_top", "f32"]],
  bindings: ["r", "rw", "rw", "rw", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { Wc: u32, bh: u32, K: u32, y0: u32, tol: f32, sharp_top: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> stack: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_med: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_mean: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_sharp: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_cov: array<u32>;

fn chan(w0: u32, ch: u32) -> u32 { return (w0 >> (8u * ch)) & 255u; }
fn rgb3(w0: u32) -> vec3<f32> { return vec3<f32>(f32(w0 & 255u), f32((w0 >> 8u) & 255u), f32((w0 >> 16u) & 255u)); }
// mode 0: clean, 1: geometric, 2: inlier (clean & dist<tol), 3: fallback inlier (geom & dist<2tol)
fn ok(w0: u32, mode: u32, med: vec3<f32>) -> bool {
  let fl = w0 >> 24u;
  let geom = (fl & 1u) != 0u; let clean = (fl & 2u) != 0u;
  if (mode == 0u) { return geom && clean; }
  if (mode == 1u) { return geom; }
  let d = dot(abs(rgb3(w0) - med), vec3<f32>(1.0 / 3.0));
  if (mode == 2u) { return geom && clean && d < p.tol; }
  return geom && d < 2.0 * p.tol;
}
fn val(w0: u32, w1: u32, ch: u32) -> u32 { if (ch < 3u) { return chan(w0, ch); } return w1; }
// 条件 mode を満たすサンプルの ch 値（昇順 kth 番目、0 始まり）を基数選択で求める
fn kth_value(base: u32, stride: u32, ch: u32, mode: u32, med: vec3<f32>, kth_in: u32) -> u32 {
  var kth = kth_in;
  var prefix = 0u;
  let npass = select(8u, 2u, ch < 3u);
  for (var ps = 0u; ps < npass; ps++) {
    let shift = (npass - 1u - ps) * 4u;
    var hist: array<u32, 16>;
    for (var b = 0u; b < 16u; b++) { hist[b] = 0u; }
    for (var k = 0u; k < p.K; k++) {
      let w0 = stack[base + k * stride]; let w1 = stack[base + k * stride + 1u];
      if (!ok(w0, mode, med)) { continue; }
      let v = val(w0, w1, ch);
      if ((v >> (shift + 4u)) == prefix) { hist[(v >> shift) & 15u] += 1u; }
    }
    var cum = 0u; var b = 0u;
    for (; b < 16u; b++) { if (cum + hist[b] > kth) { break; } cum += hist[b]; }
    b = min(b, 15u);
    kth -= cum;
    prefix = (prefix << 4u) | b;
  }
  return prefix;
}
fn count_mode(base: u32, stride: u32, mode: u32, med: vec3<f32>) -> u32 {
  var n = 0u;
  for (var k = 0u; k < p.K; k++) { if (ok(stack[base + k * stride], mode, med)) { n++; } }
  return n;
}
fn pack(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.x + 0.5, 0.0, 255.0)); let g = u32(clamp(c.y + 0.5, 0.0, 255.0)); let b = u32(clamp(c.z + 0.5, 0.0, 255.0));
  return r | (g << 8u) | (b << 16u) | (255u << 24u);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let yy = g.y;
  if (x >= p.Wc || yy >= p.bh) { return; }
  let base = (yy * p.Wc + x) * 2u;
  let stride = p.bh * p.Wc * 2u;
  let oi = (p.y0 + yy) * p.Wc + x;
  let zero = vec3<f32>(0.0);
  let cnt_c = count_mode(base, stride, 0u, zero);
  let cnt_g = count_mode(base, stride, 1u, zero);
  if (cnt_g == 0u) {
    out_med[oi] = 0u; out_mean[oi] = 0u; out_sharp[oi] = 0u; out_cov[oi] = 0u; return;
  }
  let mmode = select(1u, 0u, cnt_c > 0u);
  let mcnt = select(cnt_g, cnt_c, cnt_c > 0u);
  let kmed = (mcnt - 1u) / 2u;
  let med = vec3<f32>(f32(kth_value(base, stride, 0u, mmode, zero, kmed)),
                      f32(kth_value(base, stride, 1u, mmode, zero, kmed)),
                      f32(kth_value(base, stride, 2u, mmode, zero, kmed)));
  // インライア
  var imode = 2u;
  var ni = count_mode(base, stride, 2u, med);
  if (ni == 0u) { imode = 3u; ni = count_mode(base, stride, 3u, med); }
  var mean = med;
  var sharp = med;
  if (ni > 0u) {
    var sum = vec3<f32>(0.0);
    for (var k = 0u; k < p.K; k++) { let w0 = stack[base + k * stride]; if (ok(w0, imode, med)) { sum += rgb3(w0); } }
    mean = sum / f32(ni);
    // 鮮明度の上位 sharp_top（分位点 1 - sharp_top 以上）
    let kq = u32(ceil((1.0 - p.sharp_top) * f32(ni - 1u) - 1e-4));
    let thr = kth_value(base, stride, 3u, imode, med, min(kq, ni - 1u));
    var ts = vec3<f32>(0.0); var nt = 0u;
    for (var k = 0u; k < p.K; k++) {
      let w0 = stack[base + k * stride]; let w1 = stack[base + k * stride + 1u];
      if (ok(w0, imode, med) && w1 >= thr) { ts += rgb3(w0); nt++; }
    }
    if (nt > 0u) { sharp = ts / f32(nt); } else { sharp = mean; }
  }
  out_med[oi] = pack(med);
  out_mean[oi] = pack(mean);
  out_sharp[oi] = pack(sharp);
  out_cov[oi] = cnt_c | (cnt_g << 16u);   // 下位 16 bit: クリーン標本数, 上位: 幾何的被覆数
}`,
};
