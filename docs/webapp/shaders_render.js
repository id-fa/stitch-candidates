// shaders_render.js - 合成カーネル
// - gauss_rgba: フレームの行範囲だけをガウスぼかし（縮小時のプリフィルタ）
// - warp: フレームをキャンバスのタイル（行帯または列帯）に相似変換 / ホモグラフィ / 円筒投影で投影し、サンプルスタックに書く
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

//   vw, vh: フレームの有効サイズ（パディング領域は幾何的に無効として扱う）
//   投影 proj: 0 = 相似（s, c, sn, Tx, Ty）, 1 = ホモグラフィ（m = キャンバス → フレームの 3x3）,
//              2 = 円筒（θ = (X - X0)/fc, h = (Y - Y0)/fc, 光線 (sin θ, h, cos θ) に m = K Rᵀ を掛けて透視除算）
//   露出補正: use_expo=1 なら col = (col - off - 255 (f(y/vh) + g(x/vw)) - F_k(x, y)) / (gain (1 + v(r)))。
//   expo バッファ: [knots(nk), fy(nk*3), gx(nk*3), 局所場 F（フレームごと (gx+1)(gy+1)*3、loc_off から）, 放射項（rad_off から: 節点 nr, 値 nr*3）]（0..255 単位、放射項は比）
//   スタックは 1 標本 3 ワード: word0 = rgb+flags, word1 = 鮮明度, word2 = フェザー重み（フレーム端からの距離 / feather、0..1）
RENDER.warp = {
  fields: [["x0", "u32"], ["y0", "u32"], ["tw", "u32"], ["th", "u32"], ["W", "u32"], ["H", "u32"], ["Wq", "u32"], ["Hq", "u32"],
           ["slot", "u32"], ["sharp_off", "u32"], ["use_blur", "u32"],
           ["s", "f32"], ["c", "f32"], ["sn", "f32"], ["Tx", "f32"], ["Ty", "f32"], ["mag_lv", "u32"],
           ["vw", "u32"], ["vh", "u32"], ["use_expo", "u32"], ["nk", "u32"], ["loc_off", "u32"], ["gx", "u32"], ["gy", "u32"], ["feather", "f32"],
           ["proj", "u32"], ["fc", "f32"], ["X0", "f32"], ["Y0", "f32"], ["nr", "u32"], ["rad_off", "u32"],
           ["gain", "vec4f"], ["off", "vec4f"], ["m0", "vec4f"], ["m1", "vec4f"], ["m2", "vec4f"]],
  bindings: ["r", "r", "r", "rw", "r"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { x0: u32, y0: u32, tw: u32, th: u32, W: u32, H: u32, Wq: u32, Hq: u32, slot: u32, sharp_off: u32, use_blur: u32,
           s: f32, c: f32, sn: f32, Tx: f32, Ty: f32, mag_lv: u32, vw: u32, vh: u32, use_expo: u32, nk: u32, loc_off: u32, gx: u32, gy: u32, feather: f32,
           proj: u32, fc: f32, X0: f32, Y0: f32, nr: u32, rad_off: u32,
           gain: vec4<f32>, off: vec4<f32>, m0: vec4<f32>, m1: vec4<f32>, m2: vec4<f32> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read> blurred: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> sharpq: array<f32>;
@group(0) @binding(4) var<storage, read_write> stack: array<u32>;
@group(0) @binding(5) var<storage, read> expo: array<f32>;
fn rgba_of(c: u32) -> vec4<f32> {
  return vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), f32(c >> 24u));
}
// 折れ線プロファイル: 節点 expo[kb .. kb+nk), 値 expo[base + q*3 + c]
fn prof_k(t_in: f32, kb: u32, nk: u32, base: u32) -> vec3<f32> {
  let t = clamp(t_in, 0.0, 1.0);
  var q = 0u;
  for (var i = 1u; i + 1u < nk; i++) { if (t >= expo[kb + i]) { q = i; } }
  let k0 = expo[kb + q]; let k1 = expo[kb + q + 1u];
  let f = clamp((t - k0) / max(k1 - k0, 1e-6), 0.0, 1.0);
  let a = vec3<f32>(expo[base + q * 3u], expo[base + q * 3u + 1u], expo[base + q * 3u + 2u]);
  let b = vec3<f32>(expo[base + (q + 1u) * 3u], expo[base + (q + 1u) * 3u + 1u], expo[base + (q + 1u) * 3u + 2u]);
  return a * (1.0 - f) + b * f;
}
fn prof(t_in: f32, base: u32) -> vec3<f32> { return prof_k(t_in, 0u, p.nk, base); }
// 局所オフセット場（双一次格子 (gx+1)x(gy+1)、正規化座標 0..1）
fn local_field(xn: f32, yn: f32) -> vec3<f32> {
  let u = clamp(xn, 0.0, 1.0) * f32(p.gx); let v = clamp(yn, 0.0, 1.0) * f32(p.gy);
  let q0 = min(u32(floor(u)), p.gx - 1u); let r0 = min(u32(floor(v)), p.gy - 1u);
  let fu = clamp(u - f32(q0), 0.0, 1.0); let fv = clamp(v - f32(r0), 0.0, 1.0);
  let b = p.loc_off + (r0 * (p.gx + 1u) + q0) * 3u;
  let s1 = p.gx + 1u;
  let v00 = vec3<f32>(expo[b], expo[b + 1u], expo[b + 2u]);
  let v10 = vec3<f32>(expo[b + 3u], expo[b + 4u], expo[b + 5u]);
  let v01 = vec3<f32>(expo[b + s1 * 3u], expo[b + s1 * 3u + 1u], expo[b + s1 * 3u + 2u]);
  let v11 = vec3<f32>(expo[b + s1 * 3u + 3u], expo[b + s1 * 3u + 4u], expo[b + s1 * 3u + 5u]);
  return v00 * (1.0 - fu) * (1.0 - fv) + v10 * fu * (1.0 - fv) + v01 * (1.0 - fu) * fv + v11 * fu * fv;
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let yy = g.y;
  if (x >= p.tw || yy >= p.th) { return; }
  let o = ((p.slot * p.th + yy) * p.tw + x) * 3u;
  let X = f32(p.x0 + x); let Y = f32(p.y0 + yy);
  var fx = 0.0; var fy = 0.0; var front = true;
  if (p.proj == 0u) {
    let u = (X - p.Tx) / p.s; let v = (Y - p.Ty) / p.s;
    fx = p.c * u + p.sn * v; fy = -p.sn * u + p.c * v;
  } else if (p.proj == 1u) {
    let z = p.m2.x * X + p.m2.y * Y + p.m2.z;
    front = z > 0.0;
    let zs = select(z, 1e-9, abs(z) < 1e-9);
    fx = (p.m0.x * X + p.m0.y * Y + p.m0.z) / zs; fy = (p.m1.x * X + p.m1.y * Y + p.m1.z) / zs;
  } else {
    let th = (X - p.X0) / p.fc; let hh = (Y - p.Y0) / p.fc;
    let d = vec3<f32>(sin(th), hh, cos(th));
    let qz = dot(p.m2.xyz, d);
    front = qz > 0.0;
    let qzs = select(qz, 1e-9, abs(qz) < 1e-9);
    fx = dot(p.m0.xyz, d) / qzs; fy = dot(p.m1.xyz, d) / qzs;
  }
  let W = i32(p.W); let H = i32(p.H);
  let eps = 1e-3;
  if (!front || fx < -eps || fx > f32(i32(p.vw) - 1) + eps || fy < -eps || fy > f32(i32(p.vh) - 1) + eps) {
    stack[o] = 0u; stack[o + 1u] = 0u; stack[o + 2u] = 0u; return;
  }
  // フェザー重み: 有効領域の端からの距離（フレーム px）/ feather を 0..1 に
  var fwt = 1.0;
  if (p.feather > 0.0) {
    let de = min(min(fx, f32(p.vw - 1u) - fx), min(fy, f32(p.vh - 1u) - fy));
    fwt = clamp(de / p.feather, 0.0, 1.0);
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
  if (p.use_expo == 1u) {
    var offs = p.off.xyz;
    if (p.nk > 0u) { offs += prof((fy + 0.5) / f32(p.vh), p.nk) + prof((fx + 0.5) / f32(p.vw), p.nk + p.nk * 3u); }
    if (p.gx > 0u) { offs += local_field((fx + 0.5) / f32(p.vw), (fy + 0.5) / f32(p.vh)); }
    var gn = p.gain.xyz;
    if (p.nr > 0u) {
      // 周辺減光（乗算）: r は中心 0、隅 1
      let rr = sqrt(((fx + 0.5) / f32(p.vw) * 2.0 - 1.0) * ((fx + 0.5) / f32(p.vw) * 2.0 - 1.0) +
                    ((fy + 0.5) / f32(p.vh) * 2.0 - 1.0) * ((fy + 0.5) / f32(p.vh) * 2.0 - 1.0)) / sqrt(2.0);
      gn = gn * max(vec3<f32>(1.0) + prof_k(rr, p.rad_off, p.nr, p.rad_off + p.nr), vec3<f32>(0.2));
    }
    let cc = (col.xyz - offs) / gn;
    col = vec4<f32>(cc.x, cc.y, cc.z, col.w);
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
  // flags: bit0 幾何的有効, bit1 クリーン, bit2-7 拡大率レベル（1/12 オクターブ刻み、小さいほど細部が多い）
  let flags = 1u | select(0u, 2u, clean) | (min(p.mag_lv, 63u) << 2u);
  stack[o] = r | (gg << 8u) | (b << 16u) | (flags << 24u);
  stack[o + 1u] = bitcast<u32>(max(sh, 0.0));
  stack[o + 2u] = bitcast<u32>(fwt);
}`,
};

// 時間方向の統計。基数選択（ニブル単位）で中央値と分位点を求める
//   tiers[k]: スロット k のフレーム品質段階（0 良, 1 ボケ/ブレ, 2 ブレンド）。use_tier=1 なら、段階 ≤ L だけで min_q 標本以上ある画素は
//     それらだけを使う（L = 0 → 1 → 制限なし の順に試す）
//   motion=1: 中央値から外れた標本が第 2 のまとまりを作るとき、時間的に広く散らばっている方を背景として選ぶ。
//     また第 1 クラスタだけの中央値に取り直す（少数派の動体で中央値が背景クラスタの端に寄るのを防ぐ）
RENDER.median = {
  fields: [["Wc", "u32"], ["x0", "u32"], ["y0", "u32"], ["tw", "u32"], ["th", "u32"], ["K", "u32"], ["tol", "f32"], ["sharp_top", "f32"],
           ["res_lv", "u32"], ["anc_lo", "u32"], ["anc_hi", "u32"], ["use_tier", "u32"], ["min_q", "u32"], ["motion", "u32"]],
  bindings: ["r", "rw", "rw", "rw", "rw", "rw", "r"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { Wc: u32, x0: u32, y0: u32, tw: u32, th: u32, K: u32, tol: f32, sharp_top: f32, res_lv: u32, anc_lo: u32, anc_hi: u32,
           use_tier: u32, min_q: u32, motion: u32 }
// res_lv: 許容する拡大率レベル差（255 = 無効）。anc_lo..anc_hi: アンカーフレームのスロット範囲（anc_lo > anc_hi なら無効）
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> stack: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_med: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_mean: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_sharp: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_cov: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_blend: array<u32>;
@group(0) @binding(7) var<storage, read> tiers: array<u32>;

fn chan(w0: u32, ch: u32) -> u32 { return (w0 >> (8u * ch)) & 255u; }
fn rgb3(w0: u32) -> vec3<f32> { return vec3<f32>(f32(w0 & 255u), f32((w0 >> 8u) & 255u), f32((w0 >> 16u) & 255u)); }
fn lv_of(w0: u32) -> u32 { return (w0 >> 26u) & 63u; }
// 標本の選別条件
//   mode 0: clean, 1: geometric, 2: inlier (clean & dist<tol), 3: fallback inlier (geom & dist<2tol), 4: outlier (clean & dist>=tol)
//   lvmax: 使ってよい拡大率レベルの上限（255 なら無制限）, anc: 1 ならアンカー範囲のスロットだけ, tl: 品質段階の上限（255 なら無制限）
struct Filt { mode: u32, med: vec3<f32>, lvmax: u32, anc: u32, tl: u32 }
fn ok(w0: u32, k: u32, f: Filt) -> bool {
  let fl = w0 >> 24u;
  let geom = (fl & 1u) != 0u; let clean = (fl & 2u) != 0u;
  if (lv_of(w0) > f.lvmax) { return false; }
  if (f.anc == 1u && (k < p.anc_lo || k > p.anc_hi)) { return false; }
  if (f.tl < 255u && tiers[k] > f.tl) { return false; }
  if (f.mode == 0u) { return geom && clean; }
  if (f.mode == 1u) { return geom; }
  let d = dot(abs(rgb3(w0) - f.med), vec3<f32>(1.0 / 3.0));
  if (f.mode == 2u) { return geom && clean && d < p.tol; }
  if (f.mode == 4u) { return geom && clean && d >= p.tol; }
  return geom && d < 2.0 * p.tol;
}
fn val(w0: u32, w1: u32, ch: u32) -> u32 { if (ch < 3u) { return chan(w0, ch); } return w1; }
// 条件を満たすサンプルの ch 値（昇順 kth 番目、0 始まり）を基数選択で求める
fn kth_value(base: u32, stride: u32, ch: u32, f: Filt, kth_in: u32) -> u32 {
  var kth = kth_in;
  var prefix = 0u;
  let npass = select(8u, 2u, ch < 3u);
  for (var ps = 0u; ps < npass; ps++) {
    let shift = (npass - 1u - ps) * 4u;
    var hist: array<u32, 16>;
    for (var b = 0u; b < 16u; b++) { hist[b] = 0u; }
    for (var k = 0u; k < p.K; k++) {
      let w0 = stack[base + k * stride]; let w1 = stack[base + k * stride + 1u];
      if (!ok(w0, k, f)) { continue; }
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
fn count_f(base: u32, stride: u32, f: Filt) -> u32 {
  var n = 0u;
  for (var k = 0u; k < p.K; k++) { if (ok(stack[base + k * stride], k, f)) { n++; } }
  return n;
}
fn median_f(base: u32, stride: u32, f: Filt, cnt: u32) -> vec3<f32> {
  let kmed = (cnt - 1u) / 2u;
  return vec3<f32>(f32(kth_value(base, stride, 0u, f, kmed)), f32(kth_value(base, stride, 1u, f, kmed)), f32(kth_value(base, stride, 2u, f, kmed)));
}
// 条件を満たす標本のスロット範囲（最大 - 最小。無ければ -1）
fn spread_f(base: u32, stride: u32, f: Filt) -> i32 {
  var mn = 1000000; var mx = -1;
  for (var k = 0u; k < p.K; k++) { if (ok(stack[base + k * stride], k, f)) { mn = min(mn, i32(k)); mx = max(mx, i32(k)); } }
  return select(mx - mn, -1, mx < 0);
}
// 条件 mode を満たす標本の最小拡大率レベル + res_lv（無ければ 255）
fn lv_limit(base: u32, stride: u32, mode: u32, anc: u32) -> u32 {
  if (p.res_lv >= 255u) { return 255u; }
  var mn = 255u;
  let f = Filt(mode, vec3<f32>(0.0), 255u, anc, 255u);
  for (var k = 0u; k < p.K; k++) { let w0 = stack[base + k * stride]; if (ok(w0, k, f)) { mn = min(mn, lv_of(w0)); } }
  return select(mn + p.res_lv, 255u, mn == 255u);
}
// 品質段階の上限: 段階 ≤ 0 だけで min_q 以上あればそれ、無ければ段階 ≤ 1、無ければ制限なし
fn tier_limit(base: u32, stride: u32, mode: u32, lvmax: u32, anc: u32) -> u32 {
  if (p.use_tier == 0u) { return 255u; }
  if (count_f(base, stride, Filt(mode, vec3<f32>(0.0), lvmax, anc, 0u)) >= p.min_q) { return 0u; }
  if (count_f(base, stride, Filt(mode, vec3<f32>(0.0), lvmax, anc, 1u)) >= p.min_q) { return 1u; }
  return 255u;
}
fn pack(c: vec3<f32>) -> u32 {
  let r = u32(clamp(c.x + 0.5, 0.0, 255.0)); let g = u32(clamp(c.y + 0.5, 0.0, 255.0)); let b = u32(clamp(c.z + 0.5, 0.0, 255.0));
  return r | (g << 8u) | (b << 16u) | (255u << 24u);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let yy = g.y;
  if (x >= p.tw || yy >= p.th) { return; }
  let base = (yy * p.tw + x) * 3u;
  let stride = p.th * p.tw * 3u;
  let oi = (p.y0 + yy) * p.Wc + p.x0 + x;
  let zero = vec3<f32>(0.0);
  let cnt_c = count_f(base, stride, Filt(0u, zero, 255u, 0u, 255u));   // 被覆数（穴埋め判定用）はフィルタ前
  let cnt_g = count_f(base, stride, Filt(1u, zero, 255u, 0u, 255u));
  if (cnt_g == 0u) {
    out_med[oi] = 0u; out_mean[oi] = 0u; out_sharp[oi] = 0u; out_cov[oi] = 0u; out_blend[oi] = 0u; return;
  }
  // アンカー: 範囲内のフレームがこの画素を覆っていれば、それらだけを使う（クリーン / 幾何で別々に判定）
  let anc_c = select(0u, 1u, p.anc_lo <= p.anc_hi && count_f(base, stride, Filt(0u, zero, 255u, 1u, 255u)) > 0u);
  let anc_g = select(0u, 1u, p.anc_lo <= p.anc_hi && count_f(base, stride, Filt(1u, zero, 255u, 1u, 255u)) > 0u);
  // 解像度フィルタ: 最も細部を持つ標本の拡大率レベル + res_lv までを使う
  let lvc = lv_limit(base, stride, 0u, anc_c);
  let lvg = lv_limit(base, stride, 1u, anc_g);
  // 品質段階
  let tlc = tier_limit(base, stride, 0u, lvc, anc_c);
  let tlg = tier_limit(base, stride, 1u, lvg, anc_g);
  let mmode = select(1u, 0u, cnt_c > 0u);
  let lvm = select(lvg, lvc, cnt_c > 0u);
  let ancm = select(anc_g, anc_c, cnt_c > 0u);
  let tlm = select(tlg, tlc, cnt_c > 0u);
  let fm = Filt(mmode, zero, lvm, ancm, tlm);
  let mcnt = count_f(base, stride, fm);
  var med = median_f(base, stride, fm, mcnt);
  if (p.motion == 1u && cnt_c > 0u) {
    let med0 = med;
    let f_in1 = Filt(2u, med0, lvc, anc_c, tlc);
    let f_out = Filt(4u, med0, lvc, anc_c, tlc);
    let n_in1 = count_f(base, stride, f_in1);
    let n_out = count_f(base, stride, f_out);
    if (n_in1 > 0u) { med = median_f(base, stride, f_in1, n_in1); }
    if (n_out >= 3u && f32(n_out) >= 0.3 * f32(n_in1 + n_out)) {
      let med2 = median_f(base, stride, f_out, n_out);
      let f_in2 = Filt(2u, med2, lvc, anc_c, tlc);
      let n_in2 = count_f(base, stride, f_in2);
      if (f32(n_in2) >= 0.7 * f32(n_out)) {
        let sp1 = spread_f(base, stride, f_in1); let sp2 = spread_f(base, stride, f_in2);
        if (f32(sp2) > f32(sp1) * 1.2 + 1.0) { med = median_f(base, stride, f_in2, n_in2); }
      }
    }
  }
  // インライア
  var fi = Filt(2u, med, lvc, anc_c, tlc);
  var ni = count_f(base, stride, fi);
  if (ni == 0u) { fi = Filt(3u, med, lvg, anc_g, tlg); ni = count_f(base, stride, fi); }
  var mean = med;
  var sharp = med;
  var blend = med;
  if (ni > 0u) {
    var sum = vec3<f32>(0.0);
    for (var k = 0u; k < p.K; k++) { let w0 = stack[base + k * stride]; if (ok(w0, k, fi)) { sum += rgb3(w0); } }
    mean = sum / f32(ni);
    // フェザー合成: クリーン標本（無ければ幾何的標本）をフレーム端からの距離の重みで平均。中央値からのインライア判定は使わない
    var fsum = vec3<f32>(0.0); var fw = 0.0;
    for (var k = 0u; k < p.K; k++) {
      let w0 = stack[base + k * stride];
      if (ok(w0, k, fm)) {
        let wf = bitcast<f32>(stack[base + k * stride + 2u]);
        fsum += rgb3(w0) * wf; fw += wf;
      }
    }
    blend = select(mean, fsum / max(fw, 1e-6), fw > 1e-6);
    // 鮮明度の上位 sharp_top（分位点 1 - sharp_top 以上）
    let kq = u32(ceil((1.0 - p.sharp_top) * f32(ni - 1u) - 1e-4));
    let thr = kth_value(base, stride, 3u, fi, min(kq, ni - 1u));
    var ts = vec3<f32>(0.0); var nt = 0u;
    for (var k = 0u; k < p.K; k++) {
      let w0 = stack[base + k * stride]; let w1 = stack[base + k * stride + 1u];
      if (ok(w0, k, fi) && w1 >= thr) { ts += rgb3(w0); nt++; }
    }
    if (nt > 0u) { sharp = ts / f32(nt); } else { sharp = mean; }
  }
  out_med[oi] = pack(med);
  out_mean[oi] = pack(mean);
  out_sharp[oi] = pack(sharp);
  out_blend[oi] = pack(blend);
  out_cov[oi] = cnt_c | (cnt_g << 16u);   // 下位 16 bit: クリーン標本数, 上位: 幾何的被覆数
}`,
};
