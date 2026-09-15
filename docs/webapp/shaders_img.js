// shaders_img.js - 画像前処理カーネル（グレー化、面積平均縮小、ボックス/ガウスぼかし、
// 静止オーバーレイ検出、鮮明度、レベル画像のパック、再標本化）
//
// 規約: 画像は行優先 index = y*W + x。フレームは RGBA8 を u32 にパック（r | g<<8 | b<<16 | a<<24）。
// alpha は「クリーン（オーバーレイでない）」フラグ（255 = clean, 0 = overlay）。
// レベル画像は vec2<f32>(highpass, valid)。

const COMMON = /* wgsl */ `
fn luma_u32(c: u32) -> f32 {
  let r = f32(c & 255u); let g = f32((c >> 8u) & 255u); let b = f32((c >> 16u) & 255u);
  return floor(0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
}
fn clampi(v: i32, lo: i32, hi: i32) -> i32 { return min(max(v, lo), hi); }
`;

export const IMG = {};

// RGBA フレーム → グレー f32、面積平均で (w,h) に縮小（w=W,h=H なら等倍）
IMG.gray_area = {
  fields: [["W", "u32"], ["H", "u32"], ["w", "u32"], ["h", "u32"]],
  bindings: ["r", "rw"], wg: [16, 16, 1],
  code: COMMON + /* wgsl */ `
struct P { W: u32, H: u32, w: u32, h: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let y = g.y;
  if (x >= p.w || y >= p.h) { return; }
  let x0 = (x * p.W) / p.w; let x1 = ((x + 1u) * p.W + p.w - 1u) / p.w;
  let y0 = (y * p.H) / p.h; let y1 = ((y + 1u) * p.H + p.h - 1u) / p.h;
  var s = 0.0;
  for (var yy = y0; yy < y1; yy++) {
    for (var xx = x0; xx < x1; xx++) { s += luma_u32(src[yy * p.W + xx]); }
  }
  dst[y * p.w + x] = s / f32((x1 - x0) * (y1 - y0));
}`,
};

// f32 画像の面積平均縮小（torch adaptive_avg_pool2d と同じ窓）。src_off/dst_off は要素オフセット
IMG.area_f32 = {
  fields: [["W", "u32"], ["H", "u32"], ["w", "u32"], ["h", "u32"], ["src_off", "u32"], ["dst_off", "u32"]],
  bindings: ["r", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { W: u32, H: u32, w: u32, h: u32, src_off: u32, dst_off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let y = g.y;
  if (x >= p.w || y >= p.h) { return; }
  let x0 = (x * p.W) / p.w; let x1 = ((x + 1u) * p.W + p.w - 1u) / p.w;
  let y0 = (y * p.H) / p.h; let y1 = ((y + 1u) * p.H + p.h - 1u) / p.h;
  var s = 0.0;
  for (var yy = y0; yy < y1; yy++) {
    for (var xx = x0; xx < x1; xx++) { s += src[p.src_off + yy * p.W + xx]; }
  }
  dst[p.dst_off + y * p.w + x] = s / f32((x1 - x0) * (y1 - y0));
}`,
};

// 分離型ボックスぼかし（count_include_pad=False）。sub=1 なら dst = orig - blur（ハイパス）
IMG.box_blur = {
  fields: [["W", "u32"], ["H", "u32"], ["r", "i32"], ["axis", "u32"], ["sub", "u32"]],
  bindings: ["r", "r", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { W: u32, H: u32, r: i32, axis: u32, sub: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> orig: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = i32(g.x); let y = i32(g.y);
  let W = i32(p.W); let H = i32(p.H);
  if (x >= W || y >= H) { return; }
  var s = 0.0; var n = 0.0;
  if (p.axis == 0u) {
    for (var t = -p.r; t <= p.r; t++) { let xx = x + t; if (xx >= 0 && xx < W) { s += src[y * W + xx]; n += 1.0; } }
  } else {
    for (var t = -p.r; t <= p.r; t++) { let yy = y + t; if (yy >= 0 && yy < H) { s += src[yy * W + x]; n += 1.0; } }
  }
  var v = s / n;
  if (p.sub == 1u) { v = orig[y * W + x] - v; }
  dst[y * W + x] = v;
}`,
};

// 分離型ガウスぼかし（replicate パディング）。sigma から半径 r = ceil(3 sigma) を求める
IMG.gauss_f32 = {
  fields: [["W", "u32"], ["H", "u32"], ["sigma", "f32"], ["axis", "u32"]],
  bindings: ["r", "rw"], wg: [16, 16, 1],
  code: COMMON + /* wgsl */ `
struct P { W: u32, H: u32, sigma: f32, axis: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = i32(g.x); let y = i32(g.y);
  let W = i32(p.W); let H = i32(p.H);
  if (x >= W || y >= H) { return; }
  let r = min(12, max(1, i32(ceil(3.0 * p.sigma))));
  var s = 0.0; var n = 0.0;
  for (var t = -r; t <= r; t++) {
    let w = exp(-0.5 * (f32(t) / p.sigma) * (f32(t) / p.sigma));
    var v: f32;
    if (p.axis == 0u) { v = src[y * W + clampi(x + t, 0, W - 1)]; } else { v = src[clampi(y + t, 0, H - 1) * W + x]; }
    s += w * v; n += w;
  }
  dst[y * W + x] = s / n;
}`,
};

// 静止オーバーレイ候補: min_j |g_k - g_j| < diff かつ 勾配 > grad → 1.0（j は最大 4 フレーム、通常は k±span の 2 枚）
//   注意: 候補を k±2span まで増やして「いずれか一致」にすると背景の偶然一致が急増する（実測でマスク 14% → 41%）
// テキスト矩形 trects の内側は静止条件を外し、勾配 > grad だけで文字画素とみなす（動くティッカー用）
// 第 2 出力 flat: 静止（勾配条件なし）= 1、テキスト矩形内 = 2（ハロー膨張と組み合わせて使う）
// vk: フレーム k の有効サイズ (w, h)、vc: 候補 1..4 の有効サイズ (w1,h1,w2,h2),(w3,h3,w4,h4)。有効領域外（パディング）では静止判定しない
IMG.static_detect = {
  fields: [["W", "u32"], ["H", "u32"], ["diff", "f32"], ["grad", "f32"], ["ncand", "u32"], ["ntext", "u32"], ["trects", "vec4i", 8],
           ["vk", "vec4i"], ["vc", "vec4i", 2]],
  bindings: ["r", "r", "r", "r", "r", "rw", "rw"], wg: [16, 16, 1],
  code: COMMON + /* wgsl */ `
struct P { W: u32, H: u32, diff: f32, grad: f32, ncand: u32, ntext: u32, trects: array<vec4<i32>, 8>, vk: vec4<i32>, vc: array<vec4<i32>, 2> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> fk: array<u32>;
@group(0) @binding(2) var<storage, read> f1: array<u32>;
@group(0) @binding(3) var<storage, read> f2: array<u32>;
@group(0) @binding(4) var<storage, read> f3: array<u32>;
@group(0) @binding(5) var<storage, read> f4: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;
@group(0) @binding(7) var<storage, read_write> flat: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = i32(g.x); let y = i32(g.y);
  let W = i32(p.W); let H = i32(p.H);
  if (x >= W || y >= H) { return; }
  let i = y * W + x;
  let gk = luma_u32(fk[i]);
  let ink = x < p.vk.x && y < p.vk.y;
  var d = select(1.0, abs(gk - luma_u32(f1[i])), ink && x < p.vc[0].x && y < p.vc[0].y);
  if (p.ncand >= 2u) { d = min(d, select(1.0, abs(gk - luma_u32(f2[i])), ink && x < p.vc[0].z && y < p.vc[0].w)); }
  if (p.ncand >= 3u) { d = min(d, select(1.0, abs(gk - luma_u32(f3[i])), ink && x < p.vc[1].x && y < p.vc[1].y)); }
  if (p.ncand >= 4u) { d = min(d, select(1.0, abs(gk - luma_u32(f4[i])), ink && x < p.vc[1].z && y < p.vc[1].w)); }
  let gx = 0.5 * (luma_u32(fk[y * W + clampi(x + 1, 0, W - 1)]) - luma_u32(fk[y * W + clampi(x - 1, 0, W - 1)]));
  let gy = 0.5 * (luma_u32(fk[clampi(y + 1, 0, H - 1) * W + x]) - luma_u32(fk[clampi(y - 1, 0, H - 1) * W + x]));
  let gm = max(abs(gx), abs(gy));
  var st = d < p.diff && gm > p.grad;
  var fl = select(0.0, 1.0, d < p.diff);
  for (var k = 0u; k < p.ntext; k++) {
    let rc = p.trects[k];
    if (x >= rc.x && x < rc.x + rc.z && y >= rc.y && y < rc.y + rc.w) { fl = 2.0; if (gm > p.grad) { st = true; } }
  }
  dst[i] = select(0.0, 1.0, st);
  flat[i] = fl;
}`,
};

// マスクの時間方向クロージング: 前後 j フレーム (j=1..3) の両方でマスクされている画素はこのフレームでもマスクする
//   （テロップ上を一瞬横切る光沢アニメなどで静止検出が抜けるフレームを埋める）
IMG.mask_close = {
  fields: [["n", "u32"], ["npair", "u32"]],
  bindings: ["rw", "r", "r", "r", "r", "r", "r"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32, npair: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> frame: array<u32>;
@group(0) @binding(2) var<storage, read> a1: array<u32>;
@group(0) @binding(3) var<storage, read> b1: array<u32>;
@group(0) @binding(4) var<storage, read> a2: array<u32>;
@group(0) @binding(5) var<storage, read> b2: array<u32>;
@group(0) @binding(6) var<storage, read> a3: array<u32>;
@group(0) @binding(7) var<storage, read> b3: array<u32>;
fn masked(c: u32) -> bool { return (c >> 24u) < 128u; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let c = frame[i];
  if (masked(c)) { return; }
  var m = false;
  if (p.npair >= 1u && masked(a1[i]) && masked(b1[i])) { m = true; }
  if (p.npair >= 2u && masked(a2[i]) && masked(b2[i])) { m = true; }
  if (p.npair >= 3u && masked(a3[i]) && masked(b3[i])) { m = true; }
  if (m) { frame[i] = c & 0x00ffffffu; }
}`,
};

// 二値化: dst = src > thr ? 1 : 0
IMG.threshold = {
  fields: [["n", "u32"], ["thr", "f32"]],
  bindings: ["r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32, thr: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  dst[i] = select(0.0, 1.0, src[i] > p.thr);
}`,
};

// マスク合成 → フレームの alpha にクリーンフラグを書き込む
//   near_r  : 静止エッジ密度マスクを r で膨張したもの（> 0 なら範囲内）
//   near_rh : 同じくハロー半径 rh で膨張したもの。flat（静止 or テキスト矩形内）と AND してグロー等を含める
//   rects   : 常に除外する矩形
//   vw, vh  : フレームの有効サイズ。外側（パディング）は常に除外
IMG.finalize_mask = {
  fields: [["W", "u32"], ["H", "u32"], ["use_dens", "u32"], ["nrect", "u32"], ["vw", "u32"], ["vh", "u32"], ["rects", "vec4i", 8]],
  bindings: ["r", "r", "r", "r", "rw"], wg: [16, 16, 1],
  code: COMMON + /* wgsl */ `
struct P { W: u32, H: u32, use_dens: u32, nrect: u32, vw: u32, vh: u32, rects: array<vec4<i32>, 8> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> near_r: array<f32>;    // 静止エッジ密度を r で膨張
@group(0) @binding(2) var<storage, read> near_rh: array<f32>;   // 同 halo 半径で膨張（静止画素と AND）
@group(0) @binding(3) var<storage, read> near_rt: array<f32>;   // 同 text halo 半径で膨張（テキスト矩形内）
@group(0) @binding(4) var<storage, read> flat: array<f32>;      // 1 = 静止, 2 = テキスト矩形内
@group(0) @binding(5) var<storage, read_write> frame: array<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = i32(g.x); let y = i32(g.y);
  let W = i32(p.W); let H = i32(p.H);
  if (x >= W || y >= H) { return; }
  let i = y * W + x;
  var st = false;
  if (p.use_dens == 1u) {
    let fl = flat[i];
    st = near_r[i] > 1e-6 || (fl > 0.5 && fl < 1.5 && near_rh[i] > 1e-6) || (fl > 1.5 && near_rt[i] > 1e-6);
  }
  for (var k = 0u; k < p.nrect; k++) {
    let rc = p.rects[k];
    if (x >= rc.x && x < rc.x + rc.z && y >= rc.y && y < rc.y + rc.w) { st = true; }
  }
  if (x >= i32(p.vw) || y >= i32(p.vh)) { st = true; }
  let a = select(255u, 0u, st);
  frame[i] = (frame[i] & 0x00ffffffu) | (a << 24u);
}`,
};

// 露出補正用のセル統計: 各セル（cell x cell px）のクリーン画素の RGB 平均（0..1）とクリーン率 → vec4<f32>
IMG.cell_rgba = {
  fields: [["W", "u32"], ["H", "u32"], ["w", "u32"], ["h", "u32"], ["cell", "u32"]],
  bindings: ["r", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { W: u32, H: u32, w: u32, h: u32, cell: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let y = g.y;
  if (x >= p.w || y >= p.h) { return; }
  let x0 = x * p.cell; let y0 = y * p.cell;
  var s = vec3<f32>(0.0); var n = 0.0;
  for (var yy = y0; yy < y0 + p.cell; yy++) {
    for (var xx = x0; xx < x0 + p.cell; xx++) {
      let c = src[yy * p.W + xx];
      if ((c >> 24u) > 127u) {
        s += vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0; n += 1.0;
      }
    }
  }
  let m = select(vec3<f32>(0.0), s / max(n, 1.0), n > 0.0);
  dst[y * p.w + x] = vec4<f32>(m, n / f32(p.cell * p.cell));
}`,
};

// |ラプラシアン|（replicate パディング）
IMG.laplacian_abs = {
  fields: [["W", "u32"], ["H", "u32"]],
  bindings: ["r", "rw"], wg: [16, 16, 1],
  code: COMMON + /* wgsl */ `
struct P { W: u32, H: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = i32(g.x); let y = i32(g.y);
  let W = i32(p.W); let H = i32(p.H);
  if (x >= W || y >= H) { return; }
  let c = src[y * W + x];
  let l = src[y * W + clampi(x - 1, 0, W - 1)]; let r = src[y * W + clampi(x + 1, 0, W - 1)];
  let u = src[clampi(y - 1, 0, H - 1) * W + x]; let d = src[clampi(y + 1, 0, H - 1) * W + x];
  dst[y * W + x] = abs(l + r + u + d - 4.0 * c);
}`,
};

// フレーム alpha → f32 クリーンマスク（1 = clean）
IMG.alpha_f32 = {
  fields: [["n", "u32"]],
  bindings: ["r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  dst[i] = select(0.0, 1.0, (src[i] >> 24u) > 127u);
}`,
};

// (hp, valid) を vec2 にパック。valid は > 0.5 で二値化
IMG.pack2 = {
  fields: [["n", "u32"]],
  bindings: ["r", "r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec2<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  dst[i] = vec2<f32>(a[i], select(0.0, 1.0, b[i] > 0.5));
}`,
};

// レベル画像 vec2 を (hs, ws) に再標本化（bilinear, align_corners=False, 原点は画素境界）。valid は > 0.5
IMG.resample2 = {
  fields: [["W", "u32"], ["H", "u32"], ["w", "u32"], ["h", "u32"]],
  bindings: ["r", "rw"], wg: [16, 16, 1],
  code: COMMON + /* wgsl */ `
struct P { W: u32, H: u32, w: u32, h: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let y = g.y;
  if (x >= p.w || y >= p.h) { return; }
  let W = i32(p.W); let H = i32(p.H);
  let sx = (f32(x) + 0.5) * f32(p.W) / f32(p.w) - 0.5;
  let sy = (f32(y) + 0.5) * f32(p.H) / f32(p.h) - 0.5;
  let x0f = floor(sx); let y0f = floor(sy);
  let fx = sx - x0f; let fy = sy - y0f;
  let x0 = clampi(i32(x0f), 0, W - 1); let x1 = clampi(i32(x0f) + 1, 0, W - 1);
  let y0 = clampi(i32(y0f), 0, H - 1); let y1 = clampi(i32(y0f) + 1, 0, H - 1);
  let v = src[y0 * W + x0] * (1.0 - fx) * (1.0 - fy) + src[y0 * W + x1] * fx * (1.0 - fy)
        + src[y1 * W + x0] * (1.0 - fx) * fy + src[y1 * W + x1] * fx * fy;
  dst[y * p.w + x] = vec2<f32>(v.x, select(0.0, 1.0, v.y > 0.5));
}`,
};

// f32 配列の合計（1 ワークグループ、grid-stride）。dst[out_off] に書く
IMG.sum_f32 = {
  fields: [["n", "u32"], ["out_off", "u32"]],
  bindings: ["r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32, out_off: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var s = 0.0;
  for (var i = l.x; i < p.n; i += 256u) { s += src[i]; }
  sh[l.x] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (l.x < st) { sh[l.x] += sh[l.x + st]; }
    workgroupBarrier();
  }
  if (l.x == 0u) { dst[p.out_off] = sh[0]; }
}`,
};
