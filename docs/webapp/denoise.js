// denoise.js - アニメ調画像向けの à trous（starlet）ウェーブレットノイズ除去（WebGPU）。Python 版 wavelet_denoise.py と同じ処理
//
// 非間引きの à trous 変換（B3 スプライン [1,4,6,4,1]/16 を 2^l 間隔で分離適用、境界は反射）で詳細係数を求め、
// 軟しきい値 thr × σ_l / σ_0 で縮小して再構成する。間引きしないのでブロック歪みが出ず、平坦な塗りと輪郭線が主体の
// アニメ画像で輪郭を残して微小なノイズだけを落とせる。しきい値は最細レベルの階調で直接指定する。

const B3_NOISE = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005];

const K = {};
K.to_f32 = {
  fields: [["n", "u32"]], bindings: ["r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let c = src[i];
  dst[i] = vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), 0.0);
}`,
};
K.to_rgba = {
  fields: [["n", "u32"]], bindings: ["r", "r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> acc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> cur: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let v = clamp(acc[i] + cur[i] + 0.5, vec4<f32>(0.0), vec4<f32>(255.0));
  dst[i] = u32(v.x) | (u32(v.y) << 8u) | (u32(v.z) << 16u) | (255u << 24u);
}`,
};
// 分離型 B3 スプライン（2^l 間隔、反射境界）
K.atrous = {
  fields: [["W", "u32"], ["H", "u32"], ["step", "i32"], ["axis", "u32"]], bindings: ["r", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { W: u32, H: u32, step: i32, axis: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
fn refl(i: i32, n: i32) -> i32 {   // numpy reflect（端の画素を繰り返さない）
  var v = i;
  if (v < 0) { v = -v; }
  if (v >= n) { v = 2 * (n - 1) - v; }
  return clamp(v, 0, n - 1);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = i32(g.x); let y = i32(g.y);
  let W = i32(p.W); let H = i32(p.H);
  if (x >= W || y >= H) { return; }
  let w = array<f32, 5>(1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0);
  var s = vec4<f32>(0.0);
  for (var t = 0; t < 5; t++) {
    let o = (t - 2) * p.step;
    if (p.axis == 0u) { s += w[t] * src[y * W + refl(x + o, W)]; } else { s += w[t] * src[refl(y + o, H) * W + x]; }
  }
  dst[y * W + x] = s;
}`,
};
// acc += softthr(cur - sm, thr)
K.shrink = {
  fields: [["n", "u32"], ["thr", "f32"]], bindings: ["r", "r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32, thr: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> cur: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sm: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> acc: array<vec4<f32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let d = cur[i] - sm[i];
  acc[i] += sign(d) * max(abs(d) - vec4<f32>(p.thr), vec4<f32>(0.0));
}`,
};

/**
 * rgba: Uint8ClampedArray (W*H*4) → Uint8ClampedArray。thr: 最細レベルのしきい値（階調）、levels: レベル数
 */
export async function denoiseRGBA(gpu, rgba, W, H, thr, levels = 4) {
  const g = gpu;
  const kk = {};
  for (const [name, def] of Object.entries(K)) kk[name] = g.kernel(`dn_${name}`, def.code, def.fields, def.bindings, def.wg);
  const n = W * H;
  const L = Math.max(1, Math.min(levels, B3_NOISE.length));
  g.reserve(n * 4 * 2 + n * 16 * 4, "denoise");
  const src = g.buf(n * 4, "dn_src"), out = g.buf(n * 4, "dn_out");
  let cur = g.buf(n * 16, "dn_cur"), sm = g.buf(n * 16, "dn_sm");
  const tmp = g.buf(n * 16, "dn_tmp"), acc = g.buf(n * 16, "dn_acc");
  try {
    g.upload(src, rgba);
    g.clear(acc);
    kk.to_f32.run({ n }, [src, cur], Math.ceil(n / 256));
    for (let lv = 0; lv < L; lv++) {
      const step = 1 << lv;
      kk.atrous.run2d({ W, H, step, axis: 0 }, [cur, tmp], W, H);
      kk.atrous.run2d({ W, H, step, axis: 1 }, [tmp, sm], W, H);
      kk.shrink.run({ n, thr: thr * B3_NOISE[lv] / B3_NOISE[0] }, [cur, sm, acc], Math.ceil(n / 256));
      const t = cur; cur = sm; sm = t;
    }
    kk.to_rgba.run({ n }, [acc, cur, out], Math.ceil(n / 256));
    return new Uint8ClampedArray(await g.read(out, n * 4));
  } finally {
    for (const b of [src, out, cur, sm, tmp, acc]) g.free(b);
  }
}
