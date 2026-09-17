// frames.js - フレーム供給元と GPU 常駐キャッシュ（ストリーミング処理の基盤）
//
// 供給元 (source): { n, W, H, get(k) -> Promise<Uint8ClampedArray RGBA>, close() }
//   - 動画: <video> をシークして canvas 経由で取り出す（同じ時刻へのシークは同じフレームを返す）
//   - 画像列: createImageBitmap で都度デコード
// FrameStore: フレームの RGBA8 GPUBuffer を LRU で常駐させる。上限（maxResident 枚）を超えると
//   固定（pin）されていない最古のフレームを破棄し、必要になれば供給元から読み直す。
//   前処理で決まるマスク（alpha = クリーンフラグ）は 1 bit/画素にパックして別バッファに保持し、
//   読み直したフレームには unpack で復元する。全フレームが常駐できる場合はパックしない（従来と同じ動作）。

import { fmtMB } from "./gpu.js";

const PACK_MASK = {
  fields: [["n", "u32"]], bindings: ["r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read_write> bits: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let w = g.x; let base = w * 32u;
  if (base >= p.n) { return; }
  var v = 0u;
  for (var b = 0u; b < 32u; b++) {
    let i = base + b;
    if (i < p.n && (frame[i] >> 24u) > 127u) { v |= (1u << b); }
  }
  bits[w] = v;
}`,
};

// 放射歪み補正（shaders_img.js の undistort_rgba と同じ式）。frames.js 単体で使えるようここにも置く
const UNDISTORT = {
  fields: [["W", "u32"], ["H", "u32"], ["k1", "f32"], ["ox", "u32"], ["oy", "u32"], ["ow", "u32"], ["oh", "u32"]],
  bindings: ["r", "rw"], wg: [16, 16, 1],
  code: /* wgsl */ `
struct P { W: u32, H: u32, k1: f32, ox: u32, oy: u32, ow: u32, oh: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
fn rgba_of(c: u32) -> vec4<f32> { return vec4<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u), f32(c >> 24u)); }
fn cubw(t: f32) -> vec4<f32> {
  let t2 = t * t; let t3 = t2 * t;
  return vec4<f32>(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let y = g.y;
  if (x >= p.W || y >= p.H) { return; }
  let W = i32(p.W); let H = i32(p.H);
  let xu = f32(min(x, p.ow - 1u) + p.ox); let yu = f32(min(y, p.oh - 1u) + p.oy);
  let cx = 0.5 * f32(W - 1); let cy = 0.5 * f32(H - 1);
  let rn2 = cx * cx + cy * cy;
  let ux = xu - cx; let uy = yu - cy;
  let fct = 1.0 + p.k1 * (ux * ux + uy * uy) / rn2;
  let xd = cx + ux * fct; let yd = cy + uy * fct;
  let x0f = floor(xd); let y0f = floor(yd);
  let wx = cubw(xd - x0f); let wy = cubw(yd - y0f);
  var acc = vec4<f32>(0.0); var wsum = 0.0;
  for (var j = -1; j <= 2; j++) {
    let yy = clamp(i32(y0f) + j, 0, H - 1);
    for (var i = -1; i <= 2; i++) {
      let xx = clamp(i32(x0f) + i, 0, W - 1);
      let w = wx[i + 1] * wy[j + 1];
      acc += w * rgba_of(src[yy * W + xx]); wsum += w;
    }
  }
  acc = acc / max(wsum, 1e-6);
  let an = rgba_of(src[clamp(i32(round(yd)), 0, H - 1) * W + clamp(i32(round(xd)), 0, W - 1)]).w;
  let r = u32(clamp(acc.x + 0.5, 0.0, 255.0)); let gg = u32(clamp(acc.y + 0.5, 0.0, 255.0)); let b = u32(clamp(acc.z + 0.5, 0.0, 255.0));
  dst[y * p.W + x] = r | (gg << 8u) | (b << 16u) | (u32(an) << 24u);
}`,
};

/** k1 > 0（糸巻き）では隅が範囲外になるので、全画素が有効な中央の矩形 [x, y, w, h] を返す。k1 <= 0 なら全体 */
export function undistortCropRect(W, H, k1) {
  if (k1 <= 0) return [0, 0, W, H];
  const cx = 0.5 * (W - 1), cy = 0.5 * (H - 1), rn2 = cx * cx + cy * cy;
  let t = 1.0;
  for (let it = 0; it < 200; it++) {
    const hx = t * cx, hy = t * cy, r2 = (hx * hx + hy * hy) / rn2;
    if (hx * (1 + k1 * r2) <= cx && hy * (1 + k1 * r2) <= cy) break;
    t -= 0.005;
  }
  const x0 = Math.ceil(cx - t * cx), y0 = Math.ceil(cy - t * cy);
  return [x0, y0, W - 2 * x0, H - 2 * y0];
}

/** トップフィールド（偶数行）を残し、奇数行を上下の偶数行の平均で補間する（RGBA、破壊的） */
export function deinterlaceRGBA(rgba, W, H) {
  const row = W * 4;
  for (let y = 1; y < H; y += 2) {
    const yp = y - 1, yn = y + 1 < H ? y + 1 : y - 1;
    const o = y * row, op = yp * row, on = yn * row;
    for (let i = 0; i < row; i++) rgba[o + i] = (rgba[op + i] + rgba[on + i] + 1) >> 1;
  }
  return rgba;
}

/**
 * インターレースの櫛状パターンの割合 [縦, 横]（Python 版 comb_fraction と同じ）: 4 行にわたって明暗が交互になる画素の比率を
 * 縦方向と横方向で。プログレッシブなら縦 ≈ 横、インターレースの動きのある部分では縦 ≫ 横
 */
export function combFraction(rgba, W, H, thr = 10) {
  const g = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) g[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  const t2 = thr * thr;
  let cv = 0, ch = 0;
  for (let y = 1; y + 2 < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x;
    const c0 = (g[i - W] - g[i]) * (g[i + W] - g[i]) > t2;
    const c1 = (g[i] - g[i + W]) * (g[i + 2 * W] - g[i + W]) > t2;
    if (c0 && c1) cv++;
  }
  for (let y = 0; y < H; y++) for (let x = 1; x + 2 < W; x++) {
    const i = y * W + x;
    const c0 = (g[i - 1] - g[i]) * (g[i + 1] - g[i]) > t2;
    const c1 = (g[i] - g[i + 1]) * (g[i + 2] - g[i + 1]) > t2;
    if (c0 && c1) ch++;
  }
  return [cv / (W * H), ch / (W * H)];
}
export const isInterlaced = (cv, ch) => cv > 0.01 && cv > 3.0 * ch;

const UNPACK_MASK = {
  fields: [["n", "u32"]], bindings: ["r", "rw"], wg: [256, 1, 1],
  code: /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> bits: array<u32>;
@group(0) @binding(2) var<storage, read_write> frame: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.n) { return; }
  let bit = (bits[i >> 5u] >> (i & 31u)) & 1u;
  frame[i] = (frame[i] & 0x00ffffffu) | (select(0u, 255u, bit == 1u) << 24u);
}`,
};

export class FrameStore {
  /**
   * @param gpu Gpu
   * @param src 供給元 {n, W, H, get(k), close()}
   * @param opts {maxResident: 同時に常駐させる最大枚数, log}
   */
  constructor(gpu, src, opts) {
    this.gpu = gpu; this.src = src; this.log = opts.log || (() => {});
    this.n = src.n; this.W = src.W; this.H = src.H;
    // 各フレームの有効サイズ [w, h]（画像列でサイズが異なる場合の左上詰め。動画は全て W x H）
    this.sizes = src.sizes || Array.from({ length: this.n }, () => [this.W, this.H]);
    this.frameBytes = this.W * this.H * 4;
    this.maskBytes = Math.ceil(this.W * this.H / 32) * 4;
    this.maxResident = Math.max(1, Math.min(this.n, opts.maxResident | 0 || this.n));
    this.bufs = new Array(this.n).fill(null);
    this.masks = new Array(this.n).fill(null);
    this.maskReady = new Uint8Array(this.n);
    this.last = new Float64Array(this.n);
    this.pinned = new Set();
    this.tick = 0; this.resident = 0;
    this.decodes = 0; this.evictions = 0;
    this.kPack = gpu.kernel("pack_mask", PACK_MASK.code, PACK_MASK.fields, PACK_MASK.bindings, PACK_MASK.wg);
    this.kUnpack = gpu.kernel("unpack_mask", UNPACK_MASK.code, UNPACK_MASK.fields, UNPACK_MASK.bindings, UNPACK_MASK.wg);
    this.kUndistort = gpu.kernel("undistort_rgba", UNDISTORT.code, UNDISTORT.fields, UNDISTORT.bindings, UNDISTORT.wg);
    this.deinterlace = false;   // 読み込み時に片フィールド補間する（CPU）
    this.k1 = 0;                // 放射歪み係数（0 以外なら読み込み時に GPU で補正し、有効矩形を左上詰めにする）
    this.cropRect = null;
    this._tmp = null;
  }

  /**
   * 放射歪み補正を設定する。常駐フレームとパック済みマスクは（補正前のものなので）破棄し、以後の get() で補正して読み直す。
   * k1 > 0 では有効矩形が縮むので sizes を更新する（パディングされたフレームと同じ扱いになる）
   */
  setUndistort(k1) {
    this.k1 = k1;
    this.cropRect = undistortCropRect(this.W, this.H, k1);
    const [, , w, h] = this.cropRect;
    for (let k = 0; k < this.n; k++) this.sizes[k] = [Math.min(this.sizes[k][0], w), Math.min(this.sizes[k][1], h)];
    this.reset();
  }

  /** 常駐フレームとマスクを全て破棄する（前処理をやり直すとき） */
  reset() {
    this.gpu.submit();
    for (let k = 0; k < this.n; k++) {
      if (this.bufs[k]) { this.gpu.free(this.bufs[k]); this.bufs[k] = null; }
      if (this.masks[k]) { this.gpu.free(this.masks[k]); this.masks[k] = null; }
      this.maskReady[k] = 0;
    }
    this.resident = 0; this.pinned.clear();
  }

  /** 全フレームを常駐できる（破棄が起きない）なら true */
  get allResident() { return this.maxResident >= this.n; }
  get residentBytes() { return this.resident * this.frameBytes; }

  /** 常駐上限を変更する（減らした場合は固定されていない古いフレームから破棄） */
  setMaxResident(m) {
    this.maxResident = Math.max(1, Math.min(this.n, m | 0));
    this._makeRoom(0);
  }

  /** フレーム k の GPUBuffer（RGBA8、alpha = クリーンフラグ）。無ければ供給元から読み込む */
  async get(k) {
    let b = this.bufs[k];
    if (!b) {
      this._makeRoom(1);
      const rgba = await this.src.get(k);
      this.decodes++;
      if (this.deinterlace) deinterlaceRGBA(rgba, this.W, this.H);
      b = this.gpu.buf(this.frameBytes, `frame${k}`);
      if (this.k1 !== 0) {
        if (!this._tmp) this._tmp = this.gpu.buf(this.frameBytes, "undistort_tmp");
        this.gpu.upload(this._tmp, rgba);
        const [ox, oy, ow, oh] = this.cropRect;
        this.kUndistort.run2d({ W: this.W, H: this.H, k1: this.k1, ox, oy, ow, oh }, [this._tmp, b], this.W, this.H);
      } else {
        this.gpu.upload(b, rgba);
      }
      if (this.maskReady[k]) this.kUnpack.run({ n: this.W * this.H }, [this.masks[k], b], Math.ceil(this.W * this.H / 256));
      this.bufs[k] = b; this.resident++;
    }
    this.last[k] = ++this.tick;
    return b;
  }

  /** 常駐中のフレーム k（無ければ例外。pin 済みのフレームにだけ使う） */
  peek(k) {
    const b = this.bufs[k];
    if (!b) throw new Error(`frame ${k} is not resident`);
    this.last[k] = ++this.tick;
    return b;
  }

  pin(k) { this.pinned.add(k); }
  unpin(k) { this.pinned.delete(k); }

  /** フレーム k の alpha（マスク）を 1 bit/画素にパックして保持する（k は常駐していること） */
  packMask(k) {
    const b = this.peek(k);
    if (!this.masks[k]) this.masks[k] = this.gpu.buf(this.maskBytes, `mask${k}`);
    this.kPack.run({ n: this.W * this.H }, [b, this.masks[k]], Math.ceil(this.W * this.H / 32 / 256));
    this.maskReady[k] = 1;
  }

  /** need 枚ぶんの空きを作る。固定されていない最古のフレームから破棄する */
  _makeRoom(need) {
    while (this.resident + need > this.maxResident) {
      let best = -1, bt = Infinity;
      for (let k = 0; k < this.n; k++) {
        if (this.bufs[k] && !this.pinned.has(k) && this.last[k] < bt) { best = k; bt = this.last[k]; }
      }
      if (best < 0) {
        throw new Error(`フレームキャッシュが足りません（常駐 ${this.resident} 枚のうち固定 ${this.pinned.size} 枚、上限 ${this.maxResident} 枚）。` +
                        "memory limit を上げるか input scale を下げてください");
      }
      // 未送信のコマンドがこのバッファを参照している可能性があるので先に送信してから破棄する
      // （送信済みで未完了のコマンドがあっても destroy は安全: 完了まで実体は保持される）
      this.gpu.submit();
      this.gpu.free(this.bufs[best]);
      this.bufs[best] = null; this.resident--; this.evictions++;
    }
  }

  /** フレーム k の RGBA を CPU に読み戻す（デバッグ表示用） */
  async readRGBA(k) {
    const b = await this.get(k);
    return new Uint8ClampedArray(await this.gpu.read(b, this.frameBytes));
  }

  stats() {
    return `decoded ${this.decodes} (${(this.decodes / Math.max(1, this.n)).toFixed(2)}x of ${this.n} frames), evicted ${this.evictions}, ` +
           `resident ${this.resident}/${this.maxResident} (${fmtMB(this.residentBytes)})`;
  }

  destroy() {
    for (let k = 0; k < this.n; k++) {
      if (this.bufs[k]) { this.gpu.free(this.bufs[k]); this.bufs[k] = null; }
      if (this.masks[k]) { this.gpu.free(this.masks[k]); this.masks[k] = null; }
    }
    if (this._tmp) { this.gpu.free(this._tmp); this._tmp = null; }
    this.resident = 0; this.pinned.clear();
  }
}

// ------------------------------------------------------------ 供給元

const isVideoFile = (f) => f.type.startsWith("video/") || /\.(mp4|webm|mov|m4v|mkv)$/i.test(f.name);
export { isVideoFile };

function frameCanvas(W, H) {
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  return cv;
}

/**
 * 動画の供給元。opts: {fps, start, duration, maxFrames, inputScale, trim?, crop?}
 * 抽出時刻は開く時に決める。get(k) は times[k] へシークして描画する
 */
export async function openVideoSource(file, opts, log) {
  const url = URL.createObjectURL(file);
  const v = document.createElement("video");
  v.src = url; v.muted = true; v.preload = "auto"; v.playsInline = true;
  await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(new Error("動画を開けません（コーデック非対応の可能性）")); });
  const dur = v.duration;
  const crop = opts.crop || [0, 0, v.videoWidth, v.videoHeight];   // 元動画座標の切り出し矩形
  const W = Math.max(8, Math.round(crop[2] * opts.inputScale)), H = Math.max(8, Math.round(crop[3] * opts.inputScale));
  const cv = frameCanvas(W, H);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  let t0 = opts.start, t1 = Math.min(dur, opts.duration > 0 ? opts.start + opts.duration : dur);
  if (opts.trim) { t0 = opts.trim.t0; t1 = Math.min(dur, opts.trim.t1); }
  const step = 1.0 / opts.fps;
  const times = [];
  for (let t = t0; t < t1 - 1e-6; t += step) { times.push(t); if (opts.maxFrames > 0 && times.length >= opts.maxFrames) break; }
  log(`[load] video ${v.videoWidth}x${v.videoHeight} ${dur.toFixed(2)}s, range ${t0.toFixed(3)}-${t1.toFixed(3)}s` +
      (opts.trim ? ` (frames ${opts.trim.startFrame}-${opts.trim.endFrame} @ ${opts.trim.fps} fps)` : "") +
      (opts.crop ? `, crop ${crop.join(",")}` : "") + ` → ${times.length} frames @ ${opts.fps} fps, scale ${opts.inputScale}`);
  let closed = false;
  const get = async (k) => {
    if (closed) throw new Error("video source closed");
    await new Promise((res, rej) => {
      const to = setTimeout(() => rej(new Error(`seek timeout at ${times[k].toFixed(2)}s`)), 15000);
      v.onseeked = () => { clearTimeout(to); res(); };
      v.onerror = () => { clearTimeout(to); rej(new Error("seek error")); };
      v.currentTime = times[k];
    });
    ctx.drawImage(v, crop[0], crop[1], crop[2], crop[3], 0, 0, W, H);
    return ctx.getImageData(0, 0, W, H).data;
  };
  const close = () => { closed = true; v.removeAttribute("src"); v.load(); URL.revokeObjectURL(url); };
  return { n: times.length, W, H, get, close, kind: "video" };
}

const naturalKey = (s) => s.split(/(\d+)/).map((t) => (/^\d+$/.test(t) ? t.padStart(12, "0") : t.toLowerCase())).join("");
/** 静止画をフレーム順（ファイル名の自然順）に並べる。openImageSource と UI のサムネイル表示で共通に使う */
export function sortImageFiles(files) {
  return [...files].sort((a, b) => (naturalKey(a.name) < naturalKey(b.name) ? -1 : 1));
}

/** 画像ファイルの (幅, 高さ)。PNG / JPEG はヘッダから読む（デコード不要）。それ以外は createImageBitmap */
async function imageSize(file) {
  const head = new Uint8Array(await file.slice(0, 1 << 20).arrayBuffer());
  if (head.length > 24 && head[0] === 0x89 && head[1] === 0x50 && head[2] === 0x4e && head[3] === 0x47) {
    const dv = new DataView(head.buffer);
    return [dv.getUint32(16), dv.getUint32(20)];
  }
  if (head.length > 4 && head[0] === 0xff && head[1] === 0xd8) {
    let p = 2;
    while (p + 9 < head.length) {
      if (head[p] !== 0xff) { p++; continue; }
      const m = head[p + 1];
      if (m === 0xff) { p++; continue; }                                     // fill byte
      if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { p += 2; continue; }   // 長さ無しのマーカー
      const len = (head[p + 2] << 8) | head[p + 3];
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {   // SOFn
        return [(head[p + 7] << 8) | head[p + 8], (head[p + 5] << 8) | head[p + 6]];
      }
      p += 2 + len;
    }
  }
  const bmp = await createImageBitmap(file);
  const s = [bmp.width, bmp.height];
  bmp.close();
  return s;
}

/**
 * 画像列の供給元。opts: {every, maxFrames, inputScale}
 * サイズの異なる画像（スクリーンショット等）は引き伸ばさず、最大サイズのキャンバスに左上詰めで置き、右・下は端の画素を
 * 複製して埋める（ハイパス等の境界応答を抑えるため）。各フレームの有効サイズは sizes[k] = [w, h] で返し、
 * 複製領域は位置合わせ・合成の両方から除外される（Python 版 pad_to_common_size と同じ）
 */
export async function openImageSource(files, opts, log) {
  const list = sortImageFiles(files);
  const sel = list.filter((_, i) => i % opts.every === 0).slice(0, opts.maxFrames > 0 ? opts.maxFrames : undefined);
  if (!sel.length) throw new Error("画像がありません");
  const sizes0 = [];
  for (const f of sel) sizes0.push(await imageSize(f));
  const W0 = Math.max(...sizes0.map((s) => s[0])), H0 = Math.max(...sizes0.map((s) => s[1]));
  const sc = opts.inputScale;
  const W = Math.max(8, Math.round(W0 * sc)), H = Math.max(8, Math.round(H0 * sc));
  const sizes = sizes0.map(([w, h]) => [Math.min(W, Math.max(8, Math.round(w * sc))), Math.min(H, Math.max(8, Math.round(h * sc)))]);
  const differ = sizes0.some((s) => s[0] !== W0 || s[1] !== H0);
  log(`[load] ${sel.length} images ${W0}x${H0} → ${W}x${H}`);
  if (differ) {
    const uniq = [...new Set(sizes0.map((s) => `${s[0]}x${s[1]}`))];
    log(`[load] サイズの異なる画像を ${W0}x${H0} に揃えました（右・下を端の画素で埋め、合成には使いません）: ${uniq.join(", ")}`);
  }
  const cv = frameCanvas(W, H);
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  const get = async (k) => {
    const bmp = await createImageBitmap(sel[k]);
    const [w, h] = sizes[k];
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(bmp, 0, 0, w, h);
    if (w < W || h < H) {
      ctx.imageSmoothingEnabled = false;
      if (w < W) ctx.drawImage(bmp, bmp.width - 1, 0, 1, bmp.height, w, 0, W - w, h);            // 右: 最終列を複製
      if (h < H) ctx.drawImage(bmp, 0, bmp.height - 1, bmp.width, 1, 0, h, w, H - h);            // 下: 最終行を複製
      if (w < W && h < H) ctx.drawImage(bmp, bmp.width - 1, bmp.height - 1, 1, 1, w, h, W - w, H - h);   // 角
    }
    bmp.close();
    return ctx.getImageData(0, 0, W, H).data;
  };
  return { n: sel.length, W, H, sizes, get, close: () => {}, kind: "images" };
}
