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
      b = this.gpu.buf(this.frameBytes, `frame${k}`);
      this.gpu.upload(b, rgba);
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
  const list = [...files].sort((a, b) => (naturalKey(a.name) < naturalKey(b.name) ? -1 : 1));
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
