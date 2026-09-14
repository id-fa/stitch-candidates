// gpu.js - WebGPU 共通基盤
// - Kernel: WGSL 計算シェーダーのラッパー（uniform は動的オフセット付き 1 バッファに集約）
// - Gpu: コマンドエンコーダと uniform アリーナ。submit() でまとめて GPU へ送る
// - 読み戻し（read）はステージングバッファ経由

const UNIFORM_SLOT = 256;         // 動的オフセットの最小アライメント
const ARENA_BYTES = 16 * 1024 * 1024;

let _bufId = 1;
export const fmtMB = (b) => b >= 1073741824 ? `${(b / 1073741824).toFixed(2)}GB` : `${(b / 1048576).toFixed(0)}MB`;

export class Gpu {
  static async create(log) {
    if (!navigator.gpu) {
      if (window.isSecureContext === false)
        throw new Error("WebGPU は https:// または http://localhost でのみ使えます（現在のページは非セキュアコンテキスト）。" +
                        "http://127.0.0.1 や localhost 経由で開いてください");
      throw new Error("このブラウザは WebGPU に対応していないか、WebGPU が無効化されています。\n" +
                      "対応ブラウザ: Chrome / Edge 113+, Firefox 141+, Safari 26+\n" +
                      "Firefox: about:config で dom.webgpu.enabled を true に。Safari: 設定 → 機能フラグ → WebGPU を有効に");
    }
    let adapter = null;
    try { adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }); } catch (e) { /* null 扱い */ }
    if (!adapter) {
      try { adapter = await navigator.gpu.requestAdapter(); } catch (e) { /* null 扱い */ }
    }
    if (!adapter) throw new Error("WebGPU アダプタを取得できません。ブラウザの GPU（グラフィック）アクセラレーションが無効か、GPU / ドライバがブロックされています。\n" +
                                  "Chrome / Edge: 設定 → システム → 「グラフィック アクセラレーションが使用可能な場合は使用する」をオンにしてブラウザを再起動。" +
                                  "chrome://gpu（edge://gpu）で WebGPU の状態を確認できます。Linux では chrome://flags/#enable-unsafe-webgpu が必要な場合があります。\n" +
                                  "リモートデスクトップや仮想マシンでは GPU が使えないことがあります");
    const L = adapter.limits;
    const requiredLimits = {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.max(8, Math.min(10, L.maxStorageBuffersPerShaderStage)),
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: Math.max(256, L.maxComputeInvocationsPerWorkgroup),
    };
    let device;
    try { device = await adapter.requestDevice({ requiredLimits }); }
    catch (e) { throw new Error(`WebGPU デバイスを作成できません（${e.message || e}）。GPU ドライバの更新、または別のブラウザをお試しください`); }
    const g = new Gpu(adapter, device, log);
    let info = "";
    try {
      const ai = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
      if (ai) info = [ai.vendor, ai.architecture, ai.description].filter(Boolean).join(" / ");
    } catch (e) { /* ignore */ }
    g.info = info;
    log(`[device] WebGPU ${info || "(adapter info unavailable)"}; maxBufferSize=${(device.limits.maxBufferSize / 1048576).toFixed(0)}MB, ` +
        `maxStorageBinding=${(device.limits.maxStorageBufferBindingSize / 1048576).toFixed(0)}MB`);
    device.lost.then((info) => {
      g.lostInfo = info;
      if (info.reason !== "destroyed") log(`[device] LOST: ${info.reason} ${info.message}`);
    });
    return g;
  }

  constructor(adapter, device, log) {
    this.adapter = adapter;
    this.device = device;
    this.log = log;
    this.kernels = new Map();
    this.arenaBuf = device.createBuffer({ size: ARENA_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.arenaStaging = new ArrayBuffer(ARENA_BYTES);
    this.arenaView = new DataView(this.arenaStaging);
    this.arenaOff = 0;
    this.encoder = null;
    this.bgCache = new Map();
    this.allocBytes = 0;
    this.budgetBytes = 0;     // 確保総量の上限（0 = 無制限）。超える確保は例外にしてブラウザのクラッシュを避ける
    this.peakBytes = 0;
    this.oomError = null;     // uncapturederror で受けた out-of-memory
    this.lostInfo = null;     // device.lost の情報（喪失後は全操作を例外にする）
    device.addEventListener("uncapturederror", (ev) => {
      const e = ev.error;
      if (typeof GPUOutOfMemoryError !== "undefined" && e instanceof GPUOutOfMemoryError) {
        this.oomError = e;
        log(`[device] out-of-memory: ${e.message}`);
      } else {
        log(`[device] ${e.constructor?.name || "error"}: ${e.message}`);
      }
    });
    this.zero = this.buf(4096);
  }

  /** デバイス喪失を説明する文。OOM 起因（D3D12 の CreateCommittedResource 失敗など）なら上限の引き下げを促す */
  lostMessage() {
    const i = this.lostInfo;
    if (!i) return null;
    const oom = /OutOfMemory|E_OUTOFMEMORY|out of memory/i.test(i.message || "");
    return oom
      ? `GPU メモリ不足で GPU デバイスが失われました（Dawn 内部の確保に失敗: ${(i.message || "").split("\n")[0].trim()}）。` +
        "memory limit をこの実行の確保量より十分小さくしてください（オンボード GPU は 4〜6 GB から。「上限を確認」で測定できます）。" +
        "デバイスは次の実行時に作り直します"
      : `GPU デバイスが失われました（${i.reason}: ${(i.message || "").split("\n")[0].trim()}）。デバイスは次の実行時に作り直します`;
  }
  /** OOM / デバイス喪失が起きていれば例外にする（同期点で呼ぶ） */
  checkOom() {
    if (this.lostInfo && this.lostInfo.reason !== "destroyed") throw new Error(this.lostMessage());
    if (this.oomError) throw new Error(`GPU メモリ不足です（${this.oomError.message}）。fps / max frames / input scale を下げるか Trim で範囲を絞ってください`);
  }
  /** これから size バイト確保しても上限内か確認する（超えるなら例外） */
  reserve(size, what = "GPU バッファ") {
    this.checkOom();
    if (this.budgetBytes > 0 && this.allocBytes + size > this.budgetBytes) {
      throw new Error(`メモリ上限を超えるため中止しました: ${what} ${fmtMB(size)} を確保すると合計 ${fmtMB(this.allocBytes + size)} > 上限 ${fmtMB(this.budgetBytes)}。` +
                      "fps / max frames / input scale を下げるか Trim で範囲を絞るか、memory limit を上げてください");
    }
  }

  // ---- バッファ ----
  buf(size, label) {
    this.reserve(size, label || "buffer");
    const b = this.device.createBuffer({
      size: Math.max(16, Math.ceil(size / 16) * 16), label,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    b._id = _bufId++;
    b._size = size;
    this.allocBytes += b.size;
    if (this.allocBytes > this.peakBytes) this.peakBytes = this.allocBytes;
    return b;
  }
  free(b) {
    if (!b) return;
    this.allocBytes -= b.size;
    b.destroy();
    // 破棄したバッファを参照する bind group を捨てる
    for (const k of [...this.bgCache.keys()]) if (k.includes(`|${b._id}|`)) this.bgCache.delete(k);
  }
  upload(b, data, offset = 0) {
    this.device.queue.writeBuffer(b, offset, data.buffer ? data.buffer : data, data.byteOffset || 0, data.byteLength);
  }

  // ---- コマンド ----
  enc() {
    if (!this.encoder) this.encoder = this.device.createCommandEncoder();
    return this.encoder;
  }
  clear(b) { this.enc().clearBuffer(b); }
  copy(src, dst, size, so = 0, doff = 0) { this.enc().copyBufferToBuffer(src, so, dst, doff, size); }

  submit() {
    if (!this.encoder && this.arenaOff === 0) return;
    if (this.arenaOff > 0) {
      this.device.queue.writeBuffer(this.arenaBuf, 0, this.arenaStaging, 0, this.arenaOff);
    }
    if (this.encoder) this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    this.arenaOff = 0;
  }

  /** 保留中のコマンドを送信し、buffer[offset, offset+size) を ArrayBuffer で返す */
  async read(b, size = b._size, offset = 0) {
    this.submit();
    const size4 = Math.ceil(size / 4) * 4;
    const st = this.device.createBuffer({ size: size4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = this.device.createCommandEncoder();
    e.copyBufferToBuffer(b, offset, st, 0, size4);
    this.device.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = st.getMappedRange().slice(0, size4);
    st.unmap();
    st.destroy();
    return out;
  }
  async done() { this.submit(); await this.device.queue.onSubmittedWorkDone(); this.checkOom(); }

  // ---- uniform アリーナ ----
  _allocUniform() {
    if (this.arenaOff + UNIFORM_SLOT > ARENA_BYTES) {
      // アリーナが尽きたら一旦送信して続ける（コマンド順序は維持される）
      this.submit();
    }
    const off = this.arenaOff;
    this.arenaOff += UNIFORM_SLOT;
    return off;
  }

  // ---- カーネル ----
  /**
   * fields: [["W","u32"],["s","f32"],["rects","vec4i",8]] の順で uniform 構造体に詰める
   * bindings: 各 storage バッファのアクセス ("r" | "rw")、binding 1 から順に割り当て
   */
  kernel(name, code, fields, bindings, wg = [256, 1, 1]) {
    if (this.kernels.has(name)) return this.kernels.get(name);
    const k = new Kernel(this, name, code, fields, bindings, wg);
    this.kernels.set(name, k);
    return k;
  }
}

export class Kernel {
  constructor(gpu, name, code, fields, bindings, wg) {
    this.gpu = gpu;
    this.name = name;
    this.fields = fields;
    this.wg = wg;
    const dev = gpu.device;
    const entries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }];
    bindings.forEach((acc, i) => entries.push({
      binding: i + 1, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: acc === "r" ? "read-only-storage" : "storage" },
    }));
    this.bgl = dev.createBindGroupLayout({ entries });
    const module = dev.createShaderModule({ label: name, code });
    module.getCompilationInfo().then((ci) => {
      for (const m of ci.messages) if (m.type === "error") gpu.log(`[wgsl:${name}] ${m.lineNum}:${m.linePos} ${m.message}`);
    });
    this.pipeline = dev.createComputePipeline({
      label: name, layout: dev.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
      compute: { module, entryPoint: "main" },
    });
    // uniform レイアウト（std140 相当: スカラ 4B、vec4 は 16B 整列）
    let off = 0;
    this.layout = [];
    for (const [fname, type, count] of fields) {
      if (type === "vec4i" || type === "vec4f") {
        off = Math.ceil(off / 16) * 16;
        this.layout.push([fname, type, off, count || 1]);
        off += 16 * (count || 1);
      } else {
        this.layout.push([fname, type, off, 1]);
        off += 4;
      }
    }
    if (off > UNIFORM_SLOT) throw new Error(`uniform too large: ${name}`);
  }

  _bindGroup(buffers) {
    const key = `${this.name}|${buffers.map((b) => b._id).join("|")}|`;
    let bg = this.gpu.bgCache.get(key);
    if (!bg) {
      const entries = [{ binding: 0, resource: { buffer: this.gpu.arenaBuf, offset: 0, size: UNIFORM_SLOT } }];
      buffers.forEach((b, i) => entries.push({ binding: i + 1, resource: { buffer: b } }));
      bg = this.gpu.device.createBindGroup({ layout: this.bgl, entries });
      this.gpu.bgCache.set(key, bg);
      if (this.gpu.bgCache.size > 4096) this.gpu.bgCache.clear();
    }
    return bg;
  }

  /** params: {name: value | array}; buffers: GPUBuffer[]; nx, ny, nz: ワークグループ数 */
  run(params, buffers, nx, ny = 1, nz = 1) {
    if (nx <= 0 || ny <= 0 || nz <= 0) return;
    if (nx > 65535 || ny > 65535 || nz > 65535) throw new Error(`dispatch too large in ${this.name}: ${nx}x${ny}x${nz}`);
    const gpu = this.gpu;
    const uoff = gpu._allocUniform();
    const dv = gpu.arenaView;
    for (const [fname, type, foff, count] of this.layout) {
      const v = params[fname];
      if (v === undefined) throw new Error(`missing param ${fname} in ${this.name}`);
      const base = uoff + foff;
      if (type === "u32") dv.setUint32(base, v >>> 0, true);
      else if (type === "i32") dv.setInt32(base, v | 0, true);
      else if (type === "f32") dv.setFloat32(base, v, true);
      else if (type === "vec4i") for (let c = 0; c < count * 4; c++) dv.setInt32(base + 4 * c, (v[c] || 0) | 0, true);
      else if (type === "vec4f") for (let c = 0; c < count * 4; c++) dv.setFloat32(base + 4 * c, v[c] || 0, true);
    }
    const pass = gpu.enc().beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this._bindGroup(buffers), [uoff]);
    pass.dispatchWorkgroups(nx, ny, nz);
    pass.end();
  }

  /** 2D 画像カーネル向け: (w, h) ピクセルをワークグループで覆う */
  run2d(params, buffers, w, h) {
    this.run(params, buffers, Math.ceil(w / this.wg[0]), Math.ceil(h / this.wg[1]), 1);
  }
}
