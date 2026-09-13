// trim.js - 動画のトリム（開始/終了フレーム）、クロップ、無視矩形/テキスト矩形をプレビュー上で指定するパネル
//
// - 範囲バーの両端ハンドルをドラッグして大まかに指定し、-1 / +1 で 1 フレーム単位に微調整
// - 開始フレームと終了フレームを <video> 2 本で並べて表示（シークで追従）
// - プレビュー上のドラッグで矩形を描く。モード: crop（1 個、ハンドルで拡縮・内側ドラッグで移動）、ignore / text（複数）
// - フレームレートは requestVideoFrameCallback で実測（取れなければ fallback 値）。フレーム番号は t*fps の近似
// - 無視/テキスト矩形は内部では元動画座標で持ち、getState() でクロップ後の座標に変換して返す
// - ルーペ: プレビュー上のカーソル位置を LOUPE_ZOOM 倍で拡大表示（矩形の辺と座標も描く）。ホイールで倍率変更
// - 矩形の編集: 現在のモードの矩形の内側を押すと選択して移動、選択中の矩形のハンドルで拡縮。空いた所から描くと新規追加。
//   Ctrl（Mac は Cmd）を押しながらドラッグすると既存の矩形の上からでも新規追加。Delete / Backspace で選択中の矩形を削除

const ASPECTS = { free: null, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1, "9:16": 9 / 16, "21:9": 21 / 9 };
const COMMON_FPS = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120];
const LOUPE_PX = 180;                 // ルーペの表示サイズ（CSS px）
const LOUPE_ZOOMS = [2, 4, 8];        // ホイールで切替

function normRect(x0, y0, x1, y1) {
  const xa = Math.min(x0, x1), xb = Math.max(x0, x1), ya = Math.min(y0, y1), yb = Math.max(y0, y1);
  return [Math.round(xa), Math.round(ya), Math.round(xb - xa), Math.round(yb - ya)];
}

export class TrimPanel {
  /**
   * root: パネルを入れる要素。opts.onRects(ignoreRects, textRects): クロップ後座標の矩形を通知
   * opts.onFps(fps): 実測フレームレートを通知
   */
  constructor(root, opts = {}) {
    this.root = root;
    this.opts = opts;
    this.file = null;
    this.url = null;
    this.fps = 30;
    this.duration = 0;
    this.n = 0;
    this.W = 0; this.H = 0;
    this.start = 0; this.end = 0;
    this.crop = null;
    this.ignoreRects = [];
    this.textRects = [];
    this.mode = "crop";
    this.sel = { ignore: -1, text: -1 };   // ignore / text の選択中インデックス
    this.drag = null;
    this._hover = false;
    this._build();
    this.root.hidden = true;
  }

  // ------------------------------------------------------------ UI
  _build() {
    const r = this.root;
    r.innerHTML = `
      <div class="trim-previews">
        <div class="trim-col"><div class="hint">Start frame</div><div class="trim-wrap"><video muted playsinline preload="auto"></video><canvas class="trim-ov"></canvas><canvas class="trim-loupe" hidden></canvas></div></div>
        <div class="trim-col"><div class="hint">End frame</div><div class="trim-wrap"><video muted playsinline preload="auto"></video><canvas class="trim-ov"></canvas><canvas class="trim-loupe" hidden></canvas></div></div>
      </div>
      <div class="hint trim-info"></div>
      <div class="trim-bar"><div class="trim-range"></div><div class="trim-handle" data-h="start"></div><div class="trim-handle" data-h="end"></div></div>
      <div class="row">
        <label>Start</label><input type="number" class="t-start" step="1" style="width:80px"><button class="t-sm">-1</button><button class="t-sp">+1</button><span class="hint t-stime"></span>
        <label style="min-width:40px;margin-left:16px">End</label><input type="number" class="t-end" step="1" style="width:80px"><button class="t-em">-1</button><button class="t-ep">+1</button><span class="hint t-etime"></span>
        <span class="hint t-count" style="margin-left:16px"></span>
      </div>
      <div class="hint trim-help">
        <b><span class="sw" style="border-color:#ffe040"></span>Crop</b>: 出力に使う範囲（1 個）。ドラッグで描き、角/辺のハンドルで拡縮、内側ドラッグで移動。
        <b><span class="sw" style="border-color:#ff4040"></span>Ignore rect</b>: 常に除外する領域（固定ロゴ・ワイプなど。複数可）。
        <b><span class="sw" style="border-color:#40d0ff"></span>Text rect</b>: 動くテロップの帯（ティッカー等。矩形内は勾配の高い画素を文字としてマスク。複数可）。
        描いた矩形は内側をドラッグで移動、選択中（太線）の矩形はハンドルで拡縮。Ctrl+ドラッグで既存の矩形の上からでも新規追加、Delete キーで選択中を削除。
        Ignore / Text はクロップ後の座標に変換して「パラメータ」の ignore rects / text rects 欄へ自動で書き込まれます。
      </div>
      <div class="row">
        <label>矩形</label>
        <label style="min-width:auto"><input type="radio" name="trim-mode" value="crop" checked> Crop</label>
        <label style="min-width:auto"><input type="radio" name="trim-mode" value="ignore"> Ignore rect</label>
        <label style="min-width:auto"><input type="radio" name="trim-mode" value="text"> Text rect</label>
        <label style="min-width:auto;margin-left:12px">Aspect</label><select class="t-aspect">${Object.keys(ASPECTS).map((k) => `<option>${k}</option>`).join("")}</select>
        <button class="t-clearcrop">Clear crop</button>
        <button class="t-popi">Remove last ignore</button><button class="t-cleari">Clear ignore</button>
        <button class="t-popt">Remove last text</button><button class="t-cleart">Clear text</button>
      </div>
      <div class="row">
        <label>Crop x,y,w,h</label>
        <input type="number" class="t-cx" style="width:70px"><input type="number" class="t-cy" style="width:70px"><input type="number" class="t-cw" style="width:70px"><input type="number" class="t-ch" style="width:70px">
        <span class="hint t-rinfo"></span>
      </div>`;
    const q = (sel) => r.querySelector(sel);
    this.videos = [...r.querySelectorAll("video")];
    this.overlays = [...r.querySelectorAll("canvas.trim-ov")];
    this.loupes = [...r.querySelectorAll("canvas.trim-loupe")];
    this.loupeZoom = 4;
    for (const lp of this.loupes) { lp.width = LOUPE_PX * 2; lp.height = LOUPE_PX * 2; }   // 高 DPI 向けに 2 倍解像度
    this.bar = q(".trim-bar"); this.range = q(".trim-range");
    this.handles = { start: q('.trim-handle[data-h="start"]'), end: q('.trim-handle[data-h="end"]') };
    this.inStart = q(".t-start"); this.inEnd = q(".t-end");
    this.labStart = q(".t-stime"); this.labEnd = q(".t-etime"); this.labCount = q(".t-count");
    this.info = q(".trim-info"); this.rinfo = q(".t-rinfo");
    this.cropIn = [q(".t-cx"), q(".t-cy"), q(".t-cw"), q(".t-ch")];
    this.aspect = q(".t-aspect");
    q(".t-sm").onclick = () => this.step("start", -1); q(".t-sp").onclick = () => this.step("start", 1);
    q(".t-em").onclick = () => this.step("end", -1); q(".t-ep").onclick = () => this.step("end", 1);
    for (const [el, which] of [[this.inStart, "start"], [this.inEnd, "end"]]) {
      el.onchange = () => this.set(which, parseInt(el.value, 10));
    }
    for (const el of r.querySelectorAll('input[name="trim-mode"]')) el.onchange = () => { this.mode = el.value; this._drawOverlays(); };
    // Delete / Backspace: 選択中の矩形を削除（プレビュー上にカーソルがあるときだけ。入力欄でのキー操作は邪魔しない）
    document.addEventListener("keydown", (e) => {
      if (!this._hover || (e.key !== "Delete" && e.key !== "Backspace")) return;
      const tag = (document.activeElement && document.activeElement.tagName) || "";
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      e.preventDefault();
      this.deleteSelected();
    });
    q(".t-clearcrop").onclick = () => { this.crop = null; this.refresh(); };
    q(".t-popi").onclick = () => { this.ignoreRects.pop(); this.sel.ignore = -1; this.refresh(); };
    q(".t-cleari").onclick = () => { this.ignoreRects = []; this.sel.ignore = -1; this.refresh(); };
    q(".t-popt").onclick = () => { this.textRects.pop(); this.sel.text = -1; this.refresh(); };
    q(".t-cleart").onclick = () => { this.textRects = []; this.sel.text = -1; this.refresh(); };
    for (const el of this.cropIn) el.onchange = () => this._cropEntry();
    // 範囲バー
    this.bar.addEventListener("pointerdown", (e) => {
      const f = this._barFrame(e);
      this.barActive = Math.abs(f - this.start) <= Math.abs(f - this.end) ? "start" : "end";
      try { this.bar.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
      this.set(this.barActive, f);
    });
    this.bar.addEventListener("pointermove", (e) => { if (this.barActive) this.set(this.barActive, this._barFrame(e)); });
    this.bar.addEventListener("pointerup", () => { this.barActive = null; });
    // プレビュー上の矩形操作
    this.overlays.forEach((cv, i) => {
      cv.addEventListener("pointerdown", (e) => this._onPress(e, cv));
      cv.addEventListener("pointermove", (e) => { this._onMove(e, cv); this._loupe(e, i); });
      cv.addEventListener("pointerup", (e) => this._onRelease(e, cv));
      cv.addEventListener("pointerenter", () => { this._hover = true; });
      cv.addEventListener("pointerleave", () => { this._hover = false; this.loupes[i].hidden = true; });
      cv.addEventListener("wheel", (e) => {
        e.preventDefault();
        const k = LOUPE_ZOOMS.indexOf(this.loupeZoom);
        this.loupeZoom = LOUPE_ZOOMS[Math.max(0, Math.min(LOUPE_ZOOMS.length - 1, k + (e.deltaY < 0 ? 1 : -1)))];
        this._loupe(e, i);
      }, { passive: false });
    });
    window.addEventListener("resize", () => this._layout());
  }

  // ------------------------------------------------------------ 読み込み
  async load(file, fallbackFps = 30) {
    this.file = file;
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = URL.createObjectURL(file);
    this.root.hidden = false;
    this.info.textContent = "loading...";
    const v0 = this.videos[0], v1 = this.videos[1];
    for (const v of this.videos) v.src = this.url;
    await new Promise((res, rej) => { v0.onloadedmetadata = res; v0.onerror = () => rej(new Error("動画を開けません")); });
    this.duration = v0.duration; this.W = v0.videoWidth; this.H = v0.videoHeight;
    this.fps = await this._measureFps(v0, fallbackFps);
    this.n = Math.max(1, Math.round(this.duration * this.fps));
    this.start = 0; this.end = this.n - 1;
    this.crop = null; this.ignoreRects = []; this.textRects = [];
    this.info.textContent = `${this.W}x${this.H}, ${this.duration.toFixed(2)} s, ≈${this.fps.toFixed(3)} fps (${this.fpsSource}), ≈${this.n} frames`;
    if (this.opts.onFps) this.opts.onFps(this.fps);
    this._layout();
    this.refresh();
    await v1.play().catch(() => {}); v1.pause();
  }

  async _measureFps(v, fallback) {
    this.fpsSource = "fallback";
    if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return fallback;
    const times = [];
    v.currentTime = 0;
    await new Promise((res) => { v.onseeked = res; });
    let id = 0;
    const done = new Promise((res) => {
      const cb = (_now, meta) => {
        times.push(meta.mediaTime);
        if (times.length >= 16 || meta.mediaTime > 1.5) res(); else id = v.requestVideoFrameCallback(cb);
      };
      id = v.requestVideoFrameCallback(cb);
      setTimeout(res, 2500);
    });
    try { await v.play(); } catch (e) { return fallback; }
    await done;
    v.pause();
    if (id) try { v.cancelVideoFrameCallback(id); } catch (e) { /* ignore */ }
    const d = [];
    for (let i = 1; i < times.length; i++) { const dt = times[i] - times[i - 1]; if (dt > 1e-4) d.push(dt); }
    if (d.length < 4) return fallback;
    d.sort((a, b) => a - b);
    const fps = 1 / d[Math.floor(d.length / 2)];
    for (const c of COMMON_FPS) if (Math.abs(fps - c) / c < 0.02) { this.fpsSource = "measured"; return c; }
    this.fpsSource = "measured";
    return Math.round(fps * 1000) / 1000;
  }

  // ------------------------------------------------------------ 範囲
  frameTime(f) { return Math.min(this.duration, (f + 0.5) / this.fps); }
  set(which, f) {
    if (!Number.isFinite(f)) { this.refresh(); return; }
    f = Math.max(0, Math.min(this.n - 1, Math.round(f)));
    if (which === "start") { this.start = f; if (this.end < f) this.end = f; }
    else { this.end = f; if (this.start > f) this.start = f; }
    this.refresh();
  }
  step(which, d) { this.set(which, (which === "start" ? this.start : this.end) + d); }
  _barFrame(e) {
    const rc = this.bar.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - rc.left - 8) / Math.max(1, rc.width - 16)));
    return Math.round(t * (this.n - 1));
  }

  refresh() {
    this.inStart.value = this.start; this.inEnd.value = this.end;
    this.labStart.textContent = `${this.frameTime(this.start).toFixed(3)} s`;
    this.labEnd.textContent = `${this.frameTime(this.end).toFixed(3)} s`;
    this.labCount.textContent = `${this.end - this.start + 1} frames`;
    const pct = (f) => `${(8 + (this.n > 1 ? f / (this.n - 1) : 0) * (this.bar.clientWidth - 16))}px`;
    this.handles.start.style.left = pct(this.start); this.handles.end.style.left = pct(this.end);
    this.range.style.left = pct(this.start); this.range.style.width = `${parseFloat(pct(this.end)) - parseFloat(pct(this.start))}px`;
    this.cropIn.forEach((el, i) => { el.value = this.crop ? this.crop[i] : ""; });
    this.rinfo.textContent = `ignore: ${this.ignoreRects.length}  text: ${this.textRects.length}`;
    const ts = this.frameTime(this.start), te = this.frameTime(this.end);
    if (Math.abs(this.videos[0].currentTime - ts) > 1e-4) this.videos[0].currentTime = ts;
    if (Math.abs(this.videos[1].currentTime - te) > 1e-4) this.videos[1].currentTime = te;
    this._drawOverlays();
    if (this.opts.onRects) { const st = this.getState(); this.opts.onRects(st.ignoreRects, st.textRects); }
  }

  _layout() {
    for (const cv of this.overlays) {
      const v = cv.previousElementSibling;
      const w = v.clientWidth || 480, h = v.clientHeight || Math.round(480 * this.H / Math.max(1, this.W));
      cv.width = w; cv.height = h;
      cv.style.width = `${w}px`; cv.style.height = `${h}px`;
    }
    this._drawOverlays();
    this.refresh();
  }

  get scale() { return this.overlays[0].width / Math.max(1, this.W); }

  _drawOverlays() {
    const s = this.scale;
    for (const cv of this.overlays) {
      const g = cv.getContext("2d");
      g.clearRect(0, 0, cv.width, cv.height);
      g.lineWidth = 2;
      const rect = (r, col, dash) => { g.strokeStyle = col; g.setLineDash(dash || []); g.strokeRect(r[0] * s, r[1] * s, r[2] * s, r[3] * s); };
      this.ignoreRects.forEach((r, k) => { g.lineWidth = this.mode === "ignore" && k === this.sel.ignore ? 3 : 2; rect(r, "#ff4040"); });
      this.textRects.forEach((r, k) => { g.lineWidth = this.mode === "text" && k === this.sel.text ? 3 : 2; rect(r, "#40d0ff"); });
      g.lineWidth = 2;
      if (this.crop) {
        const [x, y, w, h] = this.crop;
        g.fillStyle = "rgba(0,0,0,0.45)";
        g.fillRect(0, 0, cv.width, y * s); g.fillRect(0, (y + h) * s, cv.width, cv.height - (y + h) * s);
        g.fillRect(0, y * s, x * s, h * s); g.fillRect((x + w) * s, y * s, cv.width - (x + w) * s, h * s);
        rect(this.crop, "#ffe040");
      }
      const selr = this._selRect();
      if (selr) {
        g.fillStyle = { crop: "#ffe040", ignore: "#ff4040", text: "#40d0ff" }[this.mode];
        for (const [hx, hy] of this._handles(selr)) g.fillRect(hx - 4, hy - 4, 8, 8);
      }
      if (this.drag && this.drag.preview) rect(this.drag.preview, { crop: "#ffe040", ignore: "#ff4040", text: "#40d0ff" }[this.drag.mode], [4, 2]);
    }
  }

  _list(kind) { return kind === "ignore" ? this.ignoreRects : this.textRects; }
  /** 現在のモードで選択中の矩形（crop モードならクロップ） */
  _selRect() {
    if (this.mode === "crop") return this.crop;
    const l = this._list(this.mode), i = this.sel[this.mode];
    return i >= 0 && i < l.length ? l[i] : null;
  }
  _setSelRect(r) {
    if (this.mode === "crop") this.crop = r;
    else { const l = this._list(this.mode), i = this.sel[this.mode]; if (i >= 0 && i < l.length) l[i] = r; }
  }
  /** 選択中の矩形を削除（crop モードならクロップ解除） */
  deleteSelected() {
    if (this.mode === "crop") { this.crop = null; }
    else { const l = this._list(this.mode), i = this.sel[this.mode]; if (i >= 0 && i < l.length) l.splice(i, 1); this.sel[this.mode] = -1; }
    this.refresh();
  }
  _handles(r) {
    if (!r) return [];
    const [x, y, w, h] = r, s = this.scale;
    const xs = [x * s, (x + w / 2) * s, (x + w) * s], ys = [y * s, (y + h / 2) * s, (y + h) * s];
    const out = [];
    for (const hy of ys) for (const hx of xs) if (!(hx === xs[1] && hy === ys[1])) out.push([hx, hy]);
    return out;   // 0..7 = 左上, 上, 右上, 左, 右, 左下, 下, 右下
  }

  // ------------------------------------------------------------ 矩形操作
  _toFrame(e, cv) {
    const rc = cv.getBoundingClientRect(), s = this.scale;
    return [Math.max(0, Math.min(this.W, (e.clientX - rc.left) / s)), Math.max(0, Math.min(this.H, (e.clientY - rc.top) / s))];
  }
  _hitHandle(e, cv) {
    const rc = cv.getBoundingClientRect(), px = e.clientX - rc.left, py = e.clientY - rc.top;
    const hs = this._handles(this._selRect());
    for (let i = 0; i < hs.length; i++) if (Math.abs(px - hs[i][0]) <= 6 && Math.abs(py - hs[i][1]) <= 6) return i;
    return -1;
  }
  _onPress(e, cv) {
    try { cv.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events */ }
    const [fx, fy] = this._toFrame(e, cv);
    const inside = (r) => r && fx >= r[0] && fx <= r[0] + r[2] && fy >= r[1] && fy <= r[1] + r[3];
    const forceNew = e.ctrlKey || e.metaKey;
    if (!forceNew) {
      // 1) 選択中の矩形のハンドル → 拡縮、内側 → 移動
      const cur = this._selRect();
      if (cur) {
        const hi = this._hitHandle(e, cv);
        const orig = [cur[0], cur[1], cur[0] + cur[2], cur[1] + cur[3]];
        if (hi >= 0) { this.drag = { mode: this.mode, edit: true, handle: hi, orig, preview: null }; return; }
        if (inside(cur)) { this.drag = { mode: this.mode, edit: true, move: [fx, fy], orig, preview: null }; return; }
      }
      // 2) 同じ種類の他の矩形の内側 → それを選択して移動（後から描いたものを優先）
      if (this.mode !== "crop") {
        const l = this._list(this.mode);
        for (let k = l.length - 1; k >= 0; k--) {
          if (inside(l[k])) {
            this.sel[this.mode] = k;
            const r = l[k];
            this.drag = { mode: this.mode, edit: true, move: [fx, fy], orig: [r[0], r[1], r[0] + r[2], r[1] + r[3]], preview: null };
            this._drawOverlays();
            return;
          }
        }
      }
    }
    // 3) 新規
    this.drag = { mode: this.mode, x0: fx, y0: fy, preview: null };
  }
  _onMove(e, cv) {
    if (!this.drag) {
      const [fx, fy] = this._toFrame(e, cv);
      const inside = (r) => r && fx >= r[0] && fx <= r[0] + r[2] && fy >= r[1] && fy <= r[1] + r[3];
      let over = this._hitHandle(e, cv) >= 0 || inside(this._selRect());
      if (!over && this.mode !== "crop") over = this._list(this.mode).some(inside);
      cv.style.cursor = over && !(e.ctrlKey || e.metaKey) ? "move" : "crosshair";
      return;
    }
    const [fx, fy] = this._toFrame(e, cv), d = this.drag, ar = ASPECTS[this.aspect.value];
    if (d.move) {
      let [x0, y0, x1, y1] = d.orig;
      const dx = Math.max(-x0, Math.min(this.W - x1, fx - d.move[0])), dy = Math.max(-y0, Math.min(this.H - y1, fy - d.move[1]));
      d.preview = normRect(x0 + dx, y0 + dy, x1 + dx, y1 + dy);
    } else if (d.handle !== undefined) {
      let [x0, y0, x1, y1] = d.orig; const hi = d.handle;
      if ([0, 3, 5].includes(hi)) x0 = fx; if ([2, 4, 7].includes(hi)) x1 = fx;
      if ([0, 1, 2].includes(hi)) y0 = fy; if ([5, 6, 7].includes(hi)) y1 = fy;
      if (ar) y1 = y0 + Math.abs(x1 - x0) / ar;
      d.preview = normRect(x0, y0, x1, y1);
    } else {
      let x1 = fx, y1 = fy;
      if (d.mode === "crop" && ar) { const w = Math.abs(x1 - d.x0); y1 = y1 >= d.y0 ? d.y0 + w / ar : d.y0 - w / ar; }
      d.preview = normRect(d.x0, d.y0, x1, y1);
    }
    this._drawOverlays();
  }
  /** カーソル位置を拡大表示するルーペ。i = プレビュー番号 */
  _loupe(e, i) {
    const lp = this.loupes[i], cv = this.overlays[i], v = this.videos[i];
    if (!this.W || v.readyState < 2) { lp.hidden = true; return; }
    const [fx, fy] = this._toFrame(e, cv);
    const z = this.loupeZoom, S = LOUPE_PX / z;        // 元動画座標での表示範囲（px）
    const sx = fx - S / 2, sy = fy - S / 2;
    const g = lp.getContext("2d"), L = lp.width, k = L / S;   // 元動画 px → ルーペ px
    g.imageSmoothingEnabled = false;
    g.fillStyle = "#000"; g.fillRect(0, 0, L, L);
    // 範囲外は黒のまま（drawImage の source は動画内にクリップする）
    const cx0 = Math.max(0, sx), cy0 = Math.max(0, sy), cx1 = Math.min(this.W, sx + S), cy1 = Math.min(this.H, sy + S);
    if (cx1 > cx0 && cy1 > cy0) {
      try { g.drawImage(v, cx0, cy0, cx1 - cx0, cy1 - cy0, (cx0 - sx) * k, (cy0 - sy) * k, (cx1 - cx0) * k, (cy1 - cy0) * k); } catch (err) { /* ignore */ }
    }
    // 矩形の辺
    g.lineWidth = 2;
    const rect = (r, col, dash) => { g.strokeStyle = col; g.setLineDash(dash || []); g.strokeRect((r[0] - sx) * k, (r[1] - sy) * k, r[2] * k, r[3] * k); };
    for (const r of this.ignoreRects) rect(r, "#ff4040");
    for (const r of this.textRects) rect(r, "#40d0ff");
    if (this.crop) rect(this.crop, "#ffe040");
    if (this.drag && this.drag.preview) rect(this.drag.preview, { crop: "#ffe040", ignore: "#ff4040", text: "#40d0ff" }[this.drag.mode], [6, 3]);
    // 十字線と座標
    g.setLineDash([]); g.strokeStyle = "rgba(255,255,255,0.9)"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(L / 2, 0); g.lineTo(L / 2, L); g.moveTo(0, L / 2); g.lineTo(L, L / 2); g.stroke();
    g.strokeStyle = "rgba(0,0,0,0.6)"; g.strokeRect(L / 2 - k / 2, L / 2 - k / 2, k, k);
    const txt = `${Math.floor(fx)}, ${Math.floor(fy)}  x${z}`;
    g.font = "bold 22px system-ui, sans-serif"; g.textBaseline = "top";
    g.fillStyle = "rgba(0,0,0,0.6)"; g.fillRect(4, 4, g.measureText(txt).width + 12, 30);
    g.fillStyle = "#fff"; g.fillText(txt, 10, 8);
    // カーソルの右下に表示。端に近ければ左/上へ反転
    const rc = cv.getBoundingClientRect();
    let px = e.clientX - rc.left + 24, py = e.clientY - rc.top + 24;
    if (px + LOUPE_PX > cv.width) px = e.clientX - rc.left - 24 - LOUPE_PX;
    if (py + LOUPE_PX > cv.height) py = e.clientY - rc.top - 24 - LOUPE_PX;
    lp.style.left = `${Math.max(0, px)}px`; lp.style.top = `${Math.max(0, py)}px`;
    lp.hidden = false;
  }
  _onRelease() {
    if (!this.drag) return;
    const d = this.drag; this.drag = null;
    let r = d.preview;
    if (!r || r[2] < 4 || r[3] < 4) { this._drawOverlays(); return; }
    const x = Math.max(0, r[0]), y = Math.max(0, r[1]);
    r = [x, y, Math.min(this.W - x, r[2]), Math.min(this.H - y, r[3])];
    if (d.edit) this._setSelRect(r);
    else if (d.mode === "crop") this.crop = r;
    else { const l = this._list(d.mode); l.push(r); this.sel[d.mode] = l.length - 1; }
    this.refresh();
  }
  _cropEntry() {
    const v = this.cropIn.map((el) => parseInt(el.value, 10));
    if (v.every(Number.isFinite) && v[2] > 0 && v[3] > 0) {
      const x = Math.max(0, v[0]), y = Math.max(0, v[1]);
      this.crop = [x, y, Math.min(this.W - x, v[2]), Math.min(this.H - y, v[3])];
    }
    this.refresh();
  }

  // ------------------------------------------------------------ 結果
  _toCropped(rects) {
    if (!this.crop) return rects.map((r) => [...r]);
    const [cx, cy, cw, ch] = this.crop, out = [];
    for (const [x, y, w, h] of rects) {
      const x0 = Math.max(x, cx), y0 = Math.max(y, cy), x1 = Math.min(x + w, cx + cw), y1 = Math.min(y + h, cy + ch);
      if (x1 > x0 && y1 > y0) out.push([x0 - cx, y0 - cy, x1 - x0, y1 - y0]);
    }
    return out;
  }
  /** 抽出に使う状態。active=false なら動画未読み込み */
  getState() {
    return {
      active: !!this.file, fps: this.fps, startFrame: this.start, endFrame: this.end,
      t0: this.n ? this.frameTime(this.start) : 0, t1: this.n ? Math.min(this.duration, (this.end + 1) / this.fps) : 0,
      crop: this.crop ? [...this.crop] : null,
      ignoreRects: this._toCropped(this.ignoreRects), textRects: this._toCropped(this.textRects),
    };
  }
  hide() { this.root.hidden = true; this.file = null; for (const v of this.videos) v.removeAttribute("src"); }
}
