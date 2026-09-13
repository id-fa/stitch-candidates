// app.js - UI、フレーム読み込み（動画シーク抽出 / 画像列）、出力

import { Gpu } from "./gpu.js";
import { Reconstructor } from "./recon.js";
import { render } from "./render.js";
import { fillHoles } from "./postfx.js";
import { TrimPanel } from "./trim.js";

const $ = (id) => document.getElementById(id);
const logEl = $("log");
function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
  console.log(msg);
}
let gpu = null, rc = null, running = false;
let selectedFiles = [];   // ファイル入力または D&D で選ばれたファイル
let outPrefix = "";       // 保存ファイル名の接頭辞（入力ファイル名から拡張子を除いたもの）
const results = {};   // name -> {canvas, w, h}

function setProgress(stage, frac) {
  $("progress").value = frac;
  $("stage").textContent = `${stage} ${(frac * 100).toFixed(0)}%`;
}

function num(id, def) { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : def; }
function int(id, def) { const v = parseInt($(id).value, 10); return Number.isFinite(v) ? v : def; }

function readArgs() {
  const parseRects = (id, label) => $(id).value.split(";").map((s) => s.trim()).filter(Boolean).map((s) => {
    const p = s.split(",").map((v) => parseInt(v, 10));
    if (p.length !== 4 || p.some((v) => !Number.isFinite(v))) throw new Error(`${label} の形式が不正です: ${s}（x,y,w,h;...）`);
    return p;
  });
  const rects = parseRects("ignoreRects", "ignore rects");
  const textRects = parseRects("textRects", "text rects");
  const cs = $("canvasScale").value.trim();
  if (cs !== "auto" && !Number.isFinite(parseFloat(cs))) throw new Error("canvas scale は auto か数値");
  return {
    model: $("model").value,
    pairs: [...new Set($("pairs").value.split(",").map((v) => parseInt(v, 10)).filter((v) => v > 0))].sort((a, b) => a - b),
    minOverlap: num("minOverlap", 0.15), coarseTol: num("coarseTol", 2.0),
    scaleMax: num("scaleMax", 0.06), scaleStep: num("scaleStep", 0.004),
    fineScale: num("fineScale", 1.0), gnIters: int("gnIters", 15), ignoreRects: rects, textRects,
    holeFill: $("holeFill").value,
    staticMask: $("staticMask").checked, staticSpan: int("staticSpan", 6), staticDiff: num("staticDiff", 0.03),
    staticGrad: num("staticGrad", 0.08), staticDilate: int("staticDilate", 7), staticHalo: int("staticHalo", 12), staticClose: int("staticClose", 3), textHalo: int("textHalo", 4),
    canvasScale: cs, inlierTol: num("inlierTol", 0.06), sharpTop: num("sharpTop", 0.3),
    stackBudgetMB: int("stackBudget", 128), levelCacheMB: int("levelCache", 768),
  };
}

// ------------------------------------------------------------ フレーム読み込み
function naturalKey(s) { return s.split(/(\d+)/).map((t) => (/^\d+$/.test(t) ? t.padStart(12, "0") : t.toLowerCase())).join(""); }

function frameCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

/** 動画からシークでフレーム抽出。各フレームは即 GPU バッファへ */
async function loadVideoFrames(file, opts, onFrame) {
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
  const frames = [];
  for (let k = 0; k < times.length; k++) {
    await new Promise((res, rej) => {
      const to = setTimeout(() => rej(new Error(`seek timeout at ${times[k].toFixed(2)}s`)), 15000);
      v.onseeked = () => { clearTimeout(to); res(); };
      v.onerror = () => { clearTimeout(to); rej(new Error("seek error")); };
      v.currentTime = times[k];
    });
    ctx.drawImage(v, crop[0], crop[1], crop[2], crop[3], 0, 0, W, H);
    const img = ctx.getImageData(0, 0, W, H);
    frames.push(onFrame(k, img.data));
    if (k % 10 === 0) setProgress("load", k / times.length);
  }
  URL.revokeObjectURL(url);
  return { W, H, frames };
}

async function loadImageFrames(files, opts, onFrame) {
  const list = [...files].sort((a, b) => (naturalKey(a.name) < naturalKey(b.name) ? -1 : 1));
  const sel = list.filter((_, i) => i % opts.every === 0).slice(0, opts.maxFrames > 0 ? opts.maxFrames : undefined);
  let W = 0, H = 0, cv = null, ctx = null;
  const frames = [];
  for (let k = 0; k < sel.length; k++) {
    const bmp = await createImageBitmap(sel[k]);
    if (k === 0) {
      W = Math.max(8, Math.round(bmp.width * opts.inputScale)); H = Math.max(8, Math.round(bmp.height * opts.inputScale));
      cv = frameCanvas(W, H); ctx = cv.getContext("2d", { willReadFrequently: true });
      log(`[load] ${sel.length} images ${bmp.width}x${bmp.height} → ${W}x${H}`);
    }
    ctx.drawImage(bmp, 0, 0, W, H);
    bmp.close();
    frames.push(onFrame(k, ctx.getImageData(0, 0, W, H).data));
    if (k % 10 === 0) setProgress("load", k / sel.length);
  }
  return { W, H, frames };
}

// ------------------------------------------------------------ 出力
function showImage(name, rgba, w, h, label) {
  let c = results[name]?.canvas;
  if (!c) {
    c = document.createElement("canvas");
    results[name] = { canvas: c };
    const tab = document.createElement("button");
    tab.textContent = label; tab.className = "tab"; tab.dataset.name = name;
    tab.onclick = () => selectTab(name);
    $("tabs").appendChild(tab);
  }
  c.width = w; c.height = h;
  c.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  results[name].w = w; results[name].h = h; results[name].label = label;
  selectTab(name);
}
function selectTab(name) {
  const view = $("view");
  view.innerHTML = "";
  view.appendChild(results[name].canvas);
  for (const b of $("tabs").querySelectorAll(".tab")) b.classList.toggle("active", b.dataset.name === name);
  $("viewInfo").textContent = `${results[name].label}: ${results[name].w} x ${results[name].h}`;
  $("dlPng").onclick = () => results[name].canvas.toBlob((blob) => download(blob, `${outPrefix}${name}.png`), "image/png");
}
function download(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
function csvButton(id, filename, text) {
  const b = $(id);
  b.disabled = false;
  b.onclick = () => download(new Blob([text], { type: "text/csv" }), filename);
}

async function makeOverlayDebug(rc) {
  const idxs = [...new Set([0, Math.floor(rc.n / 2), rc.n - 1])];
  const W = rc.W, H = rc.H, w2 = Math.floor(W / 2), h2 = Math.floor(H / 2);
  const out = new Uint8ClampedArray(w2 * idxs.length * h2 * 4);
  for (let t = 0; t < idxs.length; t++) {
    const f = await rc.readFrame(idxs[t]);
    for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
      const si = ((2 * y) * W + 2 * x) * 4, di = (y * w2 * idxs.length + t * w2 + x) * 4;
      const ov = f[si + 3] < 128;
      out[di] = ov ? 0.4 * f[si] + 153 : f[si]; out[di + 1] = ov ? 0.4 * f[si + 1] : f[si + 1];
      out[di + 2] = ov ? 0.4 * f[si + 2] : f[si + 2]; out[di + 3] = 255;
    }
  }
  showImage("debug_overlay_mask", out, w2 * idxs.length, h2, "overlay mask");
}

// ------------------------------------------------------------ 実行
async function run() {
  if (running) return;
  running = true;
  $("run").disabled = true; $("cancel").disabled = false;
  logEl.textContent = "";
  $("tabs").innerHTML = ""; $("view").innerHTML = "";
  for (const k of Object.keys(results)) delete results[k];
  const tAll = performance.now();
  try {
    const args = readArgs();
    if (!gpu) gpu = await Gpu.create(log);
    const files = selectedFiles;
    if (!files.length) throw new Error("動画ファイルまたはフレーム画像を選択（またはドロップ）してください");
    outPrefix = files[0].name.replace(/\.[^.]+$/, "") + "_";
    const opts = { fps: num("fps", 10), start: num("start", 0), duration: num("duration", 0), maxFrames: int("maxFrames", 0),
                   inputScale: num("inputScale", 1.0), every: Math.max(1, int("every", 1)) };
    const onFrame = (k, rgba) => { const b = gpu.buf(rgba.byteLength, `frame${k}`); gpu.upload(b, rgba); return b; };
    const isVideo = files.length === 1 && isVideoFile(files[0]);
    if (isVideo) {
      const st = trim.getState();
      if (st.active && st.fps > 0) { opts.trim = st; opts.crop = st.crop; }
    }
    const src = isVideo ? await loadVideoFrames(files[0], opts, onFrame) : await loadImageFrames(files, opts, onFrame);
    if (src.frames.length < 2) throw new Error("フレームが 2 枚未満です");
    log(`[load] ${src.frames.length} frames, GPU frame memory ${(src.frames.length * src.W * src.H * 4 / 1048576).toFixed(0)}MB`);
    rc = new Reconstructor(gpu, src.W, src.H, src.frames, args, log, setProgress);
    await rc.preprocess();
    await makeOverlayDebug(rc);
    const al = await rc.align();
    // CSV
    let pos = "frame,scale,theta_deg,x,y\n";
    for (let k = 0; k < rc.n; k++) pos += `${k},${al.S[k].toFixed(5)},${(al.TH[k] * 180 / Math.PI).toFixed(4)},${al.T[k * 2].toFixed(3)},${al.T[k * 2 + 1].toFixed(3)}\n`;
    csvButton("dlPositions", `${outPrefix}positions.csv`, pos);
    let prs = "i,j,scale,theta_deg,tx,ty,score,residual\n";
    al.pairs.forEach(([i, j], k) => {
      prs += `${i},${j},${al.pp[k * 4].toFixed(5)},${(al.pp[k * 4 + 1] * 180 / Math.PI).toFixed(4)},${al.pp[k * 4 + 2].toFixed(3)},${al.pp[k * 4 + 3].toFixed(3)},${al.sc[k].toFixed(4)},${al.rn[k].toFixed(3)}\n`;
    });
    csvButton("dlPairs", `${outPrefix}pairs.csv`, prs);
    window.__last = { positions: pos, pairs: prs };  // デバッグ/検証用
    if (!$("noRender").checked) {
      const res = await render(rc, al);
      if (args.holeFill !== "none") {
        // 幾何的には覆われているがクリーン標本が無い画素（テロップが常に載っていた場所）を埋める
        // クリーン標本が無い、または少なすぎる（件数 < min clean count かつ 被覆数の min clean % 未満）画素。
        // テロップに常に覆われていた場所は、わずかに残った「クリーン」標本もグロー等で汚れていることが多い
        const frac = num("minCleanPct", 25) / 100, minCnt = int("minCleanCnt", 12);
        const fill = new Uint8Array(res.Wc * res.Hc);
        for (let i = 0; i < fill.length; i++) {
          const cg = res.coverageGeom[i], cc = res.coverage[i];
          fill[i] = cg > 0 && (cc === 0 || (cc < minCnt && cc < frac * cg)) ? 1 : 0;
        }
        const t0 = performance.now();
        let nfill = 0;
        for (const img of [res.median, res.mean, res.sharp]) nfill = fillHoles(img, fill, res.Wc, res.Hc, args.holeFill);
        log(`[postfx] hole fill (${args.holeFill}): ${nfill} px, ${((performance.now() - t0) / 1000).toFixed(1)}s`);
      }
      showImage("recon_mean", res.mean, res.Wc, res.Hc, "mean");
      showImage("recon_sharp", res.sharp, res.Wc, res.Hc, "sharp");
      let cmax = 1;
      for (const v of res.coverage) if (v > cmax) cmax = v;
      const cov = new Uint8ClampedArray(res.Wc * res.Hc * 4);
      for (let i = 0; i < res.coverage.length; i++) { const v = Math.round(255 * res.coverage[i] / cmax); cov[i * 4] = cov[i * 4 + 1] = cov[i * 4 + 2] = v; cov[i * 4 + 3] = 255; }
      showImage("coverage", cov, res.Wc, res.Hc, "coverage");
      showImage("recon_median", res.median, res.Wc, res.Hc, "median");
    }
    log(`[done] total ${((performance.now() - tAll) / 1000).toFixed(1)}s`);
  } catch (e) {
    log(`[error] ${e.message || e}`);
    console.error(e);
  } finally {
    if (rc) {
      try { await gpu.done(); } catch (e) { /* ignore */ }
      if (window.__keepRc) window.__rc = rc; else rc.destroy();   // __keepRc: デバッグ用に GPU 資源を保持
      rc = null;
    }
    running = false;
    $("run").disabled = false; $("cancel").disabled = true;
    setProgress("idle", 0);
  }
}

$("run").onclick = run;
$("cancel").onclick = () => { if (rc) rc.abort = true; };
const isVideoFile = (f) => f.type.startsWith("video/") || /\.(mp4|webm|mov|m4v|mkv)$/i.test(f.name);
// Trim / Crop パネル: 無視/テキスト矩形はクロップ後座標でパラメータ欄に書き戻す。実測 fps は fps 欄の既定値にする
const trim = new TrimPanel($("trim"), {
  onRects: (ig, tx) => {
    $("ignoreRects").value = ig.map((r) => r.join(",")).join("; ");
    $("textRects").value = tx.map((r) => r.join(",")).join("; ");
  },
  onFps: (fps) => { $("fps").value = String(fps); },
});
window.__trim = trim;   // デバッグ/検証用
function setFiles(list) {
  selectedFiles = [...list].filter((f) => f.type.startsWith("video/") || f.type.startsWith("image/") || /\.(mp4|webm|mov|m4v|mkv|png|jpe?g|webp|bmp)$/i.test(f.name));
  const n = selectedFiles.length;
  $("fileInfo").textContent = n === 0 ? "" : n === 1 ? selectedFiles[0].name : `${selectedFiles[0].name} 他 ${n} files`;
  if (n === 1 && isVideoFile(selectedFiles[0])) {
    $("trimBox").hidden = false;
    trim.load(selectedFiles[0], num("fps", 30)).catch((e) => log(`[trim] ${e.message || e}`));
  } else {
    trim.hide(); $("trimBox").hidden = true;
  }
}
$("file").onchange = () => setFiles($("file").files);
// ページ全体でドロップを受け付ける（ドロップ領域はハイライト表示）
const dropEl = $("drop");
for (const ev of ["dragenter", "dragover"]) document.addEventListener(ev, (e) => { e.preventDefault(); dropEl.classList.add("over"); });
document.addEventListener("dragleave", (e) => { if (!e.relatedTarget) dropEl.classList.remove("over"); });
document.addEventListener("drop", (e) => {
  e.preventDefault();
  dropEl.classList.remove("over");
  if (e.dataTransfer?.files?.length) {
    setFiles(e.dataTransfer.files);
    try { $("file").files = e.dataTransfer.files; } catch (err) { /* 一部ブラウザでは不可 */ }
  }
});
if (!navigator.gpu) log("[warn] このブラウザは WebGPU に対応していません。Chrome / Edge 113+, Firefox 141+, Safari 26+ を使用してください");
