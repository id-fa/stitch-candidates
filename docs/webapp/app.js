// app.js - UI、フレーム読み込み（動画シーク抽出 / 画像列）、出力

import { Gpu, fmtMB } from "./gpu.js";
import { Reconstructor, DEFAULTS } from "./recon.js";
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
let gpuError = null;      // WebGPU 初期化失敗の理由（Error）。ページ表示時に判定して画面に出す

// WebGPU の利用可否をページ表示時に判定し、結果を見出し直下のバナーに表示する。
// 失敗時は実行ボタンを無効化する（Chrome の「グラフィック アクセラレーション」設定が OFF、
// 非セキュアコンテキスト、非対応ブラウザなど。「実行」まで気付けないのを避ける）
function showGpuStatus(kind, text) {
  const el = $("gpuStatus");
  el.className = `banner ${kind}`;
  el.textContent = text;
  el.hidden = false;
}
async function ensureGpu() {
  if (gpu) return gpu;
  try {
    gpu = await Gpu.create(log);
    gpuError = null;
    const g = gpu;
    g.device.lost.then((info) => {
      if (gpu === g) gpu = null;   // 次の ensureGpu() で作り直す
      if (info.reason !== "destroyed") showGpuStatus("err", g.lostMessage());
    });
    showGpuStatus("ok", `WebGPU 利用可能: ${gpu.info || "(adapter info unavailable)"}`);
    return gpu;
  } catch (e) {
    gpuError = e;
    showGpuStatus("err", `WebGPU を利用できないため、この Web 版は動作しません。\n${e.message}`);
    $("run").disabled = true;
    log(`[error] ${e.message}`);
    throw e;
  }
}
let selectedFiles = [];   // ファイル入力または D&D で選ばれたファイル
let outPrefix = "";       // 保存ファイル名の接頭辞（入力ファイル名から拡張子を除いたもの）
const results = {};   // name -> {canvas, w, h}

function setProgress(stage, frac) {
  $("progress").value = frac;
  $("stage").textContent = `${stage} ${(frac * 100).toFixed(0)}%`;
}

// メモリ上限（GB）: localStorage に保存。VRAM や空きメモリを取得する Web API は無いので利用者が設定する
try { const v = localStorage.getItem("memLimitGB"); if (v !== null && v !== "") $("memLimit").value = v; } catch (e) { /* ignore */ }
$("memLimit").onchange = () => { try { localStorage.setItem("memLimitGB", $("memLimit").value); } catch (e) { /* ignore */ } };
const memLimitBytes = () => Math.max(0, num("memLimit", 8)) * 1073741824;

// GPU バッファ総量の見積もり（recon.js / render.js の確保に合わせる）。キャンバス出力（位置合わせ後に決まる）は含まない
function estimateMemory(n, W, H, args0) {
  const args = { ...DEFAULTS, ...args0 };
  const cs = args.coarseScale;
  const hc = Math.max(16, Math.round(H * cs)), wc = Math.max(16, Math.round(W * cs));
  const h16 = Math.max(16, Math.round(hc / 4)), w16 = Math.max(16, Math.round(wc / 4));
  const Hq = Math.max(1, Math.floor(H / 4)), Wq = Math.max(1, Math.floor(W / 4));
  const frames = n * W * H * 4;
  const pre = n * (wc * hc * 8 + w16 * h16 * 8 + Hq * Wq * 4);
  const work = 6 * W * H * 4 + 3 * hc * wc * 4 + Math.max(hc * wc * 16, W * H * 8) + 16 * 1048576 * 2 +
               args.levelCacheMB * 1048576 + args.stackBudgetMB * 1048576 + 2 * W * H * 16;
  return { frames, pre, work, total: frames + pre + work, perFrame: W * H * 4 + (wc * hc * 8 + w16 * h16 * 8 + Hq * Wq * 4) };
}
// フレーム数と解像度が確定した時点（抽出前）で上限と照合する
function checkMemoryPlan(n, W, H, args) {
  const est = estimateMemory(n, W, H, args);
  const lim = memLimitBytes();
  log(`[memory] 推定 GPU バッファ ${fmtMB(est.total)}（フレーム ${fmtMB(est.frames)} + 前処理 ${fmtMB(est.pre)} + 作業領域 ${fmtMB(est.work)}、` +
      `${fmtMB(est.perFrame)}/frame）、上限 ${lim > 0 ? fmtMB(lim) : "なし"}`);
  if (lim > 0 && est.total > lim) {
    const maxN = Math.max(0, Math.floor((lim - est.work) / est.perFrame));
    if (maxN < 2) throw new Error(`メモリ上限を超えるため中止しました: ${n} frames ${W}x${H} で推定 ${fmtMB(est.total)} > 上限 ${fmtMB(lim)}。` +
                                  `作業領域（level cache MB + stack budget MB + スクラッチ ${fmtMB(est.work)}）だけで上限に達します。上限（memory limit）を上げるか level cache MB を下げてください`);
    throw new Error(`メモリ上限を超えるため中止しました: ${n} frames ${W}x${H} で推定 ${fmtMB(est.total)} > 上限 ${fmtMB(lim)}。\n` +
                    `この解像度で収まるのは約 ${maxN} frames です。fps / max frames を下げる、Trim で範囲を絞る、input scale を下げる（0.5 で 1/4）、` +
                    `または上限（memory limit）を上げてください（ブラウザがクラッシュしない範囲で）`);
  }
  return est;
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
    canvasScale: cs, inlierTol: num("inlierTol", 0.06), sharpTop: num("sharpTop", 0.3), resTol: num("resTol", 1.25),
    anchorFrame: $("anchorFrame").value.trim() === "" ? -1 : int("anchorFrame", -1), anchorWindow: int("anchorWindow", 2),
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
  if (opts.onPlan) opts.onPlan(times.length, W, H);
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
      if (opts.onPlan) opts.onPlan(sel.length, W, H);
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
  $("run").disabled = true; $("cancel").disabled = false; $("probe").disabled = true;
  logEl.textContent = "";
  $("tabs").innerHTML = ""; $("view").innerHTML = "";
  for (const k of Object.keys(results)) delete results[k];
  const tAll = performance.now();
  let runGpu = null;
  try {
    const args = readArgs();
    runGpu = await ensureGpu();
    const files = selectedFiles;
    if (!files.length) throw new Error("動画ファイルまたはフレーム画像を選択（またはドロップ）してください");
    outPrefix = files[0].name.replace(/\.[^.]+$/, "") + "_";
    const opts = { fps: num("fps", 10), start: num("start", 0), duration: num("duration", 0), maxFrames: int("maxFrames", 0),
                   inputScale: num("inputScale", 1.0), every: Math.max(1, int("every", 1)),
                   onPlan: (n, W, H) => checkMemoryPlan(n, W, H, args) };
    gpu.budgetBytes = memLimitBytes();
    gpu.peakBytes = gpu.allocBytes;
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
    log(`[done] total ${((performance.now() - tAll) / 1000).toFixed(1)}s, GPU buffer peak ${fmtMB(gpu.peakBytes)}`);
  } catch (e) {
    const lost = runGpu && runGpu.lostInfo && runGpu.lostInfo.reason !== "destroyed" ? runGpu.lostMessage() : null;
    log(`[error] ${lost || e.message || e}`);
    if (lost && (e.message || String(e)) !== lost) log(`  (直接の例外: ${e.message || e})`);
    console.error(e);
  } finally {
    if (rc) {
      try { if (gpu) await gpu.done(); } catch (e) { /* ignore */ }
      if (window.__keepRc) window.__rc = rc; else rc.destroy();   // __keepRc: デバッグ用に GPU 資源を保持
      rc = null;
    }
    running = false;
    $("run").disabled = !!gpuError; $("cancel").disabled = true; $("probe").disabled = !!gpuError;
    setProgress("idle", 0);
  }
}

$("run").onclick = run;

// ------------------------------------------------------------ メモリ上限の実測
// memory limit の値まで 512MB ずつ GPU バッファを実際に確保（clearBuffer で常駐させる）して確認する。
// createBuffer の OOM は error scope で受けられるが、Dawn 内部の確保が失敗するとデバイス喪失になるので、
// その場合はデバイスを作り直す。確保できなかった場合は上限を「確保できた量の 80%」に下げて保存する
async function probeMemory() {
  if (running) return;
  running = true;
  $("run").disabled = true; $("probe").disabled = true;
  const CH = 512 * 1048576;
  const capGB = num("memLimit", 8);
  const cap = capGB > 0 ? capGB * 1073741824 : 64 * 1073741824;
  const bufs = [];
  let total = 0, failed = null, g = null;
  try {
    g = await ensureGpu();
    log(`[probe] ${(cap / 1073741824).toFixed(1)} GB まで 512MB ずつ GPU バッファを確保して確認します（動画は読み込みません）`);
    while (total + CH <= cap + 1) {
      const dev = g.device;
      dev.pushErrorScope("out-of-memory");
      const b = dev.createBuffer({ size: CH, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const enc = dev.createCommandEncoder(); enc.clearBuffer(b); dev.queue.submit([enc.finish()]);
      const err = await Promise.race([dev.popErrorScope(), dev.lost.then(() => "lost")]);
      if (err) { failed = err === "lost" ? new Error(g.lostMessage() || "device lost") : err; try { b.destroy(); } catch (e) { /* ignore */ } break; }
      await Promise.race([dev.queue.onSubmittedWorkDone(), dev.lost.then(() => "lost")]);
      if (g.lostInfo) { failed = new Error(g.lostMessage()); break; }
      bufs.push(b); total += CH;
      setProgress("probe", total / cap);
      if (bufs.length % 4 === 0) log(`[probe] ${fmtMB(total)} OK`);
    }
  } catch (e) {
    failed = e;
  }
  for (const b of bufs) { try { b.destroy(); } catch (e) { /* ignore */ } }
  if (g && g.lostInfo) {
    // 喪失したデバイスを作り直す（gpu は lost ハンドラで null になっている）
    try { await ensureGpu(); } catch (e) { /* バナーに出ている */ }
  }
  if (failed) {
    const rec = Math.max(0.5, Math.floor(total * 0.8 / (512 * 1048576)) * 0.5);
    log(`[probe] ${fmtMB(total)} まで確保できましたが、次の 512MB で失敗しました: ${failed.message || failed}`);
    log(`[probe] memory limit を ${rec} GB に設定しました（確保できた量の 80%）。実行時はブラウザ本体や動画デコードの分も要るので、これでも失敗するならさらに下げてください`);
    $("memLimit").value = String(rec);
    $("memLimit").dispatchEvent(new Event("change"));
  } else {
    log(`[probe] ${fmtMB(total)} まで確保できました。memory limit ${capGB > 0 ? capGB + " GB" : "なし"} のままで問題ありません`);
  }
  running = false;
  $("run").disabled = !!gpuError; $("probe").disabled = !!gpuError;
  setProgress("idle", 0);
}
$("probe").onclick = probeMemory;
$("cancel").onclick = () => { if (rc) rc.abort = true; };
const isVideoFile = (f) => f.type.startsWith("video/") || /\.(mp4|webm|mov|m4v|mkv)$/i.test(f.name);
// Trim / Crop パネル: 無視/テキスト矩形はクロップ後座標でパラメータ欄に書き戻す。実測 fps は fps 欄の既定値にする
const trim = new TrimPanel($("trim"), {
  onRects: (ig, tx) => {
    $("ignoreRects").value = ig.map((r) => r.join(",")).join("; ");
    $("textRects").value = tx.map((r) => r.join(",")).join("; ");
  },
  onFps: (fps) => { $("fps").value = String(fps); },
  // アンカー: 動画のフレーム番号 → 抽出後の番号（開始フレーム基準、抽出 fps / 実 fps で換算）
  onAnchor: (frame, t) => {
    const ext = num("fps", t.fps) || t.fps;
    const idx = Math.max(0, Math.round((frame - t.start) * ext / t.fps));
    $("anchorFrame").value = String(idx);
    log(`[trim] anchor frame = ${idx} (video frame ${frame})`);
  },
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
// ページ表示時に WebGPU を初期化して可否を表示（失敗しても例外は握りつぶす。バナーとログに出ている）
ensureGpu().catch(() => {});
