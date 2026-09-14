# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image stitching tools for combining scroll/pan screenshots and reconstructing backgrounds from video. Generates multiple "nearly-correct" candidates for human visual selection rather than attempting fully automatic perfect stitching.

**Design philosophy**: Human-in-the-loop - generate many candidates, let humans choose the best one, then refine.

### Tools

1. **stitch_candidates.py** - Static image stitching (screenshots)
2. **video_strip_reconstruct.py** - Video background reconstruction with text removal (legacy, sequential model)
3. **panorama_recon.py** - Video background reconstruction v2: global alignment + temporal median (recommended for video)
4. **gui.py** - tkinter GUI for all tools above

## Main Script

`stitch_candidates.py` - Unified version combining all features:
- Multiple matching methods (phase, ncc_gray, ncc_edge, ssim)
- Vertical/horizontal/snake modes
- Ignore regions for excluding UI elements
- constantDelta variants
- refine-from mode for local re-search
- Overlap scan mode for auto-finding best overlap
- Score-based pruning (matching score + boundary similarity)

## GUI

`gui.py` - tkinter-based GUI wrapping stitch_candidates.py, video_strip_reconstruct.py and panorama_recon.py.

```bash
python gui.py
```

### Structure

- 4 tabs: **Stitch Candidates**, **Video Reconstruct**, **Panorama Reconstruct**, **Output Browser**
- Bottom panel: log output (left) + image preview (right)
- Calls `stitch_candidates.main(argv)` / `video_strip_reconstruct.main(argv)` / `panorama_recon.main(argv)` via import (not subprocess)
- stdout is redirected to the log panel via `_StdoutRedirector` during execution
- `SystemExit` from `sys.exit()` is caught and displayed as error
- Run completes → Output Browser auto-loads output directory

### Key design notes

- Both scripts expose `main(argv: Optional[List[str]] = None)` — when `argv` is `None`, they read `sys.argv` (CLI compatible); when a list is passed, they parse that instead
- GUI builds an argv list from widget state and passes it to `main(argv)`
- Execution runs in a daemon thread to keep the UI responsive
- All Browse dialogs use `initialdir` from the current path value
- Video match method uses checkbox selection (multiple methods possible), not free text
- Run buttons are `RunButton` (a coloured `tk.Button`; ttk ignores colours on the Windows theme). It greys itself when `state=DISABLED`
- Drag & drop: `App` derives from `TkinterDnD.Tk` when `tkinterdnd2` is importable (plain `tk.Tk` otherwise). `enable_drop(widget, cb)` registers a drop target; the Video Reconstruct and Panorama tabs accept a video file (→ Video) or a folder / image file (→ Frames glob) on the Input section. PyInstaller picks up tkdnd via the hooks-contrib `hook-tkinterdnd2`

## Dependencies

```bash
pip install pillow numpy
pip install opencv-python  # optional, improves performance (required for --video in panorama_recon.py)
pip install torch          # required only by panorama_recon.py (CUDA build recommended)
pip install tkinterdnd2    # optional: drag & drop of video files / frame folders onto the GUI
```

## Running

```bash
# Vertical stitching
python stitch_candidates.py -m v -o out --overlap 80,120 img/*.png

# Horizontal stitching
python stitch_candidates.py -m h -o out img/*.png

# Snake/zigzag stitching
python stitch_candidates.py -m snake --cols 4 -o out img/*.png

# Overlap scan mode (auto-find best overlap)
python stitch_candidates.py -m v -o out --overlap-scan 50,150,5 --top-n 5 img/*.png

# Refine-from mode
python stitch_candidates.py -m v -o refine_out \
  --refine-from out/v_ov120__phase__p1_dx0_dy-3.png \
  --refine-delta 2 img/*.png
```

Key parameters:
- `--overlap` - Overlap pixels (comma list, e.g., "80,120")
- `--overlap-pct` - Overlap as ratio 0.01-0.95 (e.g., "0.1,0.15,0.2")
- `--overlap-scan MIN,MAX,STEP` - Scan overlap range, output top N by score
- `--overlap-auto` - 3-stage hierarchical search (step 100 → 10 → 1)
- `--top-n` - Number of top candidates in scan/auto mode (default: 5)
- `--band` - Band size for matching (comma list)
- `--search` - Search range in pixels (comma list)
- `--ignore` / `--ignore-pct` - Ignore regions (px or %)
- `--min-overlap-ratio` - Prune by effective overlap ratio (default: 0.3)
- `--min-boundary-score` - Prune by boundary similarity SSIM (default: 0.3)
- `--exclude-method` - Exclude methods (e.g., "ssim" to speed up)
- `--refine-from` - Candidate file for local re-search
- `--refine-delta` - Search range ±n px in refine mode

## Workflow

1. **Rough search**: Wide parameter ranges (`--overlap 80,120 --band 30,40 --search 10,20`)
2. **Visual inspection**: Find nearly-correct candidate (filename contains dx/dy)
3. **Refinement**: Use `--refine-from` with small `--refine-delta`

Alternative: Use `--overlap-auto` or `--overlap-scan 50,150,5 --top-n 5` to auto-find best overlap

## Architecture

Sequential stitching model: `(1+2) + 3 + 4 + ...`

Matching methods (with score thresholds):
- `phase` - Phase correlation (FFT-based, fast), min_score=0.05
- `ncc_gray` - NCC on grayscale, min_score=0.10
- `ncc_edge` - NCC on Sobel edge map, min_score=0.05
- `ssim` - SSIM-based matching (slow but accurate), min_score=0.30

Safety features:
- Overlap auto-clipped to safe maximum
- Invalid compositions skipped
- Effective overlap ratio pruning (default: 0.3)
- Matching score pruning (low-score matches skipped)
- Boundary similarity pruning (optional, via --min-boundary-score)

Scoring (scan mode):
- Combined score = harmonic mean of matching score and boundary similarity (SSIM)
- Outputs ranked by combined score
- Shows diagnostic info when no valid candidates found

---

## Panorama Reconstruction v2 (panorama_recon.py)

`panorama_recon.py` - Recommended tool for video. Replaces the sequential "candidate generation" model with a
single deterministic answer. Motion model per frame (`--model`): `translation`, `scale` (translation + zoom,
default), `similarity` (+ rotation). Frame k pixel x maps to canvas X = S[k] R(TH[k]) x + T[k].

```bash
# Typical (all frames, GPU auto-detected, zoom handled automatically)
python panorama_recon.py --video input.mp4 --out pano_out

# Subsample long videos
python panorama_recon.py --video input.mp4 --fps 10 --start 5 --duration 20 --out pano_out

# From extracted frames, CPU only (halve the Gauss-Newton resolution for speed)
python panorama_recon.py --frames "frames/*.png" --device cpu --fine-scale 0.5 --out pano_out

# Pure pan, fastest path (FFT window search instead of Gauss-Newton)
python panorama_recon.py --video input.mp4 --model translation --out pano_out
```

### Pipeline

1. **Static overlay detection** (`--static-span/--static-diff/--static-grad/--static-dilate`):
   pixels with high gradient whose value does not change between frame i and i±span are
   screen-fixed overlays (credits, logos). They are excluded from matching and compositing.
   Check `debug_overlay_mask.png` (red = masked). If mask ratio > 50% the scroll is too slow: raise `--static-span`.
2. **Coarse alignment**: masked NCC (FFT, Padfield) over the full shift range at `--coarse-scale` (default 0.25)
   for every pair (i, i+k), k in `--pairs` (default 1,2,4). With `scale`/`similarity` models the NCC is evaluated
   on a log-scale grid (`--scale-max` 0.06, `--scale-step` 0.004) for k=1 pairs; larger k pairs search only ±3
   steps around the chain prediction.
3. **Fine alignment**:
   - `translation`: full-resolution masked NCC in a small window around the coarse estimate, parabolic subpixel.
   - `scale`/`similarity`: Gauss-Newton direct alignment on high-passed images (pyramid 0.25 → 0.5 → `--fine-scale`),
     masked, Huber-weighted, `--gn-iters` per level. Pairs inconsistent with the coarse global solution are
     re-initialised from the predicted transform. A second pass re-refines pairs with residual > 1.5 px.
4. **Global solve**: log-scale, angle and translation are solved jointly by IRLS least squares (Cauchy weights),
   so errors do not accumulate. Residuals > 1.5 px are reported in the log and `pairs.csv`. Frame 0 is the reference
   (scale 1, angle 0, position 0).
5. **Render** (row bands of `--band` rows, GPU): every frame is warped (similarity, bilinear; Gaussian prefilter
   when downscaled) to the canvas. `--canvas-scale auto` makes the most zoomed-in frame 1:1 (max detail; a
   zoom-out video therefore yields a canvas larger than the frame); a number sets the scale relative to frame 0.
   Then per pixel:
   - `recon_median.png` - temporal median of non-overlay samples (most robust, removes text/sparkles)
   - `recon_mean.png` - mean of inliers within `--inlier-tol` of the median (least noise)
   - `recon_sharp.png` - mean of the top `--sharp-top` fraction of inliers by local sharpness (crispest)
   - `coverage.png` - per-pixel count of clean samples (dark = few samples, check edges here)

### Outputs

- `recon_median.png`, `recon_mean.png`, `recon_sharp.png`, `coverage.png`
- `positions.csv` - per-frame scale, theta_deg, x, y (frame 0 = reference), subpixel
- `pairs.csv` - per-pair scale, theta_deg, tx, ty, NCC score, residual after global solve
- `debug_overlay_mask.png` - first/middle/last frame with overlay mask in red

### Notes

- Timings on an RTX 3080 Ti, 1080p 30 fps: sample.mp4 (135 frames, pan) ~55 s with `scale`, ~25 s with
  `translation`; sample3_zoom.mp4 (86 frames, pan + 1.9x zoom-out) ~48 s; sample2_jigzag.mp4 (133 frames) ~51 s.
- Direction (vertical/horizontal) is not a parameter: 2D shift is estimated directly.
- Character animation inside the shot (blinks, body motion) makes pairs across the change score ~0.6-0.7 and leaves
  residuals of a few px on those pairs; the median rendering absorbs it. Duplicate frames (24→30 fps pulldown) are
  harmless (pair shift ≈ 0).
- Frame-rate: keep the native 30 fps for zoom videos; at 6 fps the per-pair scale change can exceed `--scale-max`
  and frames lose overlap.
- `--ignore-rect x,y,w,h` adds always-excluded regions (semicolon-separated in the GUI).
- Frame range / crop (added 2026-09-13): `--start-frame N --end-frame M` (0-based, inclusive; override `--start`/`--duration`)
  and `--crop x,y,w,h` (source coordinates, applied at load; `--ignore-rect`/`--text-rect` are then in cropped coordinates).
  The GUI's Panorama tab has a "Trim / Crop..." button opening `trim_dialog.py`: start/end previews, a range bar with two
  handles, -1/+1 buttons, and drag-to-draw rectangles (crop / ignore / text) whose results fill the tab's fields. Rectangles are
  editable afterwards in both the dialog and the Web panel: press inside to select + move, handles resize the selected one,
  Ctrl+drag forces a new rectangle, Delete removes the selection.
  Thumbnails are decoded in a background thread (OpenCV), capped at 1500 (strided beyond that; other frames are seeked on demand).
- Telop / ticker residue (added 2026-09-13, same as the Web version):
  - `--text-rect x,y,w,h` (repeatable): bands with *moving* text (tickers). Inside, high-gradient pixels are masked per frame
    without the static test, grown by `--text-halo` px (default 4). Required for moving telops; the static detector cannot see them.
  - `--static-halo N` (default 12): static pixels within N px of a static edge are masked too. Catches the flat white glow
    around telop text, which has low gradient and otherwise leaks into the median. Side effect: more mask around static edges.
  - `--static-close N` (default 3): a pixel masked in both frames k-j and k+j (j=1..N) is masked in frame k. Fixes frames where a
    shine animation sweeping over the text breaks the static test for a moment.
  - `--hole-fill inpaint|blur|none` (default inpaint) with `--min-clean 12 --min-clean-pct 25`: canvas pixels that are covered
    geometrically but have no clean sample (telop always on top of them, typically at the canvas ends), or fewer than 12 clean
    samples that are also under 25% of the covering frames, are filled by push-pull interpolation (or blurred). Filled areas lose
    their real detail (a dense ticker band becomes a smooth strip). The log prints `fallback (no clean sample) px` and the fill count.
  - Do not widen the static test to k±2span "any match": background coincidences inflate the mask (14% → 41% measured).
- Moving characters / thin lines (added 2026-09-13): in-shot character motion (hair, shoulders; sample3_zoom.mp4 moves ~12 px
  between frames 2 and 3) makes the majority median wipe thin lines of the moving layer. `--anchor-frame K --anchor-window N`
  (GUI: Anchor frame / window; Web: anchor frame field, "Anchor = start/end" buttons in the Trim panel) composites pixels covered by
  frames K±N from those frames only, fixing the pose there (a seam can appear at the anchor coverage boundary). `--res-tol 1.25`
  (0 = off) restricts each pixel's samples to magnifications within the factor of the most detailed sample (zoom videos); it did
  not change sample3_zoom (closeup area is only covered by closeup frames) but is kept as a safeguard. Static detection can also
  mask still thin lines near the zoom centre: disable it or raise `--static-span` when there is no telop.
- If a static logo sits at a canvas edge covered only by frames where it is masked, the render falls back to the
  unmasked median there (logo remains). `coverage.png` shows such regions as dark.

---

## Panorama Reconstruction - WebGPU 版 (docs/webapp/)

`docs/webapp/` は `panorama_recon.py` のブラウザ移植。torch / CUDA 不要で、WebGPU 対応ブラウザ（Chrome / Edge 113+,
Firefox 141+, Safari 26+）と任意の GPU（NVIDIA / AMD / Intel）で動く。配布物は静的ファイルのみ。詳細は `docs/webapp/README.md`。

```bash
cd web && python -m http.server 8765   # ES モジュールのため file:// では動かない
# → http://127.0.0.1:8765/index.html
```

- 粗探索は FFT ではなく階層的総当たり NCC（1/16 解像度で全シフト → 1/4 解像度で ±12 px 窓）。スケール格子も
  1/16 で 0.02 刻み → 1/4 で `scale_step` 刻みの 2 段階。それ以外（静止オーバーレイ検出、Gauss-Newton、
  IRLS グローバル解、中央値 / インライア平均 / 鮮明度上位平均の合成）は Python 版と同じ手順を WGSL / JS で実装
- Gauss-Newton は全反復・全レベルを 1 コマンドバッファに詰め、4x4 正規方程式の求解まで GPU 上で行う（読み戻しはペア
  バッチごと 1 回）。MAD はヒストグラム近似
- 結果は Python 版とほぼ一致（sample.mp4: 残差中央値 0.02 px、スケール 1.0000-1.0008。sample3_zoom.mp4: 最大スケール
  1.9205 対 1.9206）。RTX 3080 Ti で sample.mp4 136 フレームが 30 s（Python + CUDA は 53 s）
- フレームは RGBA8 で GPU に置く（1080p で 8 MB / フレーム）。memory limit に収まれば全フレーム常駐、超えるならストリーミング
  （`frames.js` の `FrameStore` が LRU で必要なフレームだけ常駐させ、動画から読み直す。2026-09-14 追加）。結果は常駐時と同一
- 動画はブラウザの `<video>` シークで抽出するため、フレーム集合が Python 版と 1 フレームずれることがある
- テロップ対策（2026-09-13 追加、Python 版にも同じオプションあり）: `text rects`（動くティッカー帯を勾配でマスク）、`halo`（静止エッジ周辺の
  静止画素＝文字のグローもマスク）、`close`（前後フレームでマスクされていれば埋める＝光沢アニメ対策）、`hole fill`
  （クリーン標本が無い/少ない画素を push-pull 補間で埋める）。副作用（埋めた領域の細部消失、マスク増加）はオプションで制御
- 注意: WGSL で `a || b` をワークグループバリアの前に置くと誤コンパイルされる環境があり、`gn_accum` では `max()` で判定
- Trim / Crop パネル（`trim.js`、2026-09-13）: 動画選択時に開く。`<video>` 2 本で開始/終了フレームを表示、範囲バー + -1/+1、
  fps は requestVideoFrameCallback で実測、クロップ/無視/テキスト矩形をプレビュー上でドラッグ指定（無視/テキストはクロップ後座標で
  パラメータ欄へ書き戻す）。抽出は `drawImage` の元矩形指定でクロップ
- WebGPU の可否はページ表示時に `Gpu.create` を先行実行して判定し（`app.js` の `ensureGpu`）、`#gpuStatus` バナーに表示。
  失敗時は「実行」を無効化。`gpu.js` の例外メッセージが原因別（非対応 / 非セキュアコンテキスト / アダプタ取得不可 =
  グラフィックアクセラレーション OFF / デバイス作成失敗）の対処を含む（2026-09-14 追加）
- メモリ上限（2026-09-14 追加）: 全フレーム GPU 常駐のため 1080p で約 9 MB/frame。メインメモリ不足でブラウザがクラッシュするのを避けるため、
  `memory limit (GB)` 欄（既定 8、localStorage 保存）と照合する。`app.js` の `estimateMemory` / `checkMemoryPlan` が抽出前に総量を見積もって
  中止し、`Gpu.reserve` / `Gpu.buf` が確保時の安全網（`budgetBytes`）、`uncapturederror` の `GPUOutOfMemoryError` は `checkOom` で次の
  同期点に例外化。空きメモリ / VRAM を取得する Web API は無いので自動判定はしないが、「上限を確認」ボタン（`probeMemory`）で設定値まで
  512MB ずつ実確保して検証できる（失敗時は確保量の 80% に下げて保存）。Dawn 内部の確保で D3D12 が OOM を返すとデバイス喪失になる
  （オンボード GPU の共有メモリ予算はメインメモリよりずっと小さい。目安 4〜6 GB）。喪失は `Gpu.lostInfo` に記録し、`checkOom` と
  `run()` の catch で「GPU メモリ不足でデバイスが失われました」に差し替え、次の `ensureGpu()` で作り直す
- ストリーミング（2026-09-14 追加）: `app.js` の `planMemory` が常駐 / ストリーミングを決め、`FrameStore` の常駐上限を渡す。
  `recon.js` の前処理はフレーム順パイプライン（stage1 静止検出 → closeK → finalizeK。複数フレームを集めてカーネルを積む間は pin）、
  マスクは `packMask` で 1 bit/画素に保持し再読込時に `unpack_mask` で alpha へ復元。精密ペアは `_inFrameOrder` で (i, j) 順、
  `freeCoarse` で C4/C16 とレベルキャッシュを解放。`render.js` は行帯 / 列帯（`horiz` = キャンバスが横に伸びているか）で
  タイル化し、キャッシュが帯の被覆数より少なければ等間隔に間引く（`stride`）。旧版と出力がバイト一致することを確認済み。
  注意: FrameStore の破棄は `gpu.submit()` 後に行う（未送信コマンドの参照を壊さない）。複数フレームを `get()` で集めてから
  カーネルを積む箇所は必ず pin する（集めている間の破棄でバッファが無効になる）
- ファイル: `index.html` (UI), `app.js` (入出力), `frames.js` (フレーム供給 / GPU キャッシュ), `trim.js` (トリム/クロップ), `gpu.js` (基盤),
  `shaders_img.js` / `shaders_align.js` / `shaders_render.js` (WGSL), `recon.js` (位置合わせ), `render.js` (合成), `postfx.js` (穴埋め),
  `solve.js` (CPU 最小二乗)

---

## Video Background Reconstruction (legacy)

`video_strip_reconstruct.py` - Reconstructs backgrounds from vertical scrolling videos while removing text overlays.
Superseded by `panorama_recon.py` for most cases; kept for the strip/keyframe/template workflows below.

### Running

```bash
# From video (with ffmpeg)
python video_strip_reconstruct.py --video input.mp4 --fps 2 \
  --strip-y 980 --strip-h 100 --out outdir

# From extracted frames
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 --out outdir

# Multiple edge thresholds for candidate generation
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 --edge-thr 0.3,0.4,0.5 --out outdir

# Multiple matching methods
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 --match-method phase,ncc_gray --out outdir
```

### Key Parameters

- `--strip-y` - Top Y of horizontal strip (for vertical scroll)
- `--strip-h` - Height of strip (default: 100)
- `--strip-x` - Left X of vertical strip (for horizontal pan)
- `--strip-w` - Width of strip (default: 100)
- `--scroll-axis` - Scroll axis: vertical (default) or horizontal
- `--edge-thr` - Edge threshold(s) for text masking (comma-separated)
- `--match-method` - Matching method: phase, ncc_gray, ncc_edge, phase_gray, template, optical_flow, optical_flow_fb, optical_flow_lk
- `--min-peak` - Peak score threshold for diagnostics
- `--ignore` / `--ignore-pct` - Ignore regions (same format as stitch_candidates)
- `--scroll-dir` - Scroll direction: up, down, left, right, both (default: both)
- `--static-bg` - Static background mode (use temporal median to remove scrolling text)
- `--static-method` - Method for static-bg: median (default) or min_edge
- `--dy-region "y,h"` - Region for dy estimation (e.g., "0,200" for top 200px)
- `--uniform-dy` - Use uniform dy per frame (e.g., -50 for 50px up per frame)
- `--template-region "y,h,x,w"` - Manual template region for template matching
- `--suggest-templates` - Suggest best template regions by testing all frames
- `--suggest-n` - Number of candidates to evaluate (default: 15)
- `--jigsaw` - Jigsaw mode: intelligent reconstruction using text-free frames and clean bands (default: enabled)
- `--no-jigsaw` - Disable jigsaw mode, use basic reconstruction
- `--base-frame` - Base frame index for jigsaw mode (-1=last, -2=second-to-last). Base frame is used as foundation, other frames fill gaps.
- `--auto-exclude-text` - Auto-detect and exclude text regions from dy estimation (per-frame detection)
- `--keyframe-mode` - Keyframe selection mode: select high-quality frames and stitch with hard cuts (default: enabled)
- `--no-keyframe-mode` - Disable keyframe mode, use traditional jigsaw blending
- `--min-quality` - Minimum quality score for keyframe selection (default: 0.3)
- `--max-keyframe-gap` - Maximum frame gap between keyframes (default: 5)
- `--allow-ignore-at-edges` - Allow ignore regions at edge keyframes (first/last positioned frames preserve full content)
- `--keyframe-stitch-method` - Stitching method: `matching` (use stitch_candidates.py logic, default, outputs top 3 candidates), `position` (use estimated dy/dx)

### Output Files

- `recon_dy.png` - Result using dy as-is
- `recon_negdy.png` - Result using negated dy
- `recon_dy_e0.XX.png` - Result with specific edge-thr
- `recon_static_median.png` - Result from static-bg mode
- `recon_jigsaw_dy.png` / `recon_jigsaw_negdy.png` - Jigsaw mode results (vertical)
- `recon_jigsaw_dy_baseN.png` - Jigsaw mode with base frame N (--base-frame)
- `recon_horizontal_dx.png` / `recon_horizontal_negdx.png` - Horizontal pan mode results
- `recon_horizontal_dx_baseN.png` - Horizontal pan with base frame N
- `recon_keyframe_dy.png` / `recon_keyframe_negdy.png` - Keyframe mode results (vertical, position method)
- `recon_keyframe_dx.png` / `recon_keyframe_negdx.png` - Keyframe mode results (horizontal, position method)
- `recon_keyframe_dy_phase_b30_s10.png` - Keyframe mode with matching method (multiple candidates)
- `frame_classification.png` - Frame text classification visualization (jigsaw mode)
- `debug_positions.csv` - Per-frame dy estimation with peak scores
- `debug_jigsaw.csv` - Jigsaw mode debug info (text scores, clean ratios)
- `debug_keyframe_quality.csv` - Keyframe mode quality info (quality scores, sharpness, text, compression, noise)

### Architecture

- Extracts horizontal strip from each frame
- Estimates dy (vertical shift) using various methods
- Outputs both dy and -dy candidates (or single direction with --scroll-dir)
- Text removal: pixels with high edge strength are masked, background-like pixels prioritized

### Matching Methods

- `phase` (default) - Phase correlation on edge map
- `phase_gray` - Phase correlation on grayscale
- `ncc_gray` - NCC on grayscale (slower but robust)
- `ncc_edge` - NCC on edge map
- `template` - Template matching with automatic tracking (best for scrolling backgrounds)
- `robust` - Hybrid template + phase (recommended for jigsaw mode)
  - Uses template matching when confidence > 0.8
  - Falls back to phase correlation otherwise
  - Automatic outlier detection and correction
- `optical_flow` - Optical flow based estimation (OpenCV required)
  - Combines Farneback (dense), Lucas-Kanade (sparse), and phase correlation
  - Uses phase correlation for initial estimate (handles large shifts)
  - MAD-based outlier filtering + histogram mode for robust estimation
  - Best for complex camera movements (pan + tilt)
- `optical_flow_fb` - Farneback method only (dense optical flow)
  - Uses phase correlation as initial flow for large shift handling
- `optical_flow_lk` - Lucas-Kanade method only (sparse optical flow)
  - Bi-directional validation for accuracy

### Template Suggestion Mode

Find best `--template-region` candidates by testing tracking stability across all frames:

```bash
# Suggest template regions
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --suggest-templates --out outdir
```

Output:
- `template_candidates.png` - Visualization with regions highlighted (green=stable, yellow=moderate, red=unstable)
- `template_candidates.csv` - Detailed scores for each candidate
- Terminal output with ranked list and copy-paste ready `--template-region` values

### Jigsaw Mode

Intelligent reconstruction that analyzes each frame for text content and combines clean regions like a jigsaw puzzle:

```bash
# Jigsaw mode with robust matching (recommended)
python video_strip_reconstruct.py --video input.mp4 --fps 5 \
  --strip-y 0 --strip-h 1080 --jigsaw --match-method robust \
  --edge-thr 0.5 --out outdir

# From extracted frames
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --jigsaw --out outdir

# Base frame mode: use last frame as foundation
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --jigsaw --base-frame -1 --out outdir

# Use second-to-last frame (if last frame has scene cut issues)
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --jigsaw --base-frame -2 --out outdir

# Auto-exclude text regions from dy estimation
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --jigsaw --auto-exclude-text --out outdir

# Horizontal pan mode (use full frame width as strip)
python video_strip_reconstruct.py --video input.mp4 --fps 3 \
  --scroll-axis horizontal --strip-x 0 --strip-w 1920 --jigsaw --out outdir

# Horizontal pan with base frame
python video_strip_reconstruct.py --video input.mp4 --fps 3 \
  --scroll-axis horizontal --strip-x 0 --strip-w 1920 --jigsaw --base-frame -1 --out outdir
```

**How it works:**
1. **Frame Classification**: Each frame is scored by text density (edge strength)
   - Text-free (green): score < 0.05
   - Text-light (yellow): score 0.05-0.15
   - Text-heavy (red): score > 0.15
2. **Clean Band Detection**: Find horizontal gaps between text lines in each frame
3. **Smart dy Estimation**:
   - `robust` method combines template + phase correlation
   - Automatic outlier detection (wrong sign, too large, low peak)
   - dy smoothing (moving average) reduces jitter
4. **Weighted Blending**:
   - Each pixel weighted by (1 - text_score) × (1 - edge_strength)
   - Clean bands get 2x weight boost
   - Smooth transitions instead of hard edges

**Base Frame Mode (`--base-frame`):**
- Specified frame (e.g., -1=last, -2=second-to-last) is used as the foundation
- Base frame pixels are placed first with maximum priority
- Other frames only fill gaps (regions not covered by base frame)
- Gap-filling frames are sorted by text_score (cleanest first)
- Useful when the last frame covers most of the background area

**Auto-Exclude Text (`--auto-exclude-text`):**
- Detects text regions per-frame using edge strength (dilated mask)
- Excludes detected text regions from dy estimation
- Useful when text overlays move or change between frames
- Works with `robust` and default (phase) methods

**Output:**
- `recon_jigsaw_dy.png` / `recon_jigsaw_negdy.png` - Jigsaw-reconstructed images
- `frame_classification.png` - Visualization showing frame text scores
- `debug_jigsaw.csv` - Per-frame text scores, clean ratios, and positions

**Best for:**
- Videos with intermittent text overlays (some frames are clean)
- Scrolling subtitles/captions that pass through the frame
- Any case where text appears/disappears across frames
- Anime ending credits with repetitive patterns (use `--match-method robust`)

Workflow:
1. Run `--suggest-templates` to find stable regions
2. Copy the recommended `--template-region` value
3. Run with `--match-method template --template-region "y,h,x,w"`

### Keyframe Mode

Keyframe mode selects high-quality frames and stitches them with hard cuts (no blending). This avoids motion blur artifacts caused by blending overlapping regions with slight position errors.

```bash
# Keyframe mode for vertical scrolling (auto full frame height)
python video_strip_reconstruct.py --video input.mp4 --fps 4 \
  --jigsaw --keyframe-mode --out outdir

# Keyframe mode for horizontal panning (auto full frame width)
python video_strip_reconstruct.py --video input.mp4 --fps 10 \
  --scroll-axis horizontal --jigsaw --keyframe-mode --match-method optical_flow --out outdir

# With ignore regions - allow at edges to preserve panorama edges
python video_strip_reconstruct.py --video input.mp4 --fps 10 \
  --scroll-axis horizontal --jigsaw --keyframe-mode \
  --ignore "0,right,200,100" --allow-ignore-at-edges --out outdir

# Force intermediate keyframes if needed (e.g., every 10 frames max)
python video_strip_reconstruct.py --video input.mp4 --fps 10 \
  --scroll-axis horizontal --jigsaw --keyframe-mode --max-keyframe-gap 10 --out outdir
```

**How it works:**
1. **Quality Scoring**: Each frame is scored based on:
   - Sharpness (Laplacian variance) - 40%
   - Text density (inverse) - 30%
   - Compression artifacts (inverse) - 20%
   - Noise level (inverse) - 10%
2. **Keyframe Selection**: Greedy algorithm selects frames that:
   - Meet minimum quality threshold (--min-quality)
   - Provide coverage with minimum overlap (20 pixels)
   - Select fewest frames needed for full coverage
   - With `--max-keyframe-gap N`, forces at least every N-th frame to be a keyframe
3. **Hard Stitch**: Keyframes are stitched at overlap midpoints with no blending
4. **Auto Full Frame**: Keyframe mode automatically uses full frame dimensions (no `--strip-h` or `--strip-w` needed)

**Ignore Region Handling:**
- By default, ignore regions are applied to all keyframes (content blacked out)
- With `--allow-ignore-at-edges`, the first and last positioned keyframes preserve ignore region content
- Useful when panorama edges can only be covered by edge keyframes (e.g., logo at corner)

**Output:**
- `recon_keyframe_dy.png` / `recon_keyframe_negdy.png` - Keyframe-stitched images (vertical)
- `recon_keyframe_dx.png` / `recon_keyframe_negdx.png` - Keyframe-stitched images (horizontal)
- `debug_keyframe_quality.csv` - Per-frame quality scores and selection status

**Best for:**
- Videos with compression artifacts (noisy intermediate frames)
- Situations where weighted blending causes motion blur
- Cases where you want the sharpest possible output
- High fps videos where many frames are redundant
- Horizontal panning where 2 frames can cover the entire panorama

### Scenarios and Recommended Approaches

**Scenario A: Background scrolls, text is static (appears/disappears)**
- Use `--match-method template` for accurate dy estimation
- Template tracking follows distinctive features across frames
- May need `--template-region` for manual template specification
- Outlier detection and interpolation handle tracking loss

**Scenario B: Background is static, text scrolls**
- Use `--static-bg --static-method median` for temporal median filtering
- Works when text passes quickly and background is visible in some frames

**Scenario C: Both background and text scroll together**
- Original use case, `--match-method phase` works well
- Use `--ignore` to exclude text regions from dy estimation

**Scenario D: Horizontal pan (camera moves left/right)**
- Use `--scroll-axis horizontal` with `--strip-w <frame_width>` (e.g., `--strip-w 1920`)
- **Important**: Must use full frame width for proper panorama reconstruction
- Estimates dx (horizontal shift) instead of dy
- With keyframe mode, use `--max-keyframe-gap` to include intermediate frames
- Example: `--jigsaw --keyframe-mode --max-keyframe-gap 10`

### Known Limitations

- Phase correlation may fail with repetitive patterns (e.g., floral decorations)
- Template tracking loses target when feature exits frame (interpolation helps)
- **Higher fps does NOT always improve accuracy** - can cause more tracking errors
- Accumulated small errors in dy/dx estimation can cause visible artifacts
- Complex camera movements (pan + tilt combined) are not supported

### FPS vs Accuracy Trade-off

| fps | Pros | Cons |
|-----|------|------|
| Low (2-4) | Larger dy per frame, easier to track | Fewer frames, interpolation needed |
| High (10+) | More frames | Small dy causes tracking errors, false matches |

**Recommendation:** Start with fps=2-4, check results, adjust as needed

### Workflow for Difficult Cases

1. Start with `--match-method template` for initial dy estimation
2. Check `debug_positions.csv` for outliers or tracking loss (score drops)
3. If needed, manually adjust dy values in Python script
4. Or use `--uniform-dy` if scroll speed is constant

### Future Work

- Line-by-line extraction for high-fps videos
- Automatic text/background separation using temporal analysis
- Multi-pass refinement using keyframe positions as anchors
