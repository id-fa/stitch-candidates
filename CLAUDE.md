# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Image stitching tools for combining scroll/pan screenshots and reconstructing backgrounds from video. Generates multiple "nearly-correct" candidates for human visual selection rather than attempting fully automatic perfect stitching.

**Design philosophy**: Human-in-the-loop - generate many candidates, let humans choose the best one, then refine.

### Tools

1. **stitch_candidates.py** - Static image stitching (screenshots)
2. **video_strip_reconstruct.py** - Video background reconstruction with text removal

## Main Script

`stitch_candidates.py` - Unified version combining all features:
- Multiple matching methods (phase, ncc_gray, ncc_edge, ssim)
- Vertical/horizontal/snake modes
- Ignore regions for excluding UI elements
- constantDelta variants
- refine-from mode for local re-search
- Overlap scan mode for auto-finding best overlap
- Score-based pruning (matching score + boundary similarity)

## Dependencies

```bash
pip install pillow numpy
pip install opencv-python  # optional, improves performance
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

## Video Background Reconstruction

`video_strip_reconstruct.py` - Reconstructs backgrounds from vertical scrolling videos while removing text overlays.

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
