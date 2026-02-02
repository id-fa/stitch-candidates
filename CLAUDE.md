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

- `--strip-y` - Top Y of horizontal strip (required)
- `--strip-h` - Height of strip (default: 100)
- `--edge-thr` - Edge threshold(s) for text masking (comma-separated)
- `--match-method` - Matching method: phase, ncc_gray, ncc_edge, phase_gray, template
- `--min-peak` - Peak score threshold for diagnostics
- `--ignore` / `--ignore-pct` - Ignore regions (same format as stitch_candidates)
- `--scroll-dir` - Scroll direction: up, down, both (default: both)
- `--static-bg` - Static background mode (use temporal median to remove scrolling text)
- `--static-method` - Method for static-bg: median (default) or min_edge
- `--dy-region "y,h"` - Region for dy estimation (e.g., "0,200" for top 200px)
- `--uniform-dy` - Use uniform dy per frame (e.g., -50 for 50px up per frame)
- `--template-region "y,h,x,w"` - Manual template region for template matching

### Output Files

- `recon_dy.png` - Result using dy as-is
- `recon_negdy.png` - Result using negated dy
- `recon_dy_e0.XX.png` - Result with specific edge-thr
- `recon_static_median.png` - Result from static-bg mode
- `debug_positions.csv` - Per-frame dy estimation with peak scores

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

### Known Limitations

- Phase correlation may fail with repetitive patterns (e.g., floral decorations)
- Template tracking loses target when feature exits frame (interpolation helps)
- Higher fps improves tracking accuracy (more frames = smaller dy per frame)
- Accumulated small errors in dy estimation can cause visible artifacts

### Workflow for Difficult Cases

1. Start with `--match-method template` for initial dy estimation
2. Check `debug_positions.csv` for outliers or tracking loss (score drops)
3. If needed, manually adjust dy values in Python script
4. Or use `--uniform-dy` if scroll speed is constant
