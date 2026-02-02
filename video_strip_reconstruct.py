#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_strip_reconstruct.py

縦パン（縦スクロール）する動画から、固定の横長帯(例: 1920x100)を切り出し、
背景に合わせてフレームを縦方向に整列しつつ、テロップ（縦スクロール or 固定）を
「できる範囲で」避けて背景を復元する“プロトタイプ”です。

狙い:
- まず動くテロップを「フレーム単位で選別」ではなく「画素単位で採用/不採用」にする
- 背景の縦方向移動(dy)を推定し、帯を“背景座標系”に貼り付けて縦長画像を作る
- テロップのような高コントラスト輪郭（文字）は edge 強度で弾き、背景っぽい画素を優先

注意:
- 文字が太い・背景も高周波・ブラー等の条件だと完璧には消えません（候補生成の1手段）。
- dy 推定の符号が素材によって逆に見えるケースがあるため、dy と -dy の2候補を出します。

必須:
  pip install pillow numpy
推奨:
  pip install opencv-python

ffmpeg があると「動画→フレーム抽出」を自動でできます。

例:
  # 3秒だけ、2fpsでフレーム抽出→復元（下から100pxの帯）
  python video_strip_reconstruct.py --video input.mp4 --start 12.0 --dur 3.0 --fps 2 \
    --strip-y 980 --strip-h 100 --out outdir

  # 右上ロゴを無視（px指定: top,right,width,height）
  python video_strip_reconstruct.py --video input.mp4 --fps 2 --strip-y 980 --strip-h 100 \
    --ignore "0,right,220,120" --out outdir

  # 既に抽出済みPNGを使う
  python video_strip_reconstruct.py --frames "frames/*.png" --strip-y 980 --strip-h 100 --out outdir

出力:
  outdir/recon_dy.png      # dyをそのまま積算した候補
  outdir/recon_negdy.png   # -dyを積算した候補
  outdir/debug_positions.csv (各フレームの推定dyと累積位置)
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# -------------------------
# Ignore regions (same format as stitch_candidates.py)
# -------------------------

@dataclass(frozen=True)
class IgnoreRegion:
    top: float
    anchor: str  # "right" | "left" | "x"
    x: float
    width: float
    height: float
    unit: str    # "px" | "pct"


def parse_ignore_region(s: str, unit: str) -> IgnoreRegion:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid ignore format: {s} (expected 4 comma-separated fields)")
    top = float(parts[0])
    side = parts[1].lower()
    if side in ("right", "r"):
        return IgnoreRegion(top=top, anchor="right", x=0.0, width=float(parts[2]), height=float(parts[3]), unit=unit)
    if side in ("left", "l"):
        return IgnoreRegion(top=top, anchor="left", x=0.0, width=float(parts[2]), height=float(parts[3]), unit=unit)
    try:
        x = float(side)
    except ValueError as e:
        raise ValueError(f"Invalid ignore anchor: {parts[1]} (use right/left/or numeric X)") from e
    return IgnoreRegion(top=top, anchor="x", x=x, width=float(parts[2]), height=float(parts[3]), unit=unit)


def _resolve_region_px(region: IgnoreRegion, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    def to_px(v: float, axis: str) -> int:
        if region.unit == "px":
            return int(round(v))
        base = img_w if axis == "w" else img_h
        return int(round(base * (v / 100.0)))

    top = to_px(region.top, "h")
    w = max(0, to_px(region.width, "w"))
    h = max(0, to_px(region.height, "h"))

    y0 = max(0, top)
    y1 = min(img_h, y0 + h)

    if region.anchor == "right":
        x1 = img_w
        x0 = max(0, x1 - w)
    elif region.anchor == "left":
        x0 = 0
        x1 = min(img_w, x0 + w)
    else:
        # NOTE: pctの場合のxは “左からx%” にしたい人もいるので、ここはunitに従う
        x0 = max(0, to_px(region.x, "w") if region.unit == "pct" else int(round(region.x)))
        x1 = min(img_w, x0 + w)

    return x0, y0, x1, y1


# -------------------------
# Image helpers
# -------------------------

def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

def save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)

def to_gray_f32(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return ((0.299*r + 0.587*g + 0.114*b) / 255.0).astype(np.float32)

def apply_ignore_gray(gray: np.ndarray, regions: List[IgnoreRegion]) -> np.ndarray:
    if not regions:
        return gray
    h, w = gray.shape
    out = gray.copy()
    for reg in regions:
        x0, y0, x1, y1 = _resolve_region_px(reg, w, h)
        if x0 < x1 and y0 < y1:
            out[y0:y1, x0:x1] = 0.0
    return out

def sobel_edge(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    if HAS_CV2:
        gx = cv2.filter2D(gray, -1, kx)
        gy = cv2.filter2D(gray, -1, ky)
    else:
        gx = _conv2(gray, kx)
        gy = _conv2(gray, ky)
    mag = np.sqrt(gx*gx + gy*gy)
    p = np.percentile(mag, 99.0)
    if p > 1e-6:
        mag = np.clip(mag / p, 0.0, 1.0)
    return mag.astype(np.float32)

def _conv2(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh//2, kw//2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            patch = pad[y:y+kh, x:x+kw]
            out[y, x] = float(np.sum(patch * kernel))
    return out


# -------------------------
# Shift estimation (dy)
# -------------------------

def phase_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    """Shift (dx,dy) so that shifting b by (dx,dy) aligns it to a."""
    win_y = np.hanning(a.shape[0]).astype(np.float32)
    win_x = np.hanning(a.shape[1]).astype(np.float32)
    win = win_y[:, None] * win_x[None, :]
    A = np.fft.fft2(a * win)
    B = np.fft.fft2(b * win)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), 1e-9)
    r = np.fft.ifft2(R)
    corr = np.abs(r)
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    peak = float(corr[y, x])
    h, w = corr.shape
    dy = y if y <= h // 2 else y - h
    dx = x if x <= w // 2 else x - w
    return int(dx), int(dy), peak


def estimate_dy(prev_rgb: np.ndarray, next_rgb: np.ndarray, ignore_regions: List[IgnoreRegion],
                search: int = 40, use_edges: bool = True) -> Tuple[int, float]:
    """
    prev -> next のdyを推定する（dxは捨てる）
    - ignore_regions はフルフレーム座標でマスク
    - use_edges=True の場合はエッジ画像で位相相関（テロップの色変化に強い）
    """
    a = to_gray_f32(prev_rgb)
    b = to_gray_f32(next_rgb)
    if ignore_regions:
        a = apply_ignore_gray(a, ignore_regions)
        b = apply_ignore_gray(b, ignore_regions)

    if use_edges:
        a = sobel_edge(a)
        b = sobel_edge(b)

    dx, dy, peak = phase_correlation(a, b)
    dy = int(np.clip(dy, -search, search))
    return dy, peak


# -------------------------
# Frame extraction (ffmpeg)
# -------------------------

def extract_frames_ffmpeg(video: Path, out_dir: Path, fps: float, start: Optional[float], dur: Optional[float],
                          deinterlace: bool) -> List[Path]:
    """
    ffmpegでPNG連番を書き出す。インタレ疑いなら deinterlace=True で yadif を噛ませる。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%06d.png"

    vf = []
    if deinterlace:
        vf.append("yadif")  # 最低限
    vf.append(f"fps={fps}")

    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-vf", ",".join(vf), str(pattern)]
    cmd += ["-hide_banner", "-loglevel", "error"]

    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.png"))


# -------------------------
# Strip-based reconstruction
# -------------------------

def crop_strip(rgb: np.ndarray, y: int, h: int) -> np.ndarray:
    H, W, _ = rgb.shape
    y0 = max(0, min(H, y))
    y1 = max(0, min(H, y + h))
    return rgb[y0:y1, :, :]

def build_text_mask(strip_rgb: np.ndarray, edge_thr: float) -> np.ndarray:
    """
    テロップ（文字）を“弾きたい”ので、エッジ強度が高い画素をテキスト候補としてマスクする。
    return: True=弾く(=書かない), False=採用候補
    """
    g = to_gray_f32(strip_rgb)
    e = sobel_edge(g)
    return (e >= edge_thr), e  # mask, edge_strength

def apply_ignore_to_strip_mask(mask: np.ndarray, strip_y: int, ignore_regions: List[IgnoreRegion],
                               full_w: int, full_h: int) -> np.ndarray:
    """
    フルフレーム指定の ignore を、帯領域へ投影して mask=True にする。
    """
    if not ignore_regions:
        return mask
    out = mask.copy()
    h, w = mask.shape
    for reg in ignore_regions:
        x0, y0, x1, y1 = _resolve_region_px(reg, full_w, full_h)
        # overlap with strip band [strip_y, strip_y+h)
        sy0 = max(strip_y, y0)
        sy1 = min(strip_y + h, y1)
        if sy0 >= sy1:
            continue
        ry0 = sy0 - strip_y
        ry1 = sy1 - strip_y
        out[ry0:ry1, x0:x1] = True
    return out

def render_candidate(strips: List[np.ndarray], positions: List[int], out_path: Path,
                     edge_thr: float, ignore_regions: List[IgnoreRegion], full_w: int, full_h: int,
                     strip_y: int) -> None:
    """
    strips[i] を positions[i] (global y) に貼る。
    edgeが弱い(=背景っぽい)画素を優先し、edgeが強い(=文字っぽい)画素は書かない。
    """
    assert len(strips) == len(positions)
    strip_h = strips[0].shape[0]
    W = strips[0].shape[1]

    min_y = min(positions)
    max_y = max(positions) + strip_h
    out_h = (max_y - min_y)

    canvas = np.zeros((out_h, W, 3), dtype=np.uint8)
    best_score = np.full((out_h, W), np.inf, dtype=np.float32)  # 小さいほど優先（エッジ弱い）

    for i, (strip, gy) in enumerate(zip(strips, positions)):
        y0 = gy - min_y
        y1 = y0 + strip_h

        # 文字っぽい所を弾く
        text_mask, edge_strength = build_text_mask(strip, edge_thr=edge_thr)
        text_mask = apply_ignore_to_strip_mask(text_mask, strip_y=strip_y, ignore_regions=ignore_regions,
                                               full_w=full_w, full_h=full_h)

        # update where:
        # - not text_mask
        # - edge_strength < best_score
        # note: edge_strength is [h,w], best_score slice is [h,w]
        bs = best_score[y0:y1, :]
        upd = (~text_mask) & (edge_strength < bs)

        if np.any(upd):
            bs[upd] = edge_strength[upd]
            # broadcast upd to 3ch
            canvas_slice = canvas[y0:y1, :, :]
            canvas_slice[upd, :] = strip[upd, :]

    save_rgb(canvas, out_path)

def write_positions_csv(paths: List[Path], dys: List[int], pos1: List[int], pos2: List[int], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "frame", "dy", "pos_dy", "pos_negdy"])
        for i, p in enumerate(paths):
            dy = dys[i] if i < len(dys) else ""
            w.writerow([i, p.name, dy, pos1[i], pos2[i]])


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, default="", help="Input video path (optional if --frames is used)")
    ap.add_argument("--frames", type=str, default="", help='Glob for extracted frames, e.g. "frames/*.png"')
    ap.add_argument("--out", type=str, default="recon_out", help="Output directory")

    ap.add_argument("--fps", type=float, default=2.0, help="Extraction fps when using --video")
    ap.add_argument("--start", type=float, default=None, help="Start seconds for extraction")
    ap.add_argument("--dur", type=float, default=None, help="Duration seconds for extraction")
    ap.add_argument("--deinterlace", action="store_true", help="Apply yadif when extracting frames")

    ap.add_argument("--strip-y", type=int, required=True, help="Top Y of strip in full frame (px)")
    ap.add_argument("--strip-h", type=int, default=100, help="Height of strip (px)")

    ap.add_argument("--dy-search", type=int, default=40, help="Clamp dy to +/- this")
    ap.add_argument("--edge-thr", type=float, default=0.35, help="Edge threshold for 'text-like' masking (0..1, higher=less masked)")
    ap.add_argument("--no-edges", action="store_true", help="Use raw grayscale (not edges) for dy estimation")

    ap.add_argument("--ignore", action="append", default=[],
                    help='Ignore region (px): "top,(right|left|X),width,height"  e.g. "0,right,220,120"')
    ap.add_argument("--ignore-pct", action="append", default=[],
                    help='Ignore region (%): "top,(right|left|X),width,height"  e.g. "0,right,12,8"')

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ignore_regions: List[IgnoreRegion] = []
    for s in args.ignore:
        ignore_regions.append(parse_ignore_region(s, unit="px"))
    for s in args.ignore_pct:
        ignore_regions.append(parse_ignore_region(s, unit="pct"))

    frame_paths: List[Path] = []
    if args.frames:
        frame_paths = [Path(p) for p in sorted(glob.glob(args.frames))]
        if not frame_paths:
            raise SystemExit(f"No frames matched: {args.frames}")
    else:
        if not args.video:
            raise SystemExit("Provide --video or --frames")
        video = Path(args.video)
        tmp_frames = out_dir / "_frames"
        frame_paths = extract_frames_ffmpeg(video, tmp_frames, fps=args.fps, start=args.start, dur=args.dur, deinterlace=args.deinterlace)

    if len(frame_paths) < 2:
        raise SystemExit("Need at least 2 frames")

    # load frames
    frames_rgb = [load_rgb(p) for p in frame_paths]
    full_h, full_w, _ = frames_rgb[0].shape

    # crop strips
    strips = [crop_strip(fr, y=args.strip_y, h=args.strip_h) for fr in frames_rgb]
    # ensure consistent size
    W = strips[0].shape[1]
    Hs = strips[0].shape[0]
    if Hs != args.strip_h:
        print(f"Warning: strip-h clipped to {Hs} due to frame bounds")
    for s in strips:
        if s.shape[1] != W or s.shape[0] != Hs:
            raise SystemExit("All strips must have same size")

    # estimate dy between consecutive full frames (better than strip-only when strip is dominated by text)
    dys: List[int] = []
    peaks: List[float] = []
    use_edges = (not args.no_edges)
    for i in range(1, len(frames_rgb)):
        dy, peak = estimate_dy(frames_rgb[i-1], frames_rgb[i], ignore_regions=ignore_regions,
                               search=args.dy_search, use_edges=use_edges)
        dys.append(dy)
        peaks.append(peak)

    # two candidate position sequences: sum(dy) and sum(-dy)
    pos_dy = [0]
    pos_negdy = [0]
    for dy in dys:
        pos_dy.append(pos_dy[-1] + dy)
        pos_negdy.append(pos_negdy[-1] - dy)

    # render
    out1 = out_dir / "recon_dy.png"
    out2 = out_dir / "recon_negdy.png"
    render_candidate(strips=strips, positions=pos_dy, out_path=out1,
                     edge_thr=args.edge_thr, ignore_regions=ignore_regions,
                     full_w=full_w, full_h=full_h, strip_y=args.strip_y)
    render_candidate(strips=strips, positions=pos_negdy, out_path=out2,
                     edge_thr=args.edge_thr, ignore_regions=ignore_regions,
                     full_w=full_w, full_h=full_h, strip_y=args.strip_y)

    # debug csv
    write_positions_csv(frame_paths, dys, pos_dy, pos_negdy, out_dir / "debug_positions.csv")

    print("Done:")
    print(f"  {out1}")
    print(f"  {out2}")
    print(f"  {out_dir / 'debug_positions.csv'}")


if __name__ == "__main__":
    main()
