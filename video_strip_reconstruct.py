#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_strip_reconstruct.py

縦パン（縦スクロール）する動画から、固定の横長帯(例: 1920x100)を切り出し、
背景に合わせてフレームを縦方向に整列しつつ、テロップ（縦スクロール or 固定）を
「できる範囲で」避けて背景を復元する"プロトタイプ"です。

狙い:
- まず動くテロップを「フレーム単位で選別」ではなく「画素単位で採用/不採用」にする
- 背景の縦方向移動(dy)を推定し、帯を"背景座標系"に貼り付けて縦長画像を作る
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

  # 複数のedge-thrで候補を生成
  python video_strip_reconstruct.py --frames "frames/*.png" --strip-y 980 --strip-h 100 \
    --edge-thr 0.3,0.4,0.5 --out outdir

  # NCCベースのマッチングを追加で使用
  python video_strip_reconstruct.py --frames "frames/*.png" --strip-y 980 --strip-h 100 \
    --match-method phase,ncc_gray --out outdir

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
from typing import List, Tuple, Optional, Dict

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


def _shift_crop(a: np.ndarray, b: np.ndarray, dx: int, dy: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crop overlapping region after shifting b by (dx, dy)."""
    h, w = a.shape
    # a stays fixed, b shifts by (dx, dy)
    ax0 = max(0, dx)
    ax1 = min(w, w + dx)
    ay0 = max(0, dy)
    ay1 = min(h, h + dy)
    bx0 = max(0, -dx)
    bx1 = min(w, w - dx)
    by0 = max(0, -dy)
    by1 = min(h, h - dy)
    return a[ay0:ay1, ax0:ax1], b[by0:by1, bx0:bx1]


def template_match_dy(prev_rgb: np.ndarray, next_rgb: np.ndarray,
                      template_regions: List[Tuple[int, int, int, int]],
                      prev_offsets: List[Tuple[int, int]] = None) -> Tuple[int, float, List[Tuple[int, int]]]:
    """
    テンプレートマッチングでdy推定。複数テンプレートを追跡し、中央値を返す。

    template_regions: [(y1, y2, x1, x2), ...] - 初期テンプレート領域
    prev_offsets: [(y_offset, x_offset), ...] - 前フレームでの各テンプレートの位置オフセット

    Returns: (dy, score, new_offsets)
    """
    if HAS_CV2:
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = (0.299*prev_rgb[:,:,0] + 0.587*prev_rgb[:,:,1] + 0.114*prev_rgb[:,:,2]).astype(np.uint8)
        next_gray = (0.299*next_rgb[:,:,0] + 0.587*next_rgb[:,:,1] + 0.114*next_rgb[:,:,2]).astype(np.uint8)

    if prev_offsets is None:
        prev_offsets = [(0, 0)] * len(template_regions)

    dys = []
    scores = []
    new_offsets = []

    for i, (y1, y2, x1, x2) in enumerate(template_regions):
        # 前フレームでのテンプレート位置（追跡によるオフセットを適用）
        off_y, off_x = prev_offsets[i]
        ty1 = max(0, y1 + off_y)
        ty2 = min(prev_gray.shape[0], y2 + off_y)
        tx1 = max(0, x1 + off_x)
        tx2 = min(prev_gray.shape[1], x2 + off_x)

        if ty2 - ty1 < 20 or tx2 - tx1 < 20:
            # テンプレートが画面外
            new_offsets.append((off_y, off_x))
            continue

        template = prev_gray[ty1:ty2, tx1:tx2]

        if HAS_CV2:
            result = cv2.matchTemplate(next_gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
        else:
            # Fallback without OpenCV (slow)
            max_val, max_loc = _template_match_numpy(next_gray, template)

        if max_val > 0.7:  # 信頼できるマッチのみ採用
            dy = max_loc[1] - ty1
            dx = max_loc[0] - tx1
            dys.append(dy)
            scores.append(max_val)
            new_offsets.append((off_y + dy, off_x + dx))
        else:
            # マッチ失敗、前のオフセットを維持
            new_offsets.append((off_y, off_x))

    if dys:
        dy_median = int(np.median(dys))
        score_mean = float(np.mean(scores))
        return dy_median, score_mean, new_offsets
    else:
        return 0, 0.0, new_offsets


def _template_match_numpy(img: np.ndarray, template: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """NumPyのみでのテンプレートマッチング（遅い）"""
    th, tw = template.shape
    ih, iw = img.shape
    best_val = -1.0
    best_loc = (0, 0)

    # 粗い探索（10pxステップ）
    for y in range(0, ih - th, 10):
        for x in range(0, iw - tw, 10):
            patch = img[y:y+th, x:x+tw]
            corr = np.corrcoef(patch.flatten(), template.flatten())[0, 1]
            if corr > best_val:
                best_val = corr
                best_loc = (x, y)

    # 細かい探索
    bx, by = best_loc
    for y in range(max(0, by-10), min(ih-th, by+11)):
        for x in range(max(0, bx-10), min(iw-tw, bx+11)):
            patch = img[y:y+th, x:x+tw]
            corr = np.corrcoef(patch.flatten(), template.flatten())[0, 1]
            if corr > best_val:
                best_val = corr
                best_loc = (x, y)

    return best_val, best_loc


def select_template_regions(rgb: np.ndarray, n_templates: int = 8,
                            exclude_center: bool = True) -> List[Tuple[int, int, int, int]]:
    """
    特徴的な領域を自動選択してテンプレート候補を返す。
    エッジ強度が高い領域を優先し、中央のテキスト領域を避ける。
    左右端を優先的に選択（テキストが中央にある場合に有効）。
    """
    gray = to_gray_f32(rgb)
    edge = sobel_edge(gray)

    h, w = edge.shape
    template_h, template_w = 150, 150

    # マスク: 中央のテキスト領域を広く除外
    mask = np.ones_like(edge)
    if exclude_center:
        # 中央60%を除外（テキスト領域）
        cx = w // 2
        mask[:, int(cx-w*0.30):int(cx+w*0.30)] = 0

    # 上下端を広く除外（画面外に出やすい）
    margin_y = int(h * 0.15)
    mask[:margin_y, :] = 0
    mask[-margin_y:, :] = 0

    # エッジ強度でブロックごとのスコアを計算
    candidates = []
    step = 60
    for y in range(0, h - template_h, step):
        for x in range(0, w - template_w, step):
            region_mask = mask[y:y+template_h, x:x+template_w]
            if region_mask.mean() < 0.8:  # マスク領域を厳しく除外
                continue
            region_edge = edge[y:y+template_h, x:x+template_w]
            score = region_edge.mean()
            # 左右端にボーナス（テキストから離れている）
            if x < w * 0.25 or x > w * 0.75:
                score *= 1.5
            candidates.append((score, y, y+template_h, x, x+template_w))

    # スコア順にソートして上位を選択
    candidates.sort(reverse=True)

    # 重複を避けて選択（最小間隔を確保）
    selected = []
    min_dist = 100
    for score, y1, y2, x1, x2 in candidates:
        # 既選択と十分離れているか確認
        too_close = False
        for _, sy1, sy2, sx1, sx2 in selected:
            cy1, cx1 = (y1+y2)//2, (x1+x2)//2
            cy2, cx2 = (sy1+sy2)//2, (sx1+sx2)//2
            dist = ((cy1-cy2)**2 + (cx1-cx2)**2)**0.5
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            selected.append((score, y1, y2, x1, x2))
            if len(selected) >= n_templates:
                break

    print(f"  Selected {len(selected)} template regions")
    return [(y1, y2, x1, x2) for _, y1, y2, x1, x2 in selected]


def brute_ncc(a: np.ndarray, b: np.ndarray, search: int) -> Tuple[int, int, float]:
    """Brute-force normalized cross-correlation within ±search range."""
    best_dx, best_dy, best_score = 0, 0, -np.inf
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            a_crop, b_crop = _shift_crop(a, b, dx, dy)
            if a_crop.size == 0:
                continue
            ma = a_crop.mean()
            mb = b_crop.mean()
            a_centered = a_crop - ma
            b_centered = b_crop - mb
            denom = np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2))
            if denom < 1e-9:
                continue
            ncc = np.sum(a_centered * b_centered) / denom
            if ncc > best_score:
                best_score = ncc
                best_dx, best_dy = dx, dy
    return best_dx, best_dy, float(best_score)


@dataclass
class MatchResult:
    """Result from a single matching method."""
    method: str
    dx: int
    dy: int
    score: float


@dataclass
class TemplateTracker:
    """テンプレートマッチング用の状態を保持"""
    regions: List[Tuple[int, int, int, int]] = None
    offsets: List[Tuple[int, int]] = None


def _detect_and_fix_outliers(dys: List[int], scores: List[float]) -> List[int]:
    """
    明らかな外れ値を検出し、補間対象としてマーク（スコアを0に）。
    - 符号が多数派と逆
    - 絶対値が非常に大きい（他の2倍以上）
    - 前後と比べて急激に変化
    """
    if len(dys) < 3:
        return dys

    result = list(dys)
    nonzero = [d for d in dys if d != 0]
    if len(nonzero) < 2:
        return dys

    # 符号の多数派を特定
    positive = sum(1 for d in nonzero if d > 0)
    negative = sum(1 for d in nonzero if d < 0)
    expected_sign = 1 if positive >= negative else -1

    # 絶対値の中央値
    median_abs = np.median([abs(d) for d in nonzero])

    for i, dy in enumerate(dys):
        is_outlier = False
        reason = ""

        # 符号が逆で絶対値が大きい
        if dy != 0 and np.sign(dy) != expected_sign and abs(dy) > median_abs * 0.5:
            is_outlier = True
            reason = f"wrong sign (expected {'+' if expected_sign > 0 else '-'})"

        # 絶対値が非常に大きい（中央値の3倍以上）
        if abs(dy) > median_abs * 3:
            is_outlier = True
            reason = f"too large (median_abs={median_abs:.0f})"

        # 前後との急激な変化（前後両方と50%以上の差）
        if not is_outlier and i > 0 and i < len(dys) - 1:
            prev_dy = dys[i-1] if dys[i-1] != 0 else median_abs * expected_sign
            next_dy = dys[i+1] if dys[i+1] != 0 else median_abs * expected_sign
            if prev_dy != 0 and next_dy != 0:
                avg_neighbor = (prev_dy + next_dy) / 2
                if abs(dy - avg_neighbor) > abs(avg_neighbor) * 2:
                    is_outlier = True
                    reason = f"sudden change (neighbors avg={avg_neighbor:.0f})"

        if is_outlier:
            print(f"  Outlier at frame {i+1}: dy={dy} ({reason})")
            result[i] = 0
            scores[i] = 0.0

    return result


def _interpolate_lost_tracking(dys: List[int], scores: List[float], min_score: float = 0.5) -> List[int]:
    """
    追跡が途切れた区間を補間で埋める。
    - スコアが低い区間
    - dy=0が連続する区間（追跡が止まった可能性）
    """
    n = len(dys)
    if n == 0:
        return dys

    result = list(dys)

    # 有効なdy値のインデックスを特定
    # 条件: スコアが十分高い AND dy!=0（または前後と整合性がある）
    valid_indices = []
    for i, (dy, s) in enumerate(zip(dys, scores)):
        if s < min_score:
            continue
        # dy=0が3つ以上連続する場合は無効とみなす
        if dy == 0:
            zero_count = 1
            for j in range(i-1, -1, -1):
                if dys[j] == 0:
                    zero_count += 1
                else:
                    break
            for j in range(i+1, n):
                if dys[j] == 0:
                    zero_count += 1
                else:
                    break
            if zero_count >= 3:
                continue
        valid_indices.append(i)

    if len(valid_indices) < 2:
        # 有効な値が少なすぎる場合、非ゼロの中央値で埋める
        nonzero_dys = [d for d in dys if d != 0]
        if nonzero_dys:
            median_dy = int(np.median(nonzero_dys))
        else:
            median_dy = 0
        return [median_dy if d == 0 and scores[i] < 0.9 else d for i, d in enumerate(dys)]

    # 無効な区間を補間
    for i in range(n):
        if i not in valid_indices:
            # 前後の有効な値を探す
            prev_valid = None
            next_valid = None
            for j in range(i - 1, -1, -1):
                if j in valid_indices:
                    prev_valid = j
                    break
            for j in range(i + 1, n):
                if j in valid_indices:
                    next_valid = j
                    break

            if prev_valid is not None and next_valid is not None:
                # 線形補間
                t = (i - prev_valid) / (next_valid - prev_valid)
                result[i] = int(dys[prev_valid] + t * (dys[next_valid] - dys[prev_valid]))
            elif prev_valid is not None:
                result[i] = dys[prev_valid]
            elif next_valid is not None:
                result[i] = dys[next_valid]

    return result


def estimate_dy_multi(prev_rgb: np.ndarray, next_rgb: np.ndarray, ignore_regions: List[IgnoreRegion],
                      search: int = 40, methods: List[str] = None,
                      template_tracker: TemplateTracker = None) -> List[MatchResult]:
    """
    prev -> next のdyを複数手法で推定する

    methods: ["phase", "ncc_gray", "ncc_edge", "template"]
    returns: List of MatchResult (one per method)
    """
    if methods is None:
        methods = ["phase"]

    gray_a = to_gray_f32(prev_rgb)
    gray_b = to_gray_f32(next_rgb)
    if ignore_regions:
        gray_a = apply_ignore_gray(gray_a, ignore_regions)
        gray_b = apply_ignore_gray(gray_b, ignore_regions)

    edge_a = sobel_edge(gray_a)
    edge_b = sobel_edge(gray_b)

    results: List[MatchResult] = []

    for method in methods:
        if method == "phase":
            dx, dy, score = phase_correlation(edge_a, edge_b)
            dy = int(np.clip(dy, -search, search))
            results.append(MatchResult("phase", dx, dy, score))
        elif method == "ncc_gray":
            dx, dy, score = brute_ncc(gray_a, gray_b, search)
            results.append(MatchResult("ncc_gray", dx, dy, score))
        elif method == "ncc_edge":
            dx, dy, score = brute_ncc(edge_a, edge_b, search)
            results.append(MatchResult("ncc_edge", dx, dy, score))
        elif method == "phase_gray":
            # phase correlation on grayscale (without edge)
            dx, dy, score = phase_correlation(gray_a, gray_b)
            dy = int(np.clip(dy, -search, search))
            results.append(MatchResult("phase_gray", dx, dy, score))
        elif method == "template":
            if template_tracker is None or template_tracker.regions is None:
                # 初回: テンプレート領域を自動選択
                template_tracker.regions = select_template_regions(prev_rgb)
                template_tracker.offsets = None
            dy, score, new_offsets = template_match_dy(
                prev_rgb, next_rgb,
                template_tracker.regions,
                template_tracker.offsets
            )
            template_tracker.offsets = new_offsets
            results.append(MatchResult("template", 0, dy, score))

    return results


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

def render_static_background(strips: List[np.ndarray], out_path: Path,
                             method: str = "median") -> None:
    """
    背景が固定の場合に、時間的な中央値/最小エッジで背景を復元する。
    method: "median" or "min_edge"
    """
    # All strips should be the same size and position
    H, W, C = strips[0].shape
    stack = np.stack(strips, axis=0)  # (N, H, W, C)

    if method == "median":
        # 時間的中央値 - テキストが一時的に通過する場合に有効
        result = np.median(stack, axis=0).astype(np.uint8)
    else:
        # min_edge: 各画素で最もエッジ強度が低いフレームを選択
        N = len(strips)
        edges = np.zeros((N, H, W), dtype=np.float32)
        for i, strip in enumerate(strips):
            edges[i] = sobel_edge(to_gray_f32(strip))
        # 最小エッジのインデックス
        min_idx = np.argmin(edges, axis=0)  # (H, W)
        result = np.zeros((H, W, C), dtype=np.uint8)
        for c in range(C):
            for y in range(H):
                for x in range(W):
                    result[y, x, c] = stack[min_idx[y, x], y, x, c]

    save_rgb(result, out_path)


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

def write_positions_csv(paths: List[Path], dys: List[int], peaks: List[float],
                        pos1: List[int], pos2: List[int], out_csv: Path,
                        min_peak: float = 0.0) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "frame", "dy", "peak", "reliable", "pos_dy", "pos_negdy"])
        for i, p in enumerate(paths):
            if i < len(dys):
                dy = dys[i]
                peak = peaks[i]
                reliable = "OK" if peak >= min_peak else "LOW"
            else:
                dy = ""
                peak = ""
                reliable = ""
            w.writerow([i, p.name, dy, f"{peak:.4f}" if isinstance(peak, float) else peak,
                       reliable, pos1[i], pos2[i]])


# -------------------------
# Main
# -------------------------

def analyze_and_suggest(dys: List[int], peaks: List[float], frame_paths: List[Path],
                        method: str, fps: float) -> None:
    """
    debug_positions.csv の内容を分析し、改善のためのパラメータを提案する。
    """
    n = len(dys)
    if n < 2:
        return

    # 統計計算
    nonzero_dys = [d for d in dys if d != 0]
    zero_count = sum(1 for d in dys if d == 0)
    zero_ratio = zero_count / n

    valid_peaks = [p for p in peaks if p > 0]
    low_score_count = sum(1 for p in peaks if 0 < p < 0.7)

    if nonzero_dys:
        dy_mean = np.mean(nonzero_dys)
        dy_std = np.std(nonzero_dys)
        dy_median = np.median(nonzero_dys)
    else:
        dy_mean = dy_std = dy_median = 0

    total_movement = abs(sum(dys))

    print("\n" + "="*60)
    print("[Analysis Results]")
    print("="*60)
    print(f"  Frames: {n+1}")
    print(f"  Total movement: {total_movement}px")
    print(f"  dy mean: {dy_mean:.1f}px (std: {dy_std:.1f})")
    print(f"  Zero dy ratio: {zero_ratio*100:.1f}% ({zero_count}/{n})")
    print(f"  Low score ratio: {low_score_count}/{n}")

    # 問題検出と提案
    suggestions = []

    # Many dy=0
    if zero_ratio > 0.2:
        suggestions.append({
            'issue': 'Many dy=0 (tracking loss)',
            'suggestions': [
                f'Lower fps (current: {fps} -> suggest: {max(1, fps/2):.1f})',
                '--template-region to manually specify distinctive region',
                f'Try --uniform-dy {int(dy_median) if dy_median else "?"}',
            ]
        })

    # Unstable dy estimation
    if dy_std > abs(dy_mean) * 0.5 and abs(dy_mean) > 5:
        suggestions.append({
            'issue': 'Unstable dy estimation',
            'suggestions': [
                'Try --match-method phase (may be more stable than template)',
                '--template-region to specify stable region',
                f'Try --uniform-dy {int(dy_median)}',
            ]
        })

    # Low matching scores
    if low_score_count > n * 0.3:
        suggestions.append({
            'issue': 'Low matching scores',
            'suggestions': [
                'Change template region (choose more distinctive area)',
                '--ignore to exclude problematic regions',
                'Compare with --match-method phase,template',
            ]
        })

    # Low total movement
    expected_movement = abs(dy_median * n) if dy_median else 0
    if expected_movement > 0 and total_movement < expected_movement * 0.7:
        suggestions.append({
            'issue': f'Low total movement (expected: ~{int(expected_movement)}px)',
            'suggestions': [
                'Manually interpolate dy=0 sections',
                f'Try --uniform-dy {int(dy_median)}',
            ]
        })

    if suggestions:
        print("\n[Suggestions]")
        for i, s in enumerate(suggestions, 1):
            print(f"\n  {i}. {s['issue']}")
            for suggestion in s['suggestions']:
                print(f"     -> {suggestion}")
    else:
        print("\n  OK: No issues detected")

    # Recommended command
    if nonzero_dys and (zero_ratio > 0.1 or dy_std > abs(dy_mean) * 0.3):
        print("\n[Recommended command]")
        print(f"  python video_strip_reconstruct.py --frames \"...\" \\")
        print(f"    --strip-y 0 --strip-h 1080 \\")
        print(f"    --uniform-dy {int(-abs(dy_median))} --edge-thr 0.35 --out outdir")

    print("="*60 + "\n")


def show_diagnostics(dys: List[int], peaks: List[float], frame_paths: List[Path],
                     min_peak: float) -> None:
    """Show diagnostic info for unreliable dy estimates."""
    low_peak_indices = [i for i, p in enumerate(peaks) if p < min_peak]
    if not low_peak_indices:
        return

    print(f"\n[診断] ピークスコアが低いフレーム ({len(low_peak_indices)}件, 閾値: {min_peak:.4f}):")
    print("[Diagnostic] Frames with low peak score:")
    for i in low_peak_indices[:10]:  # show max 10
        print(f"  Frame {i+1}: {frame_paths[i+1].name} <- {frame_paths[i].name}")
        print(f"    dy={dys[i]:+3d}, peak={peaks[i]:.4f}")
    if len(low_peak_indices) > 10:
        print(f"  ... and {len(low_peak_indices) - 10} more")

    # variance check
    if len(dys) >= 3:
        dy_std = np.std(dys)
        dy_mean = np.mean(dys)
        print(f"\n[統計] dy: mean={dy_mean:.1f}, std={dy_std:.1f}")
        if dy_std > abs(dy_mean) * 0.5 and abs(dy_mean) > 1:
            print("  -> dy推定が不安定な可能性があります")
            print("  -> dy estimation may be unstable")


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
推奨ワークフロー / Recommended Workflow:
  1. --edge-thr を複数試す (例: 0.3,0.4,0.5)
  2. recon_dy.png と recon_negdy.png を確認
  3. 良い方を選び、必要に応じて --edge-thr を調整
  4. 方向が分かったら --scroll-dir up/down で絞り込み

例 / Example:
  # 両方向の候補を出力（デフォルト）
  python video_strip_reconstruct.py --video input.mp4 --fps 2 \\
    --strip-y 980 --strip-h 100 --edge-thr 0.3,0.4,0.5 --out outdir

  # 上方向スクロール（コンテンツが下に移動）のみ出力
  python video_strip_reconstruct.py --frames "frames/*.png" \\
    --strip-y 980 --strip-h 100 --scroll-dir up --out outdir

  # 下方向スクロール（コンテンツが上に移動）のみ出力
  python video_strip_reconstruct.py --frames "frames/*.png" \\
    --strip-y 980 --strip-h 100 --scroll-dir down --out outdir
""")
    ap.add_argument("--video", type=str, default="", help="Input video path (optional if --frames is used)")
    ap.add_argument("--frames", type=str, default="", help='Glob for extracted frames, e.g. "frames/*.png"')
    ap.add_argument("--out", type=str, default="recon_out", help="Output directory")

    ap.add_argument("--fps", type=float, default=2.0, help="Extraction fps when using --video")
    ap.add_argument("--start", type=float, default=None, help="Start seconds for extraction")
    ap.add_argument("--dur", type=float, default=None, help="Duration seconds for extraction")
    ap.add_argument("--deinterlace", action="store_true", help="Apply yadif when extracting frames")

    ap.add_argument("--strip-y", type=int, required=True, help="Top Y of strip in full frame (px)")
    ap.add_argument("--strip-h", type=int, default=100, help="Height of strip (px)")
    ap.add_argument("--dy-region", type=str, default="",
                    help="Region for dy estimation: 'y,h' (e.g. '0,200' for top 200px). If empty, use full frame.")

    ap.add_argument("--dy-search", type=int, default=40, help="Clamp dy to +/- this")
    ap.add_argument("--edge-thr", type=str, default="0.35",
                    help="Edge threshold(s) for 'text-like' masking. Comma-separated for multiple candidates (e.g. 0.3,0.4,0.5)")
    ap.add_argument("--no-edges", action="store_true", help="Use raw grayscale (not edges) for dy estimation")

    ap.add_argument("--min-peak", type=float, default=0.0,
                    help="Minimum peak score for reliable dy estimation. Shows diagnostic when below (default: 0=disabled)")
    ap.add_argument("--match-method", type=str, default="phase",
                    help="Matching method(s): phase, ncc_gray, ncc_edge, phase_gray, template. Comma-separated (default: phase)")
    ap.add_argument("--scroll-dir", type=str, default="both", choices=["up", "down", "both"],
                    help="Scroll direction: up (content moves down), down (content moves up), both (output both candidates, default)")
    ap.add_argument("--static-bg", action="store_true",
                    help="Static background mode: use temporal median/min-edge to remove scrolling text (background does not scroll)")
    ap.add_argument("--static-method", type=str, default="median", choices=["median", "min_edge"],
                    help="Method for static-bg mode: median (default) or min_edge")
    ap.add_argument("--uniform-dy", type=float, default=None,
                    help="Use uniform dy per frame instead of estimation (e.g. -10 for 10px up per frame)")
    ap.add_argument("--template-region", type=str, default="",
                    help="Manual template region for template matching: 'y,h,x,w' (e.g. '350,150,750,150')")

    ap.add_argument("--ignore", action="append", default=[],
                    help='Ignore region (px): "top,(right|left|X),width,height"  e.g. "0,right,220,120"')
    ap.add_argument("--ignore-pct", action="append", default=[],
                    help='Ignore region (pct): "top,(right|left|X),width,height"  e.g. "0,right,12,8"')

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse edge thresholds (multiple values supported)
    edge_thrs = [float(x.strip()) for x in args.edge_thr.split(",")]

    # Parse matching methods
    match_methods = [m.strip() for m in args.match_method.split(",")]

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
    print(f"Loading {len(frame_paths)} frames...")
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

    # Static background mode: use temporal median/min-edge
    if args.static_bg:
        print(f"Static background mode: using {args.static_method} method...")
        out_path = out_dir / f"recon_static_{args.static_method}.png"
        render_static_background(strips, out_path, method=args.static_method)
        print(f"\nDone:\n  {out_path}")
        return

    # Uniform dy mode: skip estimation
    if args.uniform_dy is not None:
        uniform_dy = args.uniform_dy
        print(f"Using uniform dy: {uniform_dy:.1f} per frame")
        n_frames = len(frames_rgb)
        dys = [uniform_dy] * (n_frames - 1)
        peaks = [1.0] * (n_frames - 1)

        pos_dy = [0]
        pos_negdy = [0]
        for dy in dys:
            pos_dy.append(pos_dy[-1] + dy)
            pos_negdy.append(pos_negdy[-1] - dy)

        # render candidates
        output_files = []
        scroll_dir = args.scroll_dir
        for edge_thr in edge_thrs:
            suffix = f"_e{edge_thr:.2f}" if len(edge_thrs) > 1 else ""
            if scroll_dir in ("up", "both"):
                out1 = out_dir / f"recon_dy{suffix}.png"
                render_candidate(strips=strips, positions=[int(p) for p in pos_dy], out_path=out1,
                                 edge_thr=edge_thr, ignore_regions=ignore_regions,
                                 full_w=full_w, full_h=full_h, strip_y=args.strip_y)
                output_files.append(out1)
            if scroll_dir in ("down", "both"):
                out2 = out_dir / f"recon_negdy{suffix}.png"
                render_candidate(strips=strips, positions=[int(p) for p in pos_negdy], out_path=out2,
                                 edge_thr=edge_thr, ignore_regions=ignore_regions,
                                 full_w=full_w, full_h=full_h, strip_y=args.strip_y)
                output_files.append(out2)

        csv_path = out_dir / "debug_positions.csv"
        write_positions_csv(frame_paths, [int(d) for d in dys], peaks,
                            [int(p) for p in pos_dy], [int(p) for p in pos_negdy], csv_path)
        print("\nDone:")
        for f in output_files:
            print(f"  {f}")
        print(f"  {csv_path}")
        return

    # Prepare frames for dy estimation (optionally crop to dy-region)
    dy_frames = frames_rgb
    if args.dy_region:
        dy_y, dy_h = [int(x) for x in args.dy_region.split(",")]
        print(f"Using dy-region: y={dy_y}, h={dy_h}")
        dy_frames = [crop_strip(fr, y=dy_y, h=dy_h) for fr in frames_rgb]

    # estimate dy between consecutive frames using multiple methods
    print(f"Estimating dy using method(s): {', '.join(match_methods)}...")

    # Store results per method
    method_results: Dict[str, Tuple[List[int], List[float]]] = {}

    for method in match_methods:
        dys: List[int] = []
        peaks: List[float] = []
        # Template method needs state tracking
        template_tracker = None
        if method == "template":
            template_tracker = TemplateTracker()
            # Manual template region if specified
            if args.template_region:
                parts = [int(x) for x in args.template_region.split(",")]
                if len(parts) == 4:
                    y, h, x, w = parts
                    template_tracker.regions = [(y, y+h, x, x+w)]
                    print(f"  Using manual template region: y={y}, h={h}, x={x}, w={w}")

        for i in range(1, len(dy_frames)):
            results = estimate_dy_multi(dy_frames[i-1], dy_frames[i],
                                        ignore_regions=ignore_regions,
                                        search=args.dy_search,
                                        methods=[method],
                                        template_tracker=template_tracker)
            if results:
                r = results[0]
                dys.append(r.dy)
                peaks.append(r.score)
            else:
                dys.append(0)
                peaks.append(0.0)

        # Template method: detect outliers and interpolate
        if method == "template":
            dys = _detect_and_fix_outliers(dys, peaks)
            dys = _interpolate_lost_tracking(dys, peaks)

        method_results[method] = (dys, peaks)

    # Use first method as primary
    primary_method = match_methods[0]
    dys, peaks = method_results[primary_method]

    # Show diagnostics if min-peak is set
    if args.min_peak > 0:
        show_diagnostics(dys, peaks, frame_paths, args.min_peak)

    # Analyze results and suggest improvements
    analyze_and_suggest(dys, peaks, frame_paths, primary_method, args.fps)

    # two candidate position sequences: sum(dy) and sum(-dy)
    pos_dy = [0]
    pos_negdy = [0]
    for dy in dys:
        pos_dy.append(pos_dy[-1] + dy)
        pos_negdy.append(pos_negdy[-1] - dy)

    # render candidates for each edge_thr
    output_files = []
    scroll_dir = args.scroll_dir
    for edge_thr in edge_thrs:
        suffix = f"_e{edge_thr:.2f}" if len(edge_thrs) > 1 else ""
        # up: content moves down (use pos_dy)
        # down: content moves up (use pos_negdy)
        if scroll_dir in ("up", "both"):
            out1 = out_dir / f"recon_dy{suffix}.png"
            render_candidate(strips=strips, positions=pos_dy, out_path=out1,
                             edge_thr=edge_thr, ignore_regions=ignore_regions,
                             full_w=full_w, full_h=full_h, strip_y=args.strip_y)
            output_files.append(out1)
        if scroll_dir in ("down", "both"):
            out2 = out_dir / f"recon_negdy{suffix}.png"
            render_candidate(strips=strips, positions=pos_negdy, out_path=out2,
                             edge_thr=edge_thr, ignore_regions=ignore_regions,
                             full_w=full_w, full_h=full_h, strip_y=args.strip_y)
            output_files.append(out2)

    # If multiple methods, also output candidates per method (using first edge_thr)
    if len(match_methods) > 1:
        primary_edge_thr = edge_thrs[0]
        for method in match_methods[1:]:
            m_dys, m_peaks = method_results[method]
            m_pos_dy = [0]
            m_pos_negdy = [0]
            for dy in m_dys:
                m_pos_dy.append(m_pos_dy[-1] + dy)
                m_pos_negdy.append(m_pos_negdy[-1] - dy)

            if scroll_dir in ("up", "both"):
                out1 = out_dir / f"recon_{method}_dy.png"
                render_candidate(strips=strips, positions=m_pos_dy, out_path=out1,
                                 edge_thr=primary_edge_thr, ignore_regions=ignore_regions,
                                 full_w=full_w, full_h=full_h, strip_y=args.strip_y)
                output_files.append(out1)
            if scroll_dir in ("down", "both"):
                out2 = out_dir / f"recon_{method}_negdy.png"
                render_candidate(strips=strips, positions=m_pos_negdy, out_path=out2,
                                 edge_thr=primary_edge_thr, ignore_regions=ignore_regions,
                                 full_w=full_w, full_h=full_h, strip_y=args.strip_y)
                output_files.append(out2)

    # debug csv (include peak scores)
    csv_path = out_dir / "debug_positions.csv"
    write_positions_csv(frame_paths, dys, peaks, pos_dy, pos_negdy, csv_path, min_peak=args.min_peak)

    print("\nDone:")
    for f in output_files:
        print(f"  {f}")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
