#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stitch_candidates.py

機能:
- 複数のマッチング手法 (phase correlation, NCC gray, NCC edge)
- 垂直/水平/スネーク(zigzag)モード
- 無視領域 (--ignore, --ignore-pct)
- constantDelta バリアント自動生成
- refine-from モード (既存候補の近傍を局所再探索)
- min-overlap-ratio による実効オーバーラップ足切り
- Windows セーフなワイルドカード展開

必須:
  pip install pillow numpy
推奨:
  pip install opencv-python
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# -------------------------
# Windows-safe wildcard expansion
# -------------------------

def expand_images(args: List[str]) -> List[Path]:
    out: List[Path] = []
    for a in args:
        if any(c in a for c in "*?[]"):
            out.extend(Path(m) for m in glob.glob(a))
        else:
            out.append(Path(a))
    return out


# -------------------------
# Ignore regions
# -------------------------

@dataclass(frozen=True)
class IgnoreRegion:
    top: float
    anchor: str  # "right" | "left" | "x"
    x: float     # used when anchor=="x" (left offset)
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
    else:  # "x"
        x0 = max(0, to_px(region.x, "w") if region.unit == "pct" else int(round(region.x)))
        x1 = min(img_w, x0 + w)

    return x0, y0, x1, y1


def apply_ignore_regions_gray(gray: np.ndarray, regions: List[IgnoreRegion]) -> np.ndarray:
    if not regions:
        return gray
    h, w = gray.shape
    out = gray.copy()
    for reg in regions:
        x0, y0, x1, y1 = _resolve_region_px(reg, w, h)
        if x0 < x1 and y0 < y1:
            out[y0:y1, x0:x1] = 0.0
    return out


# -------------------------
# Image IO / preprocessing
# -------------------------

def load_image_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.uint8)


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def to_gray_f32(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return gray.astype(np.float32)


def conv2(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            patch = padded[y:y+kh, x:x+kw]
            out[y, x] = float(np.sum(patch * kernel))
    return out


def edge_map(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    if HAS_CV2:
        gx = cv2.filter2D(gray, -1, kx)
        gy = cv2.filter2D(gray, -1, ky)
    else:
        gx = conv2(gray, kx)
        gy = conv2(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy)
    p = np.percentile(mag, 99.0)
    if p > 1e-6:
        mag = np.clip(mag / p, 0.0, 1.0)
    return mag.astype(np.float32)


# -------------------------
# Band extraction
# -------------------------

def crop_band_vertical_from_gray(prev_gray: np.ndarray, next_gray: np.ndarray, band: int) -> Tuple[np.ndarray, np.ndarray]:
    band = min(band, prev_gray.shape[0], next_gray.shape[0])
    # 幅を揃える
    min_w = min(prev_gray.shape[1], next_gray.shape[1])
    return prev_gray[-band:, :min_w], next_gray[:band, :min_w]


def crop_band_horizontal_from_gray(prev_gray: np.ndarray, next_gray: np.ndarray, band: int) -> Tuple[np.ndarray, np.ndarray]:
    band = min(band, prev_gray.shape[1], next_gray.shape[1])
    # 高さを揃える
    min_h = min(prev_gray.shape[0], next_gray.shape[0])
    return prev_gray[:min_h, -band:], next_gray[:min_h, :band]


# -------------------------
# Matching
# -------------------------

def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-9
    return float(np.sum(a0 * b0) / denom)


def compute_boundary_similarity(prev_rgb: np.ndarray, next_rgb: np.ndarray,
                                  mode: str, dx: int, dy: int, overlap: int) -> float:
    """
    境界部分のピクセル一致度を計算する（0.0〜1.0）。
    重なり領域のSSIM風スコアを返す。
    """
    ph, pw, _ = prev_rgb.shape
    nh, nw, _ = next_rgb.shape

    if mode == 'v':
        # 垂直: prev の下部 と next の上部
        eff_overlap = max(1, overlap - abs(dy))
        if eff_overlap <= 0:
            return 0.0
        prev_band = prev_rgb[-(eff_overlap):, :, :]
        next_band = next_rgb[:eff_overlap, :, :]
        # dx によるシフトを考慮
        if dx > 0:
            prev_band = prev_band[:, :min(pw, nw - dx), :]
            next_band = next_band[:, dx:dx + prev_band.shape[1], :]
        elif dx < 0:
            next_band = next_band[:, :min(nw, pw + dx), :]
            prev_band = prev_band[:, -dx:-dx + next_band.shape[1], :]
        else:
            min_w = min(pw, nw)
            prev_band = prev_band[:, :min_w, :]
            next_band = next_band[:, :min_w, :]
    else:
        # 水平: prev の右部 と next の左部
        eff_overlap = max(1, overlap - abs(dx))
        if eff_overlap <= 0:
            return 0.0
        prev_band = prev_rgb[:, -(eff_overlap):, :]
        next_band = next_rgb[:, :eff_overlap, :]
        # dy によるシフトを考慮
        if dy > 0:
            prev_band = prev_band[:min(ph, nh - dy), :, :]
            next_band = next_band[dy:dy + prev_band.shape[0], :, :]
        elif dy < 0:
            next_band = next_band[:min(nh, ph + dy), :, :]
            prev_band = prev_band[-dy:-dy + next_band.shape[0], :, :]
        else:
            min_h = min(ph, nh)
            prev_band = prev_band[:min_h, :, :]
            next_band = next_band[:min_h, :, :]

    # サイズが合わない場合は小さい方に合わせる
    h = min(prev_band.shape[0], next_band.shape[0])
    w = min(prev_band.shape[1], next_band.shape[1])
    if h <= 0 or w <= 0:
        return 0.0
    prev_band = prev_band[:h, :w, :].astype(np.float32)
    next_band = next_band[:h, :w, :].astype(np.float32)

    # SSIM風の計算（簡易版）
    # 輝度成分で計算
    prev_gray = 0.299 * prev_band[..., 0] + 0.587 * prev_band[..., 1] + 0.114 * prev_band[..., 2]
    next_gray = 0.299 * next_band[..., 0] + 0.587 * next_band[..., 1] + 0.114 * next_band[..., 2]

    mu_x = prev_gray.mean()
    mu_y = next_gray.mean()
    sigma_x = prev_gray.std() + 1e-9
    sigma_y = next_gray.std() + 1e-9
    sigma_xy = ((prev_gray - mu_x) * (next_gray - mu_y)).mean()

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))

    return float(max(0.0, min(1.0, ssim)))


def shift_crop(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape
    x0 = max(0, dx)
    y0 = max(0, dy)
    x1 = min(w, w + dx)
    y1 = min(h, h + dy)
    sx0 = max(0, -dx)
    sy0 = max(0, -dy)
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    out = np.zeros_like(img, dtype=img.dtype)
    if x0 < x1 and y0 < y1:
        out[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]
    return out


def phase_correlation(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
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


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """2つの画像パッチのSSIMスコアを計算（0.0〜1.0）"""
    mu_x = a.mean()
    mu_y = b.mean()
    sigma_x = a.std() + 1e-9
    sigma_y = b.std() + 1e-9
    sigma_xy = ((a - mu_x) * (b - mu_y)).mean()

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2))
    return float(max(0.0, min(1.0, ssim)))


def brute_ssim(a: np.ndarray, b: np.ndarray, search: int, constrain: str) -> Tuple[int, int, float]:
    """SSIMベースのブルートフォースマッチング（遅い）"""
    best_dx, best_dy, best_s = 0, 0, -1e18
    if constrain == 'v':
        dxs = [0]
        dys = range(-search, search + 1)
    elif constrain == 'h':
        dxs = range(-search, search + 1)
        dys = [0]
    else:
        dxs = range(-search, search + 1)
        dys = range(-search, search + 1)

    for dy in dys:
        for dx in dxs:
            b_shift = shift_crop(b, dx, dy)
            h, w = a.shape
            x0 = max(0, dx)
            y0 = max(0, dy)
            x1 = min(w, w + dx)
            y1 = min(h, h + dy)
            if (x1 - x0) < max(10, w // 10) or (y1 - y0) < max(10, h // 10):
                continue
            s = ssim_score(a[y0:y1, x0:x1], b_shift[y0:y1, x0:x1])
            if s > best_s:
                best_dx, best_dy, best_s = dx, dy, s
    return best_dx, best_dy, best_s


def brute_ncc(a: np.ndarray, b: np.ndarray, search: int, constrain: str) -> Tuple[int, int, float]:
    best_dx, best_dy, best_s = 0, 0, -1e18
    if constrain == 'v':
        dxs = [0]
        dys = range(-search, search + 1)
    elif constrain == 'h':
        dxs = range(-search, search + 1)
        dys = [0]
    else:
        dxs = range(-search, search + 1)
        dys = range(-search, search + 1)

    for dy in dys:
        for dx in dxs:
            b_shift = shift_crop(b, dx, dy)
            h, w = a.shape
            x0 = max(0, dx)
            y0 = max(0, dy)
            x1 = min(w, w + dx)
            y1 = min(h, h + dy)
            if (x1 - x0) < max(10, w // 10) or (y1 - y0) < max(10, h // 10):
                continue
            s = ncc_score(a[y0:y1, x0:x1], b_shift[y0:y1, x0:x1])
            if s > best_s:
                best_dx, best_dy, best_s = dx, dy, s
    return best_dx, best_dy, best_s


@dataclass
class Method:
    name: str
    fn: Callable[[np.ndarray, np.ndarray, int, str], Tuple[int, int, float]]
    preprocess: Optional[Callable[[np.ndarray], np.ndarray]] = None
    min_score: float = 0.0  # スコアがこれ未満なら足切り


def build_methods() -> List[Method]:
    methods: List[Method] = []

    def pc(a: np.ndarray, b: np.ndarray, search: int, constrain: str) -> Tuple[int, int, float]:
        dx, dy, peak = phase_correlation(a, b)
        dx = int(np.clip(dx, -search, search))
        dy = int(np.clip(dy, -search, search))
        if constrain == 'v':
            dx = 0
        elif constrain == 'h':
            dy = 0
        return dx, dy, peak

    def ncc(a: np.ndarray, b: np.ndarray, search: int, constrain: str) -> Tuple[int, int, float]:
        return brute_ncc(a, b, search=search, constrain=constrain)

    def ssim(a: np.ndarray, b: np.ndarray, search: int, constrain: str) -> Tuple[int, int, float]:
        return brute_ssim(a, b, search=search, constrain=constrain)

    methods.append(Method("phase", pc, None, min_score=0.05))
    methods.append(Method("ncc_gray", ncc, None, min_score=0.10))
    methods.append(Method("ncc_edge", ncc, edge_map, min_score=0.05))
    methods.append(Method("ssim", ssim, None, min_score=0.30))  # SSIMは0.3未満で足切り
    return methods


# -------------------------
# Compositing
# -------------------------

def _clip_overlap(mode: str, overlap: int, ph: int, pw: int, nh: int, nw: int) -> int:
    """overlap を安全な最大値に抑える"""
    if overlap < 0:
        return 0
    lim = (min(ph, nh) - 1) if mode == 'v' else (min(pw, nw) - 1)
    if lim < 0:
        return 0
    return int(min(overlap, lim))


def composite_two(prev_rgb: np.ndarray, next_rgb: np.ndarray, mode: str,
                  dx: int, dy: int, overlap: int,
                  min_overlap_ratio: float = 0.0) -> Optional[np.ndarray]:
    """
    2枚の画像を合成する。

    min_overlap_ratio > 0 の場合、実効オーバーラップが overlap * min_overlap_ratio 未満なら None を返す。
    """
    ph, pw, _ = prev_rgb.shape
    nh, nw, _ = next_rgb.shape

    overlap = _clip_overlap(mode, overlap, ph, pw, nh, nw)

    if mode == 'v':
        x = dx
        y = (ph - overlap) + dy
        effective = overlap - abs(dy)
    else:
        x = (pw - overlap) + dx
        y = dy
        effective = overlap - abs(dx)

    # 実効オーバーラップ比率チェック
    if overlap > 0 and min_overlap_ratio > 0:
        if effective < overlap * min_overlap_ratio:
            return None

    min_x = min(0, x)
    min_y = min(0, y)
    max_x = max(pw, x + nw)
    max_y = max(ph, y + nh)

    out_w = max_x - min_x
    out_h = max_y - min_y
    if out_w <= 0 or out_h <= 0:
        return None

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # prev paste (defensive clip)
    px0 = -min_x
    py0 = -min_y
    px1 = px0 + pw
    py1 = py0 + ph
    px0c, py0c = max(0, px0), max(0, py0)
    px1c, py1c = min(out_w, px1), min(out_h, py1)
    sx0, sy0 = px0c - px0, py0c - py0
    sx1, sy1 = sx0 + (px1c - px0c), sy0 + (py1c - py0c)
    if px0c < px1c and py0c < py1c and sx0 < sx1 and sy0 < sy1:
        out[py0c:py1c, px0c:px1c, :] = prev_rgb[sy0:sy1, sx0:sx1, :]

    # next paste (defensive clip)
    nx0 = x - min_x
    ny0 = y - min_y
    nx1 = nx0 + nw
    ny1 = ny0 + nh
    nx0c, ny0c = max(0, nx0), max(0, ny0)
    nx1c, ny1c = min(out_w, nx1), min(out_h, ny1)
    sx0, sy0 = nx0c - nx0, ny0c - ny0
    sx1, sy1 = sx0 + (nx1c - nx0c), sy0 + (ny1c - ny0c)
    if nx0c < nx1c and ny0c < ny1c and sx0 < sx1 and sy0 < sy1:
        out[ny0c:ny1c, nx0c:nx1c, :] = next_rgb[sy0:sy1, sx0:sx1, :]

    return out


# -------------------------
# Matching wrapper (with ignore regions)
# -------------------------

def match_pair(prev_rgb: np.ndarray, next_rgb: np.ndarray, mode: str, band: int, search: int,
               method: Method, ignore_regions: List[IgnoreRegion]) -> Tuple[int, int, float, str]:
    prev_gray = to_gray_f32(prev_rgb)
    next_gray = to_gray_f32(next_rgb)

    if ignore_regions:
        prev_gray = apply_ignore_regions_gray(prev_gray, ignore_regions)
        next_gray = apply_ignore_regions_gray(next_gray, ignore_regions)

    if mode == 'v':
        A, B = crop_band_vertical_from_gray(prev_gray, next_gray, band)
        constrain = 'v'
    else:
        A, B = crop_band_horizontal_from_gray(prev_gray, next_gray, band)
        constrain = 'h'

    if method.preprocess is not None:
        A = method.preprocess(A)
        B = method.preprocess(B)

    dx, dy, score = method.fn(A, B, search, constrain)
    label = f"{method.name}_band{band}_srch{search}_dx{dx}_dy{dy}_s{score:.4f}"
    return dx, dy, score, label


# -------------------------
# Candidate generation
# -------------------------

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def compute_overlaps_from_pct(pct_list: List[float], ref_size: int) -> List[int]:
    """パーセンテージリストからピクセル値のoverlapリストを計算"""
    overlaps = []
    for p in pct_list:
        if not (0.01 <= p <= 0.95):
            print(f"Warning: overlap-pct {p} out of range [0.01, 0.95], clipping")
            p = max(0.01, min(0.95, p))
        overlaps.append(int(round(ref_size * p)))
    return sorted(set(overlaps))  # 重複除去してソート


@dataclass
class AttemptInfo:
    """試行の情報（デバッグ用）"""
    overlap: int
    method_name: str
    band: int
    search: int
    step: int  # 何枚目で失敗したか (0=成功)
    min_score: float
    fail_reason: str  # "success", "low_score", "low_overlap", "composite_fail"


def stitch_linear_candidates(images: List[np.ndarray], mode: str, out_dir: Path,
                             overlaps: List[int], bands: List[int], searches: List[int],
                             methods: List[Method], ignore_regions: List[IgnoreRegion],
                             min_overlap_ratio: float, min_boundary_score: float = 0.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    base0 = images[0]

    success_count = 0
    attempts: List[AttemptInfo] = []

    for overlap in overlaps:
        for band in bands:
            for search in searches:
                for method in methods:
                    comp: Optional[np.ndarray] = base0
                    valid = True
                    labels = [f"ov{overlap}", method.name, f"band{band}", f"srch{search}"]
                    min_score_seen = float('inf')
                    fail_step = 0
                    fail_reason = "success"

                    for i in range(1, len(images)):
                        assert comp is not None
                        prev = comp
                        nxt = images[i]

                        dx, dy, score, label = match_pair(prev, nxt, mode, band, search, method, ignore_regions)
                        min_score_seen = min(min_score_seen, score)

                        # マッチングスコア足切り
                        if score < method.min_score:
                            valid = False
                            fail_step = i
                            fail_reason = "low_score"
                            break

                        ph, pw, _ = prev.shape
                        dx = int(np.clip(dx, -pw, pw))
                        dy = int(np.clip(dy, -ph, ph))

                        comp_next = composite_two(prev, nxt, mode, dx=dx, dy=dy, overlap=overlap,
                                                  min_overlap_ratio=min_overlap_ratio)
                        if comp_next is None:
                            valid = False
                            fail_step = i
                            fail_reason = "low_overlap"
                            break

                        # 境界一致度チェック
                        if min_boundary_score > 0:
                            boundary_sim = compute_boundary_similarity(prev, nxt, mode, dx, dy, overlap)
                            if boundary_sim < min_boundary_score:
                                valid = False
                                fail_step = i
                                fail_reason = "low_boundary"
                                break

                        comp = comp_next
                        labels.append(f"p{i}_dx{dx}_dy{dy}")

                    attempts.append(AttemptInfo(
                        overlap=overlap, method_name=method.name, band=band, search=search,
                        step=fail_step, min_score=min_score_seen, fail_reason=fail_reason
                    ))

                    if not valid or comp is None:
                        continue
                    out_path = out_dir / (f"{mode}_" + "__".join(labels).replace(":", "_") + ".png")
                    save_image(comp, out_path)
                    success_count += 1

        # constant delta variants
        for delta in range(-2, 3):
            if delta == 0:
                continue
            comp: Optional[np.ndarray] = base0
            valid = True
            for i in range(1, len(images)):
                assert comp is not None
                prev = comp
                nxt = images[i]
                comp_next = composite_two(prev, nxt, mode,
                                          dx=0 if mode == 'v' else delta,
                                          dy=delta if mode == 'v' else 0,
                                          overlap=overlap,
                                          min_overlap_ratio=min_overlap_ratio)
                if comp_next is None:
                    valid = False
                    break
                comp = comp_next
            if valid and comp is not None:
                out_path = out_dir / f"{mode}__ov{overlap}__constantDelta{delta}.png"
                save_image(comp, out_path)
                success_count += 1

    # 成功が0件の場合、惜しかったポイントを表示
    if success_count == 0:
        print("\nNo valid candidates found. Here are the best attempts:")
        # スコアでソート（高い順）
        attempts.sort(key=lambda a: a.min_score, reverse=True)
        for i, a in enumerate(attempts[:10]):
            print(f"  {i+1}. ov={a.overlap} {a.method_name} band={a.band} srch={a.search} "
                  f"score={a.min_score:.4f} failed_at=step{a.step} reason={a.fail_reason}")
        print("\nTips:")
        print("  - Try lower --min-overlap-ratio (e.g., 0.1)")
        print("  - Try different --overlap values")
        print("  - Check if images have sufficient overlap")


def reorder_snake(paths: List[Path], cols: int) -> List[Path]:
    if cols <= 0:
        raise ValueError("cols must be >= 1")
    rows = (len(paths) + cols - 1) // cols
    out: List[Path] = []
    for r in range(rows):
        row = paths[r * cols:(r + 1) * cols]
        if r % 2 == 1:
            row = list(reversed(row))
        out.extend(row)
    return out


def stitch_snake(images: List[np.ndarray], cols: int, out_dir: Path,
                 overlaps: List[int], bands: List[int], searches: List[int],
                 methods: List[Method], ignore_regions: List[IgnoreRegion],
                 min_overlap_ratio: float, min_boundary_score: float = 0.0) -> None:
    """スネークモード: 行ごとに水平連結 → 行同士を垂直連結"""
    rows = (len(images) + cols - 1) // cols

    # snake入力は既に並び替え済みなので、行合成時は毎行L->Rに戻す
    row_imgs: List[List[np.ndarray]] = []
    for r in range(rows):
        row = images[r * cols:(r + 1) * cols]
        if r % 2 == 1:
            row = list(reversed(row))
        row_imgs.append(row)

    tmp_row_dir = out_dir / "_rows"
    tmp_row_dir.mkdir(parents=True, exist_ok=True)

    row_candidates_per_setting: Dict[str, List[np.ndarray]] = {}

    for overlap in overlaps:
        for band in bands:
            for search in searches:
                for method in methods:
                    key = f"row__ov{overlap}__{method.name}__band{band}__srch{search}"
                    row_comps: List[np.ndarray] = []
                    ok = True
                    for r, row in enumerate(row_imgs):
                        comp: Optional[np.ndarray] = row[0]
                        for i in range(1, len(row)):
                            assert comp is not None
                            dx, dy, score, _ = match_pair(comp, row[i], "h", band, search, method, ignore_regions)
                            # マッチングスコア足切り
                            if score < method.min_score:
                                ok = False
                                break
                            ph, pw, _ = comp.shape
                            dx = int(np.clip(dx, -pw, pw))
                            dy = int(np.clip(dy, -ph, ph))
                            comp = composite_two(comp, row[i], "h", dx=dx, dy=dy, overlap=overlap,
                                                 min_overlap_ratio=min_overlap_ratio)
                            if comp is None:
                                ok = False
                                break
                            # 境界一致度チェック
                            if min_boundary_score > 0:
                                boundary_sim = compute_boundary_similarity(comp, row[i], "h", dx, dy, overlap)
                                if boundary_sim < min_boundary_score:
                                    ok = False
                                    break
                        if not ok or comp is None:
                            break
                        row_comps.append(comp)
                        save_image(comp, tmp_row_dir / f"{key}__r{r}.png")
                    if ok and row_comps:
                        row_candidates_per_setting[key] = row_comps

    final_dir = out_dir / "snake_final"
    final_dir.mkdir(parents=True, exist_ok=True)

    for key, row_comps in row_candidates_per_setting.items():
        parts = key.split("__")
        overlap = int(parts[1].replace("ov", ""))
        mname = parts[2]
        band = int(parts[3].replace("band", ""))
        search = int(parts[4].replace("srch", ""))

        method = next((m for m in methods if m.name == mname), methods[0])

        comp: Optional[np.ndarray] = row_comps[0]
        ok = True
        for r in range(1, len(row_comps)):
            assert comp is not None
            dx, dy, score, _ = match_pair(comp, row_comps[r], "v", band, search, method, ignore_regions)
            # マッチングスコア足切り
            if score < method.min_score:
                ok = False
                break
            ph, pw, _ = comp.shape
            dx = int(np.clip(dx, -pw, pw))
            dy = int(np.clip(dy, -ph, ph))
            comp = composite_two(comp, row_comps[r], "v", dx=dx, dy=dy, overlap=overlap,
                                 min_overlap_ratio=min_overlap_ratio)
            if comp is None:
                ok = False
                break
            # 境界一致度チェック
            if min_boundary_score > 0:
                boundary_sim = compute_boundary_similarity(comp, row_comps[r], "v", dx, dy, overlap)
                if boundary_sim < min_boundary_score:
                    ok = False
                    break

        if ok and comp is not None:
            save_image(comp, final_dir / f"snake__{key}.png")

    print(f"Done. Candidates in: {out_dir}")


# -------------------------
# Refine-from mode
# -------------------------

def parse_refine_meta(name: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    ファイル名からメタ情報を抽出する。
    例: v_ov120__phase__band40__srch20__p1_dx0_dy-3__p2_dx1_dy-2.png
    """
    ov_match = re.search(r'ov(\d+)', name)
    if not ov_match:
        raise ValueError(f"Cannot parse overlap from: {name}")
    ov = int(ov_match.group(1))

    steps = [(int(m.group(1)), int(m.group(2)))
             for m in re.finditer(r'dx(-?\d+)_dy(-?\d+)', name)]
    return ov, steps


def run_refine_from(images: List[np.ndarray], mode: str, out_dir: Path,
                    refine_from: str, refine_delta: int,
                    min_overlap_ratio: float) -> None:
    """
    既存候補の近傍を局所再探索する。
    マッチング処理は行わず、dx/dy/overlap を ±refine_delta の範囲で総当たりする。
    """
    base_name = Path(refine_from).name
    base_ov, base_steps = parse_refine_meta(base_name)

    if len(base_steps) != len(images) - 1:
        print(f"Warning: Found {len(base_steps)} steps in filename but have {len(images)} images")
        print(f"Adjusting to use {min(len(base_steps), len(images) - 1)} steps")

    out_dir.mkdir(parents=True, exist_ok=True)

    # overlap の範囲
    ov_min = max(0, base_ov - refine_delta)
    ov_max = base_ov + refine_delta + 1

    count = 0
    for ov in range(ov_min, ov_max):
        # 各ステップの dx/dy を再帰的に探索するのではなく、
        # 各ステップごとに独立して近傍を試す（組み合わせ爆発を避けるため）
        # ここでは全ステップを一度に処理する簡易版

        # 全ステップの dx/dy 組み合わせを生成すると爆発するので、
        # ステップ1つずつ順番に処理し、各ステップで候補を保存
        comp = images[0]
        step_labels: List[str] = []

        for step_idx, (dx0, dy0) in enumerate(base_steps[:len(images) - 1]):
            nxt = images[step_idx + 1]

            # このステップの近傍を探索
            for dx in range(dx0 - refine_delta, dx0 + refine_delta + 1):
                for dy in range(dy0 - refine_delta, dy0 + refine_delta + 1):
                    c = composite_two(comp, nxt, mode, dx, dy, ov, min_overlap_ratio)
                    if c is not None:
                        label = f"refine_ov{ov}_p{step_idx + 1}_dx{dx}_dy{dy}.png"
                        save_image(c, out_dir / label)
                        count += 1

            # 基準の dx/dy で合成を進める（次のステップ用）
            comp_next = composite_two(comp, nxt, mode, dx0, dy0, ov, min_overlap_ratio)
            if comp_next is None:
                break
            comp = comp_next
            step_labels.append(f"p{step_idx + 1}_dx{dx0}_dy{dy0}")

    print(f"Refine-from mode finished. Generated {count} candidates in: {out_dir}")


# -------------------------
# Overlap scan mode
# -------------------------

@dataclass
class ScanResult:
    """スキャン結果を保持"""
    overlap: int
    method_name: str
    band: int
    search: int
    match_score: float      # マッチングスコア（全ステップの最小値）
    boundary_score: float   # 境界一致度（全ステップの最小値）
    combined_score: float   # 複合スコア
    dx_dy_list: List[Tuple[int, int]]
    composite: np.ndarray


def _scan_overlaps_core(images: List[np.ndarray], mode: str,
                        overlaps: List[int], bands: List[int], searches: List[int],
                        methods: List[Method], ignore_regions: List[IgnoreRegion],
                        min_overlap_ratio: float, min_boundary_score: float,
                        verbose: bool = True) -> Tuple[List[ScanResult], List[AttemptInfo]]:
    """
    指定された overlap 値でスキャンし、結果を返す（コアロジック）。
    """
    results: List[ScanResult] = []
    failed_attempts: List[AttemptInfo] = []

    total = len(overlaps) * len(bands) * len(searches) * len(methods)
    count = 0

    for overlap in overlaps:
        for band in bands:
            for search in searches:
                for method in methods:
                    count += 1
                    if verbose and count % 50 == 0:
                        print(f"  Progress: {count}/{total} ({100*count//total}%)")

                    comp: Optional[np.ndarray] = images[0]
                    valid = True
                    min_match_score = float('inf')
                    min_boundary_seen = float('inf')
                    dx_dy_list: List[Tuple[int, int]] = []
                    fail_step = 0
                    fail_reason = "success"

                    for i in range(1, len(images)):
                        assert comp is not None
                        prev = comp
                        nxt = images[i]

                        dx, dy, score, _ = match_pair(prev, nxt, mode, band, search, method, ignore_regions)
                        min_match_score = min(min_match_score, score)

                        # マッチングスコア足切り
                        if score < method.min_score:
                            valid = False
                            fail_step = i
                            fail_reason = "low_score"
                            break

                        ph, pw, _ = prev.shape
                        dx = int(np.clip(dx, -pw, pw))
                        dy = int(np.clip(dy, -ph, ph))

                        # 境界一致度を計算
                        boundary_sim = compute_boundary_similarity(prev, nxt, mode, dx, dy, overlap)
                        min_boundary_seen = min(min_boundary_seen, boundary_sim)

                        # 境界一致度足切り
                        if min_boundary_score > 0 and boundary_sim < min_boundary_score:
                            valid = False
                            fail_step = i
                            fail_reason = "low_boundary"
                            break

                        comp_next = composite_two(prev, nxt, mode, dx=dx, dy=dy, overlap=overlap,
                                                  min_overlap_ratio=min_overlap_ratio)
                        if comp_next is None:
                            valid = False
                            fail_step = i
                            fail_reason = "low_overlap"
                            break

                        comp = comp_next
                        dx_dy_list.append((dx, dy))

                    if not valid or comp is None:
                        failed_attempts.append(AttemptInfo(
                            overlap=overlap, method_name=method.name, band=band, search=search,
                            step=fail_step, min_score=min_match_score, fail_reason=fail_reason
                        ))
                        continue

                    # 複合スコア: マッチングスコアと境界一致度の調和平均
                    if min_match_score > 0 and min_boundary_seen > 0:
                        combined = 2 * min_match_score * min_boundary_seen / (min_match_score + min_boundary_seen)
                    else:
                        combined = 0.0

                    results.append(ScanResult(
                        overlap=overlap,
                        method_name=method.name,
                        band=band,
                        search=search,
                        match_score=min_match_score,
                        boundary_score=min_boundary_seen,
                        combined_score=combined,
                        dx_dy_list=dx_dy_list,
                        composite=comp,
                    ))

    return results, failed_attempts


def run_overlap_scan(images: List[np.ndarray], mode: str, out_dir: Path,
                     overlap_min: int, overlap_max: int, overlap_step: int,
                     bands: List[int], searches: List[int],
                     methods: List[Method], ignore_regions: List[IgnoreRegion],
                     min_overlap_ratio: float, min_boundary_score: float, top_n: int) -> None:
    """
    overlap範囲をスキャンして、スコアが高い上位N件を出力する。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    overlaps = list(range(overlap_min, overlap_max + 1, overlap_step))

    print(f"Scanning overlaps: {overlap_min}-{overlap_max} step {overlap_step} ({len(overlaps)} values)")
    print(f"Methods: {[m.name for m in methods]}, Bands: {bands}, Searches: {searches}")

    results, failed_attempts = _scan_overlaps_core(
        images, mode, overlaps, bands, searches, methods, ignore_regions,
        min_overlap_ratio, min_boundary_score, verbose=True
    )

    # 有効な結果がない場合、惜しかったポイントを表示
    if len(results) == 0:
        print(f"\nNo valid candidates found. Here are the best attempts:")
        failed_attempts.sort(key=lambda a: a.min_score, reverse=True)
        for i, a in enumerate(failed_attempts[:10]):
            print(f"  {i+1}. ov={a.overlap} {a.method_name} band={a.band} srch={a.search} "
                  f"score={a.min_score:.4f} failed_at=step{a.step} reason={a.fail_reason}")
        print("\nTips:")
        print("  - Try lower --min-overlap-ratio (e.g., 0.1)")
        print("  - Try wider --overlap-scan range")
        print("  - Check if images have sufficient overlap")
        return

    # 複合スコアでソートして上位N件を出力
    results.sort(key=lambda r: r.combined_score, reverse=True)
    top_results = results[:top_n]

    print(f"\nTop {len(top_results)} results (out of {len(results)} valid):")
    for rank, r in enumerate(top_results, 1):
        steps = "_".join(f"p{i+1}_dx{dx}_dy{dy}" for i, (dx, dy) in enumerate(r.dx_dy_list))
        filename = f"scan_rank{rank:02d}_score{r.combined_score:.4f}_ov{r.overlap}__{r.method_name}__band{r.band}__srch{r.search}__{steps}.png"
        save_image(r.composite, out_dir / filename)
        print(f"  {rank}. score={r.combined_score:.4f} (match={r.match_score:.4f}, boundary={r.boundary_score:.4f}) ov={r.overlap} {r.method_name}")

    print(f"\nScan mode finished. Top {len(top_results)} candidates in: {out_dir}")


def run_overlap_auto(images: List[np.ndarray], mode: str, out_dir: Path,
                     bands: List[int], searches: List[int],
                     methods: List[Method], ignore_regions: List[IgnoreRegion],
                     min_overlap_ratio: float, min_boundary_score: float, top_n: int) -> None:
    """
    3段階の階層的スキャンでoverlapを自動探索する。

    1. 0〜画像高さ、ステップ100で粗く走査 → 上位3エリアを特定
    2. 各エリア周辺をステップ10で走査 → 上位3エリアを特定
    3. 各エリア周辺をステップ1で走査 → 最終結果を出力
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 画像サイズから探索範囲を決定
    h, w, _ = images[0].shape
    max_overlap = h if mode == "v" else w

    print(f"=== Auto Overlap Scan ===")
    print(f"Image size: {w}x{h}, Max overlap: {max_overlap}")
    print(f"Methods: {[m.name for m in methods]}, Bands: {bands}, Searches: {searches}")

    # --- 第1段階: ステップ100で粗く走査 ---
    step1 = 100
    overlaps1 = list(range(0, max_overlap + 1, step1))
    print(f"\n[Stage 1] Coarse scan: 0-{max_overlap} step {step1} ({len(overlaps1)} values)")

    results1, _ = _scan_overlaps_core(
        images, mode, overlaps1, bands, searches, methods, ignore_regions,
        min_overlap_ratio, min_boundary_score, verbose=True
    )

    if len(results1) == 0:
        print("No valid candidates found in stage 1. Try adjusting parameters.")
        return

    # 上位3エリア（overlap値）を特定
    results1.sort(key=lambda r: r.combined_score, reverse=True)
    top3_overlaps1 = sorted(set(r.overlap for r in results1[:3]))
    print(f"  Top overlaps from stage 1: {top3_overlaps1}")

    # --- 第2段階: ステップ10で中間走査 ---
    step2 = 10
    half_range2 = 50  # 前後50pxを探索
    overlaps2_set: set = set()
    for ov in top3_overlaps1:
        for o in range(max(0, ov - half_range2), min(max_overlap, ov + half_range2) + 1, step2):
            overlaps2_set.add(o)
    overlaps2 = sorted(overlaps2_set)
    print(f"\n[Stage 2] Medium scan: {len(overlaps2)} values around {top3_overlaps1}")

    results2, _ = _scan_overlaps_core(
        images, mode, overlaps2, bands, searches, methods, ignore_regions,
        min_overlap_ratio, min_boundary_score, verbose=True
    )

    if len(results2) == 0:
        print("No valid candidates found in stage 2.")
        return

    # 上位3エリアを特定
    results2.sort(key=lambda r: r.combined_score, reverse=True)
    top3_overlaps2 = sorted(set(r.overlap for r in results2[:3]))
    print(f"  Top overlaps from stage 2: {top3_overlaps2}")

    # --- 第3段階: ステップ1で精密走査 ---
    step3 = 1
    half_range3 = 5  # 前後5pxを探索
    overlaps3_set: set = set()
    for ov in top3_overlaps2:
        for o in range(max(0, ov - half_range3), min(max_overlap, ov + half_range3) + 1, step3):
            overlaps3_set.add(o)
    overlaps3 = sorted(overlaps3_set)
    print(f"\n[Stage 3] Fine scan: {len(overlaps3)} values around {top3_overlaps2} (range {overlaps3[0]}-{overlaps3[-1]})")

    results3, _ = _scan_overlaps_core(
        images, mode, overlaps3, bands, searches, methods, ignore_regions,
        min_overlap_ratio, min_boundary_score, verbose=True
    )

    if len(results3) == 0:
        print("No valid candidates found in stage 3.")
        return

    # 最終結果を出力
    results3.sort(key=lambda r: r.combined_score, reverse=True)
    top_results = results3[:top_n]

    print(f"\n=== Final Results: Top {len(top_results)} (out of {len(results3)} valid) ===")
    for rank, r in enumerate(top_results, 1):
        steps = "_".join(f"p{i+1}_dx{dx}_dy{dy}" for i, (dx, dy) in enumerate(r.dx_dy_list))
        filename = f"auto_rank{rank:02d}_score{r.combined_score:.4f}_ov{r.overlap}__{r.method_name}__band{r.band}__srch{r.search}__{steps}.png"
        save_image(r.composite, out_dir / filename)
        print(f"  {rank}. score={r.combined_score:.4f} (match={r.match_score:.4f}, boundary={r.boundary_score:.4f}) ov={r.overlap} {r.method_name}")

    print(f"\nAuto scan finished. Top {len(top_results)} candidates in: {out_dir}")


# -------------------------
# Main
# -------------------------

WORKFLOW_HELP = """\
Recommended Workflow:
  1. Rough search:   --overlap 80,120 --band 30,50 --search 10,20
                     or --overlap-pct 0.1,0.15,0.2 (10-20% of image size)
  2. Visual inspect: Find nearly-correct candidate (filename has dx/dy)
  3. Refine:         --refine-from <candidate.png> --refine-delta 2

  Alternative: Use --overlap-scan or --overlap-auto to auto-find best overlap

Examples:
  python stitch_candidates.py -m v -o out --overlap 80,120 img/*.png
  python stitch_candidates.py -m v -o out --overlap-pct 0.1,0.15 img/*.png
  python stitch_candidates.py -m v -o out --overlap-scan 50,150,5 --top-n 5 img/*.png
  python stitch_candidates.py -m v -o out --overlap-auto --top-n 5 img/*.png
  python stitch_candidates.py -m v -o refine --refine-from out/v_ov120__phase__*.png --refine-delta 2 img/*.png
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate stitching candidates (vertical/horizontal/snake) from N images.",
        epilog=WORKFLOW_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("images", nargs="+", help="Input image files (ordered). Wildcards supported on Windows.")
    ap.add_argument("-m", "--mode", choices=["v", "h", "snake"], default="v",
                    help="v=vertical, h=horizontal, snake=zigzag (needs --cols).")
    ap.add_argument("-o", "--out", default="out_candidates", help="Output directory.")
    ap.add_argument("--cols", type=int, default=0, help="Columns for snake mode (required when mode=snake).")

    ap.add_argument("--overlap", default=None, help="Overlap pixels along main axis (comma list).")
    ap.add_argument("--overlap-pct", default=None,
                    help="Overlap as ratio 0.01-0.95 (comma list). E.g., 0.1,0.15,0.2")
    ap.add_argument("--band", default="20,30,50", help="Band pixels used for matching (comma list).")
    ap.add_argument("--search", default="5,10,20", help="Search range (px) for matching (comma list).")

    ap.add_argument("--ignore", action="append", default=[],
                    help='Ignore region (px): "top,(right|left|X),width,height"')
    ap.add_argument("--ignore-pct", action="append", default=[],
                    help='Ignore region (%%): "top,(right|left|X),width,height"')

    ap.add_argument("--min-overlap-ratio", type=float, default=0.3,
                    help="Minimum effective overlap ratio (0.0-1.0). Candidates below this are skipped. Default: 0.3")
    ap.add_argument("--min-boundary-score", type=float, default=0.3,
                    help="Minimum boundary similarity (SSIM) score (0.0-1.0). Candidates below this are skipped. Default: 0.3")

    # Refine-from mode
    ap.add_argument("--refine-from", default=None,
                    help="Path to a candidate image to refine from (enables refine mode).")
    ap.add_argument("--refine-delta", type=int, default=2,
                    help="Search range ±n px around base dx/dy/overlap in refine mode.")

    # Overlap scan mode
    ap.add_argument("--overlap-scan", default=None,
                    help="Scan overlap range: MIN,MAX,STEP (e.g., 50,150,5). Outputs top N by score.")
    ap.add_argument("--overlap-auto", action="store_true",
                    help="Auto scan: 3-stage hierarchical search (step 100 -> 10 -> 1).")
    ap.add_argument("--top-n", type=int, default=6,
                    help="Number of top candidates to output in scan/auto mode. Default: 5")

    # Method selection
    ap.add_argument("--exclude-method", default=None,
                    help="Exclude matching methods (comma list). Available: phase,ncc_gray,ncc_edge,ssim")

    args = ap.parse_args()
    out_dir = Path(args.out)

    # Windows-safe wildcard expansion
    paths = expand_images(args.images)
    if len(paths) < 2:
        sys.exit("Need at least two images")

    if args.mode == "snake":
        if args.cols <= 0:
            sys.exit("mode=snake requires --cols N")
        paths = reorder_snake(paths, args.cols)

    imgs = [load_image_rgb(p) for p in paths]
    print(f"Loaded {len(imgs)} images")

    # Refine-from mode
    if args.refine_from:
        run_refine_from(
            images=imgs,
            mode=args.mode if args.mode != "snake" else "v",
            out_dir=out_dir,
            refine_from=args.refine_from,
            refine_delta=args.refine_delta,
            min_overlap_ratio=args.min_overlap_ratio,
        )
        return

    # Overlap scan mode
    if args.overlap_scan:
        parts = [int(x.strip()) for x in args.overlap_scan.split(",")]
        if len(parts) != 3:
            sys.exit("--overlap-scan requires MIN,MAX,STEP (e.g., 50,150,5)")
        overlap_min, overlap_max, overlap_step = parts

        bands = parse_int_list(args.band)
        searches = parse_int_list(args.search)
        methods = build_methods()

        # Exclude methods
        if args.exclude_method:
            exclude = [x.strip().lower() for x in args.exclude_method.split(",")]
            methods = [m for m in methods if m.name.lower() not in exclude]
            if not methods:
                sys.exit("All methods excluded. Available: phase, ncc_gray, ncc_edge")
            print(f"Using methods: {[m.name for m in methods]}")

        ignore_regions: List[IgnoreRegion] = []
        for s in args.ignore:
            ignore_regions.append(parse_ignore_region(s, unit="px"))
        for s in args.ignore_pct:
            ignore_regions.append(parse_ignore_region(s, unit="pct"))

        if args.mode == "snake":
            sys.exit("--overlap-scan does not support snake mode yet")

        run_overlap_scan(
            images=imgs,
            mode=args.mode,
            out_dir=out_dir,
            overlap_min=overlap_min,
            overlap_max=overlap_max,
            overlap_step=overlap_step,
            bands=bands,
            searches=searches,
            methods=methods,
            ignore_regions=ignore_regions,
            min_overlap_ratio=args.min_overlap_ratio,
            min_boundary_score=args.min_boundary_score,
            top_n=args.top_n,
        )
        return

    # Overlap auto mode (3-stage hierarchical search)
    if args.overlap_auto:
        bands = parse_int_list(args.band)
        searches = parse_int_list(args.search)
        methods = build_methods()

        # Exclude methods
        if args.exclude_method:
            exclude = [x.strip().lower() for x in args.exclude_method.split(",")]
            methods = [m for m in methods if m.name.lower() not in exclude]
            if not methods:
                sys.exit("All methods excluded. Available: phase, ncc_gray, ncc_edge")
            print(f"Using methods: {[m.name for m in methods]}")

        ignore_regions: List[IgnoreRegion] = []
        for s in args.ignore:
            ignore_regions.append(parse_ignore_region(s, unit="px"))
        for s in args.ignore_pct:
            ignore_regions.append(parse_ignore_region(s, unit="pct"))

        if args.mode == "snake":
            sys.exit("--overlap-auto does not support snake mode yet")

        run_overlap_auto(
            images=imgs,
            mode=args.mode,
            out_dir=out_dir,
            bands=bands,
            searches=searches,
            methods=methods,
            ignore_regions=ignore_regions,
            min_overlap_ratio=args.min_overlap_ratio,
            min_boundary_score=args.min_boundary_score,
            top_n=args.top_n,
        )
        return

    # If no overlap option specified at all, default to overlap-auto
    if not args.overlap_pct and not args.overlap:
        print("No --overlap/--overlap-pct specified, defaulting to --overlap-auto mode")
        bands = parse_int_list(args.band)
        searches = parse_int_list(args.search)
        methods = build_methods()

        if args.exclude_method:
            exclude = [x.strip().lower() for x in args.exclude_method.split(",")]
            methods = [m for m in methods if m.name.lower() not in exclude]
            if not methods:
                sys.exit("All methods excluded. Available: phase, ncc_gray, ncc_edge")
            print(f"Using methods: {[m.name for m in methods]}")

        ignore_regions: List[IgnoreRegion] = []
        for s in args.ignore:
            ignore_regions.append(parse_ignore_region(s, unit="px"))
        for s in args.ignore_pct:
            ignore_regions.append(parse_ignore_region(s, unit="pct"))

        if args.mode == "snake":
            sys.exit("--overlap-auto does not support snake mode yet")

        run_overlap_auto(
            images=imgs,
            mode=args.mode,
            out_dir=out_dir,
            bands=bands,
            searches=searches,
            methods=methods,
            ignore_regions=ignore_regions,
            min_overlap_ratio=args.min_overlap_ratio,
            min_boundary_score=args.min_boundary_score,
            top_n=args.top_n,
        )
        return

    # Normal mode
    # Compute overlaps (px or pct)
    if args.overlap_pct:
        pct_list = parse_float_list(args.overlap_pct)
        # Use first image size as reference
        h, w, _ = imgs[0].shape
        ref_size = h if args.mode == "v" else w
        overlaps = compute_overlaps_from_pct(pct_list, ref_size)
        print(f"overlap-pct {pct_list} -> {overlaps} px (ref={ref_size})")
    else:
        overlaps = parse_int_list(args.overlap)

    bands = parse_int_list(args.band)
    searches = parse_int_list(args.search)
    methods = build_methods()

    # Exclude methods
    if args.exclude_method:
        exclude = [x.strip().lower() for x in args.exclude_method.split(",")]
        methods = [m for m in methods if m.name.lower() not in exclude]
        if not methods:
            sys.exit("All methods excluded. Available: phase, ncc_gray, ncc_edge")
        print(f"Using methods: {[m.name for m in methods]}")

    ignore_regions: List[IgnoreRegion] = []
    for s in args.ignore:
        ignore_regions.append(parse_ignore_region(s, unit="px"))
    for s in args.ignore_pct:
        ignore_regions.append(parse_ignore_region(s, unit="pct"))

    if args.mode == "snake":
        stitch_snake(
            images=imgs,
            cols=args.cols,
            out_dir=out_dir,
            overlaps=overlaps,
            bands=bands,
            searches=searches,
            methods=methods,
            ignore_regions=ignore_regions,
            min_overlap_ratio=args.min_overlap_ratio,
            min_boundary_score=args.min_boundary_score,
        )
    else:
        stitch_linear_candidates(
            images=imgs,
            mode=args.mode,
            out_dir=out_dir,
            overlaps=overlaps,
            bands=bands,
            searches=searches,
            methods=methods,
            ignore_regions=ignore_regions,
            min_overlap_ratio=args.min_overlap_ratio,
            min_boundary_score=args.min_boundary_score,
        )
        print(f"Done. Candidates in: {out_dir}")


if __name__ == "__main__":
    main()
