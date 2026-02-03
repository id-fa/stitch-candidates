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

# Import matching functions from stitch_candidates.py for keyframe stitching
try:
    from stitch_candidates import (
        build_methods as sc_build_methods,
        match_pair as sc_match_pair,
        composite_two as sc_composite_two,
        compute_boundary_similarity as sc_compute_boundary_similarity,
        Method as SCMethod,
    )
    HAS_STITCH_CANDIDATES = True
except ImportError:
    HAS_STITCH_CANDIDATES = False


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


def apply_text_mask(gray: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """
    テキスト領域をマスクして0にする（dy推定から除外）

    Args:
        gray: グレースケール画像 (H, W)
        text_mask: テキスト領域マスク (H, W), True=テキスト領域

    Returns:
        テキスト領域が0になったグレースケール画像
    """
    if text_mask is None:
        return gray
    out = gray.copy()
    out[text_mask] = 0.0
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


def _mad_outlier_filter(values: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    MAD (Median Absolute Deviation) ベースの外れ値フィルタ。
    中央値から threshold * MAD 以上離れた値を除外。
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-6:
        mad = 1.0  # MADが0の場合（全て同じ値）
    mask = np.abs(values - median) <= threshold * mad * 1.4826  # 1.4826 = 正規分布のスケール係数
    return mask


def _histogram_mode(values: np.ndarray, bin_width: float = 1.0) -> float:
    """
    ヒストグラムのモード（最頻値）を推定。
    サブピクセル精度のために、ピーク周辺の加重平均を返す。
    """
    if len(values) == 0:
        return 0.0

    # ヒストグラムを作成
    min_val, max_val = np.min(values), np.max(values)
    if max_val - min_val < bin_width:
        return float(np.median(values))

    n_bins = int((max_val - min_val) / bin_width) + 1
    n_bins = max(10, min(n_bins, 200))  # 10〜200ビン

    hist, edges = np.histogram(values, bins=n_bins)

    # ピークを見つける
    peak_idx = np.argmax(hist)

    # ピーク周辺3ビンの加重平均でサブピクセル精度を得る
    start = max(0, peak_idx - 1)
    end = min(len(hist), peak_idx + 2)

    weights = hist[start:end]
    centers = (edges[start:end] + edges[start+1:end+1]) / 2

    if np.sum(weights) > 0:
        mode = np.average(centers, weights=weights)
    else:
        mode = (edges[peak_idx] + edges[peak_idx + 1]) / 2

    return float(mode)


def optical_flow_estimate(prev_rgb: np.ndarray, next_rgb: np.ndarray,
                          mask: np.ndarray = None,
                          method: str = "farneback") -> Tuple[float, float, float]:
    """
    Optical flowでフレーム間のシフト(dx, dy)を推定する。

    Args:
        prev_rgb: 前フレーム (H, W, 3)
        next_rgb: 次フレーム (H, W, 3)
        mask: 有効領域マスク (H, W), True=使用, None=全体
        method: "farneback" (密) or "lk" (疎、Lucas-Kanade)

    Returns:
        (dx, dy, confidence)
        - dx, dy: フレーム全体の推定シフト量
        - confidence: 推定の信頼度 (0-1)
    """
    if not HAS_CV2:
        # OpenCVがない場合はphase correlationにフォールバック
        gray_a = to_gray_f32(prev_rgb)
        gray_b = to_gray_f32(next_rgb)
        dx, dy, peak = phase_correlation(gray_a, gray_b)
        return float(dx), float(dy), peak

    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)

    if method == "farneback":
        # 密なオプティカルフロー（Farneback法）
        h, w = prev_gray.shape
        min_dim = min(h, w)

        # 位相相関で大まかなシフトを事前推定（大きなシフトへの対応）
        gray_a_f32 = prev_gray.astype(np.float32) / 255.0
        gray_b_f32 = next_gray.astype(np.float32) / 255.0
        edge_a = sobel_edge(gray_a_f32)
        edge_b = sobel_edge(gray_b_f32)
        init_dx, init_dy, _ = phase_correlation(edge_a, edge_b)

        # 初期フローを設定（位相相関の結果を使用）
        init_flow = np.zeros((h, w, 2), dtype=np.float32)
        init_flow[:, :, 0] = float(init_dx)
        init_flow[:, :, 1] = float(init_dy)

        # ピラミッドレベル: 画像の最小寸法に基づいて決定
        levels = max(5, min(8, int(np.log2(min_dim / 16))))

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            init_flow,  # 位相相関による初期フロー
            pyr_scale=0.5,  # ピラミッドスケール
            levels=levels,  # ピラミッドレベル数（動的設定）
            winsize=25,     # 窓サイズ
            iterations=7,   # 各レベルでの反復回数
            poly_n=7,       # ピクセル近傍サイズ
            poly_sigma=1.5, # ガウシアン標準偏差
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )
        # flow: (H, W, 2) - [dx, dy] per pixel

        # マスクを適用
        if mask is not None:
            valid = mask
        else:
            valid = np.ones(prev_gray.shape, dtype=bool)

        # 有効なフローベクトルを取得
        flow_x = flow[:, :, 0][valid]
        flow_y = flow[:, :, 1][valid]

        if len(flow_x) == 0:
            return 0.0, 0.0, 0.0

        # フローの大きさでフィルタリング（極端に小さい/大きいものを除外）
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        mag_median = np.median(magnitude)

        # 大きさが中央値の0.1〜10倍の範囲内のものを使用
        if mag_median > 0.5:  # 動きがある場合のみフィルタ
            mag_valid = (magnitude > mag_median * 0.1) & (magnitude < mag_median * 10)
            flow_x = flow_x[mag_valid]
            flow_y = flow_y[mag_valid]

        if len(flow_x) < 100:
            # フィルタ後のサンプルが少なすぎる場合は全体を使用
            flow_x = flow[:, :, 0][valid]
            flow_y = flow[:, :, 1][valid]

        # MADベースの外れ値除去
        valid_x = _mad_outlier_filter(flow_x, threshold=2.5)
        valid_y = _mad_outlier_filter(flow_y, threshold=2.5)
        valid_both = valid_x & valid_y

        if np.sum(valid_both) > 50:
            flow_x_filtered = flow_x[valid_both]
            flow_y_filtered = flow_y[valid_both]
        else:
            flow_x_filtered = flow_x
            flow_y_filtered = flow_y

        # ヒストグラムモードでロバスト推定
        dx = _histogram_mode(flow_x_filtered, bin_width=0.5)
        dy = _histogram_mode(flow_y_filtered, bin_width=0.5)

        # 信頼度計算
        # 1. フローの一貫性（MAD後の標準偏差）
        std_x = float(np.std(flow_x_filtered))
        std_y = float(np.std(flow_y_filtered))
        consistency = 1.0 / (1.0 + (std_x + std_y) / 4.0)

        # 2. フローの大きさ（動きが検出されているか）
        mag_score = min(1.0, (abs(dx) + abs(dy)) / 10.0)

        # 3. 有効ピクセル比率
        valid_ratio = len(flow_x_filtered) / max(1, len(flow_x))

        confidence = consistency * (0.3 + 0.7 * mag_score) * (0.5 + 0.5 * valid_ratio)

        return dx, dy, confidence

    elif method == "lk":
        # 疎なオプティカルフロー（Lucas-Kanade法）
        # 特徴点を検出（パラメータ強化）
        feature_params = dict(
            maxCorners=500,      # 増加
            qualityLevel=0.005,  # 下げて多くの特徴点を検出
            minDistance=7,       # 減少
            blockSize=11         # 増加
        )

        # マスクがある場合は適用
        detect_mask = None
        if mask is not None:
            detect_mask = (mask.astype(np.uint8) * 255)

        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=detect_mask, **feature_params)

        if p0 is None or len(p0) < 10:
            return 0.0, 0.0, 0.0

        # Lucas-Kanadeでトラッキング（大きなシフトに対応）
        # 画像サイズに基づいてmaxLevelを決定
        h, w = prev_gray.shape
        max_level = max(4, min(7, int(np.log2(min(h, w) / 32))))

        lk_params = dict(
            winSize=(41, 41),   # 大きな窓サイズ
            maxLevel=max_level, # 動的ピラミッドレベル
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        )
        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

        # 逆方向トラッキングでバリデーション
        p0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, p1, None, **lk_params)

        # 前後で一貫性のある点のみ使用
        if p0_back is not None:
            p0_flat = p0.reshape(-1, 2)
            p0_back_flat = p0_back.reshape(-1, 2)
            back_error = np.sqrt(np.sum((p0_flat - p0_back_flat)**2, axis=1))
            good_back = back_error < 1.0  # 1ピクセル以内の誤差
            status = status.flatten() & good_back.astype(np.uint8)

        # 成功したトラッキングのみ使用
        good_old = p0[status.flatten() == 1]
        good_new = p1[status.flatten() == 1]

        if len(good_old) < 5:
            return 0.0, 0.0, 0.0

        # シフト量を計算 (形状を (N, 2) に正規化)
        good_old = good_old.reshape(-1, 2)
        good_new = good_new.reshape(-1, 2)
        shifts = good_new - good_old

        # MADベースの外れ値除去
        valid_x = _mad_outlier_filter(shifts[:, 0], threshold=2.5)
        valid_y = _mad_outlier_filter(shifts[:, 1], threshold=2.5)
        valid_both = valid_x & valid_y

        if np.sum(valid_both) >= 5:
            shifts_filtered = shifts[valid_both]
        else:
            shifts_filtered = shifts

        # ヒストグラムモードでロバスト推定
        dx = _histogram_mode(shifts_filtered[:, 0], bin_width=0.5)
        dy = _histogram_mode(shifts_filtered[:, 1], bin_width=0.5)

        # 信頼度計算
        success_rate = len(good_old) / len(p0)
        std_x = float(np.std(shifts_filtered[:, 0]))
        std_y = float(np.std(shifts_filtered[:, 1]))
        consistency = 1.0 / (1.0 + (std_x + std_y) / 4.0)

        # 動きの大きさも考慮
        mag_score = min(1.0, (abs(dx) + abs(dy)) / 10.0)

        confidence = success_rate * consistency * (0.3 + 0.7 * mag_score)

        return dx, dy, confidence

    else:
        raise ValueError(f"Unknown optical flow method: {method}")


def optical_flow_estimate_robust(prev_rgb: np.ndarray, next_rgb: np.ndarray,
                                  text_mask: np.ndarray = None) -> Tuple[float, float, float]:
    """
    ロバストなオプティカルフロー推定。
    複数の手法を組み合わせて信頼度の高い結果を返す。
    位相相関も参照として使用。

    Args:
        prev_rgb, next_rgb: フレーム
        text_mask: テキスト領域マスク (True=テキスト、除外)

    Returns:
        (dx, dy, confidence)
    """
    # 有効領域マスク（テキスト領域を除外）
    if text_mask is not None:
        valid_mask = ~text_mask
    else:
        valid_mask = None

    # 位相相関で参照値を取得（信頼性が高い傾向）
    gray_a = to_gray_f32(prev_rgb)
    gray_b = to_gray_f32(next_rgb)
    if text_mask is not None:
        gray_a = apply_text_mask(gray_a, text_mask)
        gray_b = apply_text_mask(gray_b, text_mask)
    edge_a = sobel_edge(gray_a)
    edge_b = sobel_edge(gray_b)
    dx_phase, dy_phase, conf_phase = phase_correlation(edge_a, edge_b)

    # Farneback法で推定
    dx_fb, dy_fb, conf_fb = optical_flow_estimate(prev_rgb, next_rgb, valid_mask, method="farneback")

    # LK法で推定
    dx_lk, dy_lk, conf_lk = optical_flow_estimate(prev_rgb, next_rgb, valid_mask, method="lk")

    # 候補をリスト化
    candidates = [
        (dx_fb, dy_fb, conf_fb, "farneback"),
        (dx_lk, dy_lk, conf_lk, "lk"),
        (float(dx_phase), float(dy_phase), conf_phase * 0.8, "phase"),  # 位相相関は参照
    ]

    # 信頼度でソート
    candidates.sort(key=lambda x: x[2], reverse=True)

    # 一致度チェック: 上位2つが近い値なら信頼度を上げる
    dx1, dy1, conf1, method1 = candidates[0]
    dx2, dy2, conf2, method2 = candidates[1]

    if abs(dx1 - dx2) < 3 and abs(dy1 - dy2) < 3:
        # 上位2つが一致: 加重平均
        total_conf = conf1 + conf2
        if total_conf > 0:
            dx = (dx1 * conf1 + dx2 * conf2) / total_conf
            dy = (dy1 * conf1 + dy2 * conf2) / total_conf
            confidence = (conf1 + conf2) / 2 * 1.2  # ボーナス
            confidence = min(confidence, 1.0)
        else:
            dx, dy, confidence = dx1, dy1, conf1
    else:
        # 不一致: 最も信頼度が高いものを採用
        # ただし、位相相関と一致している場合は信頼度を上げる
        if method1 != "phase" and abs(dx1 - dx_phase) < 5 and abs(dy1 - dy_phase) < 5:
            confidence = conf1 * 1.1
            confidence = min(confidence, 1.0)
        else:
            confidence = conf1

        dx, dy = dx1, dy1

    return dx, dy, confidence


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


def find_template_candidates(rgb: np.ndarray, n_candidates: int = 20,
                              exclude_center: bool = True,
                              template_size: Tuple[int, int] = (150, 150),
                              step: int = 40) -> List[Tuple[float, int, int, int, int]]:
    """
    テンプレート候補領域を探索し、スコア付きで返す。

    Returns: [(score, y1, y2, x1, x2), ...] スコア降順
    """
    gray = to_gray_f32(rgb)
    edge = sobel_edge(gray)

    h, w = edge.shape
    template_h, template_w = template_size

    # マスク: 中央のテキスト領域を除外
    mask = np.ones_like(edge)
    if exclude_center:
        cx = w // 2
        mask[:, int(cx - w * 0.30):int(cx + w * 0.30)] = 0

    # 上下端を除外（画面外に出やすい）
    margin_y = int(h * 0.10)
    mask[:margin_y, :] = 0
    mask[-margin_y:, :] = 0

    candidates = []
    for y in range(0, h - template_h, step):
        for x in range(0, w - template_w, step):
            region_mask = mask[y:y + template_h, x:x + template_w]
            if region_mask.mean() < 0.8:
                continue
            region_edge = edge[y:y + template_h, x:x + template_w]
            score = float(region_edge.mean())
            # 左右端にボーナス
            if x < w * 0.25 or x > w * 0.75:
                score *= 1.3
            candidates.append((score, y, y + template_h, x, x + template_w))

    # スコア順にソート
    candidates.sort(reverse=True)

    # 重複を避けて選択
    selected = []
    min_dist = 80
    for score, y1, y2, x1, x2 in candidates:
        too_close = False
        for _, sy1, sy2, sx1, sx2 in selected:
            cy1, cx1 = (y1 + y2) // 2, (x1 + x2) // 2
            cy2, cx2 = (sy1 + sy2) // 2, (sx1 + sx2) // 2
            dist = ((cy1 - cy2) ** 2 + (cx1 - cx2) ** 2) ** 0.5
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            selected.append((score, y1, y2, x1, x2))
            if len(selected) >= n_candidates:
                break

    return selected


def evaluate_template_candidate(frames_rgb: List[np.ndarray],
                                 region: Tuple[int, int, int, int],
                                 max_frames: int = None) -> Dict:
    """
    1つのテンプレート候補を全フレームで追跡検証する。

    Returns: {
        'region': (y1, y2, x1, x2),
        'mean_score': float,
        'min_score': float,
        'std_score': float,
        'lost_ratio': float,  # スコア < 0.7 の割合
        'dy_std': float,      # dy推定のばらつき
        'stability': float,   # 総合安定性スコア (0-1)
    }
    """
    y1, y2, x1, x2 = region
    n_frames = len(frames_rgb)
    if max_frames and n_frames > max_frames:
        # サンプリング
        indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
        frames_sample = [frames_rgb[i] for i in indices]
    else:
        frames_sample = frames_rgb

    scores = []
    dys = []
    offsets = [(0, 0)]

    for i in range(1, len(frames_sample)):
        prev_rgb = frames_sample[i - 1]
        next_rgb = frames_sample[i]

        dy, score, new_offsets = template_match_dy(
            prev_rgb, next_rgb,
            [(y1, y2, x1, x2)],
            offsets
        )
        offsets = new_offsets
        scores.append(score)
        dys.append(dy)

    if not scores:
        return {
            'region': region,
            'mean_score': 0.0,
            'min_score': 0.0,
            'std_score': 0.0,
            'lost_ratio': 1.0,
            'dy_std': 0.0,
            'stability': 0.0,
        }

    mean_score = float(np.mean(scores))
    min_score = float(np.min(scores))
    std_score = float(np.std(scores))
    lost_ratio = sum(1 for s in scores if s < 0.7) / len(scores)

    nonzero_dys = [d for d in dys if d != 0]
    dy_std = float(np.std(nonzero_dys)) if len(nonzero_dys) > 1 else 0.0

    # 総合安定性スコア (0-1)
    # - mean_score が高いほど良い
    # - lost_ratio が低いほど良い
    # - dy_std が低いほど良い（一定の動きを期待）
    stability = mean_score * (1 - lost_ratio) * max(0, 1 - dy_std / 50)
    stability = min(1.0, max(0.0, stability))

    return {
        'region': region,
        'mean_score': mean_score,
        'min_score': min_score,
        'std_score': std_score,
        'lost_ratio': lost_ratio,
        'dy_std': dy_std,
        'stability': stability,
    }


# -------------------------
# Jigsaw reconstruction (text-aware)
# -------------------------

@dataclass
class FrameTextInfo:
    """フレームのテキスト情報"""
    frame_idx: int
    text_score: float          # 0=クリーン, 1=テキスト満載
    clean_row_mask: np.ndarray # True=クリーンな行
    clean_ratio: float         # クリーンな行の割合
    edge_map: np.ndarray       # エッジマップ
    text_mask: np.ndarray      # True=テキスト領域（2D bool配列）


def analyze_frame_text(rgb: np.ndarray, edge_thr: float = 0.15,
                       row_edge_thr: float = 0.08,
                       dilate_size: int = 5) -> FrameTextInfo:
    """
    フレームのテキスト領域を分析する。

    - 全体のテキストスコア（エッジ密度）
    - 行ごとのクリーン判定（テキスト行間を検出）
    - テキスト領域マスク（2Dブール配列）

    dilate_size: テキストマスクの膨張サイズ（テキスト周辺も含める）
    """
    gray = to_gray_f32(rgb)
    edge = sobel_edge(gray)

    h, w = edge.shape

    # 全体のテキストスコア（高エッジ画素の割合）
    text_pixels = edge > edge_thr
    text_score = float(np.mean(text_pixels))

    # 行ごとのエッジ平均（水平プロファイル）
    row_edge_mean = np.mean(edge, axis=1)  # shape: (h,)

    # クリーンな行を検出（エッジが低い行）
    clean_row_mask = row_edge_mean < row_edge_thr
    clean_ratio = float(np.mean(clean_row_mask))

    # テキスト領域マスクを生成（膨張処理でテキスト周辺も含める）
    text_mask = text_pixels.copy()
    if dilate_size > 0:
        text_mask = _dilate_mask(text_mask, dilate_size)

    return FrameTextInfo(
        frame_idx=-1,  # 後で設定
        text_score=text_score,
        clean_row_mask=clean_row_mask,
        clean_ratio=clean_ratio,
        edge_map=edge,
        text_mask=text_mask,
    )


def _dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """
    ブールマスクを膨張させる（テキスト周辺の領域も含める）
    """
    if HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        return dilated
    else:
        # Pillow版（遅いが機能する）
        from PIL import ImageFilter
        img = Image.fromarray(mask.astype(np.uint8) * 255)
        # MaxFilterで膨張を近似
        for _ in range(size):
            img = img.filter(ImageFilter.MaxFilter(3))
        return np.array(img) > 127


# -------------------------
# Frame Quality & Keyframe Selection
# -------------------------

@dataclass
class FrameQuality:
    """フレームの品質情報"""
    frame_idx: int
    quality_score: float      # 0=低品質, 1=高品質
    sharpness: float          # シャープさ（ラプラシアン分散）
    noise_score: float        # ノイズレベル（低いほど良い）
    text_score: float         # テキスト量（低いほど良い）
    compression_artifact: float  # 圧縮アーティファクト（低いほど良い）


def compute_frame_quality(rgb: np.ndarray, text_info: FrameTextInfo = None) -> FrameQuality:
    """
    フレームの品質スコアを計算する。

    品質スコア = シャープさ × (1 - テキスト量) × (1 - 圧縮ノイズ)
    """
    gray = to_gray_f32(rgb)
    h, w = gray.shape

    # 1. シャープさ（ラプラシアン分散）
    if HAS_CV2:
        gray_uint8 = (gray * 255).astype(np.uint8)
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        sharpness = float(laplacian.var())
    else:
        # 簡易ラプラシアン
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        laplacian = _conv2(gray, kernel)
        sharpness = float(np.var(laplacian))

    # 正規化（典型的な値: 100-2000）
    sharpness_norm = min(1.0, sharpness / 500.0)

    # 2. 圧縮アーティファクト（8x8ブロック境界のエッジ検出）
    if HAS_CV2 and h >= 16 and w >= 16:
        # 8x8グリッドでのエッジ強度をチェック
        block_edges = []
        for y in range(8, h - 8, 8):
            row_diff = np.abs(gray[y, :] - gray[y-1, :])
            block_edges.append(np.mean(row_diff))
        for x in range(8, w - 8, 8):
            col_diff = np.abs(gray[:, x] - gray[:, x-1])
            block_edges.append(np.mean(col_diff))

        if block_edges:
            # 全体のエッジ強度と比較して、ブロック境界が突出しているか
            overall_edge = float(np.mean(sobel_edge(gray)))
            block_edge_mean = float(np.mean(block_edges))
            # ブロック境界のエッジが全体より強い場合は圧縮アーティファクト
            compression_artifact = max(0, (block_edge_mean - overall_edge * 0.5) / 0.1)
            compression_artifact = min(1.0, compression_artifact)
        else:
            compression_artifact = 0.0
    else:
        compression_artifact = 0.0

    # 3. テキストスコア（既存の情報を使用）
    if text_info is not None:
        text_score = text_info.text_score
    else:
        edge = sobel_edge(gray)
        text_pixels = edge > 0.15
        text_score = float(np.mean(text_pixels))

    # 4. ノイズスコア（高周波成分の比率）
    edge = sobel_edge(gray)
    # 小さなエッジ（ノイズ）の比率
    small_edges = (edge > 0.02) & (edge < 0.08)
    noise_score = float(np.mean(small_edges))

    # 総合品質スコア
    quality_score = (
        sharpness_norm * 0.4 +
        (1.0 - text_score) * 0.3 +
        (1.0 - compression_artifact) * 0.2 +
        (1.0 - noise_score) * 0.1
    )

    return FrameQuality(
        frame_idx=-1,
        quality_score=quality_score,
        sharpness=sharpness_norm,
        noise_score=noise_score,
        text_score=text_score,
        compression_artifact=compression_artifact
    )


def select_keyframes(frame_qualities: List[FrameQuality],
                     positions: List[int],
                     strip_size: int,
                     min_quality: float = 0.3,
                     min_overlap: int = 20,
                     max_frame_gap: int = 0) -> List[int]:
    """
    品質の高いキーフレームを選択する。
    なるべく少ないフレームで全体をカバーし、大きな領域で繋ぐ。

    Args:
        frame_qualities: 各フレームの品質情報
        positions: 各フレームの位置（累積dy/dx）
        strip_size: ストリップのサイズ（高さまたは幅）
        min_quality: 最低品質スコア
        min_overlap: 最小オーバーラップピクセル
        max_frame_gap: 最大フレーム間隔（0=無制限、>0の場合は中間フレームを強制選択）

    Returns:
        選択されたキーフレームのインデックスリスト
    """
    n_frames = len(frame_qualities)
    if n_frames == 0:
        return []
    if n_frames == 1:
        return [0]

    # 位置の方向を検出（増加/減少）
    pos_diff = positions[-1] - positions[0]
    # 2フレームで足りる条件: 総移動量がstrip_sizeの30%未満（70%以上のオーバーラップ）
    # dy推定誤差を考慮して保守的に判定
    if abs(pos_diff) < strip_size * 0.3 and max_frame_gap <= 0:
        # 移動量が非常に小さい場合は最初と最後だけで十分
        return [0, n_frames - 1]

    # 位置で正規化したリストを作成（常に増加方向になるように）
    if pos_diff < 0:
        # 位置が減少方向 -> 反転して処理
        normalized_positions = [-p for p in positions]
    else:
        normalized_positions = positions

    # 最小オーバーラップを strip_size の比率で計算（dy推定誤差を考慮）
    # デフォルト20pxと strip_size の 20% の大きい方を使用
    effective_min_overlap = max(min_overlap, int(strip_size * 0.2))

    # グリーディアルゴリズム: 最初のフレームから始めて、
    # カバレッジを維持しながらキーフレームを選択
    keyframes = [0]
    current_idx = 0

    while current_idx < n_frames - 1:
        current_pos = normalized_positions[current_idx]
        coverage_end = current_pos + strip_size  # 現在のキーフレームがカバーできる最大位置

        # max_frame_gap が設定されている場合、探索範囲を制限
        if max_frame_gap > 0:
            search_end = min(n_frames, current_idx + max_frame_gap + 1)
        else:
            search_end = n_frames

        # 次のキーフレーム候補を探す
        best_idx = None
        best_score = -1

        for i in range(current_idx + 1, search_end):
            frame_pos = normalized_positions[i]

            # このフレームは現在のキーフレームとオーバーラップしているか？
            # オーバーラップ = 現在フレームの終端 - 次フレームの開始位置
            overlap = coverage_end - frame_pos

            if overlap < effective_min_overlap and max_frame_gap <= 0:
                # オーバーラップが足りない - ここでカバレッジが途切れる
                # (max_frame_gapが設定されている場合は無視)
                break

            # オーバーラップが strip_size を超えている場合は完全に内包されている
            # max_frame_gapが設定されている場合でも、範囲内なら選択対象
            if overlap > strip_size and max_frame_gap <= 0:
                continue

            # オーバーラップ内にあるフレームの中で評価
            quality = frame_qualities[i].quality_score

            # オーバーラップ比率（strip_sizeに対する割合）
            overlap_ratio = overlap / strip_size

            # スコア計算: オーバーラップが小さすぎる（< 30%）場合はペナルティ
            distance = i - current_idx
            if max_frame_gap > 0:
                # max_frame_gapが設定されている場合は品質を優先
                score = quality * 10 + distance
            elif overlap_ratio < 0.3:
                # オーバーラップが小さい場合は距離を優先しない（安全策）
                score = quality + overlap_ratio
            elif quality >= min_quality:
                # オーバーラップが十分な場合は遠くのフレームを優先
                score = distance * 10 + quality
            else:
                # 品質が低い場合は品質を優先
                score = quality

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            # 候補がない = 現在のキーフレームの次を選ぶ
            best_idx = current_idx + 1

        if best_idx >= n_frames:
            break

        keyframes.append(best_idx)
        current_idx = best_idx

        # 最後のフレームに到達したら終了
        if current_idx >= n_frames - 1:
            break

    # 最後のフレームを確実に含める
    if keyframes[-1] != n_frames - 1:
        keyframes.append(n_frames - 1)

    return keyframes


def render_keyframe_stitch(strips: List[np.ndarray],
                           positions: List[int],
                           keyframe_indices: List[int],
                           out_path: Path,
                           axis: str = "vertical",
                           ignore_regions: List[IgnoreRegion] = None,
                           allow_ignore_at_edges: bool = False) -> None:
    """
    キーフレームのみを使用してハードカットで繋ぐ。
    オーバーラップ部分はブレンドせず、境界でカットする。

    axis: "vertical" (dy) or "horizontal" (dx)
    ignore_regions: 無視する領域（通常は黒で埋める）
    allow_ignore_at_edges: 端のフレーム（最初/最後の位置）ではignore領域を許容する
    """
    if len(keyframe_indices) < 1:
        return

    H, W = strips[0].shape[:2]

    if axis == "vertical":
        strip_size = H  # height
        other_size = W  # width
    else:
        strip_size = W  # width
        other_size = H  # height

    # キーフレームの位置でソート
    kf_data = [(idx, positions[idx]) for idx in keyframe_indices]
    kf_data.sort(key=lambda x: x[1])

    # 端フレームのインデックスを特定
    first_kf_idx = kf_data[0][0]
    last_kf_idx = kf_data[-1][0]

    # キャンバスサイズを計算
    min_pos = kf_data[0][1]
    max_pos = kf_data[-1][1] + strip_size
    canvas_size = max_pos - min_pos

    if axis == "vertical":
        canvas = np.zeros((canvas_size, other_size, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((other_size, canvas_size, 3), dtype=np.uint8)

    # 各キーフレームの担当領域を決定
    # 隣接キーフレーム間のオーバーラップ中央で分割
    for i, (idx, pos) in enumerate(kf_data):
        strip = strips[idx].copy()
        rel_pos = pos - min_pos

        # 端フレームかどうか
        is_edge_frame = (idx == first_kf_idx or idx == last_kf_idx)

        # ignore領域を適用（端フレームで許容オプションが有効な場合は適用しない）
        if ignore_regions and not (allow_ignore_at_edges and is_edge_frame):
            for region in ignore_regions:
                # 領域をピクセルに変換
                if region.unit == "pct":
                    r_top = int(region.top / 100.0 * H)
                    r_height = int(region.height / 100.0 * H)
                    r_width = int(region.width / 100.0 * W)
                    if region.anchor == "right":
                        r_left = W - r_width
                    elif region.anchor == "left":
                        r_left = 0
                    else:
                        r_left = int(region.x / 100.0 * W)
                else:
                    r_top = int(region.top)
                    r_height = int(region.height)
                    r_width = int(region.width)
                    if region.anchor == "right":
                        r_left = W - r_width
                    elif region.anchor == "left":
                        r_left = 0
                    else:
                        r_left = int(region.x)

                # 範囲をクリップ
                r_top = max(0, min(H, r_top))
                r_bottom = max(0, min(H, r_top + r_height))
                r_left = max(0, min(W, r_left))
                r_right = max(0, min(W, r_left + r_width))

                # ignore領域を黒で埋める（後で他のフレームで埋められる可能性）
                strip[r_top:r_bottom, r_left:r_right, :] = 0

        # このフレームの担当開始位置
        if i == 0:
            start_in_strip = 0
        else:
            prev_idx, prev_pos = kf_data[i - 1]
            prev_end = prev_pos + strip_size
            overlap = prev_end - pos
            if overlap > 0:
                # オーバーラップの中央で分割
                start_in_strip = overlap // 2
            else:
                start_in_strip = 0

        # このフレームの担当終了位置
        if i == len(kf_data) - 1:
            end_in_strip = strip_size
        else:
            next_idx, next_pos = kf_data[i + 1]
            overlap = pos + strip_size - next_pos
            if overlap > 0:
                # オーバーラップの中央で分割
                end_in_strip = strip_size - (overlap - overlap // 2)
            else:
                end_in_strip = strip_size

        # キャンバスに配置
        canvas_start = rel_pos + start_in_strip
        canvas_end = rel_pos + end_in_strip

        if axis == "vertical":
            canvas[canvas_start:canvas_end, :, :] = strip[start_in_strip:end_in_strip, :, :]
        else:
            canvas[:, canvas_start:canvas_end, :] = strip[:, start_in_strip:end_in_strip, :]

    save_rgb(canvas, out_path)


def render_keyframe_stitch_matching(frames_rgb: List[np.ndarray],
                                     keyframe_indices: List[int],
                                     out_path: Path,
                                     axis: str = "vertical",
                                     ignore_regions: List[IgnoreRegion] = None,
                                     band: int = 50,
                                     search: int = 20,
                                     method_name: str = "phase",
                                     min_overlap_ratio: float = 0.3,
                                     verbose: bool = True) -> Optional[np.ndarray]:
    """
    キーフレームをマッチングベースで連結する（stitch_candidates.pyと同じロジック）。
    dy推定値を使わず、各キーフレームペア間で直接マッチングを行う。

    Args:
        frames_rgb: 全フレームのRGB画像リスト
        keyframe_indices: キーフレームのインデックス
        out_path: 出力パス
        axis: "vertical" or "horizontal"
        ignore_regions: 無視領域
        band: マッチング用バンドサイズ
        search: 探索範囲
        method_name: マッチング手法名（phase, ncc_gray, ncc_edge, ssim）
        min_overlap_ratio: 最小オーバーラップ比率
        verbose: 詳細出力

    Returns:
        合成結果のndarray、失敗時はNone
    """
    if not HAS_STITCH_CANDIDATES:
        print("Warning: stitch_candidates.py not found. Using position-based stitching.")
        return None

    if len(keyframe_indices) < 1:
        return None
    if len(keyframe_indices) == 1:
        result = frames_rgb[keyframe_indices[0]].copy()
        save_rgb(result, out_path)
        return result

    # マッチング手法を取得
    methods = sc_build_methods()
    method = next((m for m in methods if m.name == method_name), methods[0])

    mode = "v" if axis == "vertical" else "h"

    # キーフレームをインデックス順にソート（通常は時間順）
    sorted_indices = sorted(keyframe_indices)

    # 最初のキーフレームから開始
    comp = frames_rgb[sorted_indices[0]].copy()

    if verbose:
        print(f"Keyframe matching stitch: {len(sorted_indices)} frames, method={method_name}")

    # 順次マッチングして連結
    for i in range(1, len(sorted_indices)):
        kf_idx = sorted_indices[i]
        next_frame = frames_rgb[kf_idx]

        H, W = comp.shape[:2]
        nH, nW = next_frame.shape[:2]
        ref_size = H if axis == "vertical" else W
        next_size = nH if axis == "vertical" else nW

        # マッチングを一度行ってdx, dyを取得
        dx, dy, match_score, _ = sc_match_pair(comp, next_frame, mode, band, search, method, ignore_regions or [])

        if match_score < method.min_score:
            if verbose:
                print(f"  Frame {kf_idx}: Match score too low ({match_score:.4f} < {method.min_score})")
            continue

        # オーバーラップを変化させて最良の境界一致度を探索
        best_overlap = None
        best_combined = -1

        # 粗いスキャン（10%刻み）
        for ov_pct in range(20, 81, 10):
            overlap = int(min(ref_size, next_size) * ov_pct / 100)
            boundary_sim = sc_compute_boundary_similarity(comp, next_frame, mode, dx, dy, overlap)

            if boundary_sim < 0.3:
                continue

            combined = 2 * match_score * boundary_sim / (match_score + boundary_sim + 1e-9)

            if combined > best_combined:
                best_combined = combined
                best_overlap = overlap

        # 精密スキャン（最良の周辺±5%）
        if best_overlap is not None:
            fine_range = int(min(ref_size, next_size) * 0.05)
            fine_start = max(int(min(ref_size, next_size) * 0.1), best_overlap - fine_range)
            fine_end = min(int(min(ref_size, next_size) * 0.9), best_overlap + fine_range)
            for overlap in range(fine_start, fine_end + 1, max(1, fine_range // 5)):
                boundary_sim = sc_compute_boundary_similarity(comp, next_frame, mode, dx, dy, overlap)

                if boundary_sim < 0.3:
                    continue

                combined = 2 * match_score * boundary_sim / (match_score + boundary_sim + 1e-9)

                if combined > best_combined:
                    best_combined = combined
                    best_overlap = overlap

        if best_overlap is None:
            if verbose:
                print(f"  Frame {kf_idx}: No valid overlap found")
            continue

        # 合成
        comp_next = sc_composite_two(comp, next_frame, mode, dx=best_dx, dy=best_dy,
                                      overlap=best_overlap, min_overlap_ratio=min_overlap_ratio)

        if comp_next is None:
            if verbose:
                print(f"  Frame {kf_idx}: Composite failed (overlap={best_overlap}, dx={best_dx}, dy={best_dy})")
            continue

        if verbose:
            print(f"  Frame {kf_idx}: overlap={best_overlap}, dx={best_dx}, dy={best_dy}, score={best_score:.4f}")

        comp = comp_next

    save_rgb(comp, out_path)
    return comp


def render_keyframe_stitch_auto(frames_rgb: List[np.ndarray],
                                 keyframe_indices: List[int],
                                 out_path: Path,
                                 axis: str = "vertical",
                                 ignore_regions: List[IgnoreRegion] = None,
                                 reverse_order: bool = False,
                                 verbose: bool = True) -> List[Tuple[Path, float]]:
    """
    複数のマッチング手法とパラメータで候補を生成する（overlap-autoモード相当）。

    Args:
        frames_rgb: 全フレームのRGB画像リスト
        keyframe_indices: キーフレームのインデックス
        out_path: 出力ベースパス（拡張子なし）
        axis: "vertical" or "horizontal"
        ignore_regions: 無視領域
        reverse_order: True の場合、キーフレームを逆順に連結（negdy/negdx用）
        verbose: 詳細出力

    Returns:
        生成された候補の (パス, スコア) リスト
    """
    if not HAS_STITCH_CANDIDATES:
        print("Warning: stitch_candidates.py not found. Cannot use auto mode.")
        return []

    if len(keyframe_indices) < 2:
        return []

    methods = sc_build_methods()
    mode = "v" if axis == "vertical" else "h"
    # キーフレームをソート（reverse_orderの場合は逆順）
    sorted_indices = sorted(keyframe_indices, reverse=reverse_order)

    # パラメータ組み合わせ
    # キーフレームモードではより大きなバンドも試す
    bands = [50, 100, 200]
    searches = [20, 50]
    method_names = ["phase", "ncc_gray", "ncc_edge", "ssim"]

    results: List[Tuple[Path, float, np.ndarray]] = []

    if verbose:
        print(f"Auto keyframe stitch: {len(sorted_indices)} frames")
        print(f"  Methods: {method_names}, Bands: {bands}, Searches: {searches}")

    for method_name in method_names:
        method = next((m for m in methods if m.name == method_name), methods[0])

        for band in bands:
            for search in searches:
                comp = frames_rgb[sorted_indices[0]].copy()
                valid = True
                min_score = float('inf')
                dx_dy_list = []
                overlap_list = []

                fail_reason = ""
                for i in range(1, len(sorted_indices)):
                    kf_idx = sorted_indices[i]
                    next_frame = frames_rgb[kf_idx]

                    H, W = comp.shape[:2]
                    nH, nW = next_frame.shape[:2]
                    ref_size = H if axis == "vertical" else W
                    next_size = nH if axis == "vertical" else nW

                    # マッチングを一度行ってdx, dyを取得
                    dx, dy, match_score, _ = sc_match_pair(comp, next_frame, mode, band, search, method, ignore_regions or [])

                    # キーフレームモードでは閾値を大幅に緩和（フレーム数が少ないため）
                    # phaseは0.01、ncc系は-0.5まで許容
                    relaxed_min_score = 0.01 if method.name == "phase" else -0.5
                    if match_score < relaxed_min_score:
                        fail_reason = f"low_match_score ({match_score:.4f} < {relaxed_min_score})"
                        valid = False
                        break

                    if verbose and match_score < method.min_score:
                        print(f"    Note: {method_name} match_score={match_score:.4f} (below normal threshold {method.min_score}, but continuing)")

                    # オーバーラップを変化させて最良の境界一致度を探索
                    best_overlap = None
                    best_combined = -1
                    best_boundary = -1

                    for ov_pct in range(20, 81, 5):
                        overlap = int(min(ref_size, next_size) * ov_pct / 100)

                        boundary_sim = sc_compute_boundary_similarity(comp, next_frame, mode, dx, dy, overlap)
                        best_boundary = max(best_boundary, boundary_sim)
                        if boundary_sim < 0.1:  # 閾値を緩和 (0.3 -> 0.1)
                            continue

                        combined = 2 * match_score * boundary_sim / (match_score + boundary_sim + 1e-9)

                        if combined > best_combined:
                            best_combined = combined
                            best_overlap = overlap

                    if best_overlap is None:
                        fail_reason = f"no_valid_overlap (best_boundary={best_boundary:.4f}, dx={dx}, dy={dy})"
                        valid = False
                        break

                    comp_next = sc_composite_two(comp, next_frame, mode, dx=dx, dy=dy,
                                                  overlap=best_overlap, min_overlap_ratio=0.1)  # 緩和

                    if comp_next is None:
                        fail_reason = f"composite_failed (overlap={best_overlap}, dx={dx}, dy={dy})"
                        valid = False
                        break

                    min_score = min(min_score, best_combined)
                    dx_dy_list.append((dx, dy))
                    overlap_list.append(best_overlap)
                    comp = comp_next

                if not valid and verbose and fail_reason:
                    print(f"    {method_name} band={band} search={search}: FAILED - {fail_reason}")

                if valid and comp is not None:
                    # ファイル名を生成（スコアを含む）
                    overlaps_str = "_".join(str(o) for o in overlap_list)
                    dxdy_str = "_".join(f"dx{dx}dy{dy}" for dx, dy in dx_dy_list)
                    suffix = f"_score{min_score:.4f}_{method_name}_b{band}_s{search}_ov{overlaps_str}_{dxdy_str}"
                    out_file = out_path.parent / f"{out_path.stem}{suffix}.png"
                    save_rgb(comp, out_file)
                    results.append((out_file, min_score, comp))

                    if verbose:
                        print(f"  {method_name} band={band} search={search}: score={min_score:.4f}, overlaps=[{','.join(str(o) for o in overlap_list)}]")

    # スコアでソート
    results.sort(key=lambda x: x[1], reverse=True)

    # 上位3件のみを残し、それ以外は削除
    top_n = 3
    if len(results) > top_n:
        for r in results[top_n:]:
            try:
                r[0].unlink()  # ファイルを削除
            except Exception:
                pass
        if verbose:
            print(f"\nKept top {top_n} candidates, removed {len(results) - top_n} lower-scoring files")
        results = results[:top_n]

    if verbose and results:
        print(f"\nGenerated {len(results)} candidates")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r[0].name} (score={r[1]:.4f})")

    return [(r[0], r[1]) for r in results]


def classify_frames(frames_rgb: List[np.ndarray],
                    edge_thr: float = 0.15,
                    text_free_threshold: float = 0.05,
                    text_heavy_threshold: float = 0.15) -> Tuple[List[FrameTextInfo], List[int], List[int], List[int]]:
    """
    フレームをテキスト量で分類する。

    Returns:
        - frame_infos: 全フレームのテキスト情報
        - text_free_indices: テキストなしフレームのインデックス
        - text_light_indices: テキスト薄フレームのインデックス
        - text_heavy_indices: テキスト濃フレームのインデックス
    """
    frame_infos = []
    text_free = []
    text_light = []
    text_heavy = []

    for i, rgb in enumerate(frames_rgb):
        info = analyze_frame_text(rgb, edge_thr=edge_thr)
        info.frame_idx = i
        frame_infos.append(info)

        if info.text_score < text_free_threshold:
            text_free.append(i)
        elif info.text_score < text_heavy_threshold:
            text_light.append(i)
        else:
            text_heavy.append(i)

    return frame_infos, text_free, text_light, text_heavy


def find_clean_horizontal_bands(frame_infos: List[FrameTextInfo],
                                 min_band_height: int = 5,
                                 min_clean_ratio: float = 0.7) -> List[List[Tuple[int, int]]]:
    """
    各フレームからクリーンな水平バンド（行間）を検出する。

    Returns:
        各フレームの [(y_start, y_end), ...] リスト
    """
    all_bands = []

    for info in frame_infos:
        mask = info.clean_row_mask
        h = len(mask)

        bands = []
        in_band = False
        band_start = 0

        for y in range(h):
            if mask[y] and not in_band:
                band_start = y
                in_band = True
            elif not mask[y] and in_band:
                band_height = y - band_start
                if band_height >= min_band_height:
                    bands.append((band_start, y))
                in_band = False

        # 最後まで続いた場合
        if in_band:
            band_height = h - band_start
            if band_height >= min_band_height:
                bands.append((band_start, h))

        all_bands.append(bands)

    return all_bands


def estimate_dy_weighted(frames_rgb: List[np.ndarray],
                          frame_infos: List[FrameTextInfo],
                          ignore_regions: List[IgnoreRegion],
                          search: int = 40,
                          auto_exclude_text: bool = False) -> Tuple[List[int], List[float]]:
    """
    全フレーム間のdyを推定。テキストスコアが低いフレームペアを高信頼度として扱う。

    auto_exclude_text: True の場合、各フレームの text_mask を使って
                       テキスト領域を動的に除外してdy推定する
    """
    n_frames = len(frames_rgb)
    dys: List[int] = []
    peaks: List[float] = []
    confidences: List[float] = []

    for i in range(n_frames - 1):
        gray_a = to_gray_f32(frames_rgb[i])
        gray_b = to_gray_f32(frames_rgb[i + 1])
        if ignore_regions:
            gray_a = apply_ignore_gray(gray_a, ignore_regions)
            gray_b = apply_ignore_gray(gray_b, ignore_regions)

        # 自動テキスト除外: 両フレームのテキスト領域をマスク
        if auto_exclude_text:
            combined_mask = frame_infos[i].text_mask | frame_infos[i + 1].text_mask
            gray_a = apply_text_mask(gray_a, combined_mask)
            gray_b = apply_text_mask(gray_b, combined_mask)

        edge_a = sobel_edge(gray_a)
        edge_b = sobel_edge(gray_b)

        dx, dy, peak = phase_correlation(edge_a, edge_b)
        dy = int(np.clip(dy, -search, search))

        dys.append(dy)
        peaks.append(peak)

        # 信頼度: 両フレームのテキストスコアが低いほど高い
        text_score_avg = (frame_infos[i].text_score + frame_infos[i + 1].text_score) / 2
        confidence = (1.0 - text_score_avg) * peak
        confidences.append(confidence)

    # 高信頼度のdyの中央値を計算
    high_conf_threshold = np.percentile(confidences, 70) if confidences else 0
    high_conf_dys = [d for d, c in zip(dys, confidences) if c >= high_conf_threshold and d != 0]

    if high_conf_dys:
        expected_dy = int(np.median(high_conf_dys))
        expected_sign = np.sign(expected_dy)

        # 外れ値を修正
        for i in range(len(dys)):
            dy = dys[i]
            # 符号が逆、または大きく外れている場合
            if dy != 0 and expected_dy != 0:
                if np.sign(dy) != expected_sign:
                    print(f"  Frame {i}: dy={dy} -> {expected_dy} (wrong sign)")
                    dys[i] = expected_dy
                elif abs(dy - expected_dy) > abs(expected_dy) * 2:
                    print(f"  Frame {i}: dy={dy} -> {expected_dy} (outlier)")
                    dys[i] = expected_dy

    return dys, peaks


def estimate_dy_robust(frames_rgb: List[np.ndarray],
                        frame_infos: List[FrameTextInfo],
                        ignore_regions: List[IgnoreRegion],
                        search: int = 40,
                        min_peak: float = 0.1,
                        auto_exclude_text: bool = False) -> Tuple[List[int], List[float]]:
    """
    より堅牢なdy推定。テンプレートマッチングとphase correlationを組み合わせる。
    テンプレートが高信頼ならそれを優先、そうでなければphaseを使う。

    auto_exclude_text: True の場合、各フレームの text_mask を使って
                       テキスト領域を動的に除外してdy推定する
    """
    n_frames = len(frames_rgb)

    # まず全フレームでphase correlation
    phase_dys = []
    phase_peaks = []
    for i in range(n_frames - 1):
        gray_a = to_gray_f32(frames_rgb[i])
        gray_b = to_gray_f32(frames_rgb[i + 1])
        if ignore_regions:
            gray_a = apply_ignore_gray(gray_a, ignore_regions)
            gray_b = apply_ignore_gray(gray_b, ignore_regions)

        # 自動テキスト除外: 両フレームのテキスト領域をマスク
        if auto_exclude_text:
            combined_mask = frame_infos[i].text_mask | frame_infos[i + 1].text_mask
            gray_a = apply_text_mask(gray_a, combined_mask)
            gray_b = apply_text_mask(gray_b, combined_mask)

        edge_a = sobel_edge(gray_a)
        edge_b = sobel_edge(gray_b)
        dx, dy, peak = phase_correlation(edge_a, edge_b)
        dy = int(np.clip(dy, -search, search))
        phase_dys.append(dy)
        phase_peaks.append(peak)

    # テンプレートマッチングも試す
    template_dys = []
    template_peaks = []
    template_tracker = TemplateTracker()

    for i in range(n_frames - 1):
        results = estimate_dy_multi(frames_rgb[i], frames_rgb[i + 1],
                                    ignore_regions=ignore_regions,
                                    search=search,
                                    methods=["template"],
                                    template_tracker=template_tracker)
        if results and results[0].score > 0.5:
            template_dys.append(results[0].dy)
            template_peaks.append(results[0].score)
        else:
            template_dys.append(None)
            template_peaks.append(0.0)

    # 両方の結果を組み合わせる
    # - テンプレートが高信頼（score > 0.8）ならそれを使う
    # - そうでなければphase correlationを使う
    final_dys = []
    final_peaks = []
    source = []  # デバッグ用: どちらの値を使ったか

    for i in range(n_frames - 1):
        if template_dys[i] is not None and template_peaks[i] > 0.8:
            final_dys.append(template_dys[i])
            final_peaks.append(template_peaks[i])
            source.append("template")
        else:
            final_dys.append(phase_dys[i])
            final_peaks.append(phase_peaks[i])
            source.append("phase")

    # 信頼できる値から統計を計算
    # テンプレートの高信頼値を優先的に使う
    high_conf_template = [(dy, peak) for dy, peak in zip(template_dys, template_peaks)
                          if dy is not None and peak > 0.8]

    if high_conf_template:
        # テンプレートの高信頼値があればそれを基準にする
        reliable_dys = [dy for dy, _ in high_conf_template]
        median_dy = int(np.median(reliable_dys))
        print(f"  Using template-based median: {median_dy} (from {len(reliable_dys)} high-confidence values)")
    else:
        # なければphaseの値から計算（search limitに達していないもの）
        reliable_dys = [dy for dy, peak in zip(phase_dys, phase_peaks)
                        if peak > min_peak and abs(dy) < search * 0.9]
        if reliable_dys:
            median_dy = int(np.median(reliable_dys))
            print(f"  Using phase-based median: {median_dy}")
        else:
            # どちらも信頼できない場合は全体の中央値
            all_nonzero = [dy for dy in final_dys if dy != 0]
            median_dy = int(np.median(all_nonzero)) if all_nonzero else 0
            print(f"  Warning: No reliable dy values, using overall median: {median_dy}")

    expected_sign = np.sign(median_dy) if median_dy != 0 else 1

    # 外れ値検出と修正
    for i in range(len(final_dys)):
        dy = final_dys[i]
        peak = final_peaks[i]
        src = source[i]

        # 外れ値判定
        is_outlier = False
        reason = ""

        # 1. phaseでsearch limitに達している場合
        if src == "phase" and abs(dy) >= search * 0.95:
            is_outlier = True
            reason = "phase at search limit"

        # 2. 符号が逆
        elif dy != 0 and median_dy != 0 and np.sign(dy) != expected_sign:
            is_outlier = True
            reason = "wrong sign"

        # 3. 中央値から大きく外れている（3倍以上）
        elif median_dy != 0 and abs(dy) > abs(median_dy) * 3:
            is_outlier = True
            reason = "too large"

        # 4. peakが低すぎる（テンプレート失敗やphase不安定）
        elif peak < min_peak:
            is_outlier = True
            reason = "low peak"

        if is_outlier:
            print(f"  Frame {i}: dy={dy} ({src}, peak={peak:.3f}) -> {median_dy} ({reason})")
            final_dys[i] = median_dy
            final_peaks[i] = 0.5

    return final_dys, final_peaks


def _interpolate_none_values(dys: List[Optional[int]],
                              peaks: List[float]) -> List[Optional[int]]:
    """Noneの値を前後から補間する"""
    n = len(dys)
    result = list(dys)

    # 有効な値の中央値を計算
    valid_dys = [d for d in dys if d is not None]
    if not valid_dys:
        return result

    median_dy = int(np.median(valid_dys))

    for i in range(n):
        if result[i] is None:
            # 前後の有効な値を探す
            prev_val = None
            next_val = None
            for j in range(i - 1, -1, -1):
                if result[j] is not None:
                    prev_val = result[j]
                    break
            for j in range(i + 1, n):
                if result[j] is not None:
                    next_val = result[j]
                    break

            if prev_val is not None and next_val is not None:
                # 線形補間
                result[i] = (prev_val + next_val) // 2
            elif prev_val is not None:
                result[i] = prev_val
            elif next_val is not None:
                result[i] = next_val
            else:
                result[i] = median_dy

    return result


def render_jigsaw_candidate(strips: List[np.ndarray],
                             positions: List[int],
                             frame_infos: List[FrameTextInfo],
                             clean_bands: List[List[Tuple[int, int]]],
                             out_path: Path,
                             ignore_regions: List[IgnoreRegion],
                             full_w: int, full_h: int, strip_y: int,
                             prefer_clean_frames: bool = True,
                             blend_mode: str = "weighted",
                             base_frame_idx: Optional[int] = None) -> None:
    """
    ジグソーパズル方式でキャンバスに貼り付ける。

    blend_mode:
    - "priority": 優先度が低いピクセルのみ採用（従来方式）
    - "weighted": 重み付き平均でブレンド（スムーズ）

    base_frame_idx:
    - None: 従来の重み付き平均を使用
    - int: 指定フレームをベースとして最優先で配置し、他のフレームで補完
           負のインデックス可（-1=最終フレーム、-2=最後から2番目など）
    """
    assert len(strips) == len(positions) == len(frame_infos)
    strip_h = strips[0].shape[0]
    W = strips[0].shape[1]

    min_y = min(positions)
    max_y = max(positions) + strip_h
    out_h = max_y - min_y

    # base_frame_idx が指定されている場合、ベースフレーム優先モード
    if base_frame_idx is not None:
        # 負のインデックスを正規化
        if base_frame_idx < 0:
            base_frame_idx = len(strips) + base_frame_idx
        if base_frame_idx < 0 or base_frame_idx >= len(strips):
            print(f"  Warning: base_frame_idx {base_frame_idx} out of range, using last frame")
            base_frame_idx = len(strips) - 1

        # ベースフレームを最初に配置
        base_strip = strips[base_frame_idx]
        base_pos = positions[base_frame_idx]
        base_y0 = base_pos - min_y
        base_y1 = base_y0 + strip_h

        # キャンバスを初期化（ベースフレームで埋める）
        canvas = np.zeros((out_h, W, 3), dtype=np.uint8)
        canvas[base_y0:base_y1, :, :] = base_strip

        # ベースフレームがカバーしている領域をマスク
        covered = np.zeros(out_h, dtype=bool)
        covered[base_y0:base_y1] = True

        # 他のフレームで補完（ベースがカバーしていない領域のみ）
        # テキストスコアが低い順に処理
        other_indices = [i for i in range(len(strips)) if i != base_frame_idx]
        sorted_others = sorted(other_indices, key=lambda i: frame_infos[i].text_score)

        for idx in sorted_others:
            strip = strips[idx]
            gy = positions[idx]
            info = frame_infos[idx]
            bands = clean_bands[idx]

            y0 = gy - min_y
            y1 = y0 + strip_h

            # このフレームの各行についてカバー済みかチェック
            for local_y in range(strip_h):
                global_y = y0 + local_y
                if 0 <= global_y < out_h and not covered[global_y]:
                    # カバーされていない行を補完
                    canvas[global_y, :, :] = strip[local_y, :, :]
                    covered[global_y] = True

        save_rgb(canvas, out_path)
        return

    if blend_mode == "weighted":
        # 重み付き平均方式
        canvas_sum = np.zeros((out_h, W, 3), dtype=np.float64)
        weight_sum = np.zeros((out_h, W), dtype=np.float64)

        for idx in range(len(strips)):
            strip = strips[idx]
            gy = positions[idx]
            info = frame_infos[idx]
            bands = clean_bands[idx]

            y0 = gy - min_y
            y1 = y0 + strip_h

            # 重みを計算（テキストスコアとエッジ強度の逆数）
            edge = info.edge_map
            # 重み = (1 - text_score) * (1 - edge_strength)
            # テキストスコアが低く、エッジが弱いほど重みが高い
            weight = (1.0 - info.text_score) * (1.0 - np.clip(edge, 0, 0.8))

            # クリーンバンド内は重みを上げる
            for band_start, band_end in bands:
                weight[band_start:band_end, :] *= 2.0

            # ignore regions は重み0
            if ignore_regions:
                for reg in ignore_regions:
                    x0r, y0r, x1r, y1r = _resolve_region_px(reg, full_w, full_h)
                    sy0 = max(strip_y, y0r)
                    sy1 = min(strip_y + strip_h, y1r)
                    if sy0 >= sy1:
                        continue
                    ry0 = sy0 - strip_y
                    ry1 = sy1 - strip_y
                    weight[ry0:ry1, x0r:x1r] = 0.0

            # 累積
            canvas_sum[y0:y1, :, :] += strip.astype(np.float64) * weight[:, :, np.newaxis]
            weight_sum[y0:y1, :] += weight

        # 正規化
        weight_sum = np.maximum(weight_sum, 1e-6)  # ゼロ除算防止
        canvas = (canvas_sum / weight_sum[:, :, np.newaxis]).astype(np.uint8)

    else:
        # 従来の優先度方式
        canvas = np.zeros((out_h, W, 3), dtype=np.uint8)
        priority = np.full((out_h, W), np.inf, dtype=np.float32)

        sorted_indices = sorted(range(len(strips)),
                               key=lambda i: frame_infos[i].text_score)

        for idx in sorted_indices:
            strip = strips[idx]
            gy = positions[idx]
            info = frame_infos[idx]
            bands = clean_bands[idx]

            y0 = gy - min_y
            y1 = y0 + strip_h

            base_priority = info.text_score
            edge = info.edge_map
            pixel_priority = base_priority + edge * 0.5

            for band_start, band_end in bands:
                pixel_priority[band_start:band_end, :] *= 0.3

            if ignore_regions:
                for reg in ignore_regions:
                    x0r, y0r, x1r, y1r = _resolve_region_px(reg, full_w, full_h)
                    sy0 = max(strip_y, y0r)
                    sy1 = min(strip_y + strip_h, y1r)
                    if sy0 >= sy1:
                        continue
                    ry0 = sy0 - strip_y
                    ry1 = sy1 - strip_y
                    pixel_priority[ry0:ry1, x0r:x1r] = np.inf

            current_priority = priority[y0:y1, :]
            update_mask = pixel_priority < current_priority

            if np.any(update_mask):
                priority[y0:y1, :][update_mask] = pixel_priority[update_mask]
                canvas_slice = canvas[y0:y1, :, :]
                canvas_slice[update_mask, :] = strip[update_mask, :]

    save_rgb(canvas, out_path)


def smooth_dy_values(dys: List[int], window: int = 3) -> List[int]:
    """
    dy値を移動平均でスムージングする。
    """
    if len(dys) < window:
        return dys

    smoothed = []
    half = window // 2

    for i in range(len(dys)):
        start = max(0, i - half)
        end = min(len(dys), i + half + 1)
        avg = sum(dys[start:end]) / (end - start)
        smoothed.append(int(round(avg)))

    return smoothed


def jigsaw_mode(frames_rgb: List[np.ndarray], strips: List[np.ndarray],
                out_dir: Path, ignore_regions: List[IgnoreRegion],
                full_w: int, full_h: int, strip_y: int, strip_h: int,
                edge_thr: float = 0.15, search: int = 40,
                scroll_dir: str = "both",
                match_method: str = "template",
                smooth_dy: bool = True,
                base_frame_idx: Optional[int] = None,
                auto_exclude_text: bool = False,
                keyframe_mode: bool = False,
                min_quality: float = 0.3,
                max_keyframe_gap: int = 0,
                allow_ignore_at_edges: bool = False,
                keyframe_stitch_method: str = "position") -> None:
    """
    ジグソーパズルモード: テキストの有無を考慮した賢い合成

    base_frame_idx: ベースフレームのインデックス（負数可）
                    指定時はそのフレームを最優先で配置し、他のフレームで補完
    auto_exclude_text: True の場合、各フレームのテキスト領域を検出し、
                       dy推定から動的に除外する
    keyframe_mode: True の場合、品質の高いフレームを選択してハードカットで繋ぐ
                   （ブレンドしない）
    min_quality: キーフレーム選択の最低品質スコア
    keyframe_stitch_method: "position"=dy推定値を使用、"matching"=stitch_candidatesと同様にマッチング
    """
    print("\n" + "="*60)
    print("[Jigsaw Reconstruction Mode]")
    print("="*60)

    # Phase 1: フレーム分類
    print("\n[Phase 1] Classifying frames by text content...")
    frame_infos, text_free, text_light, text_heavy = classify_frames(
        strips, edge_thr=edge_thr
    )

    print(f"  Text-free frames: {len(text_free)}")
    print(f"  Text-light frames: {len(text_light)}")
    print(f"  Text-heavy frames: {len(text_heavy)}")

    if text_free:
        print(f"    Text-free indices: {text_free[:10]}{'...' if len(text_free) > 10 else ''}")

    # Phase 2: クリーンバンド検出
    print("\n[Phase 2] Finding clean horizontal bands...")
    clean_bands = find_clean_horizontal_bands(frame_infos)

    total_bands = sum(len(b) for b in clean_bands)
    print(f"  Total clean bands found: {total_bands}")

    # 統計表示
    bands_per_frame = [len(b) for b in clean_bands]
    if bands_per_frame:
        print(f"  Bands per frame: mean={np.mean(bands_per_frame):.1f}, max={max(bands_per_frame)}")

    # Phase 3: dy推定
    print(f"\n[Phase 3] Estimating dy (method: {match_method})...")
    if auto_exclude_text:
        print("  Auto-exclude text regions enabled")

    if match_method == "robust":
        # 堅牢なハイブリッド推定（テンプレート + phase）
        dys, peaks = estimate_dy_robust(frames_rgb, frame_infos, ignore_regions, search,
                                        auto_exclude_text=auto_exclude_text)
    elif match_method == "template":
        # テンプレートマッチング
        dys = []
        peaks = []
        template_tracker = TemplateTracker()

        for i in range(1, len(frames_rgb)):
            results = estimate_dy_multi(frames_rgb[i-1], frames_rgb[i],
                                        ignore_regions=ignore_regions,
                                        search=search,
                                        methods=["template"],
                                        template_tracker=template_tracker)
            if results:
                r = results[0]
                dys.append(r.dy)
                peaks.append(r.score)
            else:
                dys.append(0)
                peaks.append(0.0)

        # 外れ値を修正
        dys = _detect_and_fix_outliers(dys, peaks)
        dys = _interpolate_lost_tracking(dys, peaks)
    elif match_method in ("optical_flow", "optflow", "optical_flow_fb", "optical_flow_lk"):
        # オプティカルフロー
        dys = []
        peaks = []

        for i in range(1, len(frames_rgb)):
            # テキスト除外マスク
            text_mask = None
            if auto_exclude_text:
                text_mask = frame_infos[i-1].text_mask | frame_infos[i].text_mask

            if match_method in ("optical_flow", "optflow"):
                dx, dy, conf = optical_flow_estimate_robust(
                    frames_rgb[i-1], frames_rgb[i], text_mask
                )
            elif match_method == "optical_flow_fb":
                dx, dy, conf = optical_flow_estimate(
                    frames_rgb[i-1], frames_rgb[i],
                    mask=~text_mask if text_mask is not None else None,
                    method="farneback"
                )
            else:  # optical_flow_lk
                dx, dy, conf = optical_flow_estimate(
                    frames_rgb[i-1], frames_rgb[i],
                    mask=~text_mask if text_mask is not None else None,
                    method="lk"
                )

            dy = int(round(np.clip(dy, -search, search)))
            dys.append(dy)
            peaks.append(conf)

        # 外れ値を修正
        dys = _detect_and_fix_outliers(dys, peaks)
    else:
        # テキストスコアで重み付けした推定（phase correlation）
        dys, peaks = estimate_dy_weighted(frames_rgb, frame_infos, ignore_regions, search,
                                          auto_exclude_text=auto_exclude_text)

    # dy統計（スムージング前）
    nonzero_dys = [d for d in dys if d != 0]
    if nonzero_dys:
        dy_mean = np.mean(nonzero_dys)
        dy_std = np.std(nonzero_dys)
        print(f"  dy (raw): mean={dy_mean:.1f}, std={dy_std:.1f}")

    # スムージング（オプション）
    if smooth_dy and len(dys) >= 3:
        dys_raw = dys.copy()
        dys = smooth_dy_values(dys, window=5)
        n_changed = sum(1 for a, b in zip(dys_raw, dys) if a != b)
        print(f"  dy smoothing: {n_changed}/{len(dys)} values adjusted")

        # スムージング後の統計
        nonzero_dys = [d for d in dys if d != 0]
        if nonzero_dys:
            dy_mean = np.mean(nonzero_dys)
            dy_std = np.std(nonzero_dys)
            print(f"  dy (smoothed): mean={dy_mean:.1f}, std={dy_std:.1f}")

    # 位置計算
    pos_dy = [0]
    pos_negdy = [0]
    for dy in dys:
        pos_dy.append(pos_dy[-1] + dy)
        pos_negdy.append(pos_negdy[-1] - dy)

    # Phase 4: キーフレーム選択（オプション）またはジグソー合成
    output_files = []

    if keyframe_mode:
        print("\n[Phase 4] Keyframe selection mode...")
        # 各フレームの品質スコアを計算
        print("  Computing frame quality scores...")
        frame_qualities = []
        for i, strip in enumerate(strips):
            fq = compute_frame_quality(strip, frame_infos[i])
            fq = FrameQuality(
                frame_idx=i,
                quality_score=fq.quality_score,
                sharpness=fq.sharpness,
                noise_score=fq.noise_score,
                text_score=fq.text_score,
                compression_artifact=fq.compression_artifact
            )
            frame_qualities.append(fq)

        # 品質スコア統計
        quality_scores = [fq.quality_score for fq in frame_qualities]
        print(f"  Quality scores: min={min(quality_scores):.2f}, max={max(quality_scores):.2f}, mean={np.mean(quality_scores):.2f}")

        # 高品質フレーム数
        high_quality = sum(1 for q in quality_scores if q >= min_quality)
        print(f"  High quality frames (>={min_quality}): {high_quality}/{len(frame_qualities)}")

        # キーフレームモードではフルフレームを使用（strip-hではなくフレーム高さ）
        frame_h = frames_rgb[0].shape[0]
        print(f"  Using full frame height: {frame_h}")
        print(f"  Stitch method: {keyframe_stitch_method}")

        # dy方向のキーフレーム選択とレンダリング
        if scroll_dir in ("up", "both"):
            keyframes_dy = select_keyframes(frame_qualities, pos_dy, frame_h,
                                            min_quality=min_quality, min_overlap=20,
                                            max_frame_gap=max_keyframe_gap)
            print(f"  Selected keyframes (dy): {len(keyframes_dy)} frames")
            print(f"    Indices: {keyframes_dy}")

            if keyframe_stitch_method == "matching" and HAS_STITCH_CANDIDATES:
                # マッチングベースのスティッチング（stitch_candidates.pyと同様）
                out1_base = out_dir / "recon_keyframe_dy"
                candidates = render_keyframe_stitch_auto(
                    frames_rgb, keyframes_dy, out1_base, axis="vertical",
                    ignore_regions=ignore_regions, verbose=True
                )
                for cand_path, cand_score in candidates:
                    output_files.append(cand_path)
                if not candidates:
                    print("  Warning: No valid candidates from matching mode, falling back to position mode")
                    out1 = out_dir / "recon_keyframe_dy.png"
                    render_keyframe_stitch(frames_rgb, pos_dy, keyframes_dy, out1, axis="vertical",
                                           ignore_regions=ignore_regions,
                                           allow_ignore_at_edges=allow_ignore_at_edges)
                    output_files.append(out1)
                    print(f"  Created: {out1}")
            else:
                # 位置ベースのスティッチング（従来方式）
                out1 = out_dir / "recon_keyframe_dy.png"
                render_keyframe_stitch(frames_rgb, pos_dy, keyframes_dy, out1, axis="vertical",
                                       ignore_regions=ignore_regions,
                                       allow_ignore_at_edges=allow_ignore_at_edges)
                output_files.append(out1)
                print(f"  Created: {out1}")

        # negdy方向のキーフレーム選択とレンダリング
        if scroll_dir in ("down", "both"):
            keyframes_negdy = select_keyframes(frame_qualities, pos_negdy, frame_h,
                                               min_quality=min_quality, min_overlap=20,
                                               max_frame_gap=max_keyframe_gap)
            print(f"  Selected keyframes (negdy): {len(keyframes_negdy)} frames")
            print(f"    Indices: {keyframes_negdy}")

            if keyframe_stitch_method == "matching" and HAS_STITCH_CANDIDATES:
                out2_base = out_dir / "recon_keyframe_negdy"
                candidates = render_keyframe_stitch_auto(
                    frames_rgb, keyframes_negdy, out2_base, axis="vertical",
                    ignore_regions=ignore_regions, reverse_order=True, verbose=True
                )
                for cand_path, cand_score in candidates:
                    output_files.append(cand_path)
                if not candidates:
                    print("  Warning: No valid candidates from matching mode, falling back to position mode")
                    out2 = out_dir / "recon_keyframe_negdy.png"
                    render_keyframe_stitch(frames_rgb, pos_negdy, keyframes_negdy, out2, axis="vertical",
                                           ignore_regions=ignore_regions,
                                           allow_ignore_at_edges=allow_ignore_at_edges)
                    output_files.append(out2)
                    print(f"  Created: {out2}")
            else:
                out2 = out_dir / "recon_keyframe_negdy.png"
                render_keyframe_stitch(frames_rgb, pos_negdy, keyframes_negdy, out2, axis="vertical",
                                       ignore_regions=ignore_regions,
                                       allow_ignore_at_edges=allow_ignore_at_edges)
                output_files.append(out2)
                print(f"  Created: {out2}")

        # 品質デバッグCSV出力
        quality_csv_path = out_dir / "debug_keyframe_quality.csv"
        with quality_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "quality_score", "sharpness", "text_score",
                           "compression", "noise", "selected_dy", "selected_negdy"])
            for fq in frame_qualities:
                selected_dy = "Y" if scroll_dir in ("up", "both") and fq.frame_idx in keyframes_dy else ""
                selected_negdy = "Y" if scroll_dir in ("down", "both") and fq.frame_idx in keyframes_negdy else ""
                writer.writerow([
                    fq.frame_idx,
                    f"{fq.quality_score:.4f}",
                    f"{fq.sharpness:.4f}",
                    f"{fq.text_score:.4f}",
                    f"{fq.compression_artifact:.4f}",
                    f"{fq.noise_score:.4f}",
                    selected_dy,
                    selected_negdy
                ])
        print(f"  Quality CSV: {quality_csv_path}")

    else:
        # 従来のジグソー合成
        print("\n[Phase 4] Rendering jigsaw candidates...")

        # ベースフレームモードの場合、情報を表示
        if base_frame_idx is not None:
            # 負のインデックスを正規化して表示
            actual_idx = base_frame_idx if base_frame_idx >= 0 else len(strips) + base_frame_idx
            if actual_idx < 0 or actual_idx >= len(strips):
                actual_idx = len(strips) - 1
            print(f"  Base frame mode: using frame {actual_idx} as foundation")
            suffix = f"_base{actual_idx}"
        else:
            suffix = ""

        if scroll_dir in ("up", "both"):
            out1 = out_dir / f"recon_jigsaw_dy{suffix}.png"
            render_jigsaw_candidate(
                strips=strips, positions=pos_dy,
                frame_infos=frame_infos, clean_bands=clean_bands,
                out_path=out1, ignore_regions=ignore_regions,
                full_w=full_w, full_h=full_h, strip_y=strip_y,
                base_frame_idx=base_frame_idx
            )
            output_files.append(out1)
            print(f"  Created: {out1}")

        if scroll_dir in ("down", "both"):
            out2 = out_dir / f"recon_jigsaw_negdy{suffix}.png"
            render_jigsaw_candidate(
                strips=strips, positions=pos_negdy,
                frame_infos=frame_infos, clean_bands=clean_bands,
                out_path=out2, ignore_regions=ignore_regions,
                full_w=full_w, full_h=full_h, strip_y=strip_y,
                base_frame_idx=base_frame_idx
            )
            output_files.append(out2)
            print(f"  Created: {out2}")

    # デバッグCSV出力
    csv_path = out_dir / "debug_jigsaw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "text_score", "clean_ratio", "num_bands", "dy", "peak", "pos_dy", "pos_negdy"])
        for i, info in enumerate(frame_infos):
            dy = dys[i] if i < len(dys) else ""
            peak = peaks[i] if i < len(peaks) else ""
            writer.writerow([
                i,
                f"{info.text_score:.4f}",
                f"{info.clean_ratio:.4f}",
                len(clean_bands[i]),
                dy,
                f"{peak:.4f}" if isinstance(peak, float) else peak,
                pos_dy[i],
                pos_negdy[i]
            ])
    print(f"  Debug CSV: {csv_path}")

    # フレーム分類の可視化
    vis_path = out_dir / "frame_classification.png"
    visualize_frame_classification(strips, frame_infos, vis_path)
    print(f"  Classification visualization: {vis_path}")

    print("\n" + "="*60)
    print("Done!")
    for f in output_files:
        print(f"  {f}")
    print("="*60)


# -------------------------
# Horizontal pan mode (水平パン対応)
# -------------------------

def render_jigsaw_horizontal(strips: List[np.ndarray],
                              positions: List[int],
                              frame_infos: List[FrameTextInfo],
                              out_path: Path,
                              base_frame_idx: Optional[int] = None) -> None:
    """
    横方向パン用のジグソー合成。

    strips: 垂直ストリップのリスト (H, W, 3)
    positions: 各フレームのX位置
    """
    assert len(strips) == len(positions) == len(frame_infos)
    strip_w = strips[0].shape[1]
    H = strips[0].shape[0]

    min_x = min(positions)
    max_x = max(positions) + strip_w
    out_w = max_x - min_x

    if base_frame_idx is not None:
        # ベースフレームモード
        if base_frame_idx < 0:
            base_frame_idx = len(strips) + base_frame_idx
        if base_frame_idx < 0 or base_frame_idx >= len(strips):
            base_frame_idx = len(strips) - 1

        base_strip = strips[base_frame_idx]
        base_pos = positions[base_frame_idx]
        base_x0 = base_pos - min_x
        base_x1 = base_x0 + strip_w

        canvas = np.zeros((H, out_w, 3), dtype=np.uint8)
        canvas[:, base_x0:base_x1, :] = base_strip

        covered = np.zeros(out_w, dtype=bool)
        covered[base_x0:base_x1] = True

        other_indices = [i for i in range(len(strips)) if i != base_frame_idx]
        sorted_others = sorted(other_indices, key=lambda i: frame_infos[i].text_score)

        for idx in sorted_others:
            strip = strips[idx]
            gx = positions[idx]
            x0 = gx - min_x
            x1 = x0 + strip_w

            for local_x in range(strip_w):
                global_x = x0 + local_x
                if 0 <= global_x < out_w and not covered[global_x]:
                    canvas[:, global_x, :] = strip[:, local_x, :]
                    covered[global_x] = True
    else:
        # 重み付き平均モード
        canvas_sum = np.zeros((H, out_w, 3), dtype=np.float64)
        weight_sum = np.zeros((H, out_w), dtype=np.float64)

        for idx in range(len(strips)):
            strip = strips[idx]
            gx = positions[idx]
            info = frame_infos[idx]

            x0 = gx - min_x
            x1 = x0 + strip_w

            edge = info.edge_map
            # 垂直ストリップの場合、エッジマップの形状が異なるので調整
            if edge.shape[1] == strip_w:
                weight = (1.0 - info.text_score) * (1.0 - np.clip(edge, 0, 0.8))
            else:
                weight = np.full((H, strip_w), 1.0 - info.text_score, dtype=np.float64)

            canvas_sum[:, x0:x1, :] += strip.astype(np.float64) * weight[:, :, np.newaxis]
            weight_sum[:, x0:x1] += weight

        weight_sum = np.maximum(weight_sum, 1e-6)
        canvas = (canvas_sum / weight_sum[:, :, np.newaxis]).astype(np.uint8)

    save_rgb(canvas, out_path)


def estimate_dx_phase(prev_rgb: np.ndarray, next_rgb: np.ndarray,
                      search: int = 40) -> Tuple[int, float]:
    """
    prev -> next のdx（水平シフト）を推定する。
    """
    gray_a = to_gray_f32(prev_rgb)
    gray_b = to_gray_f32(next_rgb)
    edge_a = sobel_edge(gray_a)
    edge_b = sobel_edge(gray_b)

    dx, dy, peak = phase_correlation(edge_a, edge_b)
    dx = int(np.clip(dx, -search, search))
    return dx, peak


def jigsaw_mode_horizontal(frames_rgb: List[np.ndarray], strips: List[np.ndarray],
                            out_dir: Path, strip_x: int, strip_w: int,
                            edge_thr: float = 0.15, search: int = 40,
                            scroll_dir: str = "both",
                            base_frame_idx: Optional[int] = None,
                            auto_exclude_text: bool = False,
                            match_method: str = "phase",
                            keyframe_mode: bool = False,
                            min_quality: float = 0.3,
                            max_keyframe_gap: int = 0,
                            allow_ignore_at_edges: bool = False,
                            ignore_regions: List[IgnoreRegion] = None,
                            keyframe_stitch_method: str = "position") -> None:
    """
    水平パン用ジグソーモード

    strips: 垂直ストリップのリスト
    scroll_dir: left (コンテンツが右に移動), right (コンテンツが左に移動), both
    match_method: phase, optical_flow, optical_flow_fb, optical_flow_lk
    keyframe_mode: キーフレーム選択モード（ハードカット）
    min_quality: キーフレーム選択の最低品質スコア
    keyframe_stitch_method: "position"=dx推定値を使用、"matching"=stitch_candidatesと同様にマッチング
    """
    print("\n" + "="*60)
    print("[Horizontal Pan Jigsaw Mode]")
    print("="*60)

    # Phase 1: フレーム分類
    print("\n[Phase 1] Classifying frames by text content...")
    frame_infos = []
    for i, strip in enumerate(strips):
        info = analyze_frame_text(strip, edge_thr=edge_thr)
        info.frame_idx = i
        frame_infos.append(info)

    text_free = [i for i, info in enumerate(frame_infos) if info.text_score < 0.05]
    text_light = [i for i, info in enumerate(frame_infos) if 0.05 <= info.text_score < 0.15]
    text_heavy = [i for i, info in enumerate(frame_infos) if info.text_score >= 0.15]

    print(f"  Text-free frames: {len(text_free)}")
    print(f"  Text-light frames: {len(text_light)}")
    print(f"  Text-heavy frames: {len(text_heavy)}")

    # Phase 2: dx推定
    print(f"\n[Phase 2] Estimating dx (method: {match_method})...")

    dxs = []
    peaks = []
    for i in range(1, len(frames_rgb)):
        # テキスト除外マスク
        text_mask = None
        if auto_exclude_text:
            text_mask = frame_infos[i-1].text_mask | frame_infos[i].text_mask

        if match_method in ("optical_flow", "optflow"):
            # オプティカルフロー（ロバスト）
            dx_val, dy_val, peak = optical_flow_estimate_robust(
                frames_rgb[i-1], frames_rgb[i], text_mask
            )
            dx_val = int(round(np.clip(dx_val, -search, search)))
        elif match_method == "optical_flow_fb":
            # Farneback法
            dx_val, dy_val, peak = optical_flow_estimate(
                frames_rgb[i-1], frames_rgb[i],
                mask=~text_mask if text_mask is not None else None,
                method="farneback"
            )
            dx_val = int(round(np.clip(dx_val, -search, search)))
        elif match_method == "optical_flow_lk":
            # Lucas-Kanade法
            dx_val, dy_val, peak = optical_flow_estimate(
                frames_rgb[i-1], frames_rgb[i],
                mask=~text_mask if text_mask is not None else None,
                method="lk"
            )
            dx_val = int(round(np.clip(dx_val, -search, search)))
        else:
            # デフォルト: 位相相関
            if auto_exclude_text and text_mask is not None:
                gray_a = to_gray_f32(frames_rgb[i-1])
                gray_b = to_gray_f32(frames_rgb[i])
                if text_mask.shape == gray_a.shape:
                    gray_a = apply_text_mask(gray_a, text_mask)
                    gray_b = apply_text_mask(gray_b, text_mask)
                edge_a = sobel_edge(gray_a)
                edge_b = sobel_edge(gray_b)
                dx_val, dy_val, peak = phase_correlation(edge_a, edge_b)
                dx_val = int(np.clip(dx_val, -search, search))
            else:
                dx_val, peak = estimate_dx_phase(frames_rgb[i-1], frames_rgb[i], search)

        dxs.append(dx_val)
        peaks.append(peak)

    # dx統計
    nonzero_dxs = [d for d in dxs if d != 0]
    if nonzero_dxs:
        dx_mean = np.mean(nonzero_dxs)
        dx_std = np.std(nonzero_dxs)
        print(f"  dx (raw): mean={dx_mean:.1f}, std={dx_std:.1f}")

        # 外れ値修正
        expected_sign = np.sign(dx_mean)
        for i in range(len(dxs)):
            if dxs[i] != 0 and np.sign(dxs[i]) != expected_sign:
                median_dx = int(np.median(nonzero_dxs))
                print(f"  Frame {i}: dx={dxs[i]} -> {median_dx} (wrong sign)")
                dxs[i] = median_dx

    # スムージング
    if len(dxs) >= 3:
        dxs = smooth_dy_values(dxs, window=5)  # 同じスムージング関数を使用
        nonzero_dxs = [d for d in dxs if d != 0]
        if nonzero_dxs:
            print(f"  dx (smoothed): mean={np.mean(nonzero_dxs):.1f}, std={np.std(nonzero_dxs):.1f}")

    # 位置計算
    pos_dx = [0]
    pos_negdx = [0]
    for dx in dxs:
        pos_dx.append(pos_dx[-1] + dx)
        pos_negdx.append(pos_negdx[-1] - dx)

    # Phase 3: キーフレーム選択またはレンダリング
    output_files = []

    if keyframe_mode:
        print("\n[Phase 3] Keyframe selection mode...")
        # 各フレームの品質スコアを計算
        print("  Computing frame quality scores...")
        frame_qualities = []
        for i, strip in enumerate(strips):
            fq = compute_frame_quality(strip, frame_infos[i])
            fq = FrameQuality(
                frame_idx=i,
                quality_score=fq.quality_score,
                sharpness=fq.sharpness,
                noise_score=fq.noise_score,
                text_score=fq.text_score,
                compression_artifact=fq.compression_artifact
            )
            frame_qualities.append(fq)

        quality_scores = [fq.quality_score for fq in frame_qualities]
        print(f"  Quality scores: min={min(quality_scores):.2f}, max={max(quality_scores):.2f}, mean={np.mean(quality_scores):.2f}")

        high_quality = sum(1 for q in quality_scores if q >= min_quality)
        print(f"  High quality frames (>={min_quality}): {high_quality}/{len(frame_qualities)}")

        # キーフレームモードではフルフレームを使用（strip-wではなくフレーム幅）
        frame_w = frames_rgb[0].shape[1]
        print(f"  Using full frame width: {frame_w}")
        print(f"  Stitch method: {keyframe_stitch_method}")

        if scroll_dir in ("left", "both"):
            keyframes_dx = select_keyframes(frame_qualities, pos_dx, frame_w,
                                            min_quality=min_quality, min_overlap=20,
                                            max_frame_gap=max_keyframe_gap)
            print(f"  Selected keyframes (dx): {len(keyframes_dx)} frames")
            print(f"    Indices: {keyframes_dx}")

            if keyframe_stitch_method == "matching" and HAS_STITCH_CANDIDATES:
                out1_base = out_dir / "recon_keyframe_dx"
                candidates = render_keyframe_stitch_auto(
                    frames_rgb, keyframes_dx, out1_base, axis="horizontal",
                    ignore_regions=ignore_regions, verbose=True
                )
                for cand_path, cand_score in candidates:
                    output_files.append(cand_path)
                if not candidates:
                    print("  Warning: No valid candidates from matching mode, falling back to position mode")
                    out1 = out_dir / "recon_keyframe_dx.png"
                    render_keyframe_stitch(frames_rgb, pos_dx, keyframes_dx, out1, axis="horizontal",
                                           ignore_regions=ignore_regions,
                                           allow_ignore_at_edges=allow_ignore_at_edges)
                    output_files.append(out1)
                    print(f"  Created: {out1}")
            else:
                out1 = out_dir / "recon_keyframe_dx.png"
                render_keyframe_stitch(frames_rgb, pos_dx, keyframes_dx, out1, axis="horizontal",
                                       ignore_regions=ignore_regions,
                                       allow_ignore_at_edges=allow_ignore_at_edges)
                output_files.append(out1)
                print(f"  Created: {out1}")

        if scroll_dir in ("right", "both"):
            keyframes_negdx = select_keyframes(frame_qualities, pos_negdx, frame_w,
                                               min_quality=min_quality, min_overlap=20,
                                               max_frame_gap=max_keyframe_gap)
            print(f"  Selected keyframes (negdx): {len(keyframes_negdx)} frames")
            print(f"    Indices: {keyframes_negdx}")

            if keyframe_stitch_method == "matching" and HAS_STITCH_CANDIDATES:
                out2_base = out_dir / "recon_keyframe_negdx"
                candidates = render_keyframe_stitch_auto(
                    frames_rgb, keyframes_negdx, out2_base, axis="horizontal",
                    ignore_regions=ignore_regions, reverse_order=True, verbose=True
                )
                for cand_path, cand_score in candidates:
                    output_files.append(cand_path)
                if not candidates:
                    print("  Warning: No valid candidates from matching mode, falling back to position mode")
                    out2 = out_dir / "recon_keyframe_negdx.png"
                    render_keyframe_stitch(frames_rgb, pos_negdx, keyframes_negdx, out2, axis="horizontal",
                                           ignore_regions=ignore_regions,
                                           allow_ignore_at_edges=allow_ignore_at_edges)
                    output_files.append(out2)
                    print(f"  Created: {out2}")
            else:
                out2 = out_dir / "recon_keyframe_negdx.png"
                render_keyframe_stitch(frames_rgb, pos_negdx, keyframes_negdx, out2, axis="horizontal",
                                       ignore_regions=ignore_regions,
                                       allow_ignore_at_edges=allow_ignore_at_edges)
                output_files.append(out2)
                print(f"  Created: {out2}")

        # 品質デバッグCSV出力
        quality_csv_path = out_dir / "debug_keyframe_quality.csv"
        with quality_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "quality_score", "sharpness", "text_score",
                           "compression", "noise", "selected_dx", "selected_negdx"])
            for fq in frame_qualities:
                selected_dx = "Y" if scroll_dir in ("left", "both") and fq.frame_idx in keyframes_dx else ""
                selected_negdx = "Y" if scroll_dir in ("right", "both") and fq.frame_idx in keyframes_negdx else ""
                writer.writerow([
                    fq.frame_idx,
                    f"{fq.quality_score:.4f}",
                    f"{fq.sharpness:.4f}",
                    f"{fq.text_score:.4f}",
                    f"{fq.compression_artifact:.4f}",
                    f"{fq.noise_score:.4f}",
                    selected_dx,
                    selected_negdx
                ])
        print(f"  Quality CSV: {quality_csv_path}")

    else:
        print("\n[Phase 3] Rendering jigsaw candidates...")

        if base_frame_idx is not None:
            actual_idx = base_frame_idx if base_frame_idx >= 0 else len(strips) + base_frame_idx
            if actual_idx < 0 or actual_idx >= len(strips):
                actual_idx = len(strips) - 1
            print(f"  Base frame mode: using frame {actual_idx} as foundation")
            suffix = f"_base{actual_idx}"
        else:
            suffix = ""

        if scroll_dir in ("left", "both"):
            out1 = out_dir / f"recon_horizontal_dx{suffix}.png"
            render_jigsaw_horizontal(
                strips=strips, positions=pos_dx,
                frame_infos=frame_infos, out_path=out1,
                base_frame_idx=base_frame_idx
            )
            output_files.append(out1)
            print(f"  Created: {out1}")

        if scroll_dir in ("right", "both"):
            out2 = out_dir / f"recon_horizontal_negdx{suffix}.png"
            render_jigsaw_horizontal(
                strips=strips, positions=pos_negdx,
                frame_infos=frame_infos, out_path=out2,
                base_frame_idx=base_frame_idx
            )
            output_files.append(out2)
            print(f"  Created: {out2}")

    # デバッグCSV
    csv_path = out_dir / "debug_horizontal.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "text_score", "dx", "peak", "pos_dx", "pos_negdx"])
        for i in range(len(frame_infos)):
            dx = dxs[i] if i < len(dxs) else ""
            peak = peaks[i] if i < len(peaks) else ""
            writer.writerow([
                i,
                f"{frame_infos[i].text_score:.4f}",
                dx,
                f"{peak:.4f}" if isinstance(peak, float) else peak,
                pos_dx[i],
                pos_negdx[i]
            ])
    print(f"  Debug CSV: {csv_path}")

    print("\n" + "="*60)
    print("Done!")
    for f in output_files:
        print(f"  {f}")
    print("="*60)


def visualize_frame_classification(strips: List[np.ndarray],
                                    frame_infos: List[FrameTextInfo],
                                    out_path: Path,
                                    max_frames: int = 50) -> None:
    """
    フレーム分類の可視化: テキストスコアに応じて色分けした帯を並べる
    """
    n_frames = min(len(strips), max_frames)
    strip_h, W, _ = strips[0].shape

    # サムネイル幅
    thumb_w = 80
    thumb_h = int(strip_h * thumb_w / W)

    # キャンバス
    cols = 10
    rows = (n_frames + cols - 1) // cols
    canvas_h = rows * (thumb_h + 20)
    canvas_w = cols * (thumb_w + 10)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i in range(n_frames):
        row = i // cols
        col = i % cols
        x = col * (thumb_w + 10) + 5
        y = row * (thumb_h + 20) + 5

        # サムネイル
        if HAS_CV2:
            thumb = cv2.resize(strips[i], (thumb_w, thumb_h))
        else:
            img = Image.fromarray(strips[i])
            img = img.resize((thumb_w, thumb_h), Image.LANCZOS)
            thumb = np.array(img)

        canvas[y:y+thumb_h, x:x+thumb_w] = thumb

        # テキストスコアに応じた枠色
        info = frame_infos[i]
        if info.text_score < 0.05:
            color = (0, 200, 0)  # 緑: テキストなし
        elif info.text_score < 0.15:
            color = (200, 200, 0)  # 黄: テキスト薄
        else:
            color = (200, 0, 0)  # 赤: テキスト濃

        # 枠を描画
        canvas[y:y+2, x:x+thumb_w] = color
        canvas[y+thumb_h-2:y+thumb_h, x:x+thumb_w] = color
        canvas[y:y+thumb_h, x:x+2] = color
        canvas[y:y+thumb_h, x+thumb_w-2:x+thumb_w] = color

    save_rgb(canvas, out_path)


def visualize_template_candidates(rgb: np.ndarray,
                                   candidates: List[Dict],
                                   out_path: Path,
                                   top_n: int = 10) -> None:
    """
    候補領域を可視化した画像を保存する。
    """
    img = rgb.copy()
    h, w = img.shape[:2]

    # 上位N件を描画
    for rank, cand in enumerate(candidates[:top_n], 1):
        y1, y2, x1, x2 = cand['region']
        stability = cand['stability']

        # 色: 安定性に応じて緑→黄→赤
        if stability >= 0.7:
            color = (0, 255, 0)  # 緑
        elif stability >= 0.4:
            color = (255, 255, 0)  # 黄
        else:
            color = (255, 0, 0)  # 赤

        # 矩形を描画
        thickness = 3 if rank <= 3 else 2
        img[y1:y1 + thickness, x1:x2] = color
        img[y2 - thickness:y2, x1:x2] = color
        img[y1:y2, x1:x1 + thickness] = color
        img[y1:y2, x2 - thickness:x2] = color

        # ランク番号を左上に描画（簡易的に矩形で）
        label_h, label_w = 20, 25
        ly, lx = max(0, y1 - label_h), x1
        img[ly:ly + label_h, lx:lx + label_w] = color

    save_rgb(img, out_path)


def suggest_templates_mode(frames_rgb: List[np.ndarray],
                            out_dir: Path,
                            n_candidates: int = 15,
                            max_eval_frames: int = None) -> None:
    """
    テンプレート候補を探索・検証・可視化するモード。
    """
    print(f"\n[Template Suggestion Mode]")
    print(f"  Analyzing {len(frames_rgb)} frames...")

    # 1. 候補領域を探索
    print(f"\n  Step 1: Finding candidate regions...")
    first_frame = frames_rgb[0]
    candidates_raw = find_template_candidates(first_frame, n_candidates=n_candidates)
    print(f"    Found {len(candidates_raw)} candidates")

    # 2. 各候補を全フレームで検証
    print(f"\n  Step 2: Evaluating stability across all frames...")
    candidates_evaluated = []
    for i, (score, y1, y2, x1, x2) in enumerate(candidates_raw):
        print(f"    [{i + 1}/{len(candidates_raw)}] Region y={y1},h={y2 - y1},x={x1},w={x2 - x1}...", end="", flush=True)
        result = evaluate_template_candidate(
            frames_rgb,
            (y1, y2, x1, x2),
            max_frames=max_eval_frames
        )
        result['edge_score'] = score
        candidates_evaluated.append(result)
        print(f" stability={result['stability']:.3f}")

    # 3. 安定性でソート
    candidates_evaluated.sort(key=lambda x: x['stability'], reverse=True)

    # 4. 結果を出力
    print(f"\n  Step 3: Results (sorted by stability)")
    print("=" * 80)
    print(f"{'Rank':<5} {'Stability':<10} {'Mean Score':<12} {'Lost %':<8} {'dy_std':<8} {'--template-region':<25}")
    print("-" * 80)

    for rank, cand in enumerate(candidates_evaluated[:10], 1):
        y1, y2, x1, x2 = cand['region']
        region_str = f'"{y1},{y2 - y1},{x1},{x2 - x1}"'
        lost_pct = cand['lost_ratio'] * 100
        print(f"{rank:<5} {cand['stability']:<10.3f} {cand['mean_score']:<12.3f} {lost_pct:<8.1f} {cand['dy_std']:<8.1f} {region_str:<25}")

    print("=" * 80)

    # ベスト候補のコマンド例
    if candidates_evaluated:
        best = candidates_evaluated[0]
        y1, y2, x1, x2 = best['region']
        print(f"\n  Recommended command:")
        print(f'    --template-region "{y1},{y2 - y1},{x1},{x2 - x1}"')

        if best['stability'] < 0.5:
            print(f"\n  Warning: Best stability is low ({best['stability']:.3f})")
            print(f"    Consider using --match-method phase instead")

    # 5. 可視化画像を保存
    vis_path = out_dir / "template_candidates.png"
    visualize_template_candidates(first_frame, candidates_evaluated, vis_path)
    print(f"\n  Visualization saved: {vis_path}")

    # 6. 詳細CSVを保存
    csv_path = out_dir / "template_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["rank", "y", "h", "x", "w", "stability", "mean_score", "min_score", "std_score", "lost_ratio", "dy_std", "edge_score"])
        for rank, cand in enumerate(candidates_evaluated, 1):
            y1, y2, x1, x2 = cand['region']
            writer.writerow([
                rank, y1, y2 - y1, x1, x2 - x1,
                f"{cand['stability']:.4f}",
                f"{cand['mean_score']:.4f}",
                f"{cand['min_score']:.4f}",
                f"{cand['std_score']:.4f}",
                f"{cand['lost_ratio']:.4f}",
                f"{cand['dy_std']:.2f}",
                f"{cand['edge_score']:.4f}",
            ])
    print(f"  Details saved: {csv_path}")


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
        elif method == "optical_flow" or method == "optflow":
            # オプティカルフローによるdy推定
            dx, dy, confidence = optical_flow_estimate_robust(prev_rgb, next_rgb)
            dy = int(round(np.clip(dy, -search, search)))
            dx = int(round(dx))
            results.append(MatchResult("optical_flow", dx, dy, confidence))
        elif method == "optical_flow_fb" or method == "optflow_fb":
            # Farneback法のみ
            dx, dy, confidence = optical_flow_estimate(prev_rgb, next_rgb, method="farneback")
            dy = int(round(np.clip(dy, -search, search)))
            dx = int(round(dx))
            results.append(MatchResult("optical_flow_fb", dx, dy, confidence))
        elif method == "optical_flow_lk" or method == "optflow_lk":
            # Lucas-Kanade法のみ
            dx, dy, confidence = optical_flow_estimate(prev_rgb, next_rgb, method="lk")
            dy = int(round(np.clip(dy, -search, search)))
            dx = int(round(dx))
            results.append(MatchResult("optical_flow_lk", dx, dy, confidence))

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
    """水平ストリップを抽出（垂直スクロール用）"""
    H, W, _ = rgb.shape
    y0 = max(0, min(H, y))
    y1 = max(0, min(H, y + h))
    return rgb[y0:y1, :, :]


def crop_vertical_strip(rgb: np.ndarray, x: int, w: int) -> np.ndarray:
    """垂直ストリップを抽出（水平パン用）"""
    H, W, _ = rgb.shape
    x0 = max(0, min(W, x))
    x1 = max(0, min(W, x + w))
    return rgb[:, x0:x1, :]


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

  # テンプレート候補を探索（全フレームで安定性を検証）
  python video_strip_reconstruct.py --frames "frames/*.png" \\
    --strip-y 0 --strip-h 1080 --suggest-templates --out outdir

  # ジグソーモード: テキストの有無を考慮した賢い合成
  python video_strip_reconstruct.py --frames "frames/*.png" \\
    --strip-y 0 --strip-h 1080 --jigsaw --out outdir
""")
    ap.add_argument("--video", type=str, default="", help="Input video path (optional if --frames is used)")
    ap.add_argument("--frames", type=str, default="", help='Glob for extracted frames, e.g. "frames/*.png"')
    ap.add_argument("--out", type=str, default="recon_out", help="Output directory")

    ap.add_argument("--fps", type=float, default=2.0, help="Extraction fps when using --video")
    ap.add_argument("--start", type=float, default=None, help="Start seconds for extraction")
    ap.add_argument("--dur", type=float, default=None, help="Duration seconds for extraction")
    ap.add_argument("--deinterlace", action="store_true", help="Apply yadif when extracting frames")

    ap.add_argument("--strip-y", type=int, default=None, help="Top Y of strip in full frame (px) - for vertical scroll")
    ap.add_argument("--strip-h", type=int, default=100, help="Height of strip (px)")
    ap.add_argument("--strip-x", type=int, default=None, help="Left X of strip in full frame (px) - for horizontal pan")
    ap.add_argument("--strip-w", type=int, default=100, help="Width of strip (px)")
    ap.add_argument("--scroll-axis", type=str, default="vertical", choices=["vertical", "horizontal"],
                    help="Scroll axis: vertical (default, use --strip-y) or horizontal (use --strip-x)")
    ap.add_argument("--dy-region", type=str, default="",
                    help="Region for dy estimation: 'y,h' (e.g. '0,200' for top 200px). If empty, use full frame.")

    ap.add_argument("--dy-search", type=int, default=40, help="Clamp dy/dx to +/- this")
    ap.add_argument("--edge-thr", type=str, default="0.35",
                    help="Edge threshold(s) for 'text-like' masking. Comma-separated for multiple candidates (e.g. 0.3,0.4,0.5)")
    ap.add_argument("--no-edges", action="store_true", help="Use raw grayscale (not edges) for dy estimation")

    ap.add_argument("--min-peak", type=float, default=0.0,
                    help="Minimum peak score for reliable dy estimation. Shows diagnostic when below (default: 0=disabled)")
    ap.add_argument("--match-method", type=str, default="phase",
                    help="Matching method(s): phase, ncc_gray, ncc_edge, phase_gray, template. Comma-separated (default: phase)")
    ap.add_argument("--scroll-dir", type=str, default="both", choices=["up", "down", "left", "right", "both"],
                    help="Scroll direction: up (content moves down), down (content moves up), both (output both candidates, default)")
    ap.add_argument("--static-bg", action="store_true",
                    help="Static background mode: use temporal median/min-edge to remove scrolling text (background does not scroll)")
    ap.add_argument("--static-method", type=str, default="median", choices=["median", "min_edge"],
                    help="Method for static-bg mode: median (default) or min_edge")
    ap.add_argument("--uniform-dy", type=float, default=None,
                    help="Use uniform dy per frame instead of estimation (e.g. -10 for 10px up per frame)")
    ap.add_argument("--template-region", type=str, default="",
                    help="Manual template region for template matching: 'y,h,x,w' (e.g. '350,150,750,150')")
    ap.add_argument("--suggest-templates", action="store_true",
                    help="Suggest best template regions by testing stability across all frames")
    ap.add_argument("--suggest-n", type=int, default=15,
                    help="Number of template candidates to evaluate (default: 15)")
    ap.add_argument("--jigsaw", action="store_true", default=True,
                    help="Jigsaw mode: intelligent reconstruction using text-free frames and clean bands (default: enabled)")
    ap.add_argument("--no-jigsaw", action="store_true",
                    help="Disable jigsaw mode, use basic reconstruction")
    ap.add_argument("--base-frame", type=int, default=None,
                    help="Base frame index for jigsaw mode (-1=last, -2=second-to-last, etc). Base frame is used as foundation, other frames fill gaps.")
    ap.add_argument("--auto-exclude-text", action="store_true",
                    help="Auto-detect and exclude text regions from dy estimation (per-frame detection)")
    ap.add_argument("--keyframe-mode", action="store_true", default=True,
                    help="Keyframe selection mode: select high-quality frames and stitch with hard cuts (default: enabled)")
    ap.add_argument("--no-keyframe-mode", action="store_true",
                    help="Disable keyframe mode, use traditional jigsaw blending")
    ap.add_argument("--min-quality", type=float, default=0.3,
                    help="Minimum quality score for keyframe selection (default: 0.3)")
    ap.add_argument("--max-keyframe-gap", type=int, default=5,
                    help="Maximum frame gap between keyframes (0=auto, >0 forces intermediate frames, default: 5)")
    ap.add_argument("--allow-ignore-at-edges", action="store_true",
                    help="Allow ignore regions at edge frames (first/last positioned keyframes)")
    ap.add_argument("--keyframe-stitch-method", choices=["position", "matching"], default="matching",
                    help="Keyframe stitching method: 'matching' uses stitch_candidates.py logic (default), 'position' uses estimated dy/dx")

    ap.add_argument("--ignore", action="append", default=[],
                    help='Ignore region (px): "top,(right|left|X),width,height"  e.g. "0,right,220,120"')
    ap.add_argument("--ignore-pct", action="append", default=[],
                    help='Ignore region (pct): "top,(right|left|X),width,height"  e.g. "0,right,12,8"')

    args = ap.parse_args()

    # Handle --no-keyframe-mode flag
    if args.no_keyframe_mode:
        args.keyframe_mode = False

    # Handle --no-jigsaw flag
    if args.no_jigsaw:
        args.jigsaw = False

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

    # Suggest templates mode
    if args.suggest_templates:
        suggest_templates_mode(frames_rgb, out_dir, n_candidates=args.suggest_n)
        return

    # crop strips (horizontal or vertical depending on scroll axis)
    if args.scroll_axis == "horizontal":
        # 水平パンモード: 垂直ストリップを抽出
        if args.strip_x is None:
            args.strip_x = 0  # デフォルトは左端
        strips = [crop_vertical_strip(fr, x=args.strip_x, w=args.strip_w) for fr in frames_rgb]
        strip_dim = "width"
        expected_size = args.strip_w
    else:
        # 垂直スクロールモード: 水平ストリップを抽出
        if args.strip_y is None:
            args.strip_y = 0  # デフォルトは上端
        strips = [crop_strip(fr, y=args.strip_y, h=args.strip_h) for fr in frames_rgb]
        strip_dim = "height"
        expected_size = args.strip_h

    # ensure consistent size
    if args.scroll_axis == "horizontal":
        actual_size = strips[0].shape[1]  # width
        for s in strips:
            if s.shape[1] != actual_size or s.shape[0] != strips[0].shape[0]:
                raise SystemExit("All strips must have same size")
        if actual_size != expected_size:
            print(f"Warning: strip-w clipped to {actual_size} due to frame bounds")
    else:
        W = strips[0].shape[1]
        Hs = strips[0].shape[0]
        if Hs != args.strip_h:
            print(f"Warning: strip-h clipped to {Hs} due to frame bounds")
        for s in strips:
            if s.shape[1] != W or s.shape[0] != Hs:
                raise SystemExit("All strips must have same size")

    # Jigsaw mode
    if args.jigsaw:
        if args.scroll_axis == "horizontal":
            # 水平パンモード
            jigsaw_mode_horizontal(
                frames_rgb=frames_rgb,
                strips=strips,
                out_dir=out_dir,
                strip_x=args.strip_x, strip_w=args.strip_w,
                edge_thr=edge_thrs[0],
                search=args.dy_search,
                scroll_dir=args.scroll_dir,
                base_frame_idx=args.base_frame,
                auto_exclude_text=args.auto_exclude_text,
                match_method=match_methods[0],
                keyframe_mode=args.keyframe_mode,
                min_quality=args.min_quality,
                max_keyframe_gap=args.max_keyframe_gap,
                allow_ignore_at_edges=args.allow_ignore_at_edges,
                ignore_regions=ignore_regions,
                keyframe_stitch_method=args.keyframe_stitch_method
            )
        else:
            # 垂直スクロールモード
            jigsaw_mode(
                frames_rgb=frames_rgb,
                strips=strips,
                out_dir=out_dir,
                ignore_regions=ignore_regions,
                full_w=full_w, full_h=full_h,
                strip_y=args.strip_y, strip_h=args.strip_h,
                edge_thr=edge_thrs[0],
                search=args.dy_search,
                scroll_dir=args.scroll_dir,
                match_method=match_methods[0],
                base_frame_idx=args.base_frame,
                auto_exclude_text=args.auto_exclude_text,
                keyframe_mode=args.keyframe_mode,
                min_quality=args.min_quality,
                max_keyframe_gap=args.max_keyframe_gap,
                allow_ignore_at_edges=args.allow_ignore_at_edges,
                keyframe_stitch_method=args.keyframe_stitch_method
            )
        return

    # (strips already cropped above)

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
