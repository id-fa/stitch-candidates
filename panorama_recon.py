#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
panorama_recon.py

動画（スクロール/パン）から背景を再構成するツール。
既存の逐次連結方式とは異なり、以下の設計で「1枚の正解」を出す。

1. グローバル位置合わせ
   - 隣接ペアだけでなく (i, i+1), (i, i+2), (i, i+4) ... のペアで
     マスク付き NCC（FFT、Padfield 法）によりサブピクセル平行移動を推定
   - 全フレームの位置を最小二乗（IRLS で外れ値に頑健）で一括解決
   - 逐次連結の累積誤差が原理的に発生しない
2. 静止オーバーレイ検出
   - 画面座標で静止している高エッジ画素（クレジット文字、ロゴ）を
     時間差分から自動検出し、位置合わせと合成の両方から除外
3. 時間的中央値による合成
   - 整列済みスタックのピクセルごとの中央値で文字や光の粒子を除去
   - 中央値に近い候補の中で最もシャープなフレームの画素を採用した版も出力

運動モデルはフレームごとの相似変換（平行移動 + ズーム、オプションで回転）。
粗探索はスケール格子 × マスク付き NCC、精密化は Gauss-Newton 直接法（マスク・Huber 重み付き）。

必須:
  pip install pillow numpy torch
推奨:
  pip install opencv-python   (動画読み込みに使用。無い場合は --frames を使う)
  CUDA 対応 GPU（無ければ CPU で動作。遅いが結果は同じ）
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    sys.exit("panorama_recon.py には torch が必要です: pip install torch")


def log(msg: str) -> None:
    print(msg, flush=True)


# -------------------------
# フレーム読み込み
# -------------------------

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def expand_globs(args: List[str]) -> List[Path]:
    out: List[Path] = []
    for a in args:
        if any(ch in a for ch in "*?["):
            out.extend(Path(p) for p in sorted(glob.glob(a), key=_natural_key))
        else:
            out.append(Path(a))
    return out


def apply_crop(img: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """crop = (x, y, w, h)。None ならそのまま。範囲は画像内にクリップ。"""
    if crop is None:
        return img
    x, y, w, h = crop
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        sys.exit(f"--crop {crop} が画像 {W}x{H} の外か小さすぎます")
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def load_frames_from_paths(paths: List[Path], every: int, max_frames: int,
                           start_frame: Optional[int] = None, end_frame: Optional[int] = None,
                           crop: Optional[Tuple[int, int, int, int]] = None) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    s0 = start_frame or 0
    for k, p in enumerate(paths):
        if k < s0 or (end_frame is not None and k > end_frame):
            continue
        if (k - s0) % every != 0:
            continue
        frames.append(apply_crop(np.array(Image.open(p).convert("RGB")), crop))
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def pad_to_common_size(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], Optional[List[Tuple[int, int]]]]:
    """
    サイズの異なるフレーム（スクリーンショット等）を最大サイズに揃える。右・下を端の画素の複製で埋め（ハイパス等の
    境界応答を抑えるため黒ではなく複製）、元のサイズを返す。全て同じサイズなら sizes は None。
    """
    sizes = [(f.shape[0], f.shape[1]) for f in frames]
    H = max(h for h, _ in sizes)
    W = max(w for _, w in sizes)
    if all(s == (H, W) for s in sizes):
        return frames, None
    out = []
    for f, (h, w) in zip(frames, sizes):
        out.append(np.pad(f, ((0, H - h), (0, W - w), (0, 0)), mode="edge") if (h, w) != (H, W) else f)
    return out, sizes


def load_frames_from_video(path: Path, every: int, fps: Optional[float],
                           start: float, duration: Optional[float],
                           max_frames: int,
                           start_frame: Optional[int] = None, end_frame: Optional[int] = None,
                           crop: Optional[Tuple[int, int, int, int]] = None) -> Tuple[List[np.ndarray], float]:
    """start_frame / end_frame（両端含む）が指定されていれば秒指定より優先。crop は (x, y, w, h)。"""
    try:
        import cv2  # type: ignore
    except ImportError:
        sys.exit("--video には opencv-python が必要です。無い場合は ffmpeg でフレーム抽出して --frames を使ってください")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"動画を開けません: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, every)
    if fps and fps > 0:
        step = max(1, int(round(src_fps / fps)))
    start_f = int(round(start * src_fps))
    end_f = None if duration is None else start_f + int(round(duration * src_fps))
    if start_frame is not None:
        start_f = max(0, int(start_frame))
    if end_frame is not None:
        end_f = int(end_frame) + 1
    if start_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    frames: List[np.ndarray] = []
    idx = start_f
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if end_f is not None and idx >= end_f:
            break
        if (idx - start_f) % step == 0:
            frames.append(apply_crop(np.ascontiguousarray(bgr[:, :, ::-1]), crop))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, src_fps / step


# -------------------------
# 実写向け前処理: インターレース検出 / デインターレース、レンズ歪み補正
# -------------------------

def comb_fraction(gray: np.ndarray, thr: float = 10.0) -> Tuple[float, float]:
    """
    インターレースの櫛状パターンの割合 (縦, 横)。4 行にわたって明暗が交互になる画素（(g[y-1]-g[y])(g[y+1]-g[y]) > thr² が
    隣り合う 2 行で連続）の比率を縦方向と、同じ計算を横方向（列）で行ったものを返す。プログレッシブ素材のテクスチャは
    等方的なので縦 ≈ 横、インターレースの動きのある部分では縦 ≫ 横になる。
    """
    g = gray.astype(np.float32)

    def alt(a: np.ndarray) -> float:
        d1 = a[:-2] - a[1:-1]
        d2 = a[2:] - a[1:-1]
        c = (d1 * d2) > thr * thr          # 行 y が上下と同符号で外れる（極値）
        return float((c[:-1] & c[1:]).mean())  # 隣り合う 2 行で連続
    return alt(g), alt(g.T)


def is_interlaced(cv: float, ch: float) -> bool:
    return cv > 0.01 and cv > 3.0 * ch


def deinterlace_frame(img: np.ndarray) -> np.ndarray:
    """トップフィールド（偶数行）を残し、奇数行を上下の偶数行の平均で補間する（片フィールド線形補間）。"""
    out = img.astype(np.float32)
    H = out.shape[0]
    odd_rows = np.arange(1, H, 2)
    prev_rows = odd_rows - 1
    next_rows = np.minimum(odd_rows + 1, H - 1)
    next_rows = np.where(next_rows % 2 == 1, prev_rows, next_rows)   # 末尾の奇数行は上の行をそのまま
    out[odd_rows] = 0.5 * (out[prev_rows] + out[next_rows])
    return (out + 0.5).astype(np.uint8)


def undistort_tensor(x: torch.Tensor, k1: float, mode: str = "bilinear") -> torch.Tensor:
    """
    (C, H, W) の放射歪み補正。モデル: x_d = c + (x_u - c)(1 + k1 r_u²)、r_u = |x_u - c| / 半対角。
    補正画像 U(x_u) = I(x_d(x_u)) をサンプリングで作る。範囲外は 0（k1 > 0 の隅）。
    """
    C, H, W = x.shape
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    rn = math.hypot(cx, cy)
    ys, xs = torch.meshgrid(torch.arange(H, device=x.device, dtype=torch.float32),
                            torch.arange(W, device=x.device, dtype=torch.float32), indexing="ij")
    ux, uy = (xs - cx) / rn, (ys - cy) / rn
    fct = 1.0 + k1 * (ux * ux + uy * uy)
    xd = cx + (xs - cx) * fct
    yd = cy + (ys - cy) * fct
    grid = torch.stack([2.0 * xd / (W - 1) - 1.0, 2.0 * yd / (H - 1) - 1.0], dim=-1)[None]
    return F.grid_sample(x[None], grid, mode=mode, padding_mode="zeros", align_corners=True)[0]


def undistort_crop_rect(W: int, H: int, k1: float) -> Tuple[int, int, int, int]:
    """k1 > 0（糸巻き）では隅が範囲外になるので、全画素が有効な中央の矩形 (x, y, w, h) を返す。k1 <= 0 なら全体。"""
    if k1 <= 0:
        return 0, 0, W, H
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    rn = math.hypot(cx, cy)
    t = 1.0
    for _ in range(200):
        hx, hy = t * cx, t * cy
        r2 = (hx * hx + hy * hy) / (rn * rn)
        if hx * (1.0 + k1 * r2) <= cx and hy * (1.0 + k1 * r2) <= cy:
            break
        t -= 0.005
    x0, y0 = int(math.ceil(cx - t * cx)), int(math.ceil(cy - t * cy))
    return x0, y0, W - 2 * x0, H - 2 * y0


def undistort_frames(frames: List[np.ndarray], k1: float, dev: torch.device) -> List[np.ndarray]:
    H, W = frames[0].shape[:2]
    x0, y0, w, h = undistort_crop_rect(W, H, k1)
    out = []
    for f in frames:
        t = torch.from_numpy(f).to(dev).permute(2, 0, 1).float()
        u = undistort_tensor(t, k1, mode="bicubic").clamp(0, 255)
        u = u[:, y0:y0 + h, x0:x0 + w]
        out.append((u + 0.5).byte().permute(1, 2, 0).cpu().numpy())
    return out


# -------------------------
# 画像処理ユーティリティ (torch)
# -------------------------

def box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """(..., H, W) に対する分離型ボックスぼかし。k は奇数。"""
    if k <= 1:
        return x
    shp = x.shape
    x4 = x.reshape(1, -1, shp[-2], shp[-1])
    y = F.avg_pool2d(x4, (1, k), stride=1, padding=(0, k // 2), count_include_pad=False)
    y = F.avg_pool2d(y, (k, 1), stride=1, padding=(k // 2, 0), count_include_pad=False)
    return y.reshape(shp)


def dilate(mask: torch.Tensor, k: int) -> torch.Tensor:
    """(H, W) bool/float マスクの膨張。"""
    if k <= 1:
        return mask
    m = mask.float()[None, None]
    return F.max_pool2d(m, k, stride=1, padding=k // 2)[0, 0] > 0.5


def highpass(gray: torch.Tensor, k: int) -> torch.Tensor:
    return gray - box_blur(gray, k)


def gradient_mag(gray: torch.Tensor) -> torch.Tensor:
    g = gray[None, None]
    gp = F.pad(g, (1, 1, 1, 1), mode="replicate")[0, 0]
    gx = (gp[1:-1, 2:] - gp[1:-1, :-2]) * 0.5
    gy = (gp[2:, 1:-1] - gp[:-2, 1:-1]) * 0.5
    return torch.maximum(gx.abs(), gy.abs())


def laplacian_abs(gray: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                     device=gray.device, dtype=gray.dtype)[None, None]
    g = F.pad(gray[None, None], (1, 1, 1, 1), mode="replicate")
    return F.conv2d(g, k)[0, 0].abs()


def odd(n: float) -> int:
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def downsample_area(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """(H, W) → (h, w) 面積平均縮小。"""
    return F.interpolate(x[None, None], size=size, mode="area")[0, 0]



# -------------------------
# マスク付き NCC (Padfield 2012, FFT)
# -------------------------

def masked_ncc_surface(A: torch.Tensor, MA: torch.Tensor,
                       B: torch.Tensor, MB: torch.Tensor,
                       P: int, Q: int, nmin: float) -> torch.Tensor:
    """
    全シフト d に対する corr(d) = Σ_x A(x+d) B(x) ベースの正規化相互相関。
    A, B のサイズは異なってもよい。P >= hA+hB, Q >= wA+wB なら折り返しなし。
    戻り値 ncc[P, Q]: index → shift は (idx if idx < hA else idx - P)。
    d = p_B - p_A（B の画素 x が A の画素 x+d に対応）。
    N < nmin の位置は -2 で無効化。
    """
    X = torch.stack([A * MA, A * A * MA, MA])
    Y = torch.stack([MB, B * MB, B * B * MB])
    FX = torch.fft.rfft2(X, s=(P, Q))
    FY = torch.conj(torch.fft.rfft2(Y, s=(P, Q)))
    prods = torch.stack([
        FX[2] * FY[0],  # N   = corr(MA, MB)
        FX[0] * FY[0],  # Sf  = corr(A MA, MB)
        FX[2] * FY[1],  # Sm  = corr(MA, B MB)
        FX[1] * FY[0],  # Sff = corr(A² MA, MB)
        FX[2] * FY[2],  # Smm = corr(MA, B² MB)
        FX[0] * FY[1],  # Sfm = corr(A MA, B MB)
    ])
    C = torch.fft.irfft2(prods, s=(P, Q))
    N, Sf, Sm, Sff, Smm, Sfm = C[0], C[1], C[2], C[3], C[4], C[5]
    Nc = N.clamp(min=1.0)
    num = Sfm - Sf * Sm / Nc
    vf = (Sff - Sf * Sf / Nc).clamp(min=0)
    vm = (Smm - Sm * Sm / Nc).clamp(min=0)
    ncc = num / torch.sqrt(vf * vm + 1e-12)
    ncc = torch.where(N < nmin, torch.full_like(ncc, -2.0), ncc)
    return ncc


def _subpix(cm: float, c0: float, cp: float) -> float:
    den = cm - 2.0 * c0 + cp
    if den >= 0 or abs(den) < 1e-12:
        return 0.0
    return float(max(-0.5, min(0.5, 0.5 * (cm - cp) / den)))


def peak_full(ncc: torch.Tensor, hA: int, wA: int) -> Tuple[int, int, float]:
    """全面サーフェスの最大位置 → (dx, dy, score)。"""
    P, Q = ncc.shape
    flat = int(torch.argmax(ncc).item())
    py, px = divmod(flat, Q)
    dy = py if py < hA else py - P
    dx = px if px < wA else px - Q
    return dx, dy, float(ncc[py, px].item())


def peak_window(ncc: torch.Tensor, r: int) -> Tuple[float, float, float]:
    """±r の窓内の最大位置（サブピクセル補間付き） → (dx, dy, score)。"""
    P, Q = ncc.shape
    idx = torch.arange(-r, r + 1, device=ncc.device)
    win = ncc[(idx % P)[:, None], (idx % Q)[None, :]]
    flat = int(torch.argmax(win).item())
    wy, wx = divmod(flat, 2 * r + 1)
    score = float(win[wy, wx].item())
    w = win.cpu().numpy()
    sx = _subpix(w[wy, wx - 1], score, w[wy, wx + 1]) if 0 < wx < 2 * r else 0.0
    sy = _subpix(w[wy - 1, wx], score, w[wy + 1, wx]) if 0 < wy < 2 * r else 0.0
    return float(wx - r + sx), float(wy - r + sy), score


def resample(x: torch.Tensor, s: float, is_mask: bool = False) -> Tuple[torch.Tensor, float]:
    """(H, W) を倍率 s で再標本化（原点は画素境界 (-0.5,-0.5)）。戻り値: (画像, 実効倍率)。"""
    h, w = x.shape
    hs, ws = max(8, int(round(h * s))), max(8, int(round(w * s)))
    if (hs, ws) == (h, w):
        return x, 1.0
    y = F.interpolate(x[None, None], size=(hs, ws), mode="bilinear",
                      align_corners=False, antialias=(s < 1.0))[0, 0]
    if is_mask:
        y = (y > 0.5).float()
    return y, hs / h


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """(..., H, W) の分離型ガウスぼかし。"""
    if sigma <= 0.05:
        return x
    r = max(1, int(math.ceil(3.0 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    shp = x.shape
    x4 = x.reshape(1, -1, shp[-2], shp[-1])
    x4 = F.pad(x4, (r, r, r, r), mode="replicate")
    y = F.conv2d(x4, k.view(1, 1, 1, -1).expand(x4.shape[1], 1, 1, -1), groups=x4.shape[1])
    y = F.conv2d(y, k.view(1, 1, -1, 1).expand(x4.shape[1], 1, -1, 1), groups=x4.shape[1])
    return y.reshape(shp)


def push_pull_fill(img: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """
    (C, H, W) の未知画素（known=False）を push-pull 補間で埋める。
    push: 重み付き平均で 1/2 縮小を繰り返す。pull: 粗いレベルを双一次拡大し、重みの足りない画素に混ぜる。
    """
    v = img * known[None].float()
    w = known.float()[None]
    vals, wts = [v], [w]
    while vals[-1].shape[-1] > 1 or vals[-1].shape[-2] > 1:
        v, w = vals[-1], wts[-1]
        h, wd = v.shape[-2:]
        ph, pw = h % 2, wd % 2
        vp = F.pad(v[None], (0, pw, 0, ph), mode="replicate")[0]
        wp = F.pad(w[None], (0, pw, 0, ph), mode="replicate")[0]
        sv = F.avg_pool2d(vp[None], 2)[0] * 4.0
        sw = F.avg_pool2d(wp[None], 2)[0] * 4.0
        nw = (sw / 4.0).clamp(max=1.0)
        nv = torch.where(sw > 0, sv / sw.clamp(min=1e-6), torch.zeros_like(sv)) * nw
        vals.append(nv)
        wts.append(nw)
    for lv in range(len(vals)):
        vals[lv] = torch.where(wts[lv] > 0, vals[lv] / wts[lv].clamp(min=1e-6), vals[lv])
    for lv in range(len(vals) - 2, -1, -1):
        up = F.interpolate(vals[lv + 1][None], size=vals[lv].shape[-2:], mode="bilinear", align_corners=False)[0]
        w = wts[lv]
        vals[lv] = w * vals[lv] + (1.0 - w) * up
    return vals[0]


# -------------------------
# グローバル最小二乗
# -------------------------

def solve_positions(n: int, pairs: List[Tuple[int, int]], d: np.ndarray, w: np.ndarray,
                    iters: int = 5, c: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    p_j - p_i = d_ij を重み w で最小二乗（p は任意次元）。IRLS (Cauchy) で外れ値を減衰。
    戻り値: positions (n,D), residual norm (m,), final weights (m,)
    """
    m = len(pairs)
    D = d.shape[1]
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    wt = w.astype(np.float64).copy()
    pos = np.zeros((n, D))
    rn = np.zeros(m)
    for _ in range(iters):
        A = np.zeros((m + 1, n))
        b = np.zeros((m + 1, D))
        A[np.arange(m), ii] = -wt
        A[np.arange(m), jj] = wt
        b[:m] = wt[:, None] * d
        A[m, 0] = float(m)  # p_0 = 0 のアンカー
        pos = np.linalg.lstsq(A, b, rcond=None)[0]
        res = pos[jj] - pos[ii] - d
        rn = np.linalg.norm(res, axis=1)
        wt = w / (1.0 + (rn / c) ** 2)
    return pos, rn, wt


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def sim_matrix(s: float, th: float, tx: float, ty: float) -> np.ndarray:
    """x_i = s R(th) x_j + t を 3x3 同次行列で。"""
    c, sn = math.cos(th), math.sin(th)
    return np.array([[s * c, -s * sn, tx], [s * sn, s * c, ty], [0.0, 0.0, 1.0]])


def norm_matrix(W: int, H: int) -> np.ndarray:
    """画素座標（画素中心、0..W-1）→ 正規化座標（中心 0、半長辺 1）。ホモグラフィの条件数を揃えるために使う。"""
    f = 0.5 * max(W, H)
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    return np.array([[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]])


def apply_h(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,2) 点に 3x3 同次変換を適用（透視除算）。"""
    p = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1) @ Hm.T
    return p[:, :2] / p[:, 2:3]


def apply_h_t(Hm: np.ndarray, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch 版: (X, Y) → (x, y, z)（z は透視除算前の第 3 成分。z <= 0 はカメラの後ろ）。"""
    h = [float(v) for v in Hm.reshape(-1)]
    z = h[6] * X + h[7] * Y + h[8]
    zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    x = (h[0] * X + h[1] * Y + h[2]) / zs
    y = (h[3] * X + h[4] * Y + h[5]) / zs
    return x, y, z


def rot_from_rodrigues(v: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1.0 - math.cos(th)) * (Kx @ Kx)


def rodrigues_from_rot(R: np.ndarray) -> np.ndarray:
    c = max(-1.0, min(1.0, 0.5 * (np.trace(R) - 1.0)))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return v * th


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def focals_from_homography(Hm: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    主点原点の画素座標でのペアホモグラフィ x_i = H x_j から焦点距離の候補 (f_j, f_i) を推定する
    （Szeliski & Shum の方法。OpenCV の focalsFromHomography と同じ式）。推定できない場合は None。
    """
    h = Hm.reshape(-1) / Hm[2, 2]
    f1 = f0 = None
    d1 = h[6] * h[7]
    d2 = (h[7] - h[6]) * (h[7] + h[6])
    v1 = -(h[0] * h[1] + h[3] * h[4]) / d1 if abs(d1) > 1e-12 else None
    v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2 if abs(d2) > 1e-12 else None
    if v1 is not None and v2 is not None and v1 < v2:
        v1, v2 = v2, v1
    if v1 is not None and v1 > 0 and v2 is not None and v2 > 0:
        f1 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 is not None and v1 > 0:
        f1 = math.sqrt(v1)
    d1 = h[0] * h[3] + h[1] * h[4]
    d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4]
    v1 = -h[2] * h[5] / d1 if abs(d1) > 1e-12 else None
    v2 = (h[5] * h[5] - h[2] * h[2]) / d2 if abs(d2) > 1e-12 else None
    if v1 is not None and v2 is not None and v1 < v2:
        v1, v2 = v2, v1
    if v1 is not None and v1 > 0 and v2 is not None and v2 > 0:
        f0 = math.sqrt(v1 if abs(d1) > abs(d2) else v2)
    elif v1 is not None and v1 > 0:
        f0 = math.sqrt(v1)
    return f1, f0


class Alignment:
    """
    フレーム k の画素 x → キャンバス X。
    - kind="sim": X = S[k] R(TH[k]) x + T[k]（translation / scale / similarity モデル）
    - kind="homography": X = π(Hm[k] x)（平面キャンバス、フレーム 0 の座標系）
    - kind="cyl": 円筒投影。光線 d = R[k] K^-1 x、θ = atan2(d.x, d.z), h = d.y / sqrt(d.x^2 + d.z^2)、
      X = fc θ + X0, Y = fc h + Y0（fc は円筒半径 px）
    S は各フレームの中心での局所倍率（解像度フィルタ・プリフィルタ・canvas scale に使う）。
    """

    def __init__(self, S: np.ndarray, TH: np.ndarray, T: np.ndarray, kind: str = "sim",
                 Hm: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
                 K: Optional[np.ndarray] = None, fc: float = 1.0, X0: float = 0.0, Y0: float = 0.0) -> None:
        self.S, self.TH, self.T = S, TH, T
        self.kind = kind
        self.Hm = Hm            # (n,3,3) 画素座標のホモグラフィ（kind="homography"）
        self.R = R              # (n,3,3) 回転（kind="cyl"）
        self.K = K              # 3x3 内部パラメータ（kind="cyl"）
        self.fc, self.X0, self.Y0 = fc, X0, Y0

    @property
    def n(self) -> int:
        return len(self.S)

    def corners(self, k: int, H: int, W: int) -> np.ndarray:
        """フレームの有効領域の境界点のキャンバス座標 (M,2)。sim では四隅、その他は境界を細かくサンプル。"""
        if self.kind == "sim":
            pts = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [-0.5, H - 0.5], [W - 0.5, H - 0.5]])
            return (self.S[k] * (rot2(self.TH[k]) @ pts.T)).T + self.T[k]
        m = 32
        t = np.linspace(0.0, 1.0, m + 1)
        xs = -0.5 + t * W
        ys = -0.5 + t * H
        pts = np.concatenate([np.stack([xs, np.full_like(xs, -0.5)], 1), np.stack([xs, np.full_like(xs, H - 0.5)], 1),
                              np.stack([np.full_like(ys, -0.5), ys], 1), np.stack([np.full_like(ys, W - 0.5), ys], 1)])
        return self.to_canvas(k, pts)

    def to_canvas(self, k: int, pts: np.ndarray) -> np.ndarray:
        if self.kind == "sim":
            return (self.S[k] * (rot2(self.TH[k]) @ pts.T)).T + self.T[k]
        if self.kind == "homography":
            return apply_h(self.Hm[k], pts)
        assert self.R is not None and self.K is not None
        Kinv = np.linalg.inv(self.K)
        p = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        d = (self.R[k] @ (Kinv @ p.T)).T
        th = np.arctan2(d[:, 0], d[:, 2])
        hh = d[:, 1] / np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
        return np.stack([self.fc * th + self.X0, self.fc * hh + self.Y0], 1)

    def to_frame(self, k: int, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """キャンバス座標 → フレーム k の画素座標 (fx, fy, front)。front はカメラ前方（None なら常に有効）。"""
        if self.kind == "sim":
            s, th = float(self.S[k]), float(self.TH[k])
            c, sn = math.cos(th), math.sin(th)
            u = (X - float(self.T[k, 0])) / s
            v = (Y - float(self.T[k, 1])) / s
            return c * u + sn * v, -sn * u + c * v, None
        if self.kind == "homography":
            fx, fy, z = apply_h_t(np.linalg.inv(self.Hm[k]), X, Y)
            return fx, fy, z > 0
        assert self.R is not None and self.K is not None
        th = (X - self.X0) / self.fc
        hh = (Y - self.Y0) / self.fc
        M = self.K @ self.R[k].T
        m = [float(v) for v in M.reshape(-1)]
        dx, dy, dz = torch.sin(th), hh, torch.cos(th)
        qz = m[6] * dx + m[7] * dy + m[8] * dz
        qzs = torch.where(qz.abs() < 1e-9, torch.full_like(qz, 1e-9), qz)
        fx = (m[0] * dx + m[1] * dy + m[2] * dz) / qzs
        fy = (m[3] * dx + m[4] * dy + m[5] * dz) / qzs
        return fx, fy, qz > 0

    def scaled_shifted(self, g: float, origin: np.ndarray) -> "Alignment":
        """キャンバス倍率 g と原点移動を適用した新しい Alignment。sim は従来と同じ演算順（S*=g, T*=g, T-=origin）。"""
        if self.kind == "sim":
            S, T = self.S * g, self.T * g
            T = T - origin
            return Alignment(S, self.TH.copy(), T)
        if self.kind == "homography":
            A = np.array([[g, 0.0, -origin[0]], [0.0, g, -origin[1]], [0.0, 0.0, 1.0]])
            return Alignment(self.S * g, self.TH.copy(), self.T * g - origin, "homography", Hm=np.einsum("ab,kbc->kac", A, self.Hm))
        return Alignment(self.S * g, self.TH.copy(), self.T * g - origin, "cyl", R=self.R, K=self.K,
                         fc=self.fc * g, X0=self.X0 * g - origin[0], Y0=self.Y0 * g - origin[1])

    def frame_to_frame(self, src: int, dst: int) -> np.ndarray:
        """フレーム src の画素 → フレーム dst の画素、3x3 同次行列（cyl は回転から、局所的にはホモグラフィ）。"""
        if self.kind == "sim":
            Hs = sim_matrix(self.S[src], self.TH[src], self.T[src, 0], self.T[src, 1])
            Hd = sim_matrix(self.S[dst], self.TH[dst], self.T[dst, 0], self.T[dst, 1])
            return np.linalg.inv(Hd) @ Hs
        if self.kind == "homography":
            return np.linalg.inv(self.Hm[dst]) @ self.Hm[src]
        assert self.R is not None and self.K is not None
        return self.K @ self.R[dst].T @ self.R[src] @ np.linalg.inv(self.K)


def local_scale_of_h(Hm: np.ndarray, x: float, y: float) -> float:
    """ホモグラフィの点 (x, y) における局所倍率 sqrt(|det J|)。"""
    p = np.array([x, y, 1.0])
    q = Hm @ p
    z = q[2]
    J = (Hm[:2, :2] * z - np.outer(q[:2], Hm[2, :2])) / (z * z)
    return float(math.sqrt(abs(np.linalg.det(J))))


# -------------------------
# 本体
# -------------------------

class Reconstructor:
    def __init__(self, frames: List[np.ndarray], device: torch.device, args,
                 sizes: Optional[List[Tuple[int, int]]] = None) -> None:
        self.frames = frames
        self.n = len(frames)
        self.H, self.W = frames[0].shape[:2]
        self.dev = device
        self.args = args
        for k, f in enumerate(frames):
            if f.shape[:2] != (self.H, self.W):
                sys.exit(f"フレーム {k} のサイズが異なります: {f.shape[:2]} != {(self.H, self.W)}"
                         "（pad_to_common_size で揃えてください）")
        # 各フレームの有効サイズ (h, w)。パディングされたフレームは右・下の複製領域を無効として扱う
        self.sizes: List[Tuple[int, int]] = list(sizes) if sizes is not None else [(self.H, self.W)] * self.n
        self.padded = any(s != (self.H, self.W) for s in self.sizes)
        self.exposure: Optional[Dict[str, Any]] = None   # 露出補正 {gain (n,3), offset (n,3), profile ((K,3),(K,3)) | None}
        # キャッシュ
        self.gray_u8: List[np.ndarray] = []          # フル解像度グレー (uint8)
        self.overlay: List[np.ndarray] = []          # 静止オーバーレイ + 無視領域 (bool, H×W)
        self.sharp_q: List[np.ndarray] = []          # 鮮明度マップ 1/4 スケール (float16)
        self.coarse_hp: List[torch.Tensor] = []      # 粗解像度ハイパス (device)
        self.coarse_valid: List[torch.Tensor] = []   # 粗解像度有効マスク (device)
        self._full_cache: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._level_cache: "OrderedDict[Tuple[int, float], Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self.ignore_rects: List[Tuple[int, int, int, int]] = args.ignore_rects
        self._last_outlier = 0.0          # 直近の GN の構造化残差の割合（|r| > 3 MAD）
        self._last_mad = 0.0              # 直近の GN の残差 MAD
        self._last_rel = 1.0              # 直近の GN の相対残差（レンズ歪み探索の目的関数）
        self.pair_mats = np.zeros((0, 3, 3))    # 有効ペアの変換行列 x_i = pm x_j（align で設定）
        self.pair_outlier = np.zeros(0)
        self.frame_tier: Optional[np.ndarray] = None   # フレーム品質の段階（detect_blend_frames / estimate_quality で設定）
        self.blend_rho = np.full(self.n, np.nan)
        self.blend_alpha = np.full(self.n, np.nan)

    # ---- 前処理 ----
    def _gray_tensor(self, k: int) -> torch.Tensor:
        return torch.from_numpy(self.gray_u8[k]).to(self.dev).float() / 255.0

    def preprocess(self) -> None:
        a = self.args
        t0 = time.time()
        H, W = self.H, self.W
        for f in self.frames:
            g = (0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]).astype(np.uint8)
            self.gray_u8.append(g)

        ignore = torch.zeros((H, W), dtype=torch.bool, device=self.dev)
        for (x, y, w, h) in self.ignore_rects:
            ignore[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = True
        in_text = torch.zeros((H, W), dtype=torch.bool, device=self.dev)
        for (x, y, w, h) in a.text_rects:
            in_text[max(0, y):min(H, y + h), max(0, x):min(W, x + w)] = True
        has_text = bool(a.text_rects)

        span = a.static_span
        ratios = []
        use_static = a.static_mask and self.n > 1
        # 有効領域（パディングされたフレームのみ）: 右・下の複製領域は位置合わせ・合成の両方から除外する
        self.valid_t: List[Optional[torch.Tensor]] = []
        for k in range(self.n):
            hk, wk = self.sizes[k]
            if (hk, wk) == (H, W):
                self.valid_t.append(None)
            else:
                v = torch.zeros((H, W), dtype=torch.bool, device=self.dev)
                v[:hk, :wk] = True
                self.valid_t.append(v)
        for k in range(self.n):
            g = self._gray_tensor(k)
            if use_static or has_text:
                gm = gradient_mag(g)
                if use_static:
                    cands = [j for j in (k - span, k + span) if 0 <= j < self.n]
                    if not cands:
                        cands = [max(0, min(self.n - 1, k + (span if k == 0 else -span)))]
                    diff = None
                    for j in cands:
                        dj = (g - self._gray_tensor(j)).abs()
                        # どちらかの無効領域（パディング）では静止判定しない
                        for v in (self.valid_t[k], self.valid_t[j]):
                            if v is not None:
                                dj = torch.where(v, dj, torch.ones_like(dj))
                        diff = dj if diff is None else torch.minimum(diff, dj)
                    flat = diff < a.static_diff          # 静止（勾配条件なし）
                    edge = flat & (gm > a.static_grad)   # 静止エッジ
                else:
                    flat = torch.zeros_like(g, dtype=torch.bool)
                    edge = flat.clone()
                if has_text:
                    # テキスト矩形内は静止条件なしで勾配の高い画素を文字とみなす（動くティッカー用）
                    edge = edge | (in_text & (gm > a.static_grad))
                dens = box_blur(edge.float(), 5) > 0.3
                static = dilate(dens, a.static_dilate)
                # ハロー: 静止エッジ周辺の静止画素（文字のグロー等）とテキスト矩形内の周辺画素も除外
                if a.static_halo > 0:
                    static = static | (flat & ~in_text & dilate(dens, 2 * a.static_halo + 1))
                if has_text and a.text_halo > 0:
                    static = static | (in_text & dilate(dens, 2 * a.text_halo + 1))
            else:
                static = torch.zeros_like(g, dtype=torch.bool)
            ov = static | ignore
            vk = self.valid_t[k]
            if vk is not None:
                ov = ov | ~vk
            self.overlay.append(ov.cpu().numpy())
        # 時間方向クロージング: 前後 j フレーム（両側）でマスクされている画素はこのフレームでもマスク（光沢アニメ等の抜け対策）
        nclose = max(0, min(3, int(a.static_close)))
        if nclose > 0 and (use_static or has_text):
            for k in range(self.n):
                m = self.overlay[k]
                for j in range(1, nclose + 1):
                    if k - j >= 0 and k + j < self.n:
                        m = m | (self.overlay[k - j] & self.overlay[k + j])
                self.overlay[k] = m
        for k in range(self.n):
            hk, wk = self.sizes[k]
            ratios.append(float(self.overlay[k][:hk, :wk].mean()))   # 比率は有効領域内で
            g = self._gray_tensor(k)
            ov = torch.from_numpy(self.overlay[k]).to(self.dev)
            # 鮮明度（ラプラシアン絶対値のボックス平均）を 1/4 で保持
            # --sharp-sigma > 0: 先に軽くぼかしてノイズ / 粒子のラプラシアン応答を抑える（実写。ノイズの多いフレームが「鮮明」と
            # 誤判定されるのを防ぐ。本物のエッジの応答は 1/√(1+σ²) 程度にしか下がらない）
            gs = gaussian_blur(g, float(a.sharp_sigma)) if float(a.sharp_sigma) > 0 else g
            s = box_blur(laplacian_abs(gs), 15)
            self.sharp_q.append(downsample_area(s, (max(1, H // 4), max(1, W // 4))).cpu().numpy().astype(np.float16))
            # 粗解像度ハイパスとマスク
            hc, wc = max(16, int(round(H * a.coarse_scale))), max(16, int(round(W * a.coarse_scale)))
            gc = downsample_area(g, (hc, wc))
            k_hp = odd(max(3, 0.08 * min(hc, wc)))
            self.coarse_hp.append(highpass(gc, k_hp))
            vc = downsample_area((~ov).float(), (hc, wc)) > 0.5
            self.coarse_valid.append(vc.float())
        self.hc, self.wc = self.coarse_hp[0].shape
        self.sx, self.sy = self.wc / W, self.hc / H
        mean_ratio = float(np.mean(ratios))
        log(f"[preprocess] {self.n} frames {W}x{H}" + (" (padded, sizes differ)" if self.padded else "") +
            f", coarse {self.wc}x{self.hc}, overlay mask mean ratio={mean_ratio:.3f}, {time.time() - t0:.1f}s")
        if mean_ratio > 0.5:
            log("  警告: オーバーレイマスクが 50% を超えています。--static-span を大きくするか "
                "--static-diff を小さくしてください（スクロールが遅い可能性）")

    def _full(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """フル解像度のハイパス画像と有効マスク（LRU キャッシュ）。"""
        if k in self._full_cache:
            self._full_cache.move_to_end(k)
            return self._full_cache[k]
        g = self._gray_tensor(k)
        k_hp = odd(max(3, 0.08 * min(self.H, self.W)))
        hp = highpass(g, k_hp)
        valid = (~torch.from_numpy(self.overlay[k]).to(self.dev)).float()
        self._full_cache[k] = (hp, valid)
        while len(self._full_cache) > 12:
            self._full_cache.popitem(last=False)
        return hp, valid

    def _level(self, k: int, lv: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gauss-Newton 用: 倍率 lv のハイパス画像（軽くぼかし）と有効マスク。"""
        key = (k, lv)
        if key in self._level_cache:
            self._level_cache.move_to_end(key)
            return self._level_cache[key]
        hp, valid = self._full(k)
        if lv < 0.999:
            size = (max(8, int(round(self.H * lv))), max(8, int(round(self.W * lv))))
            hp_l = downsample_area(hp, size)
            va_l = (downsample_area(valid, size) > 0.5).float()
        else:
            hp_l, va_l = hp, valid
        hp_l = gaussian_blur(hp_l, 1.0)
        self._level_cache[key] = (hp_l, va_l)
        while len(self._level_cache) > 24:
            self._level_cache.popitem(last=False)
        return hp_l, va_l

    # ---- ペア推定（平行移動） ----
    def coarse_estimate(self, i: int, j: int) -> Tuple[float, float, float]:
        A, MA = self.coarse_hp[i], self.coarse_valid[i]
        B, MB = self.coarse_hp[j], self.coarse_valid[j]
        h, w = A.shape
        ncc = masked_ncc_surface(A, MA, B, MB, 2 * h, 2 * w, nmin=self.args.min_overlap * h * w)
        dx, dy, sc = peak_full(ncc, h, w)
        return float(dx), float(dy), sc

    def fine_estimate(self, i: int, j: int, d0: Tuple[int, int], r: int) -> Optional[Tuple[float, float, float]]:
        A, MA = self._full(i)
        B, MB = self._full(j)
        H, W = self.H, self.W
        dx, dy = d0
        by0, by1 = max(0, -dy), min(H, H - dy)
        bx0, bx1 = max(0, -dx), min(W, W - dx)
        if by1 - by0 < 4 * r + 16 or bx1 - bx0 < 4 * r + 16:
            return None
        Bc, MBc = B[by0:by1, bx0:bx1], MB[by0:by1, bx0:bx1]
        Ac, MAc = A[by0 + dy:by1 + dy, bx0 + dx:bx1 + dx], MA[by0 + dy:by1 + dy, bx0 + dx:bx1 + dx]
        pad = 2 * r + 8
        area = (by1 - by0) * (bx1 - bx0)
        ncc = masked_ncc_surface(Ac, MAc, Bc, MBc, by1 - by0 + pad, bx1 - bx0 + pad, nmin=0.1 * area)
        ddx, ddy, sc = peak_window(ncc, r)
        return dx + ddx, dy + ddy, sc

    # ---- ペア推定（スケール付き） ----
    def coarse_estimate_scaled(self, i: int, j: int, scales: List[float]) -> Tuple[float, float, float, float]:
        """
        スケール格子 × マスク付き NCC。戻り値 (s, tx, ty, score) は粗解像度の
        x_i = s * x_j + t モデル（原点は画素中心座標系）。
        """
        A, MA = self.coarse_hp[i], self.coarse_valid[i]
        hA, wA = A.shape
        nmin = self.args.min_overlap * hA * wA
        cands: List[Tuple[float, float, float]] = []
        scores: List[float] = []
        for s in scales:
            Bs, s_eff = resample(self.coarse_hp[j], s)
            MBs, _ = resample(self.coarse_valid[j], s, is_mask=True)
            hB, wB = Bs.shape
            ncc = masked_ncc_surface(A, MA, Bs, MBs, hA + hB, wA + wB, nmin)
            dx, dy, sc = peak_full(ncc, hA, wA)
            cands.append((s_eff, float(dx), float(dy)))
            scores.append(sc)
        k = int(np.argmax(scores))
        s_eff, dx, dy = cands[k]
        ls = math.log(s_eff)
        if 0 < k < len(scales) - 1:
            step = math.log(cands[k + 1][0]) - math.log(cands[k][0])
            ls += _subpix(scores[k - 1], scores[k], scores[k + 1]) * step
        s_ref = math.exp(ls)
        # 画素境界原点の再標本化 → 画素中心座標モデルへの補正
        return s_ref, dx + 0.5 * (s_eff - 1.0), dy + 0.5 * (s_eff - 1.0), scores[k]

    def gn_refine(self, i: int, j: int, p0: np.ndarray, model: str) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Gauss-Newton 直接法による相似変換の精密化。
        p = (log s, theta, tx, ty), x_i = s R(theta) x_j + t（フル解像度、画素中心座標）。
        戻り値 (p, score, n_valid)。
        """
        a = self.args
        ls, th, tx, ty = float(p0[0]), float(p0[1]), float(p0[2]), float(p0[3])
        levels = sorted(set([lv for lv in (0.25, 0.5, 1.0) if lv <= a.fine_scale + 1e-9] + [a.fine_scale]))
        if model == "translation":
            cols = [2, 3]
        elif model == "scale":
            cols = [0, 2, 3]
        else:
            cols = [0, 1, 2, 3]
        score = 0.0
        n_valid = 0
        valid = None
        Bw = None
        mad = 1e-6
        for lv in levels:
            As, MAs = self._level(i, lv)
            Bs, MBs = self._level(j, lv)
            hs, ws = As.shape
            hb, wb = Bs.shape
            # 勾配画像
            Bp = F.pad(Bs[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
            Bx = (Bp[1:-1, 2:] - Bp[1:-1, :-2]) * 0.5
            By = (Bp[2:, 1:-1] - Bp[:-2, 1:-1]) * 0.5
            Bstack = torch.stack([Bs, MBs, Bx, By])[None]
            ys, xs = torch.meshgrid(torch.arange(hs, device=self.dev, dtype=torch.float32),
                                    torch.arange(ws, device=self.dev, dtype=torch.float32), indexing="ij")
            # レベル座標系での平行移動: x^lv = lv (x^f + 0.5) - 0.5 より
            # t_lv = lv t + (0.5 lv - 0.5) + (1 - lv) s R (0.5, 0.5)
            c0, sn0 = math.cos(th), math.sin(th)
            s0 = math.exp(ls)
            off = 0.5 * lv - 0.5
            tlx = lv * tx + off + (1.0 - lv) * s0 * 0.5 * (c0 - sn0)
            tly = lv * ty + off + (1.0 - lv) * s0 * 0.5 * (sn0 + c0)
            for _ in range(a.gn_iters):
                s = math.exp(ls)
                c, sn = math.cos(th), math.sin(th)
                u = (xs - tlx) / s
                v = (ys - tly) / s
                Wx = c * u + sn * v
                Wy = -sn * u + c * v
                grid = torch.stack([2.0 * Wx / (wb - 1) - 1.0, 2.0 * Wy / (hb - 1) - 1.0], dim=-1)[None]
                smp = F.grid_sample(Bstack, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
                Bw, MBw, Bxw, Byw = smp[0], smp[1], smp[2], smp[3]
                inside = (Wx >= 0) & (Wx <= wb - 1) & (Wy >= 0) & (Wy <= hb - 1)
                valid = (MAs > 0.5) & (MBw > 0.5) & inside
                n_valid = int(valid.sum().item())
                if n_valid < 0.02 * hs * ws:
                    return None
                r = (As - Bw)[valid]
                Jls = -(Bxw * Wx + Byw * Wy)[valid]
                Jth = (Bxw * Wy - Byw * Wx)[valid]
                Jtx = -(Bxw * c - Byw * sn)[valid] / s
                Jty = -(Bxw * sn + Byw * c)[valid] / s
                Jall = [Jls, Jth, Jtx, Jty]
                J = torch.stack([Jall[cc] for cc in cols], dim=1)  # (N, P)
                mad = torch.median(r.abs()).item() * 1.4826 + 1e-6
                cth = 1.345 * mad
                wgt = torch.where(r.abs() <= cth, torch.ones_like(r), cth / r.abs())
                Jw = J * wgt[:, None]
                Hm = Jw.T @ J
                g = Jw.T @ r
                Hm = Hm + torch.eye(len(cols), device=self.dev) * 1e-6 * Hm.diagonal().mean()
                try:
                    delta = torch.linalg.solve(Hm, g).cpu().numpy()
                except Exception:
                    return None
                if not np.all(np.isfinite(delta)):
                    return None
                upd = np.zeros(4)
                upd[cols] = delta
                ls += float(upd[0])
                th += float(upd[1])
                tlx += float(upd[2])
                tly += float(upd[3])
                if abs(upd[0]) < 1e-6 and abs(upd[1]) < 1e-6 and abs(upd[2]) < 2e-3 and abs(upd[3]) < 2e-3:
                    break
            # レベル座標 → フル解像度座標へ戻す
            s = math.exp(ls)
            c, sn = math.cos(th), math.sin(th)
            tx = (tlx - off - (1.0 - lv) * s * 0.5 * (c - sn)) / lv
            ty = (tly - off - (1.0 - lv) * s * 0.5 * (sn + c)) / lv
            # 最終スコア: 有効領域の NCC
            if valid is None or Bw is None:
                return None
            av = As[valid]
            bv = Bw[valid]
            av = av - av.mean()
            bv = bv - bv.mean()
            score = float((av * bv).sum() / (torch.sqrt((av * av).sum() * (bv * bv).sum()) + 1e-12))
            # 構造化残差の割合（視差 / 動体の診断用）: |r| > 3 MAD の画素比率
            self._last_outlier = float(((As - Bw)[valid].abs() > 3.0 * mad).float().mean().item()) if n_valid > 0 else 0.0
            self._last_mad = float(mad)
        return np.array([ls, th, tx, ty]), score, n_valid

    def gn_refine_h(self, i: int, j: int, G0: np.ndarray) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        Gauss-Newton 直接法によるホモグラフィの精密化（8 自由度）。
        G は正規化座標（norm_matrix）で x_j = π(G x_i)（フレーム i の画素 → フレーム j）。戻り値の行列も同じ規約。
        戻り値 (G, score, n_valid)。
        """
        a = self.args
        G = np.array(G0, dtype=np.float64).copy()
        G = G / G[2, 2]
        levels = sorted(set([lv for lv in (0.25, 0.5, 1.0) if lv <= a.fine_scale + 1e-9] + [a.fine_scale]))
        score = 0.0
        n_valid = 0
        W, H = self.W, self.H
        f_full = 0.5 * max(W, H)
        cx_full, cy_full = 0.5 * (W - 1), 0.5 * (H - 1)
        mad = 1e-6
        for lv in levels:
            As, MAs = self._level(i, lv)
            Bs, MBs = self._level(j, lv)
            hs, ws = As.shape
            hb, wb = Bs.shape
            Bp = F.pad(Bs[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
            Bx = (Bp[1:-1, 2:] - Bp[1:-1, :-2]) * 0.5
            By = (Bp[2:, 1:-1] - Bp[:-2, 1:-1]) * 0.5
            Bstack = torch.stack([Bs, MBs, Bx, By])[None]
            ys, xs = torch.meshgrid(torch.arange(hs, device=self.dev, dtype=torch.float32),
                                    torch.arange(ws, device=self.dev, dtype=torch.float32), indexing="ij")
            # レベル座標 x^lv = lv (x + 0.5) - 0.5 → 正規化座標は全レベルで同じ
            f_lv = lv * f_full
            cx_lv, cy_lv = lv * (cx_full + 0.5) - 0.5, lv * (cy_full + 0.5) - 0.5
            xn = (xs - cx_lv) / f_lv
            yn = (ys - cy_lv) / f_lv
            fb = f_lv
            cxb, cyb = cx_lv, cy_lv
            valid = None
            Bw = None
            for _ in range(a.gn_iters):
                g = [float(v) for v in G.reshape(-1)]
                Z = g[6] * xn + g[7] * yn + 1.0
                Zs = torch.where(Z.abs() < 1e-6, torch.full_like(Z, 1e-6), Z)
                un = (g[0] * xn + g[1] * yn + g[2]) / Zs
                vn = (g[3] * xn + g[4] * yn + g[5]) / Zs
                Wx = un * fb + cxb
                Wy = vn * fb + cyb
                grid = torch.stack([2.0 * Wx / (wb - 1) - 1.0, 2.0 * Wy / (hb - 1) - 1.0], dim=-1)[None]
                smp = F.grid_sample(Bstack, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
                Bw, MBw, Bxw, Byw = smp[0], smp[1], smp[2], smp[3]
                inside = (Wx >= 0) & (Wx <= wb - 1) & (Wy >= 0) & (Wy <= hb - 1) & (Z > 0.05)
                valid = (MAs > 0.5) & (MBw > 0.5) & inside
                n_valid = int(valid.sum().item())
                if n_valid < 0.02 * hs * ws:
                    return None
                r = (As - Bw)[valid]
                xv, yv, iz = xn[valid], yn[valid], 1.0 / Zs[valid]
                uv, vv = un[valid], vn[valid]
                bx = Bxw[valid] * fb
                by = Byw[valid] * fb
                # J = ∇B · ∂W/∂g（= -∂r/∂g。gn_refine と同じ規約: δ = (JᵀJ)⁻¹ Jᵀ r を加える）
                # ∂u/∂g: (x/Z, y/Z, 1/Z, 0, 0, 0, -x u/Z, -y u/Z), ∂v/∂g: (0, 0, 0, x/Z, y/Z, 1/Z, -x v/Z, -y v/Z)
                J = torch.stack([
                    bx * xv * iz, bx * yv * iz, bx * iz,
                    by * xv * iz, by * yv * iz, by * iz,
                    -(bx * uv + by * vv) * xv * iz, -(bx * uv + by * vv) * yv * iz], dim=1)
                mad = torch.median(r.abs()).item() * 1.4826 + 1e-6
                cth = 1.345 * mad
                wgt = torch.where(r.abs() <= cth, torch.ones_like(r), cth / r.abs())
                Jw = J * wgt[:, None]
                Hm = Jw.T @ J
                gv = Jw.T @ r
                Hm = Hm + torch.diag(Hm.diagonal()) * 1e-4 + torch.eye(8, device=self.dev) * 1e-9 * Hm.diagonal().mean()
                try:
                    delta = torch.linalg.solve(Hm, gv).cpu().numpy()
                except Exception:
                    return None
                if not np.all(np.isfinite(delta)):
                    return None
                G = G + np.concatenate([delta, [0.0]]).reshape(3, 3)
                if np.abs(delta).max() < 3e-6:
                    break
            if valid is None or Bw is None:
                return None
            av = As[valid]
            bv = Bw[valid]
            av = av - av.mean()
            bv = bv - bv.mean()
            score = float((av * bv).sum() / (torch.sqrt((av * av).sum() * (bv * bv).sum()) + 1e-12))
            self._last_outlier = float(((As - Bw)[valid].abs() > 3.0 * mad).float().mean().item()) if n_valid > 0 else 0.0
            self._last_mad = float(mad)
            # 相対残差（レンズ歪み探索の目的関数）: 画像のぼけ / 拡大に依らないよう A 自身の振幅で正規化
            self._last_rel = float(torch.median((As - Bw)[valid].abs()).item() / (torch.median(As[valid].abs()).item() + 1e-6))
        return G, score, n_valid

    # ---- 位置合わせ ----
    def align(self) -> Tuple[Alignment, List[Tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
        """
        戻り値: Alignment, pairs, pair params (m,4)=(s, theta, tx, ty), scores (m,), residuals (m,)
        """
        a = self.args
        n = self.n
        pairs: List[Tuple[int, int]] = []
        for k in a.pair_offsets:
            for i in range(n - k):
                pairs.append((i, i + k))
        m = len(pairs)
        model = a.model

        t0 = time.time()
        pc = np.zeros((m, 4))  # 粗推定 (s, th, tx, ty) フル解像度単位
        pc[:, 0] = 1.0
        sc = np.zeros(m)
        if model == "translation":
            for r_, (i, j) in enumerate(pairs):
                dx, dy, s_ = self.coarse_estimate(i, j)
                pc[r_, 2:4] = (dx / self.sx, dy / self.sy)
                sc[r_] = s_
        else:
            K = int(round(a.scale_max / a.scale_step))
            full_grid = [math.exp(k * a.scale_step) for k in range(-K, K + 1)]
            # k=1 ペアは全格子、k>1 ペアは連鎖予測の近傍のみ
            chain_ls = np.zeros(n)
            order = sorted(range(m), key=lambda r_: pairs[r_][1] - pairs[r_][0])
            for r_ in order:
                i, j = pairs[r_]
                if j - i == 1 or j - i == min(a.pair_offsets):
                    grid = full_grid
                else:
                    pred = chain_ls[j] - chain_ls[i]
                    grid = [math.exp(pred + k * a.scale_step) for k in range(-3, 4)]
                s_, dx, dy, s_sc = self.coarse_estimate_scaled(i, j, grid)
                corr = 0.5 * (s_ - 1.0) * (1.0 - 1.0 / self.sx)
                pc[r_] = (s_, 0.0, dx / self.sx + corr, dy / self.sy + corr)
                sc[r_] = s_sc
                if j - i == min(a.pair_offsets):
                    chain_ls[j] = chain_ls[i] + math.log(s_)
        wc = np.clip(sc, 0.05, 1.0) ** 2
        # ブレンドフレーム（二重像）を含むペアは位置が定まらないので重みを落とす（k=2 のペアで連鎖はつながる）
        bad_frame = np.zeros(n, dtype=bool)
        if self.frame_tier is not None:
            bad_frame = self.frame_tier == 2
        pair_bad = np.array([bad_frame[i] or bad_frame[j] for (i, j) in pairs])
        wc[pair_bad] *= 1e-3
        # 離れたペア（k > 最小間隔）の粗推定が隣接ペアの連鎖と矛盾していれば重みを落とす。
        # 重なりの無いペアでも滑らかな輪郭が偶然合って高い NCC が出ることがあり（静止画列で顕著）、
        # そのままグローバル解に入れると正しい隣接ペアまで「不整合」扱いになって精密化の初期値が壊れる
        k_min = min(a.pair_offsets)
        # 連鎖 = 最小間隔のペア + ブレンドフレームを跨ぐ 2 倍間隔のペア（ブレンドフレームのペアは重みが無いので、跨がないと連鎖が切れる）
        chain = [r_ for r_ in range(m) if pairs[r_][1] - pairs[r_][0] == k_min or
                 (pairs[r_][1] - pairs[r_][0] == 2 * k_min and not pair_bad[r_] and bad_frame[pairs[r_][0] + k_min])]
        n_far_bad = 0
        if len(chain) < m and len(chain) >= n - 1:
            S_h, TH_h, T_h, _ = self._solve_global(n, [pairs[r_] for r_ in chain], pc[chain], wc[chain], model)
            chain_set = set(chain)
            for r_ in range(m):
                if r_ in chain_set or pair_bad[r_]:
                    continue
                i, j = pairs[r_]
                d_pair = S_h[i] * (rot2(TH_h[i]) @ pc[r_, 2:4])
                err = float(np.linalg.norm(d_pair - (T_h[j] - T_h[i])))
                tol_far = max(32.0, 8.0 * (j - i) / self.sx)
                if err > tol_far:
                    wc[r_] *= 1e-3
                    n_far_bad += 1
        # 粗解の整合性チェック（平行移動成分のみ、フル解像度 px）
        S_c, TH_c, T_c, rn_c = self._solve_global(n, pairs, pc, wc, model)
        n_bad = int((rn_c > a.coarse_tol / self.sx).sum())
        log(f"[coarse] {m} pairs, model={model}, score mean={sc.mean():.3f}, "
            f"residual median={np.median(rn_c) * self.sx:.2f}px (coarse), inconsistent pairs={n_bad}"
            + (f", far pairs inconsistent with chain={n_far_bad}" if n_far_bad else "") +
            f", {time.time() - t0:.1f}s")

        # 精密推定
        t0 = time.time()
        pf = np.zeros((m, 4))
        sf = np.zeros(m)
        ok = np.ones(m, dtype=bool)
        outl = np.zeros(m)                      # 構造化残差の割合（|r| > 3 MAD）: 視差 / 動体の診断
        Hn = np.tile(np.eye(3), (m, 1, 1))      # homography: ペア変換 x_i = Hn x_j（正規化座標）
        Nm = norm_matrix(self.W, self.H)
        Nm_inv = np.linalg.inv(Nm)
        f_full = 0.5 * max(self.W, self.H)

        def sim_to_G(p0: np.ndarray) -> np.ndarray:
            """(ls, th, tx, ty): x_i = s R x_j + t → G（正規化座標、x_j = G x_i）"""
            Hpix = sim_matrix(math.exp(p0[0]), p0[1], p0[2], p0[3])
            return Nm @ np.linalg.inv(Hpix) @ Nm_inv

        def h_to_pf(Hpix: np.ndarray) -> np.ndarray:
            """画素座標のペアホモグラフィ → 概略の (s, th, tx, ty)（CSV / ログ用）"""
            s_ = local_scale_of_h(Hpix, 0.5 * (self.W - 1), 0.5 * (self.H - 1))
            return np.array([s_, math.atan2(Hpix[1, 0], Hpix[0, 0]), Hpix[0, 2], Hpix[1, 2]])

        if model == "translation":
            r_small = int(math.ceil(1.0 / a.coarse_scale)) + 2
            r_big = max(8, int(math.ceil(3.0 / a.coarse_scale)))
            for r_, (i, j) in enumerate(pairs):
                if rn_c[r_] <= a.coarse_tol / self.sx:
                    d0, r = pc[r_, 2:4], r_small
                else:
                    d0, r = T_c[j] - T_c[i], r_big
                res = self.fine_estimate(i, j, (int(round(d0[0])), int(round(d0[1]))), r)
                if res is None:
                    ok[r_] = False
                    continue
                pf[r_] = (1.0, 0.0, res[0], res[1])
                sf[r_] = res[2]
        else:
            for r_, (i, j) in enumerate(pairs):
                if rn_c[r_] <= a.coarse_tol / self.sx:
                    p0 = np.array([math.log(pc[r_, 0]), pc[r_, 1], pc[r_, 2], pc[r_, 3]])
                else:
                    # グローバル解からの予測で初期化: x_i = (S_j/S_i) R(TH_j-TH_i) x_j + R(-TH_i)(T_j-T_i)/S_i
                    rel_s = S_c[j] / S_c[i]
                    rel_th = TH_c[j] - TH_c[i]
                    rel_t = rot2(-TH_c[i]) @ (T_c[j] - T_c[i]) / S_c[i]
                    p0 = np.array([math.log(rel_s), rel_th, rel_t[0], rel_t[1]])
                if model == "homography":
                    resh = self.gn_refine_h(i, j, sim_to_G(p0))
                    if resh is None:
                        ok[r_] = False
                        continue
                    G_, s_, _ = resh
                    Hn[r_] = np.linalg.inv(G_)
                    Hn[r_] /= Hn[r_][2, 2]
                    pf[r_] = h_to_pf(Nm_inv @ Hn[r_] @ Nm)
                    sf[r_] = s_
                    outl[r_] = self._last_outlier
                    continue
                res = self.gn_refine(i, j, p0, model)
                if res is None:
                    ok[r_] = False
                    continue
                p_, s_, _ = res
                pf[r_] = (math.exp(p_[0]), p_[1], p_[2], p_[3])
                sf[r_] = s_
                outl[r_] = self._last_outlier
        pairs_ok = [p for p, o in zip(pairs, ok) if o]
        pf, sf, outl, Hn = pf[ok], sf[ok], outl[ok], Hn[ok]
        if len(pairs_ok) < n - 1:
            log(f"  警告: 有効ペアが少なすぎます ({len(pairs_ok)} / {m})")
        pair_bad_ok = pair_bad[ok]
        wf = np.clip(sf, 0.05, 1.0) ** 2
        wf[pair_bad_ok] *= 1e-3
        Hk = None
        if model == "homography":
            Hk, rn_n = self._solve_global_h(n, pairs_ok, Hn, wf)
            rn = rn_n * f_full
            S, TH, T = self._sim_summary_of_h(Hk, Nm, Nm_inv)
        else:
            S, TH, T, rn = self._solve_global(n, pairs_ok, pf, wf, model)
        # 2パス目: 残差の大きいペアをグローバル解から再初期化して再精密化（局所解からの脱出）
        if model != "translation":
            redo = [k for k in range(len(pairs_ok)) if rn[k] > 1.5 and not pair_bad_ok[k]]
            n_fixed = 0
            for k in redo:
                i, j = pairs_ok[k]
                if model == "homography":
                    assert Hk is not None
                    G0 = np.linalg.inv(Hk[j]) @ Hk[i]           # x_j = H_j^-1 H_i x_i
                    resh = self.gn_refine_h(i, j, G0)
                    if resh is None:
                        continue
                    G_, s_, _ = resh
                    cand_h = np.linalg.inv(G_)
                    cand_h /= cand_h[2, 2]
                    pred = np.linalg.inv(Hk[i]) @ Hk[j]           # 予測 x_i = H_i^-1 H_j x_j
                    ctr = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
                    d_old = np.linalg.norm(apply_h(Hn[k], ctr) - apply_h(pred, ctr))
                    d_new = np.linalg.norm(apply_h(cand_h, ctr) - apply_h(pred, ctr))
                    if d_new < d_old and s_ >= sf[k] - 0.05:
                        Hn[k] = cand_h
                        pf[k] = h_to_pf(Nm_inv @ cand_h @ Nm)
                        sf[k] = s_
                        outl[k] = self._last_outlier
                        n_fixed += 1
                    continue
                rel_t = rot2(-TH[i]) @ (T[j] - T[i]) / S[i]
                p0 = np.array([math.log(S[j] / S[i]), TH[j] - TH[i], rel_t[0], rel_t[1]])
                res = self.gn_refine(i, j, p0, model)
                if res is None:
                    continue
                p_, s_, _ = res
                cand = np.array([math.exp(p_[0]), p_[1], p_[2], p_[3]])
                # グローバル解との整合が良くなり、スコアが大きく落ちなければ採用
                d_old = np.linalg.norm(S[i] * (rot2(TH[i]) @ pf[k, 2:4]) - (T[j] - T[i]))
                d_new = np.linalg.norm(S[i] * (rot2(TH[i]) @ cand[2:4]) - (T[j] - T[i]))
                if d_new < d_old and s_ >= sf[k] - 0.05:
                    pf[k] = cand
                    sf[k] = s_
                    outl[k] = self._last_outlier
                    n_fixed += 1
            if redo:
                wf = np.clip(sf, 0.05, 1.0) ** 2
                wf[pair_bad_ok] *= 1e-3
                if model == "homography":
                    Hk, rn_n = self._solve_global_h(n, pairs_ok, Hn, wf)
                    rn = rn_n * f_full
                    S, TH, T = self._sim_summary_of_h(Hk, Nm, Nm_inv)
                else:
                    S, TH, T, rn = self._solve_global(n, pairs_ok, pf, wf, model)
                log(f"  2パス目: 残差 1.5px 超 {len(redo)} ペアを再精密化、{n_fixed} 件更新")
        if model == "homography":
            # 位置合わせに失敗したフレーム（そのフレームの全ペアの残差の中央値が 20 px 超）: テクスチャの乏しいカットなどで
            # ホモグラフィが暴走したもの。合成から外し（段階 2）、位置は前後から補間する（キャンバスとパン角を汚さないため）
            med_res = np.full(n, np.nan)
            for k in range(n):
                rk = [rn[q] for q, (i, j) in enumerate(pairs_ok) if (i == k or j == k) and not pair_bad_ok[q]]
                if rk:
                    med_res[k] = float(np.median(rk))
            wild = np.nan_to_num(med_res, nan=0.0) > 20.0
            if wild.any():
                log(f"  位置合わせ失敗フレーム（ペア残差の中央値 > 20 px）{int(wild.sum())} frames: "
                    + ",".join(str(k) for k in np.flatnonzero(wild)[:20]) + "（合成から外し、位置は前後から補間）")
                bad_frame = bad_frame | wild
                if self.frame_tier is None:
                    self.frame_tier = np.zeros(n, dtype=np.int32)
                self.frame_tier[wild] = 2
                pair_bad_ok = np.array([bad_frame[i] or bad_frame[j] for (i, j) in pairs_ok])
        if bad_frame.any():
            if model == "homography":
                assert Hk is not None
                if bad_frame[0] and not bad_frame.all():
                    # 基準（単位行列）のフレーム 0 が使えないので、最初の良いフレームを基準にし直す（フレーム 0 のペアには重みが無く、
                    # 他のフレームはフレーム 0 に対して任意の射影変換だけ自由に動けてしまうため）
                    ref = int(np.flatnonzero(~bad_frame)[0])
                    Href_inv = np.linalg.inv(Hk[ref])
                    Hk = np.einsum("ab,kbc->kac", Href_inv, Hk)
                    Hk /= Hk[:, 2:3, 2:3]
                    log(f"  基準フレームを {ref} に変更（フレーム 0 は位置合わせ失敗 / ブレンド）")
                self._interp_bad_frames(bad_frame, Hk)
                S, TH, T = self._sim_summary_of_h(Hk, Nm, Nm_inv)
            else:
                self._interp_bad_frames(bad_frame, S, TH, T)
        rn_good = rn[~pair_bad_ok] if (~pair_bad_ok).any() else rn
        log(f"[fine]   {len(pairs_ok)} pairs, score mean={sf.mean():.3f}, residual median={np.median(rn_good):.2f}px, "
            f"max={rn_good.max():.2f}px, {time.time() - t0:.1f}s")
        sus = [(pairs_ok[k], rn[k], sf[k]) for k in range(len(pairs_ok)) if rn[k] > 1.5 and not pair_bad_ok[k]]
        if sus:
            log(f"  残差 1.5px 超のペア: {len(sus)} 件（重みを下げて解決済み）")
            for (i, j), r_, s_ in sus[:10]:
                log(f"    ({i},{j}) residual={r_:.2f}px score={s_:.3f}")
        step = np.diff(T, axis=0)
        if len(step):
            log(f"  1フレームあたりの移動: dx mean={step[:, 0].mean():+.2f} dy mean={step[:, 1].mean():+.2f} "
                f"(min {np.linalg.norm(step, axis=1).min():.2f} / max {np.linalg.norm(step, axis=1).max():.2f} px)")
        if model != "translation":
            log(f"  スケール: min={S.min():.4f} max={S.max():.4f} (frame0=1), "
                f"回転: max |θ|={np.degrees(np.abs(TH).max()):.3f}°")
        if model != "translation" and len(outl):
            med_o, max_o = float(np.median(outl)) * 100, float(outl.max()) * 100
            log(f"  構造化残差（|r| > 3 MAD の画素）: median={med_o:.1f}% max={max_o:.1f}%")
            if med_o > 10.0:
                log("  注意: 位置合わせ後も残差の大きい画素が多い（動体 / キャラクターの動き、または視差 = カメラの平行移動）。"
                    "実写で前景が二重になるなら --ignore-rect で除外するか、--anchor-frame で姿を固定してください")
        # ペア変換の 3x3 行列（画素座標、x_i = pm x_j）: 露出補正・品質推定・ブレンドフレーム検出で使う
        pm = np.zeros((len(pairs_ok), 3, 3))
        for k in range(len(pairs_ok)):
            pm[k] = (Nm_inv @ Hn[k] @ Nm) if model == "homography" else sim_matrix(pf[k, 0], pf[k, 1], pf[k, 2], pf[k, 3])
        self.pair_mats, self.pair_outlier = pm, outl
        if model == "homography":
            assert Hk is not None
            Hpix = np.einsum("ab,kbc,cd->kad", Nm_inv, Hk, Nm)
            al = Alignment(S, TH, T, "homography", Hm=Hpix)
            al = self._choose_projection(al, pairs_ok, pm, sf, rn, bad_frame)
            return al, pairs_ok, pf, sf, rn
        return Alignment(S, TH, T), pairs_ok, pf, sf, rn

    def _sim_summary_of_h(self, Hk: np.ndarray, Nm: np.ndarray, Nm_inv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """各フレームのホモグラフィ（正規化座標）から中心での局所倍率 S、回転角 TH、中心のキャンバス位置 T を求める（ログ / 解像度フィルタ用）。"""
        n = Hk.shape[0]
        S, TH, T = np.ones(n), np.zeros(n), np.zeros((n, 2))
        cx, cy = 0.5 * (self.W - 1), 0.5 * (self.H - 1)
        for k in range(n):
            Hp = Nm_inv @ Hk[k] @ Nm
            S[k] = local_scale_of_h(Hp, cx, cy)
            TH[k] = math.atan2(Hp[1, 0], Hp[0, 0])
            T[k] = apply_h(Hp, np.array([[cx, cy]]))[0] - np.array([cx, cy])
        return S, TH, T

    @staticmethod
    def _interp_bad_frames(bad: np.ndarray, *arrays: np.ndarray) -> None:
        """
        位置の定まらないフレーム（ブレンドフレーム: そのペアは全て重みを落としてある）の値を、前後の良いフレームの平均で置き換える
        （合成には使わないが、パン角・キャンバス範囲・回転平均を汚さないため）。arrays は先頭軸がフレームの配列（S, TH, T, Hk, R など）。
        """
        n = len(bad)
        good = [k for k in range(n) if not bad[k]]
        if not good:
            return
        for k in range(n):
            if not bad[k]:
                continue
            prev = max([g for g in good if g < k], default=None)
            nxt = min([g for g in good if g > k], default=None)
            for arr in arrays:
                if prev is not None and nxt is not None:
                    arr[k] = 0.5 * (arr[prev] + arr[nxt])
                else:
                    arr[k] = arr[prev if prev is not None else nxt]
                if arr.ndim == 3 and arr.shape[1:] == (3, 3):
                    if abs(arr[k][2, 2]) > 1e-12 and not np.allclose(arr[k] @ arr[k].T, np.eye(3), atol=0.5):
                        arr[k] /= arr[k][2, 2]                       # ホモグラフィは正規化
                    else:
                        arr[k] = orthonormalize(arr[k])              # 回転は直交化

    def _choose_projection(self, al: Alignment, pairs: List[Tuple[int, int]], pm: np.ndarray,
                           sf: np.ndarray, rn: np.ndarray, bad_frame: Optional[np.ndarray] = None) -> Alignment:
        """
        ホモグラフィモデルの投影面を決める。--projection planar ならそのまま。cylindrical / auto では
        ペアホモグラフィから焦点距離を推定（Szeliski & Shum）、回転平均で各フレームの回転を解き、円筒投影の Alignment を返す。
        auto: 総パン角が --cyl-min-span 度を超えるときだけ円筒（それ以下は平面の方が歪みが少ない）。
        回転モデルの残差がホモグラフィより大幅に大きければ平行移動（視差）があるので、auto では平面のまま。
        """
        a = self.args
        proj = a.projection
        if proj == "planar":
            return al
        n = al.n
        W, H = self.W, self.H
        cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
        C = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
        C_inv = np.linalg.inv(C)
        fs = []
        for r_, (i, j) in enumerate(pairs):
            if sf[r_] < 0.3:
                continue
            Hc = C @ pm[r_] @ C_inv
            f1, f0 = focals_from_homography(Hc)
            if f1 is not None and f0 is not None and 0.2 * max(W, H) < f1 < 20 * max(W, H) and 0.2 * max(W, H) < f0 < 20 * max(W, H):
                fs.append(math.sqrt(f0 * f1))
        if a.focal is not None and a.focal > 0:
            f = float(a.focal)
            log(f"[projection] 焦点距離 {f:.1f} px（--focal 指定）")
        elif len(fs) >= max(3, n // 8):
            f = float(np.median(fs))
            log(f"[projection] 焦点距離の推定: {f:.1f} px（{len(fs)} ペア、四分位 {np.percentile(fs, 25):.0f}-{np.percentile(fs, 75):.0f}、"
                f"水平画角 {2 * math.degrees(math.atan(0.5 * W / f)):.1f}°）")
        else:
            log(f"[projection] 焦点距離を推定できるペアが少ない（{len(fs)} 件）ため平面投影を使います")
            return al
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        K_inv = np.linalg.inv(K)
        Rp = [orthonormalize(K_inv @ pm[r_] @ K) for r_ in range(len(pairs))]
        w = np.clip(sf, 0.05, 1.0) ** 2
        R, rn_rot = self._solve_rotations(n, pairs, Rp, w)
        good_pair = np.ones(len(pairs), dtype=bool)
        if bad_frame is not None and bad_frame.any():
            self._interp_bad_frames(bad_frame, R)
            good_pair = np.array([not (bad_frame[i] or bad_frame[j]) for (i, j) in pairs])
        rn_rot_px = rn_rot[good_pair] * f
        rn_h = rn[good_pair]
        # 総パン角: 各フレームの光軸の frame 0 からの角度
        axes = np.array([R[k] @ np.array([0.0, 0.0, 1.0]) for k in range(n)])
        yaw = np.degrees(np.arctan2(axes[:, 0], axes[:, 2]))
        pitch = np.degrees(np.arcsin(np.clip(axes[:, 1], -1, 1)))
        span = float(max(yaw.max() - yaw.min(), pitch.max() - pitch.min()))
        log(f"[projection] 回転モデルの残差 median={np.median(rn_rot_px):.2f}px max={rn_rot_px.max():.2f}px"
            f"（ホモグラフィ median={np.median(rn_h):.2f}px）、パン角 yaw {yaw.min():+.1f}..{yaw.max():+.1f}° "
            f"pitch {pitch.min():+.1f}..{pitch.max():+.1f}°")
        parallax = np.median(rn_rot_px) > max(1.0, 3.0 * np.median(rn_h))
        if parallax:
            log("  注意: 回転だけでは説明できない残差があります（カメラの平行移動 = 視差）")
        if proj == "auto":
            if span < a.cyl_min_span:
                log(f"[projection] パン角 {span:.1f}° < {a.cyl_min_span}° なので平面投影のままにします")
                return al
            if parallax and span < 2 * a.cyl_min_span:
                log("[projection] 視差があるため平面投影のままにします（--projection cylindrical で強制できます）")
                return al
        log(f"[projection] 円筒投影（半径 = 焦点距離 {f:.1f} px）")
        S = np.ones(n)
        TH = np.array([math.atan2(R[k][1, 0], R[k][0, 0]) for k in range(n)])
        alc = Alignment(S, TH, np.zeros((n, 2)), "cyl", R=R, K=K, fc=f)
        ctr = np.array([[cx, cy]])
        for k in range(n):
            alc.T[k] = alc.to_canvas(k, ctr)[0]
        alc.T -= alc.T[0]
        return alc

    def _solve_global_h(self, n: int, pairs: List[Tuple[int, int]], Hp: np.ndarray, w: np.ndarray,
                        iters: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        ペアホモグラフィ x_i = Hp[r] x_j（正規化座標）から各フレームの H_k（x_k → キャンバス = フレーム 0 の正規化座標）を
        解く。残差はフレーム j の 4 隅と中心の 5 点を H_i Hp と H_j で写した差（正規化座標）。IRLS (Cauchy) で外れ値を減衰。
        初期値は最小間隔ペアの連鎖。数値微分による Gauss-Newton（未知数 8(n-1)、GPU の密行列で解く）。
        戻り値: Hk (n,3,3), 残差ノルム (m,)（正規化座標）
        """
        m = len(pairs)
        Hk = np.tile(np.eye(3), (n, 1, 1))
        chain: Dict[int, Tuple[int, np.ndarray]] = {}
        k_min = min(j - i for i, j in pairs)
        for r_, (i, j) in enumerate(pairs):
            if j - i == k_min and (j not in chain or w[r_] > chain[j][0]):
                chain[j] = (w[r_], Hp[r_])
        for j in range(1, n):
            if j in chain:
                Hk[j] = Hk[j - k_min] @ chain[j][1]
            else:
                Hk[j] = Hk[j - 1].copy()
        W, H = self.W, self.H
        N = norm_matrix(W, H)
        pts = apply_h(N, np.array([[-0.5, -0.5], [W - 0.5, -0.5], [-0.5, H - 0.5], [W - 0.5, H - 0.5],
                                   [0.5 * (W - 1), 0.5 * (H - 1)]]))
        pts3 = np.concatenate([pts, np.ones((5, 1))], axis=1)   # (5,3)
        f_full = 0.5 * max(W, H)
        c_cauchy = 2.0 / f_full
        dev = self.dev
        P = 8 * (n - 1)

        def proj(M: np.ndarray) -> np.ndarray:
            q = pts3 @ M.T
            return (q[:, :2] / q[:, 2:3]).reshape(-1)

        def param_to_h(k: int, delta: np.ndarray) -> np.ndarray:
            return Hk[k] + np.concatenate([delta, [0.0]]).reshape(3, 3)

        # 残差はフレーム i の座標で測る（H_i^-1 H_j とペア推定 Hp の差）: フレーム 0 の平面座標で測ると長いパンの遠いフレームで
        # 透視除算が発散して残差が爆発する
        wt = w.astype(np.float64).copy()
        rn = np.zeros(m)
        eps = 1e-6
        target = [proj(Hp[r_]) for r_ in range(m)]
        for it in range(iters):
            A_blocks = []
            res_all = np.zeros((m, 10))
            for r_, (i, j) in enumerate(pairs):
                Hi_inv = np.linalg.inv(Hk[i])
                base = proj(Hi_inv @ Hk[j])
                res_all[r_] = base - target[r_]
                Jb = np.zeros((10, 16))
                if i > 0:
                    for q in range(8):
                        d = np.zeros(8)
                        d[q] = eps
                        Jb[:, q] = (proj(np.linalg.inv(param_to_h(i, d)) @ Hk[j]) - base) / eps
                if j > 0:
                    for q in range(8):
                        d = np.zeros(8)
                        d[q] = eps
                        Jb[:, 8 + q] = (proj(Hi_inv @ param_to_h(j, d)) - base) / eps
                A_blocks.append(Jb)
            rn = np.sqrt((res_all ** 2).reshape(m, 5, 2).sum(2).mean(1))
            # Cauchy 尺度は大きい値から絞る（連鎖の初期値に外れ値ペアが入っていても、多数決で正しい解に移れるように）
            c_it = c_cauchy * max(1.0, 32.0 / (2.0 ** it))
            wt = w / (1.0 + (rn / c_it) ** 2)
            AtA = torch.zeros((P, P), dtype=torch.float64, device=dev)
            Atb = torch.zeros(P, dtype=torch.float64, device=dev)
            for r_, (i, j) in enumerate(pairs):
                Jb = torch.from_numpy(A_blocks[r_]).to(dev)
                rb = torch.from_numpy(res_all[r_]).to(dev)
                cols = []
                if i > 0:
                    cols.append((8 * (i - 1), 0))
                if j > 0:
                    cols.append((8 * (j - 1), 8))
                for (ca, oa) in cols:
                    for (cb, ob) in cols:
                        AtA[ca:ca + 8, cb:cb + 8] += wt[r_] * (Jb[:, oa:oa + 8].T @ Jb[:, ob:ob + 8])
                    Atb[ca:ca + 8] -= wt[r_] * (Jb[:, oa:oa + 8].T @ rb)
            AtA = AtA + torch.diag(AtA.diagonal()) * 1e-4 + torch.eye(P, dtype=torch.float64, device=dev) * 1e-10
            try:
                delta = torch.linalg.solve(AtA, Atb).cpu().numpy()
            except Exception:
                break
            if not np.all(np.isfinite(delta)):
                break
            for k in range(1, n):
                Hk[k] = param_to_h(k, delta[8 * (k - 1):8 * k])
                Hk[k] /= Hk[k][2, 2]
            if np.abs(delta).max() < 1e-7:
                break
        res_all = np.zeros((m, 10))
        for r_, (i, j) in enumerate(pairs):
            res_all[r_] = proj(np.linalg.inv(Hk[i]) @ Hk[j]) - target[r_]
        rn = np.sqrt((res_all ** 2).reshape(m, 5, 2).sum(2).mean(1))
        return Hk, rn

    @staticmethod
    def _solve_rotations(n: int, pairs: List[Tuple[int, int]], Rp: List[np.ndarray], w: np.ndarray,
                         iters: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        ペア回転 R_ij（光線 d_i = R_ij d_j）からフレームごとの R_k（d_world = R_k d_k）を解く（回転平均）。
        R_j = R_i R_ij。小角度近似で δ_j - δ_i = log(R_i R_ij R_j^T) を IRLS で解いて左から掛ける。
        戻り値: R (n,3,3), 残差角 (m,) rad
        """
        R = np.tile(np.eye(3), (n, 1, 1))
        k_min = min(j - i for i, j in pairs)
        chain: Dict[int, Tuple[float, np.ndarray]] = {}
        for r_, (i, j) in enumerate(pairs):
            if j - i == k_min and (j not in chain or w[r_] > chain[j][0]):
                chain[j] = (float(w[r_]), Rp[r_])
        for j in range(1, n):
            R[j] = R[j - k_min] @ chain[j][1] if j in chain else R[j - 1].copy()
        rn = np.zeros(len(pairs))
        for _ in range(iters):
            d = np.zeros((len(pairs), 3))
            for r_, (i, j) in enumerate(pairs):
                d[r_] = rodrigues_from_rot(R[i] @ Rp[r_] @ R[j].T)
            rn = np.linalg.norm(d, axis=1)
            pos, _, _ = solve_positions(n, pairs, d, w, iters=3, c=math.radians(0.2))
            pos -= pos[0]
            for k in range(n):
                R[k] = orthonormalize(rot_from_rodrigues(pos[k]) @ R[k])
            if rn.max() < 1e-7:
                break
        for r_, (i, j) in enumerate(pairs):
            rn[r_] = float(np.linalg.norm(rodrigues_from_rot(R[i] @ Rp[r_] @ R[j].T)))
        return R, rn

    @staticmethod
    def _solve_global(n: int, pairs: List[Tuple[int, int]], pp: np.ndarray, w: np.ndarray,
                      model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ペア変換 x_i = s_ij R(th_ij) x_j + t_ij からフレームごとの (S, TH, T) を解く。
        S_j = S_i s_ij, TH_j = TH_i + th_ij, T_j = T_i + S_i R(TH_i) t_ij
        """
        if model == "translation":
            S = np.ones(n)
            TH = np.zeros(n)
        else:
            ls = solve_positions(n, pairs, np.log(pp[:, 0:1]), w, c=0.01)[0][:, 0]
            S = np.exp(ls)
            if model == "similarity":
                TH = solve_positions(n, pairs, pp[:, 1:2], w, c=0.005)[0][:, 0]
            else:
                TH = np.zeros(n)
        d = np.zeros((len(pairs), 2))
        for r_, (i, _j) in enumerate(pairs):
            d[r_] = S[i] * (rot2(TH[i]) @ pp[r_, 2:4])
        T, rn, _ = solve_positions(n, pairs, d, w, c=2.0)
        return S, TH, T, rn

    # ---- 露出補正 ----
    # 節点位置（正規化座標 0..1）: プレイヤーのグラデーション等は端で急に変わるので端に密に置く
    EXPO_KNOTS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0]
    # 周辺減光プロファイルの節点（正規化半径 r: 中心 0、隅 1）。減光は周辺で急に強まるので外側に密
    RAD_KNOTS = [0.0, 0.25, 0.45, 0.6, 0.72, 0.82, 0.9, 0.96, 1.0]

    @staticmethod
    def _hat(t: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        """折れ線補間の基底（ハット関数）。t (...) → (..., K)"""
        K = knots.shape[0]
        t = t.clamp(0.0, 1.0)
        q0 = (torch.searchsorted(knots, t.reshape(-1).contiguous(), right=True) - 1).clamp(0, K - 2).reshape(t.shape)
        k0, k1 = knots[q0], knots[q0 + 1]
        f = ((t - k0) / (k1 - k0)).clamp(0.0, 1.0)
        out = torch.zeros(t.shape + (K,), dtype=t.dtype, device=t.device)
        out.scatter_(-1, q0[..., None], (1.0 - f)[..., None])
        out.scatter_(-1, (q0 + 1)[..., None], f[..., None])
        return out

    def _exposure_at(self, k: int, fx: torch.Tensor, fy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        フレーム k の画素座標 (fx, fy) における (オフセット (3, ...), 1/ゲイン (3, 1, 1))。
        補正後 = (I - オフセット) / ゲイン。値は 0..1 スケール。
        """
        e = self.exposure
        assert e is not None
        hk, wk = self.sizes[k]
        off = torch.from_numpy(e["offset"][k]).to(fx.device, torch.float32)         # (3,)
        out = off[:, None, None].expand(3, *fx.shape).clone()
        if e["profile"] is not None:
            knots = torch.tensor(self.EXPO_KNOTS, device=fx.device, dtype=torch.float32)
            fy_ = torch.from_numpy(e["profile"][0]).to(fx.device, torch.float32)   # (K, 3)
            fx_ = torch.from_numpy(e["profile"][1]).to(fx.device, torch.float32)
            By = self._hat((fy + 0.5) / hk, knots)                                  # (..., K)
            Bx = self._hat((fx + 0.5) / wk, knots)
            out = out + (By @ fy_ + Bx @ fx_).movedim(-1, 0)
        if e["local"] is not None:
            # フレームごとの局所オフセット場（双一次格子）
            Gx, Gy = e["local_grid"]
            Fk = torch.from_numpy(e["local"][k]).to(fx.device, torch.float32)       # ((Gx+1)*(Gy+1), 3)
            idx, wgt = self._grid_basis((fx + 0.5) / wk, (fy + 0.5) / hk, Gx, Gy)  # (..., 4)
            out = out + (Fk[idx] * wgt[..., None]).sum(-2).movedim(-1, 0)
        invg = torch.from_numpy(1.0 / e["gain"][k]).to(fx.device, torch.float32)
        if e.get("radial") is not None:
            # 周辺減光（乗算）: ゲイン g_k (1 + v(r))、r は中心 0、隅 1
            rk = torch.tensor(self.RAD_KNOTS, device=fx.device, dtype=torch.float32)
            rv = torch.from_numpy(e["radial"]).to(fx.device, torch.float32)          # (KR, 3)
            rr = torch.sqrt(((fx + 0.5) / wk * 2.0 - 1.0) ** 2 + ((fy + 0.5) / hk * 2.0 - 1.0) ** 2) / math.sqrt(2.0)
            vig = 1.0 + (self._hat(rr, rk) @ rv).movedim(-1, 0)                     # (3, ...)
            return out, invg[:, None, None] / vig.clamp(min=0.2)
        return out, invg[:, None, None]

    @staticmethod
    def _grid_basis(xn: torch.Tensor, yn: torch.Tensor, Gx: int, Gy: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """正規化座標 (0..1) → 双一次格子 (Gx+1)×(Gy+1) の節点インデックス (..., 4) と重み (..., 4)"""
        u = (xn.clamp(0.0, 1.0) * Gx)
        v = (yn.clamp(0.0, 1.0) * Gy)
        q0 = u.floor().clamp(0, Gx - 1)
        r0 = v.floor().clamp(0, Gy - 1)
        fu = (u - q0).clamp(0.0, 1.0)
        fv = (v - r0).clamp(0.0, 1.0)
        q0l, r0l = q0.long(), r0.long()
        base = r0l * (Gx + 1) + q0l
        idx = torch.stack([base, base + 1, base + Gx + 1, base + Gx + 2], dim=-1)
        wgt = torch.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv], dim=-1)
        return idx, wgt

    def estimate_exposure(self, pairs: List[Tuple[int, int]], pp: np.ndarray, sc: np.ndarray,
                          out_dir: Optional[Path] = None) -> None:
        """
        フレーム間の明るさの違い（露出・フェード・プレイヤーの周辺グラデーションなど）を重なりから推定する。
        モデル: I_k(x, y) = g_k L(X) + a_k + f(y/h_k) + g(x/w_k)
          g_k, a_k: フレームごとのゲインとオフセット（チャンネル別）、f, g: 全フレーム共通の加算プロファイル
          （有効サイズで正規化した座標の折れ線。節点は端に密）。
        ペア (i, j) の重なりを 32 px ブロックに分け、クリーン画素のブロック平均の差 I_i - I_j を観測として
        全パラメータを最小二乗（IRLS, Cauchy）で解く。L は (I_i + I_j)/2 で近似。Σ a_k = 0, Σ (g_k - 1) = 0 で
        全体の明るさを保ち、g_k は 1 へ、プロファイルは 0 と滑らかさへ弱く正則化する。合成時に (I - a_k - f - g) / g_k。
        """
        a = self.args
        n, H, W, dev = self.n, self.H, self.W, self.dev
        cell, blk = 8, 4                                  # 8 px セル平均 → 4x4 セル = 32 px ブロック
        h8, w8 = max(2, H // cell), max(2, W // cell)
        use_profile = a.exposure_profile != "off"
        knots = torch.tensor(self.EXPO_KNOTS, device=dev, dtype=torch.float64)
        K = knots.shape[0] if use_profile else 0
        t0 = time.time()
        M: List[torch.Tensor] = []
        V: List[torch.Tensor] = []
        for k in range(n):
            img = torch.from_numpy(self.frames[k]).to(dev).permute(2, 0, 1).float() / 255.0
            clean = torch.from_numpy(~self.overlay[k]).to(dev).float()
            vk = downsample_area(clean, (h8, w8))
            mk = torch.stack([downsample_area(img[c] * clean, (h8, w8)) for c in range(3)]) / vk.clamp(min=1e-6)[None]
            M.append(mk)
            V.append(vk)
        ys, xs = torch.meshgrid(torch.arange(h8, device=dev, dtype=torch.float32),
                                torch.arange(w8, device=dev, dtype=torch.float32), indexing="ij")
        Xi = (xs + 0.5) * cell - 0.5      # セル中心のフル解像度画素座標
        Yi = (ys + 0.5) * cell - 0.5

        def pool(t: torch.Tensor) -> torch.Tensor:
            t4 = t[None, None] if t.dim() == 2 else t[None]
            return F.avg_pool2d(t4, blk)[0]

        obs = []   # (fi, fj, mi (m,3), mj (m,3), xni, yni, xnj, ynj, w)
        n_pairs_used = 0
        for r_, (i, j) in enumerate(pairs):
            if sc[r_] < a.exposure_min_score:
                continue
            if a.model == "homography":
                Xj, Yj, _z = apply_h_t(np.linalg.inv(self.pair_mats[r_]), Xi, Yi)
            else:
                s, th, tx, ty = (float(v) for v in pp[r_])
                c, sn = math.cos(th), math.sin(th)
                # x_i = s R x_j + t  →  x_j = R^T (x_i - t) / s
                u = (Xi - tx) / s
                v = (Yi - ty) / s
                Xj = c * u + sn * v
                Yj = -sn * u + c * v
            uj = (Xj + 0.5) / cell - 0.5
            vj = (Yj + 0.5) / cell - 0.5
            grid = torch.stack([2.0 * uj / (w8 - 1) - 1.0, 2.0 * vj / (h8 - 1) - 1.0], dim=-1)[None]
            src = torch.cat([M[j], V[j][None]])[None]
            smp = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
            Mj, Vj = smp[:3], smp[3]
            inside = (uj >= 0) & (uj <= w8 - 1) & (vj >= 0) & (vj <= h8 - 1)
            w = ((V[i] > 0.5) & (Vj > 0.5) & inside).float()
            wb = pool(w)[0]
            if float(wb.max()) < 0.7:
                continue
            inv = 1.0 / wb.clamp(min=1e-6)
            Mi_b = pool(M[i] * w[None]) * inv[None]
            Mj_b = pool(Mj * w[None]) * inv[None]
            Xi_b, Yi_b = pool(Xi * w)[0] * inv, pool(Yi * w)[0] * inv
            Xj_b, Yj_b = pool(Xj * w)[0] * inv, pool(Yj * w)[0] * inv
            keep = wb >= 0.7
            if int(keep.sum()) == 0:
                continue
            n_pairs_used += 1
            hi, wi = self.sizes[i]
            hj, wj = self.sizes[j]
            obs.append((i, j, Mi_b.permute(1, 2, 0)[keep].double(), Mj_b.permute(1, 2, 0)[keep].double(),
                        ((Xi_b + 0.5) / wi)[keep].double(), ((Yi_b + 0.5) / hi)[keep].double(),
                        ((Xj_b + 0.5) / wj)[keep].double(), ((Yj_b + 0.5) / hj)[keep].double(), wb[keep].double()))
        if not obs:
            log("[exposure] 重なりの観測が得られないため露出補正を行いません")
            return
        fi = torch.cat([torch.full((o[2].shape[0],), o[0], device=dev, dtype=torch.long) for o in obs])
        fj = torch.cat([torch.full((o[2].shape[0],), o[1], device=dev, dtype=torch.long) for o in obs])
        Mi = torch.cat([o[2] for o in obs])          # (N, 3)
        Mj = torch.cat([o[3] for o in obs])
        xni, yni = torch.cat([o[4] for o in obs]), torch.cat([o[5] for o in obs])
        xnj, ynj = torch.cat([o[6] for o in obs]), torch.cat([o[7] for o in obs])
        w0 = torch.cat([o[8] for o in obs])
        N = Mi.shape[0]
        # 局所オフセット場: フレームごとの双一次格子（長辺 --exposure-local セル）。フレーム数が多いと未知数が増えすぎるので上限あり
        Gx = Gy = 0
        if int(a.exposure_local) > 0:
            if n > 100:
                log(f"[exposure] フレーム数 {n} > 100 のため局所オフセット場は使いません（--exposure-local 0 と同じ）")
            else:
                Gl = int(a.exposure_local)
                if W >= H:
                    Gx, Gy = Gl, max(1, int(round(Gl * H / W)))
                else:
                    Gy, Gx = Gl, max(1, int(round(Gl * W / H)))
        NG = (Gx + 1) * (Gy + 1) if Gx > 0 else 0
        # 放射状（周辺減光）項: 全フレーム共通の乗算プロファイル v(r)、I = g_k (1 + v(r)) L + ...。r は中心 0、隅 1
        use_radial = a.exposure_radial == "on"
        rknots = torch.tensor(self.RAD_KNOTS, device=dev, dtype=torch.float64)
        KR = rknots.shape[0] if use_radial else 0
        base_r = 2 * n + 2 * K + n * NG
        # 未知数: [δg_0..δg_{n-1}, a_0..a_{n-1}, f_0..f_{K-1}, g_0..g_{K-1}, F_0 (NG), ..., F_{n-1} (NG), v_0..v_{KR-1}]（チャンネル別に解く）
        P = base_r + KR
        d = Mi - Mj                                   # (N, 3)
        Lbar = 0.5 * (Mi + Mj)                        # (N, 3)
        # 位置に依らない部分の疎な行: 列と値（チャンネル共通）
        cols_c = [fi + n, fj + n]
        vals_c = [torch.ones(N, device=dev, dtype=torch.float64), -torch.ones(N, device=dev, dtype=torch.float64)]
        if use_profile:
            # 折れ線基底は隣り合う 2 節点だけが非零（疎な行にしないと N × Q² が爆発する）
            def hat2(t: torch.Tensor, base: int, sign: float) -> None:
                t = t.clamp(0.0, 1.0)
                q0 = (torch.searchsorted(knots, t.contiguous(), right=True) - 1).clamp(0, K - 2)
                f = ((t - knots[q0]) / (knots[q0 + 1] - knots[q0])).clamp(0.0, 1.0)
                cols_c.extend([base + q0, base + q0 + 1])
                vals_c.extend([sign * (1.0 - f), sign * f])
            hat2(yni, 2 * n, 1.0)
            hat2(ynj, 2 * n, -1.0)
            hat2(xni, 2 * n + K, 1.0)
            hat2(xnj, 2 * n + K, -1.0)
        if NG > 0:
            baseF = 2 * n + 2 * K
            for (xn_, yn_, fk, sign) in ((xni, yni, fi, 1.0), (xnj, ynj, fj, -1.0)):
                idx, wgt = self._grid_basis(xn_, yn_, Gx, Gy)                 # (N, 4)
                for q in range(4):
                    cols_c.append(baseF + fk * NG + idx[:, q])
                    vals_c.append(sign * wgt[:, q])
        cols_common = torch.stack(cols_c, 1)          # (N, Q)
        vals_common = torch.stack(vals_c, 1)
        # 正則化
        reg = torch.zeros(P, dtype=torch.float64, device=dev)
        # ゲインは 1 へ、オフセットは 0 へ（弱く）。フレーム固定の縦ランプは「等間隔に並ぶフレームごとの定数」でも
        # 「共通プロファイルの直線」でも同じ観測になる（ゲージ不定）ので、オフセットのリッジをプロファイルより強くして
        # 共通プロファイル側に寄せる（スクリーンショットのプレイヤーグラデーションはこちら。定数だけの場合はそのまま残る）
        reg[:n] = 1e-2 * N / n
        reg[n:2 * n] = 1e-2 * N / n
        # 放射項の観測行（チャンネル依存: L̄ を掛ける）。中心の節点 v(0) = 0 をゲージとして強く固定（ゲインと縮退するため）
        rad_cols: List[torch.Tensor] = []
        rad_hat: List[torch.Tensor] = []
        if use_radial:
            def rhat(xn_: torch.Tensor, yn_: torch.Tensor, sign: float) -> None:
                rr = (torch.sqrt(((xn_ - 0.5) * 2.0) ** 2 + ((yn_ - 0.5) * 2.0) ** 2) / math.sqrt(2.0)).clamp(0.0, 1.0)
                q0 = (torch.searchsorted(rknots, rr.contiguous(), right=True) - 1).clamp(0, KR - 2)
                f = ((rr - rknots[q0]) / (rknots[q0 + 1] - rknots[q0])).clamp(0.0, 1.0)
                rad_cols.extend([base_r + q0, base_r + q0 + 1])
                rad_hat.extend([sign * (1.0 - f), sign * f])
            rhat(xni, yni, 1.0)
            rhat(xnj, ynj, -1.0)
            reg[base_r:] = 1e-4 * N / KR
            reg[base_r] += float(N)
        gauge = torch.zeros((P, P), dtype=torch.float64, device=dev)
        for lo, hi in ((0, n), (n, 2 * n)):
            ind = torch.zeros(P, dtype=torch.float64, device=dev)
            ind[lo:hi] = 1.0
            gauge += ind[:, None] * ind[None, :] * float(N)      # Σ δg_k = 0, Σ a_k = 0
        smooth = torch.zeros((P, P), dtype=torch.float64, device=dev)
        if use_radial:
            for t in range(KR - 2):
                h1, h2 = float(rknots[t + 1] - rknots[t]), float(rknots[t + 2] - rknots[t + 1])
                row = torch.zeros(P, dtype=torch.float64, device=dev)
                row[base_r + t], row[base_r + t + 1], row[base_r + t + 2] = 1.0 / h1, -(1.0 / h1 + 1.0 / h2), 1.0 / h2
                row *= 0.5 * (h1 + h2)
                smooth += row[:, None] * row[None, :] * (1e-2 * N / KR)
        if use_profile:
            reg[2 * n:2 * n + 2 * K] = 1e-4 * N / K
            kn = knots
            for base in (2 * n, 2 * n + K):
                for t in range(K - 2):
                    h1, h2 = float(kn[t + 1] - kn[t]), float(kn[t + 2] - kn[t + 1])
                    row = torch.zeros(P, dtype=torch.float64, device=dev)
                    row[base + t], row[base + t + 1], row[base + t + 2] = 1.0 / h1, -(1.0 / h1 + 1.0 / h2), 1.0 / h2
                    row *= 0.5 * (h1 + h2)
                    smooth += row[:, None] * row[None, :]
            smooth *= 1e-2 * N / K
        if NG > 0:
            # 局所場: 0 へのリッジ（観測の無い節点は補正しない）と隣接節点の差の平滑化。定数成分は a_k と重なるが、リッジで a_k 側に寄る
            baseF = 2 * n + 2 * K
            reg[baseF:] = 1e-3 * N / n
            lam_l = 3e-2 * N / n
            for k in range(n):
                b0 = baseF + k * NG
                for r in range(Gy + 1):
                    for q in range(Gx + 1):
                        p = b0 + r * (Gx + 1) + q
                        for pn in ((p + 1) if q < Gx else None, (p + Gx + 1) if r < Gy else None):
                            if pn is None:
                                continue
                            smooth[p, p] += lam_l
                            smooth[pn, pn] += lam_l
                            smooth[p, pn] -= lam_l
                            smooth[pn, p] -= lam_l
        cthr = 0.03                                   # Cauchy スケール（0..1 単位 ≈ 8 階調）
        theta = torch.zeros((P, 3), dtype=torch.float64, device=dev)
        res = d.clone()
        for ch in range(3):
            cols = torch.cat([fi[:, None], fj[:, None], cols_common], 1)            # (N, Q+2)
            vals = torch.cat([Lbar[:, ch:ch + 1], -Lbar[:, ch:ch + 1], vals_common], 1)
            if use_radial:
                cols = torch.cat([cols, torch.stack(rad_cols, 1)], 1)
                vals = torch.cat([vals, torch.stack(rad_hat, 1) * Lbar[:, ch:ch + 1]], 1)
            outer = cols[:, :, None].expand(-1, -1, cols.shape[1])
            inner = cols[:, None, :].expand(-1, cols.shape[1], -1)
            wt = w0.clone()
            dc = d[:, ch]
            th_c = torch.zeros(P, dtype=torch.float64, device=dev)
            r = dc.clone()
            for _ in range(5):
                AtA = torch.zeros((P, P), dtype=torch.float64, device=dev)
                prod = (vals[:, :, None] * vals[:, None, :]) * wt[:, None, None]
                AtA.index_put_((outer.reshape(-1), inner.reshape(-1)), prod.reshape(-1), accumulate=True)
                Atb = torch.zeros(P, dtype=torch.float64, device=dev)
                Atb.index_put_((cols.reshape(-1),), (vals * (wt * dc)[:, None]).reshape(-1), accumulate=True)
                AtA = AtA + torch.diag(reg) + gauge + smooth
                th_c = torch.linalg.solve(AtA, Atb)
                r = (vals * th_c[cols]).sum(1) - dc
                wt = w0 / (1.0 + (r / cthr) ** 2)
            theta[:, ch] = th_c
            res[:, ch] = r
        gain = (1.0 + theta[:n]).cpu().numpy()                        # (n, 3)
        offset = theta[n:2 * n].cpu().numpy()                          # (n, 3) 0..1 単位
        profile = (theta[2 * n:2 * n + K].cpu().numpy(), theta[2 * n + K:2 * n + 2 * K].cpu().numpy()) if use_profile else None
        local = theta[2 * n + 2 * K:base_r].reshape(n, NG, 3).cpu().numpy() if NG > 0 else None
        radial = theta[base_r:].cpu().numpy() if use_radial else None
        self.exposure = {"gain": gain, "offset": offset, "profile": profile, "local": local, "local_grid": (Gx, Gy), "radial": radial}
        mad0 = float(d.abs().mean(1).median()) * 255.0
        mad1 = float(res.abs().mean(1).median()) * 255.0
        msg = (f"[exposure] {n_pairs_used} pairs, {N} blocks, 重なりの差（中央値）{mad0:.2f} → {mad1:.2f} 階調, "
               f"gain {gain.min():.3f}-{gain.max():.3f}, offset {offset.min() * 255:+.1f}..{offset.max() * 255:+.1f} 階調")
        if radial is not None:
            msg += f", vignette (corner gain) {1.0 + radial[-1].mean():.3f}"
        if profile is not None:
            py, px = profile[0].mean(1) * 255.0, profile[1].mean(1) * 255.0
            msg += f", profile y {py.min():+.1f}..{py.max():+.1f} / x {px.min():+.1f}..{px.max():+.1f} 階調"
        if local is not None:
            lm = local.mean(2) * 255.0
            msg += f", local field ({Gx}x{Gy} cells) {lm.min():+.1f}..{lm.max():+.1f} 階調"
        log(msg + f", {time.time() - t0:.1f}s")
        if out_dir is not None:
            with open(out_dir / "exposure.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "gain_r", "gain_g", "gain_b", "offset_r", "offset_g", "offset_b"])
                for k in range(n):
                    wr.writerow([k] + [f"{v:.4f}" for v in gain[k]] + [f"{v * 255:.2f}" for v in offset[k]])
                if profile is not None:
                    wr.writerow([])
                    wr.writerow(["knot", "fy_r", "fy_g", "fy_b", "gx_r", "gx_g", "gx_b"])
                    for q in range(K):
                        wr.writerow([f"{self.EXPO_KNOTS[q]:.2f}"] + [f"{v * 255:.2f}" for v in profile[0][q]] +
                                    [f"{v * 255:.2f}" for v in profile[1][q]])
                if local is not None:
                    wr.writerow([])
                    wr.writerow(["local_frame", "knot_x", "knot_y", "r", "g", "b"])
                if radial is not None:
                    wr.writerow([])
                    wr.writerow(["radial_knot", "vignette_r", "vignette_g", "vignette_b"])
                    for q in range(KR):
                        wr.writerow([f"{self.RAD_KNOTS[q]:.2f}"] + [f"{1.0 + v:.4f}" for v in radial[q]])
                    for k in range(n):
                        for r in range(Gy + 1):
                            for q in range(Gx + 1):
                                wr.writerow([k, q, r] + [f"{v * 255:.2f}" for v in local[k][r * (Gx + 1) + q]])
            if os.environ.get("PANORAMA_EXPOSURE_DEBUG"):
                with open(out_dir / "exposure_obs.csv", "w", newline="", encoding="utf-8") as f:
                    wr = csv.writer(f)
                    wr.writerow(["i", "j", "xi", "yi", "xj", "yj", "d_r", "d_g", "d_b", "res_r", "res_g", "res_b", "w"])
                    for q in range(N):
                        wr.writerow([int(fi[q]), int(fj[q]), f"{float(xni[q]):.4f}", f"{float(yni[q]):.4f}",
                                     f"{float(xnj[q]):.4f}", f"{float(ynj[q]):.4f}"] +
                                    [f"{float(v) * 255:.2f}" for v in d[q]] + [f"{float(v) * 255:.2f}" for v in res[q]] +
                                    [f"{float(w0[q]):.3f}"])

    # ---- フレーム品質（実写: ボケ / ブレ / ブレンドフレーム） ----
    def detect_blend_frames(self) -> None:
        """
        ブレンドフレーム（フレームレート変換で前後 2 フレームが位置合わせなしに混ざったもの = 二重像）を検出し、
        self.frame_tier に 2 を立てる（位置合わせの前に呼ぶ。align はそのペアの重みを落とす）。
        粗解像度ハイパスで hp_k ≈ α hp_{k-1} + (1-α) hp_{k+1} を最小二乗で当てはめ、残差 ρ = |hp_k - fit| / (|a-b|/2) が
        --blend-thr 未満なら判定。通常のフレームは中間位置の単像なので ρ ≈ 1、複製フレームは α ≈ 0 / 1 なので除外。
        """
        a = self.args
        n = self.n
        self.blend_rho = np.full(n, np.nan)
        self.blend_alpha = np.full(n, np.nan)
        tier = np.zeros(n, dtype=np.int32)
        if n >= 3:
            for k in range(1, n - 1):
                common = (self.coarse_valid[k] > 0.5) & (self.coarse_valid[k - 1] > 0.5) & (self.coarse_valid[k + 1] > 0.5)
                if int(common.sum().item()) < 0.1 * self.hc * self.wc:
                    continue
                c_ = self.coarse_hp[k][common]
                a_, b_ = self.coarse_hp[k - 1][common], self.coarse_hp[k + 1][common]
                dab = a_ - b_
                den2 = float((dab * dab).sum().item())
                if den2 < 1e-6 or float(dab.pow(2).mean().sqrt().item()) < 0.004:
                    continue                                     # 前後がほぼ同じ（静止 / 複製）なら判定できない
                al_ = float(((c_ - b_) * dab).sum().item() / den2)
                fit = al_ * a_ + (1.0 - al_) * b_
                self.blend_rho[k] = float((c_ - fit).pow(2).mean().sqrt().item() / (0.5 * dab.pow(2).mean().sqrt().item()))
                self.blend_alpha[k] = al_
                if self.blend_rho[k] < a.blend_thr and 0.15 < al_ < 0.85:
                    tier[k] = 2
        self.frame_tier = tier
        idx = np.flatnonzero(tier == 2)
        log(f"[blend] ブレンドフレーム（ρ < {a.blend_thr}）{len(idx)} frames"
            + (": " + ",".join(str(k) for k in idx[:20]) + "（位置合わせの重みを下げ、合成から外します）" if len(idx) else ""))

    def estimate_quality(self, pairs: List[Tuple[int, int]], sc: np.ndarray, al: Alignment,
                         out_dir: Optional[Path] = None) -> None:
        """
        フレームごとの相対鮮明度とブレンドフレームを判定し、self.frame_tier（0: 良, 1: ボケ / ブレ, 2: ブレンド）を設定する。
        鮮明度: 位置合わせ済みペアの重なりで鮮明度マップ（1/4）の平均の比 log(s_i / s_j) を観測し、フレームごとの log 鮮明度 q_k を
          最小二乗（IRLS）で解く（同じ被写体で比べるので内容の違いに依らない）。近傍 ±quality_window フレームの中央値を基準に、
          比が --quality-thr 未満のフレームを段階 1 とする。
        ブレンド: フレーム k を前後のフレームの混合 α a + (1-α) b（位置合わせなし）で説明できる（ρ < --blend-thr）なら段階 2
          （フレームレート変換で 2 フレームが混ざったもの。複製フレームは α ≈ 0 / 1、通常のフレームは ρ ≈ 1 なので対象外）。
        """
        a = self.args
        n, dev = self.n, self.dev
        t0 = time.time()
        Hq, Wq = self.sharp_q[0].shape
        W, H = self.W, self.H
        tier = np.zeros(n, dtype=np.int32)
        q = np.zeros(n)
        ratio = np.ones(n)
        do_q = a.quality == "on"
        do_b = a.blend_detect == "on"
        ys, xs = torch.meshgrid(torch.arange(Hq, device=dev, dtype=torch.float32),
                                torch.arange(Wq, device=dev, dtype=torch.float32), indexing="ij")
        Xi = (xs + 0.5) * (W / Wq) - 0.5
        Yi = (ys + 0.5) * (H / Hq) - 0.5
        Sq = [torch.from_numpy(self.sharp_q[k].astype(np.float32)).to(dev) for k in range(n)]
        Cq = [(downsample_area(torch.from_numpy(~self.overlay[k]).to(dev).float(), (Hq, Wq)) > 0.5).float() for k in range(n)]

        def warp_q(k: int, M: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """M: このフレームの画素 → フレーム k の画素。フレーム k の (鮮明度, クリーン) をこの 1/4 格子へ引く"""
            Xj, Yj, z = apply_h_t(M, Xi, Yi)
            uj = (Xj + 0.5) * (Wq / W) - 0.5
            vj = (Yj + 0.5) * (Hq / H) - 0.5
            grid = torch.stack([2.0 * uj / (Wq - 1) - 1.0, 2.0 * vj / (Hq - 1) - 1.0], dim=-1)[None]
            smp = F.grid_sample(torch.stack([Sq[k], Cq[k]])[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
            inside = (uj >= 0) & (uj <= Wq - 1) & (vj >= 0) & (vj <= Hq - 1) & (z > 0)
            return smp[0], (smp[1] > 0.5) & inside, inside

        n_obs = 0
        if do_q:
            obs_pairs: List[Tuple[int, int]] = []
            d_obs: List[float] = []
            w_obs: List[float] = []
            cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
            for r_, (i, j) in enumerate(pairs):
                if sc[r_] < a.exposure_min_score:
                    continue
                M = self.pair_mats[r_]                          # x_i = M x_j
                if abs(math.log(max(1e-6, local_scale_of_h(M, cx, cy)))) > 0.05:
                    continue                                    # 拡大率が違うと鮮明度の比較にならない（ズーム）
                sj, cj, _ = warp_q(j, np.linalg.inv(M))
                common = (Cq[i] > 0.5) & cj
                nc = int(common.sum().item())
                if nc < 0.05 * Hq * Wq:
                    continue
                mi = float(Sq[i][common].mean().item())
                mj = float(sj[common].mean().item())
                if mi <= 1e-6 or mj <= 1e-6:
                    continue
                obs_pairs.append((i, j))
                d_obs.append(math.log(mj) - math.log(mi))       # q_j - q_i
                w_obs.append(math.sqrt(nc / (Hq * Wq)))
            n_obs = len(obs_pairs)
            if n_obs >= n - 1:
                pos, _, _ = solve_positions(n, obs_pairs, np.array(d_obs)[:, None], np.array(w_obs), iters=5, c=0.15)
                q = pos[:, 0] - float(np.median(pos[:, 0]))
                Wn = max(1, int(a.quality_window))
                for k in range(n):
                    lo, hi = max(0, k - Wn), min(n, k + Wn + 1)
                    ratio[k] = math.exp(q[k] - float(np.median(q[lo:hi])))
                tier[ratio < a.quality_thr] = 1
            else:
                log(f"[quality] 鮮明度の比較ができるペアが少ない（{n_obs} 件）ため、ボケ / ブレの判定は行いません")
        n_blend = int((self.frame_tier == 2).sum()) if self.frame_tier is not None else 0
        if self.frame_tier is not None:
            tier[self.frame_tier == 2] = 2          # ブレンドフレーム（detect_blend_frames で判定済み）は優先
        rho, alpha = self.blend_rho, self.blend_alpha
        self.frame_tier = tier
        n1 = int((tier == 1).sum())
        msg = f"[quality] {time.time() - t0:.1f}s"
        if do_q:
            msg = (f"[quality] 相対鮮明度: {n_obs} 観測, 比 {ratio.min():.2f}..{ratio.max():.2f}, "
                   f"ボケ / ブレ判定（< {a.quality_thr}）{n1} frames" + (": " + ",".join(str(k) for k in np.flatnonzero(tier == 1)[:20]) if n1 else ""))
        if do_b:
            msg += f"; ブレンドフレーム（ρ < {a.blend_thr}）{n_blend} frames" + (": " + ",".join(str(k) for k in np.flatnonzero(tier == 2)[:20]) if n_blend else "")
        log(msg + f", {time.time() - t0:.1f}s")
        if n1 + n_blend > 0:
            log(f"  段階の高いフレームは、良いフレームだけで {a.quality_min_samples} 標本以上ある画素では合成に使いません（足りない画素では使います）")
        if out_dir is not None:
            with open(out_dir / "quality.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.writer(f)
                wr.writerow(["frame", "log_sharp", "ratio_to_local", "blend_rho", "blend_alpha", "tier"])
                for k in range(n):
                    wr.writerow([k, f"{q[k]:+.4f}", f"{ratio[k]:.3f}", "" if np.isnan(rho[k]) else f"{rho[k]:.3f}",
                                 "" if np.isnan(alpha[k]) else f"{alpha[k]:.3f}", int(tier[k])])

    # ---- レンズ歪み ----
    def gn_refine_hk(self, i: int, j: int, G0: np.ndarray, k1_0: float, levels: Tuple[float, ...] = (0.25, 0.5)
                     ) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        ホモグラフィ + 放射歪み k1 の同時推定（9 自由度）。両フレームに同じ k1 が掛かっているとして、
        A の（歪んだ）画素 x_d → 歪み除去 x_u → G → 歪み付与 → B をサンプルする。A を再標本化しないので、
        補正画像のぼけによる残差の減少（k1 の探索でぼける方向に引かれる）が起きない。k1 の微分は数値微分。
        戻り値 (G, k1, score)。G は歪みなし正規化座標での x_j = π(G x_i)。
        """
        a = self.args
        G = np.array(G0, dtype=np.float64).copy()
        G = G / G[2, 2]
        k1 = float(k1_0)
        W, H = self.W, self.H
        f_full = 0.5 * max(W, H)
        cx_full, cy_full = 0.5 * (W - 1), 0.5 * (H - 1)
        kap_scale = (f_full / math.hypot(cx_full, cy_full)) ** 2   # 正規化座標での歪み係数 κ = k1 (f / 半対角)²
        score = 0.0
        dev = self.dev
        for lv in levels:
            As, MAs = self._level(i, lv)
            Bs, MBs = self._level(j, lv)
            hs, ws = As.shape
            hb, wb = Bs.shape
            Bp = F.pad(Bs[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
            Bx = (Bp[1:-1, 2:] - Bp[1:-1, :-2]) * 0.5
            By = (Bp[2:, 1:-1] - Bp[:-2, 1:-1]) * 0.5
            Bstack = torch.stack([Bs, MBs, Bx, By])[None]
            ys, xs = torch.meshgrid(torch.arange(hs, device=dev, dtype=torch.float32),
                                    torch.arange(ws, device=dev, dtype=torch.float32), indexing="ij")
            f_lv = lv * f_full
            cx_lv, cy_lv = lv * (cx_full + 0.5) - 0.5, lv * (cy_full + 0.5) - 0.5
            xdn = (xs - cx_lv) / f_lv          # 歪んだ正規化座標
            ydn = (ys - cy_lv) / f_lv

            def warp(Gm: np.ndarray, k1v: float) -> Tuple[torch.Tensor, ...]:
                kap = k1v * kap_scale
                # 歪み除去（固定点反復）: x_u = x_d / (1 + κ |x_u|²)
                xu, yu = xdn, ydn
                for _ in range(6):
                    r2 = xu * xu + yu * yu
                    xu, yu = xdn / (1.0 + kap * r2), ydn / (1.0 + kap * r2)
                g = [float(v) for v in Gm.reshape(-1)]
                Z = g[6] * xu + g[7] * yu + 1.0
                Zs = torch.where(Z.abs() < 1e-6, torch.full_like(Z, 1e-6), Z)
                un = (g[0] * xu + g[1] * yu + g[2]) / Zs
                vn = (g[3] * xu + g[4] * yu + g[5]) / Zs
                fct = 1.0 + kap * (un * un + vn * vn)   # 歪み付与
                Wx = un * fct * f_lv + cx_lv
                Wy = vn * fct * f_lv + cy_lv
                grid = torch.stack([2.0 * Wx / (wb - 1) - 1.0, 2.0 * Wy / (hb - 1) - 1.0], dim=-1)[None]
                smp = F.grid_sample(Bstack, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
                inside = (Wx >= 0) & (Wx <= wb - 1) & (Wy >= 0) & (Wy <= hb - 1) & (Z > 0.05)
                return xu, yu, un, vn, Zs, smp[0], smp[1], smp[2], smp[3], inside

            eps_k = 2e-3
            for _ in range(a.gn_iters):
                xu, yu, un, vn, Zs, Bw, MBw, Bxw, Byw, inside = warp(G, k1)
                valid = (MAs > 0.5) & (MBw > 0.5) & inside
                n_valid = int(valid.sum().item())
                if n_valid < 0.02 * hs * ws:
                    return None
                r = (As - Bw)[valid]
                xv, yv, iz = xu[valid], yu[valid], 1.0 / Zs[valid]
                uv, vv = un[valid], vn[valid]
                # 歪み付与の局所倍率は 1 + O(κ) なので G の微分では無視（k1 の微分が数値なので収束に影響しない）
                bx = Bxw[valid] * f_lv
                by = Byw[valid] * f_lv
                Jg = torch.stack([
                    bx * xv * iz, bx * yv * iz, bx * iz,
                    by * xv * iz, by * yv * iz, by * iz,
                    -(bx * uv + by * vv) * xv * iz, -(bx * uv + by * vv) * yv * iz], dim=1)
                _, _, _, _, _, Bw2, _, _, _, _ = warp(G, k1 + eps_k)
                Jk = -((As - Bw2)[valid] - r) / eps_k
                J = torch.cat([Jg, Jk[:, None]], dim=1)
                mad = torch.median(r.abs()).item() * 1.4826 + 1e-6
                cth = 1.345 * mad
                wgt = torch.where(r.abs() <= cth, torch.ones_like(r), cth / r.abs())
                Jw = J * wgt[:, None]
                Hm = Jw.T @ J
                gv = Jw.T @ r
                Hm = Hm + torch.diag(Hm.diagonal()) * 1e-4 + torch.eye(9, device=dev) * 1e-9 * Hm.diagonal().mean()
                try:
                    delta = torch.linalg.solve(Hm, gv).cpu().numpy()
                except Exception:
                    return None
                if not np.all(np.isfinite(delta)):
                    return None
                G = G + np.concatenate([delta[:8], [0.0]]).reshape(3, 3)
                k1 = float(np.clip(k1 + float(delta[8]), -0.6, 0.6))
                if np.abs(delta[:8]).max() < 3e-6 and abs(delta[8]) < 1e-5:
                    break
            _, _, _, _, _, Bw, MBw, _, _, inside = warp(G, k1)
            valid = (MAs > 0.5) & (MBw > 0.5) & inside
            av = As[valid]
            bv = Bw[valid]
            av = av - av.mean()
            bv = bv - bv.mean()
            score = float((av * bv).sum() / (torch.sqrt((av * av).sum() * (bv * bv).sum()) + 1e-12))
        return G, k1, score

    def search_lens_k1(self) -> float:
        """
        放射歪み k1 を推定する。間隔の大きいペアを数組選び、ホモグラフィと k1 を同時に Gauss-Newton で解いて
        （1/4 → 1/2 解像度）、スコアの高いペアの k1 の中央値を採る。ペア間のばらつきが大きい / 値が小さいときは 0。
        """
        a = self.args
        n = self.n
        t0 = time.time()
        koff = max(a.pair_offsets)
        if n <= koff:
            return 0.0
        nsel = min(8, n - koff)
        starts = np.linspace(0, n - 1 - koff, nsel).round().astype(int)
        sel = sorted(set((int(s), int(s) + koff) for s in starts))
        K = int(round(a.scale_max / a.scale_step))
        grid = [math.exp(k * a.scale_step) for k in range(-K, K + 1)]
        Nm = norm_matrix(self.W, self.H)
        Nm_inv = np.linalg.inv(Nm)
        ests: List[Tuple[float, float, float, Tuple[int, int]]] = []   # (k1, score, motion px, pair)
        for (i, j) in sel:
            s_, dx, dy, s_sc = self.coarse_estimate_scaled(i, j, grid)
            if s_sc < 0.2:
                continue
            corr = 0.5 * (s_ - 1.0) * (1.0 - 1.0 / self.sx)
            tx, ty = dx / self.sx + corr, dy / self.sy + corr
            Hpix = sim_matrix(s_, 0.0, tx, ty)
            G0 = Nm @ np.linalg.inv(Hpix) @ Nm_inv
            res = self.gn_refine_hk(i, j, G0, 0.0)
            if res is None:
                continue
            _, k1, score = res
            if score < 0.3:
                continue
            ests.append((k1, score, math.hypot(tx, ty), (i, j)))
        if len(ests) < 3:
            log(f"[lens] k1 を推定できるペアが少ない（{len(ests)} 件）ため歪み補正は行いません, {time.time() - t0:.1f}s")
            return 0.0
        ks = np.array([e[0] for e in ests])
        k1 = float(np.median(ks))
        q1, q3 = float(np.percentile(ks, 25)), float(np.percentile(ks, 75))
        log(f"[lens] k1 の同時推定: {len(ests)} ペア, median={k1:+.4f} (四分位 {q1:+.4f}..{q3:+.4f}), "
            + " ".join(f"({i},{j})={k:+.3f}" for k, _s, _m, (i, j) in ests) + f", {time.time() - t0:.1f}s")
        if (q3 - q1) > max(0.02, 0.5 * abs(k1)):
            log("[lens] ペア間のばらつきが大きいため歪み補正は行いません（k1 = 0）")
            return 0.0
        if abs(k1) < 0.015:
            log("[lens] 歪みは無視できる大きさです（k1 = 0）")
            return 0.0
        return k1

    # ---- 合成 ----
    def render(self, al: Alignment) -> Dict[str, np.ndarray]:
        a = self.args
        H, W, n = self.H, self.W, self.n
        # キャンバス倍率: auto = 最も寄った（細部が最も多い）フレームが等倍になる倍率（円筒投影では frame 0 の中心が等倍）
        g = (1.0 / al.S.min()) if a.canvas_scale == "auto" else float(a.canvas_scale)
        al_s = al.scaled_shifted(g, np.zeros(2))
        allc = np.concatenate([al_s.corners(k, *self.sizes[k]) for k in range(n)])
        origin = np.floor(allc.min(axis=0) + 0.5)
        al2 = al.scaled_shifted(g, origin)
        S, T = al2.S, al2.T
        boxes = [al2.corners(k, *self.sizes[k]) for k in range(n)]   # 有効領域の境界（パディングは含めない）
        Wc = int(math.ceil(max(b[:, 0].max() for b in boxes) + 0.5))
        Hc = int(math.ceil(max(b[:, 1].max() for b in boxes) + 0.5))
        if Wc * Hc > 400_000_000:
            sys.exit(f"キャンバス {Wc}x{Hc} が大きすぎます（--projection cylindrical か --canvas-scale を小さく）")
        # フレーム品質の段階（0: 良, 1: ボケ / ブレ, 2: ブレンドフレーム等）。段階の低いフレームだけで標本数が足りる画素ではそれらだけを使う
        tiers = getattr(self, "frame_tier", None)
        use_tier = tiers is not None and int(np.max(tiers)) > 0
        log(f"[render] canvas {Wc}x{Hc}, canvas scale={g:.4f}, band={a.band}, projection={al.kind}" +
            (f", quality tiers (min samples {a.quality_min_samples})" if use_tier else "") +
            (", motion reject" if a.motion_reject else ""))
        t0 = time.time()
        out_med = np.zeros((Hc, Wc, 3), dtype=np.uint8)
        out_sharp = np.zeros((Hc, Wc, 3), dtype=np.uint8)
        out_mean = np.zeros((Hc, Wc, 3), dtype=np.uint8)
        feather = float(a.feather)
        out_blend = np.zeros((Hc, Wc, 3), dtype=np.uint8) if feather > 0 else None
        coverage = np.zeros((Hc, Wc), dtype=np.uint16)
        coverage_g = np.zeros((Hc, Wc), dtype=np.uint16)
        holes = 0
        dev = self.dev
        xs_all = torch.arange(Wc, device=dev, dtype=torch.float32)
        nan = torch.tensor(float("nan"), device=dev)
        for y0 in range(0, Hc, a.band):
            y1 = min(Hc, y0 + a.band)
            bh = y1 - y0
            ks = [k for k in range(n) if boxes[k][:, 1].min() < y1 and boxes[k][:, 1].max() > y0]
            vals, aux, fw = [], [], []
            ys = torch.arange(y0, y1, device=dev, dtype=torch.float32)
            Xg, Yg = torch.meshgrid(xs_all, ys, indexing="xy")  # (bh, Wc)
            for k in ks:
                s = float(S[k])
                fx, fy, front = al2.to_frame(k, Xg, Yg)
                grid = torch.stack([2.0 * fx / (W - 1) - 1.0, 2.0 * fy / (H - 1) - 1.0], dim=-1)[None]
                img = torch.from_numpy(self.frames[k]).to(dev).permute(2, 0, 1).float()[None] / 255.0
                if s < 0.9:
                    img = gaussian_blur(img, 0.5 * math.sqrt(1.0 / (s * s) - 1.0))
                clean = torch.from_numpy(~self.overlay[k]).to(dev).float()
                sh = torch.from_numpy(self.sharp_q[k].astype(np.float32)).to(dev)
                sh = F.interpolate(sh[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                # 幾何的有効: パディングされたフレームは有効領域の内側だけ（境界は従来のフレーム端と同じ扱い）
                vk = self.valid_t[k]
                geom = vk.float() if vk is not None else torch.ones_like(clean)
                ax = torch.stack([geom, clean, sh])[None]
                smp = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
                if self.exposure is not None:
                    off, invg = self._exposure_at(k, fx, fy)
                    smp = (smp - off) * invg
                vals.append(smp)
                axs = F.grid_sample(ax, grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
                if front is not None:
                    axs = torch.cat([axs[0:1] * front.float()[None], axs[1:]])   # カメラの後ろ側（透視除算の z <= 0）は無効
                aux.append(axs)
                if feather > 0:
                    # フェザー重み: フレームの有効領域の端からの距離（フレーム px）を feather で飽和
                    hk, wk = self.sizes[k]
                    dist_e = torch.minimum(torch.minimum(fx, wk - 1 - fx), torch.minimum(fy, hk - 1 - fy))
                    fw.append((dist_e / feather).clamp(0.0, 1.0))
            if not ks:
                holes += bh * Wc
                continue
            V = torch.stack(vals)          # K,3,bh,Wc
            AX = torch.stack(aux)          # K,3,bh,Wc
            G = AX[:, 0] > 0.999           # 幾何的に有効
            C = G & (AX[:, 1] > 0.5)       # オーバーレイ除外後
            Sh = AX[:, 2]
            cnt_c = C.sum(0)               # 被覆数（穴埋め判定用）は解像度フィルタ前の値
            cnt_g = G.sum(0)
            if a.anchor_frame is not None and a.anchor_frame >= 0:
                # アンカーフレーム: 指定フレーム ±window が覆う画素は、それらのフレームだけで合成する
                # （キャラクターの髪や体が動くカットで、多数決の中央値が動く層の細線を消すのを防ぐ）
                lo, hi = a.anchor_frame - a.anchor_window, a.anchor_frame + a.anchor_window
                is_anc = torch.tensor([lo <= k <= hi for k in ks], device=dev)[:, None, None]
                Ca = C & is_anc
                has_c = Ca.any(0)
                C = torch.where(has_c[None], Ca, C)
                Ga = G & is_anc
                has_g = Ga.any(0)
                G = torch.where(has_g[None], Ga, G)
            if a.res_tol > 0:
                # 解像度を考慮: 画素ごとに最も拡大率の小さい（細部を持つ）標本を基準に、
                # 拡大率がその res_tol 倍以内の標本だけを統計に使う（ズーム動画で引きのボケた標本が多数派になるのを防ぐ）
                Mk = torch.tensor([float(S[k]) for k in ks], device=dev)[:, None, None].expand(len(ks), bh, Wc)
                big = torch.full_like(Mk, float("inf"))
                mc = torch.where(C, Mk, big).amin(0)
                C = C & (Mk <= mc[None] * a.res_tol)
                mg = torch.where(G, Mk, big).amin(0)
                G = G & (Mk <= mg[None] * a.res_tol)
            if use_tier:
                # フレーム品質: 良いフレーム（段階 0）だけで --quality-min-samples 以上の標本がある画素はそれらだけを使い、
                # 足りなければ段階 1（ボケ / ブレ）を戻し、それでも足りなければ全部使う（被覆を犠牲にしない）
                tk = torch.tensor([int(tiers[k]) for k in ks], device=dev)[:, None, None]
                mq = int(a.quality_min_samples)
                for lvl in (1, 0):
                    Cq = C & (tk <= lvl)
                    C = torch.where((Cq.sum(0) >= mq)[None], Cq, C)
                    Gq = G & (tk <= lvl)
                    G = torch.where((Gq.sum(0) >= mq)[None], Gq, G)
            Vc = torch.where(C[:, None], V, nan)
            med_c = torch.nanmedian(Vc, dim=0).values
            Vg = torch.where(G[:, None], V, nan)
            med_g = torch.nanmedian(Vg, dim=0).values
            med = torch.where((cnt_c > 0)[None], med_c, med_g)
            med = torch.nan_to_num(med, nan=0.0)
            if a.motion_reject:
                # 動体の判定: 中央値から外れた標本がまとまった第 2 クラスタを作り、それが時間的に広い範囲に散らばっているなら
                # そちらが背景（動く物は連続した区間にしか現れない。止まっている物が多数派になっても背景を選べる）
                dist0 = (V - med[None]).abs().mean(1)
                out = C & (dist0 >= a.inlier_tol)
                in1 = C & (dist0 < a.inlier_tol)
                cnt_o = out.sum(0)
                # 第 1 クラスタだけの中央値に置き換える: 少数派の動体が 20-40% あると全体の中央値は背景クラスタの端に寄って
                # 数階調暗く / 明るくなる（通過した帯が見える）ので、外れ値を除いてから取り直す
                med1 = torch.nanmedian(torch.where(in1[:, None], V, nan), dim=0).values
                med = torch.where((in1.sum(0) > 0)[None], torch.nan_to_num(med1, nan=0.0), med)
                cand = (cnt_o >= 3) & (cnt_o.float() >= 0.3 * C.sum(0).float())
                if bool(cand.any()):
                    Vo = torch.where(out[:, None], V, nan)
                    med2 = torch.nan_to_num(torch.nanmedian(Vo, dim=0).values, nan=0.0)
                    in2 = C & ((V - med2[None]).abs().mean(1) < a.inlier_tol)
                    tight = in2.sum(0).float() >= 0.7 * cnt_o.float()
                    kidx = torch.tensor([float(k) for k in ks], device=dev)[:, None, None].expand(len(ks), bh, Wc)
                    neg = torch.full_like(kidx, -1.0)
                    pos = torch.full_like(kidx, 1e9)

                    def spread(mask: torch.Tensor) -> torch.Tensor:
                        return torch.where(mask, kidx, neg).amax(0) - torch.where(mask, kidx, pos).amin(0)
                    choose2 = cand & tight & (spread(in2) > spread(in1) * 1.2 + 1.0)
                    med2 = torch.nan_to_num(torch.nanmedian(torch.where(in2[:, None], V, nan), dim=0).values, nan=0.0)
                    med = torch.where(choose2[None], med2, med)
            # インライア: クリーンかつ中央値に近い
            dist = (V - med[None]).abs().mean(1)
            inl = C & (dist < a.inlier_tol)
            fallback = inl.sum(0) == 0
            inl = torch.where(fallback[None], G & (dist < a.inlier_tol * 2), inl)
            cnt_i = inl.sum(0).clamp(min=1)
            mean = (V * inl[:, None]).sum(0) / cnt_i[None]
            mean = torch.where((inl.sum(0) > 0)[None], mean, med)
            # 鮮明度上位のインライアのみ平均（単一フレーム選択だと圧縮ノイズを拾うため）
            Sn = torch.where(inl, Sh, nan)
            thr = torch.nanquantile(Sn, 1.0 - a.sharp_top, dim=0)
            top = inl & (Sh >= torch.nan_to_num(thr, nan=float("inf"))[None])
            top = torch.where((top.sum(0) > 0)[None], top, inl)
            cnt_t = top.sum(0).clamp(min=1)
            sharp = (V * top[:, None]).sum(0) / cnt_t[None]
            sharp = torch.where((top.sum(0) > 0)[None], sharp, med)
            holes += int((cnt_g == 0).sum().item())

            def to_u8(t: torch.Tensor) -> np.ndarray:
                return (t.clamp(0, 1) * 255.0 + 0.5).byte().permute(1, 2, 0).cpu().numpy()
            out_med[y0:y1] = to_u8(med)
            out_mean[y0:y1] = to_u8(mean)
            if out_blend is not None:
                # フェザー合成: クリーン標本をフレーム端からの距離で重み付けして平均（境界の明るさの段差をなだらかにする）。
                # 中央値からのインライア判定は使わない: 2 標本しか無い境界付近で片方が外れ値扱いになると、重みがほぼ 0 の
                # 標本だけが残って正規化で全重みになり、画素ごとに供給源が切り替わる粒状ノイズが出る
                Fw = torch.stack(fw) * C.float()
                sw = Fw.sum(0)
                blend = (V * Fw[:, None]).sum(0) / sw.clamp(min=1e-6)[None]
                blend = torch.where((sw > 1e-6)[None], blend, mean)
                out_blend[y0:y1] = to_u8(blend)
            out_sharp[y0:y1] = to_u8(sharp)
            coverage[y0:y1] = cnt_c.cpu().numpy().astype(np.uint16)
            coverage_g[y0:y1] = cnt_g.cpu().numpy().astype(np.uint16)
        fallback = int(((coverage_g > 0) & (coverage == 0)).sum())
        log(f"[render] done {time.time() - t0:.1f}s, uncovered px={holes}, fallback (no clean sample) px={fallback}")
        res = {"median": out_med, "sharp": out_sharp, "mean": out_mean, "coverage": coverage, "coverage_geom": coverage_g}
        if out_blend is not None:
            res["blend"] = out_blend
        return res

    # ---- 後処理: クリーン標本の無い/少ない画素の穴埋め ----
    def fill_holes(self, res: Dict[str, np.ndarray]) -> int:
        """
        幾何的には覆われているのにクリーン標本が無い画素（テロップが常に載っていた場所）と、
        クリーン標本が「min_clean 件未満かつ被覆数の min_clean_pct % 未満」の画素を周囲から埋める。
        --hole-fill inpaint: push-pull 補間、blur: ぼかした値で置換。戻り値: 埋めた画素数。
        """
        a = self.args
        cg, cc = res["coverage_geom"].astype(np.int32), res["coverage"].astype(np.int32)
        fill = (cg > 0) & ((cc == 0) | ((cc < a.min_clean) & (cc < a.min_clean_pct / 100.0 * cg)))
        n = int(fill.sum())
        if n == 0 or a.hole_fill == "none":
            return 0
        fill_t = torch.from_numpy(fill).to(self.dev)
        for key in ("median", "mean", "sharp", "blend"):
            if key not in res:
                continue
            img = torch.from_numpy(res[key]).to(self.dev).permute(2, 0, 1).float()  # 3,H,W
            if a.hole_fill == "blur":
                b = box_blur(box_blur(img, 2 * 8 + 1), 2 * 8 + 1)
                out = torch.where(fill_t[None], b, img)
            else:
                out = push_pull_fill(img, ~fill_t)
            res[key] = (out.clamp(0, 255) + 0.5).byte().permute(1, 2, 0).cpu().numpy()
        return n

    def save_debug_overlay(self, out_dir: Path) -> None:
        idxs = sorted(set([0, self.n // 2, self.n - 1]))
        tiles = []
        for k in idxs:
            f = self.frames[k].copy()
            m = self.overlay[k]
            f[m] = (0.4 * f[m] + np.array([153, 0, 0])).astype(np.uint8)
            tiles.append(Image.fromarray(f).reduce(2))
        w = sum(t.width for t in tiles)
        h = max(t.height for t in tiles)
        canvas = Image.new("RGB", (w, h))
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0))
            x += t.width
        canvas.save(out_dir / "debug_overlay_mask.png")


# -------------------------
# CLI
# -------------------------

def parse_rect(s: str) -> Tuple[int, int, int, int]:
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--ignore-rect は x,y,w,h 形式")
    return parts[0], parts[1], parts[2], parts[3]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="動画からのグローバル整列＋時間的中央値による背景再構成")
    src = ap.add_argument_group("入力")
    src.add_argument("--video", type=str, help="入力動画")
    src.add_argument("--frames", nargs="+", help="フレーム画像（glob 可）")
    src.add_argument("--fps", type=float, default=None, help="動画から抽出する fps（省略時は --every）")
    src.add_argument("--every", type=int, default=1, help="N フレームごとに使用（default: 1）")
    src.add_argument("--start", type=float, default=0.0, help="動画の開始秒")
    src.add_argument("--duration", type=float, default=None, help="動画の使用秒数")
    src.add_argument("--max-frames", type=int, default=0, help="最大フレーム数（0=無制限）")
    src.add_argument("--start-frame", type=int, default=None, help="開始フレーム番号（0 始まり。--start より優先）")
    src.add_argument("--end-frame", type=int, default=None, help="終了フレーム番号（両端含む。--duration より優先）")
    src.add_argument("--crop", type=parse_rect, default=None,
                    help="読み込み時に切り出す矩形 x,y,w,h（元動画座標）。--ignore-rect / --text-rect はクロップ後の座標")
    ap.add_argument("--out", type=str, required=True, help="出力ディレクトリ")
    ap.add_argument("--device", type=str, default="auto", help="cuda / cpu / auto")

    al = ap.add_argument_group("位置合わせ")
    al.add_argument("--model", type=str, default=None, choices=["translation", "scale", "similarity", "homography"],
                    help="運動モデル: translation（平行移動）, scale（＋ズーム, default）, similarity（＋回転）, "
                         "homography（射影変換 8 自由度。実写のカメラ回転向け。--live-action の既定）")
    al.add_argument("--pairs", type=str, default="1,2,4", help="ペアのフレーム間隔（default: 1,2,4）")
    al.add_argument("--coarse-scale", type=float, default=0.25, help="粗探索の縮小率（default: 0.25）")
    al.add_argument("--coarse-tol", type=float, default=2.0, help="粗解の不整合とみなす残差 px（粗解像度）")
    al.add_argument("--min-overlap", type=float, default=0.15, help="粗探索で許容する最小重なり比率")
    al.add_argument("--scale-max", type=float, default=0.06,
                    help="粗探索のペアあたり最大 log スケール差（default: 0.06 ≈ ±6%%）")
    al.add_argument("--scale-step", type=float, default=0.004, help="粗探索の log スケール刻み（default: 0.004）")
    al.add_argument("--fine-scale", type=float, default=1.0,
                    help="Gauss-Newton 精密化の最終解像度倍率（CPU なら 0.5 推奨）")
    al.add_argument("--gn-iters", type=int, default=15, help="Gauss-Newton の各レベル反復回数")
    al.add_argument("--ignore-rect", type=parse_rect, action="append", default=[],
                    help="常に除外する矩形 x,y,w,h（複数指定可）")
    al.add_argument("--text-rect", type=parse_rect, action="append", default=[],
                    help="動くテロップ帯 x,y,w,h（複数指定可）。矩形内は静止条件なしで勾配の高い画素を文字としてマスク")
    al.add_argument("--text-halo", type=int, default=4, help="テキスト矩形内で文字マスクを広げる半径 px（default: 4）")

    st = ap.add_argument_group("静止オーバーレイ検出")
    st.add_argument("--no-static-mask", action="store_true", help="静止オーバーレイ検出を無効化")
    st.add_argument("--static-span", type=int, default=6, help="比較するフレーム間隔（default: 6）")
    st.add_argument("--static-diff", type=float, default=0.03, help="静止とみなす輝度差（0-1, default: 0.03）")
    st.add_argument("--static-grad", type=float, default=0.08, help="オーバーレイとみなす勾配下限（default: 0.08）")
    st.add_argument("--static-dilate", type=int, default=7, help="マスク膨張サイズ（default: 7）")
    st.add_argument("--static-halo", type=int, default=12,
                    help="静止エッジの周囲 N px 以内で静止している画素もマスク（文字のグロー対策, default: 12, 0=無効）")
    st.add_argument("--static-close", type=int, default=3,
                    help="前後 N フレームの両方でマスクされている画素をマスク（光沢アニメ対策, 0-3, default: 3）")

    ex = ap.add_argument_group("露出補正（フレーム間の明るさの違い）")
    ex.add_argument("--exposure", type=str, default="auto", choices=["auto", "on", "off"],
                    help="重なりからフレームごとの明るさ（露出・周辺減光）の違いを推定して補正する。"
                         "auto: 画像列（--frames）では on、動画では off（default: auto）")
    ex.add_argument("--exposure-profile", type=str, default=None, choices=["on", "off"],
                    help="全フレーム共通の周辺減光プロファイル（画像端に密な節点の折れ線, x と y）も推定する（default: on）。"
                         "off ならフレームごとのゲインとオフセットのみ")
    ex.add_argument("--exposure-min-score", type=float, default=0.2,
                    help="露出推定に使うペアの最小マッチスコア（default: 0.2）")
    ex.add_argument("--exposure-local", type=int, default=6,
                    help="フレームごとの局所オフセット場（双一次格子）の長辺セル数。湯気・光の当たり方など位置によって違う明るさの差を"
                         "重なりから推定して補正する。0 で無効（default: 6。フレーム数 > 100 では自動的に無効）")
    ex.add_argument("--denoise", type=float, default=0.0,
                    help="出力画像に à trous ウェーブレットのノイズ除去（wavelet_denoise.py）を掛けて *_dn.png も出力する。"
                         "値は最細レベルのしきい値（階調、目安 6-12）。0 で無効（default: 0）")
    ex.add_argument("--feather", type=float, default=-1.0,
                    help="recon_blend.png: 各フレームの端から N px でなだらかに重みを落として重ね合わせる（露出補正後に残る差を"
                         "境界でぼかす）。0 で出力しない、-1 = auto（画像列ではフレーム短辺の 1/4、動画では 0）")

    la = ap.add_argument_group("実写向け（--live-action で既定値をまとめて切替。個別指定が優先）")
    la.add_argument("--live-action", action="store_true",
                    help="実写に特化: model=homography, projection=auto, lens-k1=auto, quality=on, sharp-sigma=1, blend-detect=on, "
                         "deinterlace=auto, exposure=on（動画でも）, exposure-radial=on, motion-reject=on")
    la.add_argument("--projection", type=str, default=None, choices=["auto", "planar", "cylindrical"],
                    help="homography モデルの投影面。planar: フレーム 0 の平面, cylindrical: 焦点距離を推定して円筒に投影（長いパン向け）, "
                         "auto: パン角が --cyl-min-span 度を超えたら円筒（default: planar。--live-action では auto）")
    la.add_argument("--cyl-min-span", type=float, default=35.0, help="projection=auto で円筒投影に切り替えるパン角 度（default: 35）")
    la.add_argument("--focal", type=float, default=None, help="焦点距離 px（円筒投影用。省略時はペアホモグラフィから推定）")
    la.add_argument("--lens-k1", type=str, default=None,
                    help="放射歪み係数 k1（x_d = c + (x_u - c)(1 + k1 r²)、r は半対角で正規化）。auto: 数ペアで探索, off, または数値"
                         "（default: off。--live-action では auto）。補正後の画像で位置合わせと合成を行う")
    la.add_argument("--quality", type=str, default=None, choices=["on", "off"],
                    help="フレームごとの相対鮮明度を重なりから推定し、近傍より --quality-thr 未満にボケた / ブレたフレームを、"
                         "良いフレームだけで足りる画素では合成に使わない（default: off。--live-action では on）")
    la.add_argument("--quality-thr", type=float, default=0.75, help="鮮明度の比（近傍中央値に対する）の下限（default: 0.75）")
    la.add_argument("--quality-window", type=int, default=8, help="鮮明度の基準にする近傍フレーム数 ±N（default: 8）")
    la.add_argument("--quality-min-samples", type=int, default=3,
                    help="良いフレームだけで合成するのに必要な画素あたりの標本数。足りない画素では低品質フレームも使う（default: 3）")
    la.add_argument("--sharp-sigma", type=float, default=None,
                    help="鮮明度マップを取る前のガウスぼかし σ px。ノイズ / 粒子を鮮明さと誤認しないため（default: 0。--live-action では 1.0）")
    la.add_argument("--blend-detect", type=str, default=None, choices=["on", "off"],
                    help="前後フレームの平均で説明できるブレンドフレーム（フレームレート変換の混ざり）を検出して合成から外す"
                         "（default: off。--live-action では on）")
    la.add_argument("--blend-thr", type=float, default=0.4, help="ブレンド判定 ρ = |f - (a+b)/2| / (|a-b|/2) の上限（default: 0.4）")
    la.add_argument("--deinterlace", type=str, default=None, choices=["auto", "on", "off"],
                    help="インターレース素材の櫛状パターンを検出して片フィールド補間する。auto: 櫛の割合が 1%% を超えたら"
                         "（default: off。--live-action では auto）")
    la.add_argument("--exposure-radial", type=str, default=None, choices=["on", "off"],
                    help="露出補正に全フレーム共通の周辺減光（放射状の乗算プロファイル）を加える（default: off。--live-action では on）")
    la.add_argument("--motion-reject", type=str, default=None, choices=["on", "off"],
                    help="画素ごとに中央値から外れた標本が第 2 のまとまりを作るとき、時間的に広く散らばっている方を背景として選ぶ"
                         "（途中で止まった動体が多数派になるのを防ぐ。default: off。--live-action では on）")

    rd = ap.add_argument_group("合成")
    rd.add_argument("--canvas-scale", type=str, default="auto",
                    help="キャンバス倍率: auto（最も寄ったフレームが等倍）または数値（frame0 基準、1=frame0 等倍）")
    rd.add_argument("--band", type=int, default=64, help="合成時の行バンド幅（メモリ調整用）")
    rd.add_argument("--inlier-tol", type=float, default=0.06, help="中央値からの許容差（0-1, default: 0.06）")
    rd.add_argument("--anchor-frame", type=int, default=None,
                    help="アンカーフレーム番号（抽出後の番号）。このフレーム ±--anchor-window が覆う画素はそれらだけで合成する"
                         "（動くキャラクターをその姿で確定させる）")
    rd.add_argument("--anchor-window", type=int, default=2, help="アンカーの前後フレーム数（default: 2）")
    rd.add_argument("--res-tol", type=float, default=1.25,
                    help="解像度を考慮した中央値: 画素ごとに最も細部を持つ標本の拡大率 × この値以内の標本だけを使う"
                         "（ズーム動画で細線が消えるのを防ぐ。default: 1.25, 0=無効）")
    rd.add_argument("--sharp-top", type=float, default=0.3,
                    help="recon_sharp で平均するインライアの鮮明度上位比率（default: 0.3）")
    rd.add_argument("--hole-fill", type=str, default="inpaint", choices=["inpaint", "blur", "none"],
                    help="クリーン標本が無い/少ない画素の穴埋め: inpaint（push-pull 補間, default）, blur, none")
    rd.add_argument("--min-clean", type=int, default=12,
                    help="穴埋め対象とみなすクリーン標本数の上限（件数未満かつ --min-clean-pct 未満で対象, default: 12）")
    rd.add_argument("--min-clean-pct", type=float, default=25.0,
                    help="穴埋め対象とみなすクリーン標本の被覆数に対する割合 %%（default: 25）")
    rd.add_argument("--no-render", action="store_true", help="位置推定のみ行い画像を出力しない")
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_parser()
    a = ap.parse_args(argv)
    a.pair_offsets = sorted(set(int(v) for v in a.pairs.split(",") if v.strip()))
    a.static_mask = not a.no_static_mask
    a.ignore_rects = a.ignore_rect
    a.text_rects = a.text_rect
    if not a.video and not a.frames:
        ap.error("--video または --frames を指定してください")
    # --live-action: 個別に指定されていない実写向けオプションの既定値をまとめて切り替える
    live = bool(a.live_action)
    a.model = a.model or ("homography" if live else "scale")
    a.projection = a.projection or ("auto" if live else "planar")
    a.lens_k1 = a.lens_k1 or ("auto" if live else "off")
    a.quality = a.quality or ("on" if live else "off")
    a.sharp_sigma = float(a.sharp_sigma) if a.sharp_sigma is not None else (1.0 if live else 0.0)
    a.blend_detect = a.blend_detect or ("on" if live else "off")
    a.deinterlace = a.deinterlace or ("auto" if live else "off")
    a.exposure_radial = a.exposure_radial or ("on" if live else "off")
    # 実写では周辺減光は放射項（乗算）で扱い、画面固定の加算プロファイル（プレイヤーのグラデーション用）は既定で使わない
    a.exposure_profile = a.exposure_profile or ("off" if live else "on")
    a.motion_reject = (a.motion_reject or ("on" if live else "off")) == "on"
    if live and a.exposure == "auto":
        a.exposure = "on"
    if a.lens_k1 not in ("auto", "off"):
        try:
            float(a.lens_k1)
        except ValueError:
            ap.error("--lens-k1 は auto / off / 数値")
    if a.projection != "planar" and a.model != "homography":
        log(f"[warn] --projection {a.projection} は model=homography のときだけ有効です（{a.model} では平面）")
        a.projection = "planar"
    if live:
        log("[live-action] model=%s projection=%s lens-k1=%s quality=%s sharp-sigma=%g blend-detect=%s deinterlace=%s "
            "exposure=%s profile=%s radial=%s motion-reject=%s" % (a.model, a.projection, a.lens_k1, a.quality, a.sharp_sigma, a.blend_detect,
                                                                  a.deinterlace, a.exposure, a.exposure_profile, a.exposure_radial,
                                                                  "on" if a.motion_reject else "off"))
    if a.canvas_scale != "auto":
        try:
            float(a.canvas_scale)
        except ValueError:
            ap.error("--canvas-scale は auto か数値")

    if a.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(a.device)
    log(f"[device] {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    t_all = time.time()
    if a.video:
        frames, eff_fps = load_frames_from_video(Path(a.video), a.every, a.fps, a.start, a.duration, a.max_frames,
                                                 a.start_frame, a.end_frame, a.crop)
        log(f"[load] {len(frames)} frames from {a.video} (effective {eff_fps:.2f} fps)")
    else:
        paths = expand_globs(a.frames)
        frames = load_frames_from_paths(paths, a.every, a.max_frames, a.start_frame, a.end_frame, a.crop)
        log(f"[load] {len(frames)} frames from {len(paths)} files")
    if len(frames) < 2:
        sys.exit("フレームが 2 枚未満です")
    if a.deinterlace != "off":
        # 櫛状パターンの割合を等間隔の最大 12 フレームで測る
        idxs = sorted(set(np.linspace(0, len(frames) - 1, min(12, len(frames))).round().astype(int).tolist()))
        combs = [comb_fraction((0.299 * frames[k][:, :, 0] + 0.587 * frames[k][:, :, 1] + 0.114 * frames[k][:, :, 2])) for k in idxs]
        cv, ch = float(np.median([c[0] for c in combs])), float(np.median([c[1] for c in combs]))
        do_di = a.deinterlace == "on" or (a.deinterlace == "auto" and is_interlaced(cv, ch))
        log(f"[deinterlace] 櫛状パターンの割合 縦 {cv * 100:.2f}% / 横 {ch * 100:.2f}%（中央値、max 縦 {max(c[0] for c in combs) * 100:.2f}%）"
            + ("（インターレースと判断、片フィールド補間します）" if do_di else "（プログレッシブ）"))
        if do_di:
            frames = [deinterlace_frame(f) for f in frames]
    frames, sizes = pad_to_common_size(frames)
    if sizes is not None:
        uniq = sorted(set(sizes))
        log(f"[load] サイズの異なる画像を {frames[0].shape[1]}x{frames[0].shape[0]} に揃えました（右・下を端の画素で埋め、"
            f"合成には使いません）: " + ", ".join(f"{w}x{h}" for h, w in uniq))
    exposure_on = a.exposure == "on" or (a.exposure == "auto" and not a.video)
    if a.feather < 0:
        a.feather = 0.0 if a.video else 0.25 * min(frames[0].shape[0], frames[0].shape[1])

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    k1 = 0.0
    if a.lens_k1 not in ("auto", "off"):
        k1 = float(a.lens_k1)
    rc = Reconstructor(frames, dev, a, sizes)
    rc.preprocess()
    if a.lens_k1 == "auto":
        if sizes is not None:
            log("[lens] サイズの異なる画像列では k1 の探索を行いません")
        else:
            k1 = rc.search_lens_k1()
    if abs(k1) > 1e-9:
        if sizes is not None:
            sys.exit("--lens-k1 はサイズの同じフレーム列にだけ使えます")
        t_u = time.time()
        frames = undistort_frames(frames, k1, dev)
        x0, y0, wu, hu = undistort_crop_rect(rc.W, rc.H, k1)
        log(f"[lens] k1={k1:+.4f} で歪み補正（bicubic）"
            + (f"、有効領域 {wu}x{hu} に切り出し（--ignore-rect / --text-rect は補正前の座標のまま）" if (x0 or y0) else "")
            + f", {time.time() - t_u:.1f}s")
        if x0 or y0:
            a.ignore_rects = [(x - x0, y - y0, w, h) for (x, y, w, h) in a.ignore_rects]
            a.text_rects = [(x - x0, y - y0, w, h) for (x, y, w, h) in a.text_rects]
        rc = Reconstructor(frames, dev, a, sizes)
        rc.preprocess()
    rc.save_debug_overlay(out_dir)
    if a.blend_detect == "on":
        rc.detect_blend_frames()
    al, pairs, pp, sc, rn = rc.align()
    if a.quality == "on" or a.blend_detect == "on":
        rc.estimate_quality(pairs, sc, al, out_dir)
    if exposure_on:
        rc.estimate_exposure(pairs, pp, sc, out_dir)

    with open(out_dir / "positions.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if al.kind == "homography":
            extra = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
        elif al.kind == "cyl":
            extra = ["yaw_deg", "pitch_deg", "roll_deg", "focal_px"]
        else:
            extra = []
        wr.writerow(["frame", "scale", "theta_deg", "x", "y"] + extra)
        for k in range(rc.n):
            row = [k, f"{al.S[k]:.5f}", f"{math.degrees(al.TH[k]):.4f}", f"{al.T[k, 0]:.3f}", f"{al.T[k, 1]:.3f}"]
            if al.kind == "homography":
                assert al.Hm is not None
                row += [f"{v:.7g}" for v in al.Hm[k].reshape(-1)]
            elif al.kind == "cyl":
                assert al.R is not None and al.K is not None
                ax = al.R[k] @ np.array([0.0, 0.0, 1.0])
                row += [f"{math.degrees(math.atan2(ax[0], ax[2])):.4f}", f"{math.degrees(math.asin(max(-1.0, min(1.0, ax[1])))):.4f}",
                        f"{math.degrees(al.TH[k]):.4f}", f"{al.K[0, 0]:.2f}"]
            wr.writerow(row)
    with open(out_dir / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["i", "j", "scale", "theta_deg", "tx", "ty", "score", "residual", "outlier_pct"])
        for k, (i, j) in enumerate(pairs):
            wr.writerow([i, j, f"{pp[k, 0]:.5f}", f"{math.degrees(pp[k, 1]):.4f}", f"{pp[k, 2]:.3f}", f"{pp[k, 3]:.3f}",
                         f"{sc[k]:.4f}", f"{rn[k]:.3f}", f"{rc.pair_outlier[k] * 100:.2f}"])

    if not a.no_render:
        res = rc.render(al)
        nfill = rc.fill_holes(res)
        if nfill:
            log(f"[postfx] hole fill ({a.hole_fill}): {nfill} px")
        Image.fromarray(res["median"]).save(out_dir / "recon_median.png")
        Image.fromarray(res["sharp"]).save(out_dir / "recon_sharp.png")
        Image.fromarray(res["mean"]).save(out_dir / "recon_mean.png")
        if "blend" in res:
            Image.fromarray(res["blend"]).save(out_dir / "recon_blend.png")
            log(f"[output] {out_dir / 'recon_blend.png'} (feather {a.feather:.0f} px)")
        if a.denoise > 0:
            from wavelet_denoise import denoise_rgb
            t_dn = time.time()
            for key in ("median", "mean", "sharp", "blend"):
                if key in res:
                    Image.fromarray(denoise_rgb(res[key], thr=a.denoise)).save(out_dir / f"recon_{key}_dn.png")
            log(f"[denoise] à trous wavelet thr={a.denoise:g} → recon_*_dn.png, {time.time() - t_dn:.1f}s")
        cov = res["coverage"].astype(np.float32)
        cov = (255.0 * np.clip(cov / max(1.0, float(cov.max())), 0, 1)).astype(np.uint8)
        Image.fromarray(cov).save(out_dir / "coverage.png")
        log(f"[output] {out_dir / 'recon_median.png'}")
        log(f"[output] {out_dir / 'recon_sharp.png'}")
        log(f"[output] {out_dir / 'recon_mean.png'}")
    log(f"[done] total {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
