#!/usr/bin/env python3
"""
wavelet_denoise.py - アニメ調画像向けのウェーブレット（à trous / starlet）ノイズ除去

非間引き（undecimated）の à trous ウェーブレット変換（B3 スプライン [1,4,6,4,1]/16 を 2^l 間隔で分離適用）で
画像を詳細係数 (level 0..L-1) と最粗成分に分け、詳細係数を軟しきい値 k·σ_l で縮小して再構成する。
間引きしないのでブロック歪みが出ず、平坦な塗りと輪郭線が主体のアニメ画像で、輪郭（大きな係数）を残して
微小なノイズ（JPEG のモスキート、合成の境界の粒状ノイズ）だけを落とせる。

しきい値は最細レベルで thr（0..255 の階調）、粗いレベルほど B3 スプラインのノイズ伝播係数 (0.889, 0.200, 0.086,
0.041, ...) に比例して小さくする（白色ノイズならこの比率で減るため）。アニメ画像は平坦部で差分がほぼ 0 になり
MAD による σ の自動推定が働かないので、しきい値を直接指定する（既定 8 階調。粒状ノイズや JPEG のモスキートは
これで消え、輪郭線は係数が大きいので残る）。

使い方:
  python wavelet_denoise.py in.png out.png [--thr 8] [--levels 4]
モジュールとして: denoise_rgb(img_u8 (H,W,3), thr=8.0, levels=4) -> uint8
"""

from __future__ import annotations

import argparse

import numpy as np

# B3 スプライン à trous の各レベルのノイズ標準偏差（入力の白色ノイズ σ=1 に対する詳細係数の σ）
_B3_NOISE = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005]
_KERNEL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _atrous_blur(x: np.ndarray, step: int) -> np.ndarray:
    """(H, W, C) を 2^l 間隔の 5 タップ B3 スプラインで分離ぼかし（境界は反射）。"""
    pad = 2 * step
    y = np.pad(x, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    H = x.shape[0]
    out = np.zeros_like(x)
    for t, w in enumerate(_KERNEL):
        off = (t - 2) * step + pad
        out += w * y[off:off + H]
    y = np.pad(out, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    W = x.shape[1]
    out2 = np.zeros_like(x)
    for t, w in enumerate(_KERNEL):
        off = (t - 2) * step + pad
        out2 += w * y[:, off:off + W]
    return out2


def estimate_sigma(img: np.ndarray) -> float:
    """ノイズ σ（0..255 単位）の推定: 輝度の水平差分の MAD から。"""
    g = img[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    d = np.diff(g, axis=1)
    mad = float(np.median(np.abs(d - np.median(d))))
    return mad / 0.6745 / np.sqrt(2.0)


def denoise_rgb(img_u8: np.ndarray, thr: float = 8.0, levels: int = 4) -> np.ndarray:
    """
    (H, W, 3) uint8 → (H, W, 3) uint8。thr: 最細レベルの軟しきい値（階調）。レベル l のしきい値は thr × σ_l / σ_0。
    """
    x = img_u8[..., :3].astype(np.float32)
    if thr <= 0:
        return img_u8[..., :3].copy()
    levels = max(1, min(levels, len(_B3_NOISE)))
    acc = np.zeros_like(x)
    cur = x
    for lv in range(levels):
        sm = _atrous_blur(cur, 1 << lv)
        det = cur - sm
        thr_l = thr * _B3_NOISE[lv] / _B3_NOISE[0]
        # 軟しきい値（輪郭の大きな係数は thr だけ縮んで残り、ノイズは 0 になる）
        det = np.sign(det) * np.maximum(np.abs(det) - thr_l, 0.0)
        acc += det
        cur = sm
    acc += cur
    return np.clip(acc + 0.5, 0, 255).astype(np.uint8)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="アニメ調画像向け à trous ウェーブレットノイズ除去")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--thr", type=float, default=8.0, help="最細レベルのしきい値（階調, default: 8。大きいほど強く除去）")
    ap.add_argument("--levels", type=int, default=4, help="ウェーブレットのレベル数（default: 4）")
    a = ap.parse_args(argv)
    from PIL import Image
    img = np.asarray(Image.open(a.input).convert("RGB"))
    out = denoise_rgb(img, thr=a.thr, levels=a.levels)
    Image.fromarray(out).save(a.output)
    print(f"[denoise] thr={a.thr} levels={a.levels} (noise sigma estimate {estimate_sigma(img):.2f}) -> {a.output}")


if __name__ == "__main__":
    main()
