# stitch_candidates

スクロール/パンニングで取得したスクリーンショットを連結するツール

A tool for stitching scroll/pan screenshots

---

## 概要 / Overview

### 🇯🇵 日本語

`stitch_candidates.py` は、複数のスクリーンショットや分割画像を自動連結し、**「惜しい候補」を大量に生成して目視で選ぶ**ためのツールです。

完全自動で1枚の正解を出すことは目的としていません。

- 「あと1〜3pxずらせば合う」
- 「境目はほぼ正しい」

という**人間が判断できるレベルの候補を確実に出す**ことに特化しています。

### 🇺🇸 English

`stitch_candidates.py` automatically stitches multiple screenshots or image fragments and **generates many "almost-correct" candidates for visual inspection**.

It is **not** designed to output a single perfect image automatically.

Instead, it focuses on reliably producing candidates that are:

- Just 1–3 pixels away from perfect
- Good enough for a human to finalize

---

## 設計思想 / Design Philosophy

### 日本語

- 完全自動より **Human-in-the-loop**
- 失敗より **スキップ**
- 正解より **候補**
- 最後の1pxは **人間が決める**

Photoshop / GIMP 等での最終調整を前提としています。

### English

- Human-in-the-loop by design
- Skip invalid cases instead of crashing
- Generate candidates, not a single "answer"
- Humans decide the final pixel alignment

---

## インストール / Installation

```bash
pip install pillow numpy
pip install opencv-python   # optional but recommended
```

---

## 基本的な使い方 / Basic Usage

### 垂直連結 / Vertical stitching

```bash
python stitch_candidates.py -m v -o out img/*.png
```

### 水平連結 / Horizontal stitching

```bash
python stitch_candidates.py -m h -o out img/*.png
```

### スネーク(ジグザグ)連結 / Zigzag (snake) stitching

```bash
python stitch_candidates.py -m snake --cols 4 -o out img/*.png
```

---

## パラメータ / Parameters

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `-m, --mode` | `v` (垂直), `h` (水平), `snake` | `v` (vertical), `h` (horizontal), `snake` |
| `-o, --out` | 出力ディレクトリ | Output directory |
| `--overlap` | 重なりピクセル数（カンマ区切り） | Overlap pixels (comma list) |
| `--overlap-pct` | 重なり比率 0.01-0.95 | Overlap as ratio 0.01-0.95 |
| `--band` | マッチングに使うバンドサイズ | Band size for matching |
| `--search` | 位置合わせの探索範囲(px) | Search range for alignment |
| `--cols` | スネークモードの列数 | Columns for snake mode |
| `--ignore` | 無視領域 (px) | Ignore region (px) |
| `--ignore-pct` | 無視領域 (%) | Ignore region (%) |
| `--min-overlap-ratio` | 実効オーバーラップ比率の下限 (default: 0.3) | Minimum effective overlap ratio (default: 0.3) |
| `--min-boundary-score` | 境界一致度(SSIM)の下限 (default: 0.3) | Minimum boundary similarity (default: 0.3) |
| `--exclude-method` | 除外するメソッド (例: ssim) | Exclude methods (e.g., ssim) |
| `--refine-from` | 微調整の基準となる候補画像 | Candidate image for refinement |
| `--refine-delta` | 微調整時の探索範囲 ±n px | Search range ±n px for refinement |
| `--overlap-scan` | overlap範囲スキャン: MIN,MAX,STEP | Scan overlap range: MIN,MAX,STEP |
| `--overlap-auto` | 3段階階層探索 (step 100→10→1) | 3-stage hierarchical search (step 100→10→1) |
| `--top-n` | スキャン/オートモードで出力する上位N件 (default: 5) | Top N candidates in scan/auto mode (default: 5) |

---

## 推奨ワークフロー / Recommended Workflow

### Step 1: 荒探索 / Rough search

```bash
python stitch_candidates.py -m v -o out \
  --overlap 80,100,120 \
  --band 30,40,50 \
  --search 10,20 \
  img/*.png
```

### Step 2: 惜しい候補を見つける / Find a nearly-correct candidate

出力ファイル名の例 / Example filename:

```
v_ov120__phase__band40__srch20__p1_dx0_dy-3__p2_dx1_dy-2.png
```

オフセットが **±1〜5px** の範囲なら、正解に非常に近い状態です。

If offsets are within **±1–5 pixels**, you are already very close.

### Step 3: 微調整 / Refinement

```bash
python stitch_candidates.py -m v -o refine_out \
  --refine-from out/v_ov120__phase__band40__srch20__p1_dx0_dy-3.png \
  --refine-delta 2 \
  img/*.png
```

**ポイント / Key ideas:**

- `--refine-from` で基準候補を指定
- `--refine-delta` で探索範囲を小さく（±2px程度）

---

## 自動スキャンモード / Auto Scan Mode

overlap範囲を自動スキャンして、スコアの高い上位N件を出力します。

Automatically scans overlap range and outputs top N candidates by score.

```bash
# 手動範囲指定 / Manual range
python stitch_candidates.py -m v -o out \
  --overlap-scan 50,150,5 \
  --top-n 5 \
  img/*.png

# 全自動3段階探索 / Fully automatic 3-stage search
python stitch_candidates.py -m v -o out \
  --overlap-auto \
  --top-n 5 \
  img/*.png
```

**`--overlap-auto` の動作 / How --overlap-auto works:**

1. **Stage 1**: 0〜画像高さをステップ100で粗く走査 → 上位3エリア特定
2. **Stage 2**: 各エリア±50px周辺をステップ10で走査 → 上位3エリア特定
3. **Stage 3**: 各エリア±5px周辺をステップ1で精密走査 → 最終結果出力

1. **Stage 1**: Coarse scan 0 to image height, step 100 → Top 3 areas
2. **Stage 2**: Scan ±50px around each area, step 10 → Top 3 areas
3. **Stage 3**: Fine scan ±5px around each area, step 1 → Final output

**スコアリング / Scoring:**

- マッチングスコア（位相相関/NCC）と境界一致度（SSIM風）の調和平均
- Harmonic mean of matching score and boundary similarity (SSIM-like)

**出力ファイル名 / Output filename:**

```
scan_rank01_score0.892_ov120__phase__band30__srch10__p1_dx0_dy-2.png
auto_rank01_score0.892_ov120__phase__band30__srch10__p1_dx0_dy-2.png
```

---

## constantDelta（隠し機能）/ constantDelta (Secret Weapon)

スクリプトは自動的に以下の候補も生成します：

The script automatically generates candidates where:

```
全画像を一定の dx / dy でずらして連結
all images are stitched using a constant dx / dy
```

以下のケースで非常に効果的です / Extremely effective when:

- 全体的に一定方向にずれている
- 継ぎ目が常に同じ方向にずれている

---

## 安定性 / Stability

### 日本語

- overlap が大きすぎる場合は **自動的に安全な最大値にクリップ**
- 出力サイズが不正になる候補は **自動スキップ**
- マッチングスコアが低い場合は **自動スキップ** (phase < 0.05, ncc_gray < 0.10, ncc_edge < 0.05, ssim < 0.30)
- 境界一致度が低い場合もスキップ可能 (`--min-boundary-score`)
- 有効な候補が0件の場合、**診断情報を表示**
- NumPy の broadcast エラーで停止しません

### English

- Overlap is automatically clipped to a safe maximum
- Invalid compositions are skipped
- Low matching scores are pruned (phase < 0.05, ncc_gray < 0.10, ncc_edge < 0.05, ssim < 0.30)
- Low boundary similarity can also be pruned (`--min-boundary-score`)
- Shows **diagnostic info** when no valid candidates found
- The script will not crash due to NumPy broadcast errors

---

## 適したユースケース / When to Use

### ✅ 適している / Good fit

- スクロールキャプチャの再構成
- パンニング背景の連結
- 手動仕上げを前提とした半自動ワークフロー

### ❌ 適さない / Not ideal

- 完全自動で1枚の正解が欲しい場合
- 強いパースペクティブ歪みがある場合
- フレームごとに非線形な変形がある場合

---

## マッチング手法 / Matching Methods

| 手法 / Method | 説明 / Description | 足切り閾値 / Min Score |
|--------------|-------------------|----------------------|
| `phase` | 位相相関（FFTベース、高速） / Phase correlation (FFT-based, fast) | 0.05 |
| `ncc_gray` | グレースケールNCC / NCC on grayscale | 0.10 |
| `ncc_edge` | エッジマップNCC（Sobel） / NCC on Sobel edge map | 0.05 |
| `ssim` | SSIMベース（低速だが高精度） / SSIM-based (slow but accurate) | 0.30 |

スコアが閾値未満のマッチングは自動的にスキップされます。

Matches with scores below the threshold are automatically skipped.

**高速化 / Speed up:**

SSIMは計算コストが高いため、`--exclude-method ssim` で除外できます。

SSIM is computationally expensive. Use `--exclude-method ssim` to skip it.

---

## ライセンス / License

MIT License

---

## まとめ / Summary

> **多くの良い候補を生成し、人間が最良のものを選ぶ**

> **Generate many good answers, then let humans choose the best one.**

画像エディタ（Photoshop、GIMP等）との併用を推奨します。

Works best when combined with image editors like Photoshop or GIMP.

---

## 関連 / Link

[AIにつなぎ画像合成（スティッチ）ツールを作ってもらった - ふぁメモ](https://fa.hatenadiary.jp/entry/20260202/1770030091)

