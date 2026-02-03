# video_strip_reconstruct

動画のスクロール（縦パン/横パン）から背景を復元するツール

A tool for reconstructing backgrounds from scrolling/panning videos

---

## 概要 / Overview

### 日本語

`video_strip_reconstruct.py` は、スクロールする動画からストリップを切り出し、背景を復元するツールです。

- テロップ（字幕・ロゴ等）を**画素単位で除去**
- dy/dx（移動量）を自動推定して連結
- **キーフレームモード**で高品質なフレームを選択し、ハードカットで連結
- **stitch_candidates.py と同様のマッチングロジック**で正確な位置合わせ
- 複数の候補を生成し、人間が最良のものを選択

### English

`video_strip_reconstruct.py` extracts strips from scrolling videos and reconstructs the background.

- Removes text overlays (subtitles, logos) at the **pixel level**
- Automatically estimates dy/dx (displacement) for stitching
- **Keyframe mode** selects high-quality frames and stitches with hard cuts
- **Same matching logic as stitch_candidates.py** for accurate alignment
- Generates multiple candidates for human selection

---

## 設計思想 / Design Philosophy

### 日本語

- テロップはエッジ強度で判別し、背景っぽい画素を優先
- 推定の符号が逆になるケースがあるため、**dy/-dy または dx/-dx の2候補**を出力
- キーフレームモードでは品質スコア（シャープネス、ノイズ、テキスト量）に基づいてフレームを選択
- 完璧な復元より**良い候補の生成**を目指す

### English

- Text is identified by edge strength; background-like pixels are prioritized
- Outputs **both dy/-dy or dx/-dx candidates** since sign may be inverted
- Keyframe mode selects frames based on quality score (sharpness, noise, text density)
- Aims to **generate good candidates** rather than perfect reconstruction

---

## インストール / Installation

```bash
pip install pillow numpy
pip install opencv-python   # optional but recommended
```

ffmpegがあると動画から直接フレーム抽出が可能です。

ffmpeg enables direct frame extraction from videos.

---

## 基本的な使い方 / Basic Usage

### 動画から直接処理 / Direct from video

```bash
# 縦スクロール（jigsawモード + キーフレームモードがデフォルト）
python video_strip_reconstruct.py --video input.mp4 --fps 2 \
  --strip-y 0 --strip-h 1080 --out outdir

# 横スクロール
python video_strip_reconstruct.py --video input.mp4 --fps 2 \
  --scroll-axis horizontal --strip-x 0 --strip-w 1920 --out outdir
```

### 抽出済みフレームから / From extracted frames

```bash
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --out outdir
```

### 従来モードを使用する場合 / Using Traditional Mode

```bash
# ジグソーモードを無効化
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --no-jigsaw --out outdir

# キーフレームモードのみ無効化（ブレンドモード）
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --no-keyframe-mode --out outdir
```

---

## パラメータ / Parameters

### 入力 / Input

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `--video` | 入力動画パス | Input video path |
| `--frames` | フレーム画像のglob | Glob pattern for frame images |
| `--fps` | フレーム抽出レート (default: 2) | Frame extraction rate |
| `--start` | 抽出開始位置（秒） | Start position in seconds |
| `--dur` | 抽出時間（秒） | Duration in seconds |
| `--deinterlace` | インターレース解除 | Apply deinterlacing (yadif) |

### ストリップ設定 / Strip Settings

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `--scroll-axis` | スクロール軸: vertical/horizontal (default: vertical) | Scroll axis |
| `--strip-y` | 縦スクロール時のストリップ上端Y座標 | Top Y for vertical scroll |
| `--strip-h` | ストリップの高さ (default: 100) | Height of strip |
| `--strip-x` | 横スクロール時のストリップ左端X座標 | Left X for horizontal scroll |
| `--strip-w` | ストリップの幅 (default: 100) | Width of strip |

### キーフレームモード / Keyframe Mode

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `--keyframe-mode` | キーフレームモード（デフォルト有効） | Keyframe mode (default: enabled) |
| `--no-keyframe-mode` | キーフレームモードを無効化 | Disable keyframe mode |
| `--min-quality` | 最低品質スコア (default: 0.3) | Minimum quality score |
| `--max-keyframe-gap` | 最大フレーム間隔 (default: 5) | Maximum frame gap |
| `--allow-ignore-at-edges` | 端フレームでignore領域を許容 | Allow ignore regions at edge frames |
| `--keyframe-stitch-method` | スティッチ方法: matching/position (default: matching) | Stitching method |

### マッチング設定 / Matching Settings

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `--dy-search` | dy探索範囲 ±n px (default: 40) | Search range for dy ±n px |
| `--match-method` | マッチング手法 (default: phase) | Matching method(s) |
| `--min-peak` | ピークスコア閾値（診断用） | Peak score threshold for diagnostics |
| `--no-edges` | エッジなしでdy推定 | Use grayscale (not edges) for dy estimation |
| `--dy-region` | dy推定に使う領域 "y,h" | Region for dy estimation "y,h" |
| `--template-region` | テンプレート領域 "y,h,x,w" | Manual template region "y,h,x,w" |
| `--suggest-templates` | テンプレート候補を探索・評価 | Suggest best template regions |
| `--suggest-n` | 評価する候補数 (default: 15) | Number of candidates to evaluate |
| `--uniform-dy` | 均一なdy値を直接指定 | Use uniform dy per frame |

### テロップ除去設定 / Text Removal Settings

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `--edge-thr` | エッジ閾値（カンマ区切りで複数可） | Edge threshold(s), comma-separated |
| `--ignore` | 無視領域 (px) | Ignore region in pixels |
| `--ignore-pct` | 無視領域 (%) | Ignore region in percentage |

### スクロール方向・モード / Scroll Direction & Mode

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `--scroll-dir` | スクロール方向: up/down/left/right/both (default: both) | Scroll direction |
| `--jigsaw` | ジグソーモード（デフォルト有効） | Jigsaw mode (default: enabled) |
| `--no-jigsaw` | ジグソーモードを無効化 | Disable jigsaw mode |
| `--static-bg` | 背景固定モード（時間的中央値で復元） | Static background mode |
| `--static-method` | static-bgの手法: median/min_edge | Method for static-bg mode |

### 出力 / Output

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `-o, --out` | 出力ディレクトリ (default: recon_out) | Output directory |

---

## キーフレームモード詳細 / Keyframe Mode Details

### 概要 / Overview

キーフレームモードは、品質の高いフレームを選択し、ハードカット（ブレンドなし）で連結します。

Keyframe mode selects high-quality frames and stitches them with hard cuts (no blending).

### 品質スコア / Quality Score

各フレームは以下の指標で評価されます：

Each frame is evaluated by:

| 指標 | 説明 | Description |
|-----|------|-------------|
| シャープネス | ラプラシアン分散 | Laplacian variance |
| テキスト量 | エッジ強度の割合 | Edge strength ratio |
| ノイズレベル | 高周波成分の量 | High-frequency content |
| 圧縮アーティファクト | ブロックノイズ | Block noise |

### スティッチ方法 / Stitching Methods

#### matching（デフォルト）

`stitch_candidates.py` と同じロジックを使用：

Uses the same logic as `stitch_candidates.py`:

1. 各キーフレームペア間でマッチング（phase, ncc_gray, ncc_edge, ssim）
2. オーバーラップを20%〜80%の範囲でスキャン
3. マッチングスコアと境界一致度の調和平均でスコアリング
4. 上位3候補のみを出力（低スコアは自動削除）
5. 順次合成: (1+2) + 3 + 4 + ...

#### position

dy/dx推定値をそのまま使用してオーバーラップ中点でカット。

Uses estimated dy/dx directly and cuts at overlap midpoints.

### 出力ファイル名 / Output Filenames

マッチングモードの出力ファイル名にはスコアとパラメータが含まれます：

Matching mode output filenames include score and parameters:

```
recon_keyframe_dy_score0.5234_phase_b100_s20_ov432_dx0dy-15.png
```

形式 / Format: `{base}_score{スコア}_{手法}_b{バンド}_s{探索範囲}_ov{オーバーラップ}_{dx/dy}.png`

---

## マッチング手法 / Matching Methods

| 手法 / Method | 説明 / Description |
|--------------|-------------------|
| `phase` | エッジ画像での位相相関（デフォルト） / Phase correlation on edge map (default) |
| `phase_gray` | グレースケールでの位相相関 / Phase correlation on grayscale |
| `ncc_gray` | グレースケールNCC / NCC on grayscale |
| `ncc_edge` | エッジマップNCC / NCC on edge map |
| `ssim` | SSIMベースマッチング（遅いが精度高） / SSIM-based matching (slow but accurate) |
| `template` | テンプレートマッチング追跡 / Template matching with tracking |
| `robust` | template + phase のハイブリッド / Hybrid template + phase |
| `optical_flow` | オプティカルフロー / Optical flow |

### キーフレームモードでのマッチング

キーフレームモード（`--keyframe-stitch-method matching`）では、4手法すべてを試行：

- phase
- ncc_gray
- ncc_edge
- ssim

バンドサイズ [50, 100, 200] × 探索範囲 [20, 50] の組み合わせで候補を生成し、スコア上位3件を出力します。

---

## 出力ファイル / Output Files

### 基本出力 / Basic Output

| ファイル | 説明 | Description |
|---------|------|-------------|
| `recon_dy.png` | dyをそのまま積算した結果 | Result with dy as-is |
| `recon_negdy.png` | -dyを積算した結果 | Result with negated dy |
| `recon_horizontal_dx.png` | 横スクロール、dxを積算 | Horizontal scroll, dx as-is |
| `recon_horizontal_negdx.png` | 横スクロール、-dxを積算 | Horizontal scroll, negated dx |

### キーフレームモード出力 / Keyframe Mode Output

| ファイル | 説明 | Description |
|---------|------|-------------|
| `recon_keyframe_dy_score*.png` | 縦、マッチングモード（スコア付き、上位3件） | Vertical, matching mode (top 3) |
| `recon_keyframe_negdy_score*.png` | 縦、逆方向（スコア付き、上位3件） | Vertical reverse (top 3) |
| `recon_keyframe_dx_score*.png` | 横、マッチングモード | Horizontal, matching mode |
| `recon_keyframe_negdx_score*.png` | 横、逆方向 | Horizontal reverse |

### デバッグ出力 / Debug Output

| ファイル | 説明 | Description |
|---------|------|-------------|
| `debug_positions.csv` | 各フレームの推定情報 | Estimation info per frame |
| `debug_jigsaw.csv` | ジグソーモードのデバッグ情報 | Jigsaw mode debug info |
| `debug_keyframe_quality.csv` | キーフレーム品質情報 | Keyframe quality info |
| `frame_classification.png` | フレーム分類の可視化 | Frame classification visualization |

---

## 推奨ワークフロー / Recommended Workflow

### 縦スクロール動画 / Vertical Scrolling Video

```bash
# Step 1: jigsawモードで実行（キーフレームモードがデフォルト）
python video_strip_reconstruct.py --video input.mp4 --fps 3 \
  --strip-y 0 --strip-h 1080 --jigsaw --out outdir

# Step 2: 出力を確認し、最適な候補を選択
# Check output and select the best candidate

# Step 3: 必要に応じてパラメータを調整
# Adjust parameters if needed
python video_strip_reconstruct.py --video input.mp4 --fps 3 \
  --strip-y 0 --strip-h 1080 --jigsaw \
  --ignore-pct "0,right,12,8" --out outdir
```

### 横スクロール動画 / Horizontal Scrolling Video

```bash
python video_strip_reconstruct.py --video input.mp4 --fps 3 \
  --scroll-axis horizontal --strip-x 0 --strip-w 1920 \
  --jigsaw --out outdir
```

---

## 無視領域 / Ignore Regions

ロゴやウォーターマークなど、常に存在する要素を無視できます。

Ignore persistent elements like logos or watermarks.

### ピクセル指定 / Pixel coordinates

```bash
# 右上のロゴを無視（top=0, right, width=220, height=120）
--ignore "0,right,220,120"

# 左下のロゴを無視（top=900, left, width=150, height=80）
--ignore "900,left,150,80"
```

### パーセント指定 / Percentage

```bash
# 右上12%x8%を無視
--ignore-pct "0,right,12,8"
```

### --allow-ignore-at-edges

キーフレームモードで、端のフレーム（最初/最後の位置）ではignore領域を適用しない：

In keyframe mode, don't apply ignore regions to edge frames (first/last positioned):

```bash
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --jigsaw \
  --ignore-pct "0,right,12,8" --allow-ignore-at-edges --out outdir
```

---

## シナリオ別アプローチ / Scenario-based Approaches

### シナリオA: 背景がスクロール、テキストは固定（出現/消失）

**Scenario A: Background scrolls, text is static (appears/disappears)**

```bash
# jigsawモード + キーフレームモード（推奨）
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --jigsaw --out outdir
```

### シナリオB: 背景固定、テキストがスクロール

**Scenario B: Background is static, text scrolls**

```bash
# static-bgモードで時間的中央値を計算
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --static-bg --static-method median --out outdir
```

### シナリオC: 背景とテキストが一緒にスクロール

**Scenario C: Both background and text scroll together**

```bash
# ignore領域でテキストを除外
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --jigsaw \
  --ignore-pct "30,25,50,40" --out outdir
```

### シナリオD: 横スクロール動画

**Scenario D: Horizontal scrolling video**

```bash
python video_strip_reconstruct.py --frames "frames/*.png" \
  --scroll-axis horizontal --strip-x 0 --strip-w 1920 \
  --jigsaw --out outdir
```

---

## 既知の制限事項 / Known Limitations

### マッチングの問題 / Matching Issues

- **繰り返しパターン**: 花柄装飾等で位相相関が誤検出しやすい
- **Repetitive patterns**: Phase correlation prone to errors with floral decorations, etc.

- **フレーム間の変化が大きい**: オーバーラップが少ないとマッチングスコアが低下
- **Large inter-frame changes**: Low overlap leads to low matching scores

### キーフレームモードの制限 / Keyframe Mode Limitations

- フレーム数が少なすぎると候補が生成されない場合がある
- Too few frames may result in no candidates being generated

- マッチングスコアが非常に低い場合、位置ベースモードにフォールバック
- Falls back to position mode if matching scores are very low

### fpsと精度のトレードオフ / FPS vs Accuracy Trade-off

| fps | メリット / Pros | デメリット / Cons |
|-----|----------------|-------------------|
| 低 (2-4) | dyが大きく追跡しやすい | フレーム少 |
| 高 (10+) | フレーム数が多い | 小さいdyで誤検出増加 |

**推奨: fps=2-4 から開始**

**Recommendation: Start with fps=2-4**

---

## stitch_candidates.py との関係 / Relationship with stitch_candidates.py

`--keyframe-stitch-method matching` では、`stitch_candidates.py` の関数を直接インポートして使用します：

With `--keyframe-stitch-method matching`, functions from `stitch_candidates.py` are directly imported:

- `build_methods()` - マッチング手法の構築
- `match_pair()` - ペアマッチング
- `composite_two()` - 2枚合成
- `compute_boundary_similarity()` - 境界一致度計算

合成は順次方式: (1+2) + 3 + 4 + ...

Sequential stitching: (1+2) + 3 + 4 + ...

---

## トラブルシューティング / Troubleshooting

### マッチングモードで候補が生成されない

```
Warning: No valid candidates from matching mode, falling back to position mode
```

原因と対策：

1. **マッチングスコアが低い**: フレーム間のオーバーラップが不足
   - fpsを下げてフレーム間の移動量を増やす
   - `--keyframe-stitch-method position` を試す

2. **境界一致度が低い**: フレーム内容が大きく異なる
   - `--max-keyframe-gap` を小さくする

### 結果がズレている / Result is misaligned

1. `debug_positions.csv` でdy/dx値を確認
2. `debug_keyframe_quality.csv` で選択されたキーフレームを確認
3. `--keyframe-stitch-method position` を試す
4. 必要に応じて `--uniform-dy` で手動指定

### テキストが残る / Text remains in result

1. `--edge-thr` を上げる（0.4〜0.5）
2. テキストがない瞬間が少ない場合は限界
3. 後処理で画像編集ソフトを使用

---

## ライセンス / License

MIT License
