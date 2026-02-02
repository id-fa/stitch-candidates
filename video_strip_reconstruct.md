# video_strip_reconstruct

動画の縦スクロール（縦パン）から背景を復元するツール

A tool for reconstructing backgrounds from vertical scrolling/panning videos

---

## 概要 / Overview

### 日本語

`video_strip_reconstruct.py` は、縦スクロールする動画から横長の帯（ストリップ）を切り出し、背景を復元するツールです。

- テロップ（字幕・ロゴ等）を**画素単位で除去**
- dy（縦方向移動量）を自動推定して連結
- 複数の候補を生成し、人間が最良のものを選択

### English

`video_strip_reconstruct.py` extracts horizontal strips from vertically scrolling videos and reconstructs the background.

- Removes text overlays (subtitles, logos) at the **pixel level**
- Automatically estimates dy (vertical displacement) for stitching
- Generates multiple candidates for human selection

---

## 設計思想 / Design Philosophy

### 日本語

- テロップはエッジ強度で判別し、背景っぽい画素を優先
- dy推定の符号が逆になるケースがあるため、**dy と -dy の2候補**を出力
- 完璧な復元より**良い候補の生成**を目指す

### English

- Text is identified by edge strength; background-like pixels are prioritized
- Outputs **both dy and -dy candidates** since sign may be inverted depending on content
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
python video_strip_reconstruct.py --video input.mp4 --fps 2 \
  --strip-y 980 --strip-h 100 --out outdir
```

### 抽出済みフレームから / From extracted frames

```bash
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 --out outdir
```

### 複数候補を生成 / Generate multiple candidates

```bash
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 \
  --edge-thr 0.3,0.4,0.5 --out outdir
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
| `--strip-y` | ストリップの上端Y座標 (必須) | Top Y coordinate of strip (required) |
| `--strip-h` | ストリップの高さ (default: 100) | Height of strip |

### マッチング設定 / Matching Settings

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `--dy-search` | dy探索範囲 ±n px (default: 40) | Search range for dy ±n px |
| `--match-method` | マッチング手法 (default: phase) | Matching method(s) |
| `--min-peak` | ピークスコア閾値（診断用） | Peak score threshold for diagnostics |
| `--no-edges` | エッジなしでdy推定 | Use grayscale (not edges) for dy estimation |
| `--dy-region` | dy推定に使う領域 "y,h" | Region for dy estimation "y,h" |
| `--template-region` | テンプレート領域 "y,h,x,w" | Manual template region "y,h,x,w" |
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
| `--scroll-dir` | スクロール方向: up/down/both (default: both) | Scroll direction |
| `--static-bg` | 背景固定モード（時間的中央値で復元） | Static background mode |
| `--static-method` | static-bgの手法: median/min_edge | Method for static-bg mode |

### 出力 / Output

| パラメータ | 説明 | Description |
|-----------|------|-------------|
| `-o, --out` | 出力ディレクトリ (default: recon_out) | Output directory |

---

## マッチング手法 / Matching Methods

| 手法 / Method | 説明 / Description |
|--------------|-------------------|
| `phase` | エッジ画像での位相相関（デフォルト） / Phase correlation on edge map (default) |
| `phase_gray` | グレースケールでの位相相関 / Phase correlation on grayscale |
| `ncc_gray` | グレースケールNCC / NCC on grayscale |
| `ncc_edge` | エッジマップNCC / NCC on edge map |
| `template` | テンプレートマッチング追跡 / Template matching with tracking |

### template メソッドについて / About template method

テンプレートマッチングは、背景がスクロールする動画で最も正確なdy推定ができます。

Template matching provides the most accurate dy estimation for scrolling backgrounds.

- 特徴的な領域を自動選択して追跡 / Automatically selects and tracks distinctive regions
- 外れ値検出と補間で追跡ロスを補正 / Outlier detection and interpolation handle tracking loss
- `--template-region "y,h,x,w"` で手動指定も可能 / Manual specification available

```bash
# 自動テンプレート選択 / Automatic template selection
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --match-method template --out outdir

# 手動テンプレート指定 / Manual template specification
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --match-method template \
  --template-region "350,150,750,150" --out outdir
```

複数指定可能（カンマ区切り）：

Multiple methods can be specified (comma-separated):

```bash
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 \
  --match-method phase,ncc_gray --out outdir
```

---

## 推奨ワークフロー / Recommended Workflow

### Step 1: 荒探索 / Rough search

```bash
python video_strip_reconstruct.py --video input.mp4 --fps 2 \
  --strip-y 980 --strip-h 100 \
  --edge-thr 0.3,0.4,0.5 --out outdir
```

### Step 2: 候補確認 / Check candidates

出力ファイル / Output files:
- `recon_dy_e0.30.png` - dy方向、edge-thr=0.30
- `recon_negdy_e0.30.png` - -dy方向、edge-thr=0.30
- `recon_dy_e0.40.png` - dy方向、edge-thr=0.40
- ...

### Step 3: 最適なedge-thrで再実行 / Re-run with optimal edge-thr

```bash
python video_strip_reconstruct.py --video input.mp4 --fps 2 \
  --strip-y 980 --strip-h 100 \
  --edge-thr 0.35 --out final
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

---

## 診断機能 / Diagnostics

`--min-peak` を指定すると、dy推定が不安定なフレームを検出します。

Use `--min-peak` to detect frames with unreliable dy estimation.

```bash
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 \
  --min-peak 0.1 --out outdir
```

出力例 / Example output:

```
[診断] ピークスコアが低いフレーム (3件, 閾値: 0.1000):
  Frame 5: frame_000005.png <- frame_000004.png
    dy=+12, peak=0.0823
  Frame 12: frame_000012.png <- frame_000011.png
    dy=-3, peak=0.0654

[統計] dy: mean=8.2, std=4.5
  -> dy推定が不安定な可能性があります
```

---

## 出力ファイル / Output Files

| ファイル | 説明 | Description |
|---------|------|-------------|
| `recon_dy.png` | dyをそのまま積算した結果 | Result with dy as-is |
| `recon_negdy.png` | -dyを積算した結果 | Result with negated dy |
| `recon_dy_e0.XX.png` | edge-thr=0.XX での dy結果 | dy result with edge-thr=0.XX |
| `recon_{method}_dy.png` | 特定手法での dy結果 | dy result from specific method |
| `recon_static_median.png` | static-bgモードの結果 | Result from static-bg mode |
| `debug_positions.csv` | 各フレームの推定情報 | Estimation info per frame |

### debug_positions.csv の内容 / Contents

```csv
index,frame,dy,peak,reliable,pos_dy,pos_negdy
0,frame_000001.png,,,,,0,0
1,frame_000002.png,8,0.2134,OK,8,-8
2,frame_000003.png,7,0.1987,OK,15,-15
3,frame_000004.png,9,0.0823,LOW,24,-24
```

---

## シナリオ別アプローチ / Scenario-based Approaches

### シナリオA: 背景がスクロール、テキストは固定（出現/消失）

**Scenario A: Background scrolls, text is static (appears/disappears)**

```bash
# templateメソッドで正確なdy推定
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --match-method template \
  --edge-thr 0.35 --out outdir
```

- テンプレート追跡が最も効果的 / Template tracking is most effective
- 追跡が途切れた区間は自動補間 / Lost tracking is auto-interpolated
- 必要に応じて `--template-region` で手動指定 / Use `--template-region` if needed

### シナリオB: 背景固定、テキストがスクロール

**Scenario B: Background is static, text scrolls**

```bash
# static-bgモードで時間的中央値を計算
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --static-bg --static-method median --out outdir
```

- テキストが一時的に通過する場合に有効 / Effective when text passes temporarily
- `min_edge` は最小エッジ強度の画素を選択 / `min_edge` selects pixels with lowest edge strength

### シナリオC: 背景とテキストが一緒にスクロール

**Scenario C: Both background and text scroll together**

```bash
# 従来の方法（位相相関）が有効
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 980 --strip-h 100 --match-method phase \
  --ignore-pct "30,25,50,40" --out outdir
```

- `--ignore` でテキスト領域を除外 / Use `--ignore` to exclude text regions

### シナリオD: スクロール速度が既知

**Scenario D: Scroll speed is known**

```bash
# 均一なdyを直接指定
python video_strip_reconstruct.py --frames "frames/*.png" \
  --strip-y 0 --strip-h 1080 --uniform-dy -50 \
  --edge-thr 0.35 --out outdir
```

- 最も確実な方法 / Most reliable method
- 事前にフレーム間の移動量を計測しておく / Pre-measure displacement between frames

---

## 適したユースケース / When to Use

### 適している / Good fit

- 縦スクロール動画からの背景抽出
- テロップ付きの動画からのクリーンな背景取得
- 一定速度でパンする映像

### 適さない / Not ideal

- 横スクロール動画（水平パン）
- 不規則な動きのある映像
- 極端に短い動画（フレーム数が少なすぎる）

---

## 既知の制限事項 / Known Limitations

### dy推定の問題 / dy Estimation Issues

- **位相相関の弱点**: 繰り返しパターン（花柄装飾等）で誤検出しやすい
- **Phase correlation weakness**: Prone to errors with repetitive patterns (floral decorations, etc.)

- **テンプレート追跡のロス**: 追跡対象がフレーム外に出ると途切れる
- **Template tracking loss**: Tracking fails when target exits frame

- **累積誤差**: dyの小さな誤差が累積して画質劣化の原因になる
- **Accumulated errors**: Small dy errors accumulate and cause quality degradation

### 改善のヒント / Tips for Improvement

- **fpsを上げる**: フレーム間のdyが小さくなり、追跡が安定
- **Increase fps**: Smaller dy per frame, more stable tracking

- **テンプレート位置を調整**: 特徴的で長く画面に残る領域を選ぶ
- **Adjust template position**: Choose distinctive regions that stay visible longer

- **手動でdy調整**: `debug_positions.csv` を確認して異常値を修正
- **Manual dy adjustment**: Check `debug_positions.csv` and fix outliers

---

## stitch_candidates.py との違い / Difference from stitch_candidates.py

| 項目 | video_strip_reconstruct | stitch_candidates |
|-----|------------------------|-------------------|
| 入力 | 動画 or フレーム連番 | 静止画（スクリーンショット等） |
| 目的 | テロップ除去 + 背景復元 | 画像連結（候補生成） |
| オーバーラップ | 自動推定（dy） | 手動指定（overlap） |
| テキスト処理 | エッジベースで除去 | なし |

---

## ライセンス / License

MIT License

---

## トラブルシューティング / Troubleshooting

### 結果がズレている / Result is misaligned

1. `debug_positions.csv` でdy値を確認
2. 異常値（0や極端に大きな値）がないかチェック
3. `--match-method template` を試す
4. 必要に応じて `--uniform-dy` で手動指定

### テンプレート追跡が失敗する / Template tracking fails

1. `--template-region` で特徴的な領域を手動指定
2. fpsを上げてフレーム数を増やす
3. 追跡対象が画面内に長く留まる領域を選ぶ

### テキストが残る / Text remains in result

1. `--edge-thr` を上げる（0.4〜0.5）
2. テキストがない瞬間が少ない場合は限界
3. 後処理で画像編集ソフトを使用

### 位相相関がdy=0を返す / Phase correlation returns dy=0

1. 背景が本当に動いていない可能性を確認
2. `--match-method template` に切り替え
3. `--ignore` で問題のある領域を除外

---

## まとめ / Summary

> **縦スクロール動画から、テロップを避けて背景を復元する**

> **Reconstruct backgrounds from vertical scrolling videos while avoiding text overlays**

完璧な結果より、良い候補を生成することを目指しています。

Aims to generate good candidates rather than perfect results.

### 検証予定 / TODO

- [ ] fpsを上げた場合の精度向上を検証
- [ ] Verify accuracy improvement with higher fps
