#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trim_dialog.py - 動画のトリム（開始/終了フレーム）、クロップ、無視矩形/テキスト矩形を
画面を見ながら指定するダイアログ（gui.py の Panorama タブから使用）。

- 範囲バーの両端ハンドルをドラッグして大まかに指定し、-1 / +1 ボタンで 1 フレーム単位に微調整
- 開始フレームと終了フレームのプレビューを並べて表示
- プレビュー上でドラッグして矩形を描く。モード: Crop（1 個、辺/角のハンドルで調整可）、
  Ignore rect / Text rect（複数、Remove last / Clear で削除）
- 無視矩形とテキスト矩形は内部では元動画座標で持ち、OK 時にクロップ後の座標へ変換して返す

必要: opencv-python, pillow
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None
from PIL import Image, ImageTk

Rect = Tuple[int, int, int, int]

THUMB_W = 480          # プレビュー幅（px）
MAX_THUMBS = 1500      # サムネイルキャッシュの最大枚数（超える場合は間引き）
ASPECTS = {"free": None, "16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0, "9:16": 9 / 16, "21:9": 21 / 9}


def _norm_rect(x0: float, y0: float, x1: float, y1: float) -> Rect:
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return int(round(xa)), int(round(ya)), int(round(xb - xa)), int(round(yb - ya))


class TrimCropDialog(tk.Toplevel):
    """result: None（キャンセル）または dict(start_frame, end_frame, crop, ignore_rects, text_rects, fps, n_frames)"""

    def __init__(self, master, video_path: str, initial: Optional[Dict] = None):
        super().__init__(master)
        self.title(f"Trim / Crop - {video_path}")
        self.result: Optional[Dict] = None
        self.video_path = video_path
        initial = initial or {}
        if cv2 is None:
            raise RuntimeError("opencv-python が必要です")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap                       # 単発シーク用（メインスレッド）
        self.scale = THUMB_W / self.W
        self.cw, self.ch = THUMB_W, max(1, int(round(self.H * self.scale)))
        self.stride = max(1, -(-self.n // MAX_THUMBS))
        self._thumbs: Dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._decoded = 0
        self._done = False

        self.start = max(0, min(self.n - 1, int(initial.get("start_frame") or 0)))
        end0 = initial.get("end_frame")
        self.end = max(self.start, min(self.n - 1, int(end0) if end0 is not None else self.n - 1))
        self.crop: Optional[Rect] = initial.get("crop")
        # 無視/テキスト矩形は元動画座標で保持（初期値はクロップ後座標なので戻す）
        ox, oy = (self.crop[0], self.crop[1]) if self.crop else (0, 0)
        self.ignore_rects: List[Rect] = [(x + ox, y + oy, w, h) for (x, y, w, h) in initial.get("ignore_rects", [])]
        self.text_rects: List[Rect] = [(x + ox, y + oy, w, h) for (x, y, w, h) in initial.get("text_rects", [])]

        self._build()
        self._start_decode()
        self._refresh_all()
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    # ------------------------------------------------------------ UI
    def _build(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        # プレビュー 2 枚
        pf = ttk.Frame(top)
        pf.pack(fill=tk.X)
        self._canvases: Dict[str, tk.Canvas] = {}
        self._photos: Dict[str, Optional[ImageTk.PhotoImage]] = {"start": None, "end": None}
        for which, title in (("start", "Start frame"), ("end", "End frame")):
            col = ttk.Frame(pf)
            col.pack(side=tk.LEFT, padx=(0, 8))
            ttk.Label(col, text=title).pack(anchor=tk.W)
            c = tk.Canvas(col, width=self.cw, height=self.ch, bg="#202020", highlightthickness=1, cursor="crosshair")
            c.pack()
            c.bind("<ButtonPress-1>", lambda e, w=which: self._on_press(e, w))
            c.bind("<B1-Motion>", lambda e, w=which: self._on_drag(e, w))
            c.bind("<ButtonRelease-1>", lambda e, w=which: self._on_release(e, w))
            c.bind("<Motion>", lambda e, w=which: self._on_move(e, w))
            self._canvases[which] = c
        self.info_var = tk.StringVar()
        ttk.Label(top, textvariable=self.info_var, foreground="#666666").pack(anchor=tk.W, pady=(2, 0))

        # 範囲バー
        self.bar = tk.Canvas(top, height=34, bg="#e8e8e8", highlightthickness=0, cursor="sb_h_double_arrow")
        self.bar.pack(fill=tk.X, pady=(6, 2))
        self.bar.bind("<ButtonPress-1>", self._bar_press)
        self.bar.bind("<B1-Motion>", self._bar_drag)
        self.bar.bind("<Configure>", lambda _e: self._draw_bar())
        self._bar_active: Optional[str] = None

        # 開始/終了の数値と微調整
        rf = ttk.Frame(top)
        rf.pack(fill=tk.X, pady=2)
        self.start_var, self.end_var = tk.StringVar(), tk.StringVar()
        self.start_lab, self.end_lab = tk.StringVar(), tk.StringVar()
        for which, var, lab, title in (("start", self.start_var, self.start_lab, "Start"), ("end", self.end_var, self.end_lab, "End")):
            f = ttk.Frame(rf)
            f.pack(side=tk.LEFT, padx=(0, 24))
            ttk.Label(f, text=f"{title}:").pack(side=tk.LEFT)
            e = ttk.Entry(f, textvariable=var, width=7)
            e.pack(side=tk.LEFT, padx=4)
            e.bind("<Return>", lambda _e, w=which: self._entry_apply(w))
            e.bind("<FocusOut>", lambda _e, w=which: self._entry_apply(w))
            ttk.Button(f, text="-1", width=3, command=lambda w=which: self._step(w, -1)).pack(side=tk.LEFT)
            ttk.Button(f, text="+1", width=3, command=lambda w=which: self._step(w, +1)).pack(side=tk.LEFT, padx=(2, 4))
            ttk.Label(f, textvariable=lab, foreground="#666666").pack(side=tk.LEFT)
        ttk.Button(rf, text="Set start = end - 1", command=lambda: self._set("start", self.end - 1)).pack(side=tk.LEFT)
        self.count_var = tk.StringVar()
        ttk.Label(rf, textvariable=self.count_var).pack(side=tk.LEFT, padx=12)

        # 矩形モード
        mf = ttk.LabelFrame(top, text="Rectangles (drag on preview)")
        mf.pack(fill=tk.X, pady=4)
        r1 = ttk.Frame(mf)
        r1.pack(fill=tk.X, padx=4, pady=2)
        self.mode_var = tk.StringVar(value="crop")
        for v, t in (("crop", "Crop"), ("ignore", "Ignore rect"), ("text", "Text rect")):
            ttk.Radiobutton(r1, text=t, variable=self.mode_var, value=v).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(r1, text="Aspect:").pack(side=tk.LEFT, padx=(12, 2))
        self.aspect_var = tk.StringVar(value="free")
        ttk.Combobox(r1, textvariable=self.aspect_var, values=list(ASPECTS.keys()), state="readonly", width=6).pack(side=tk.LEFT)
        ttk.Button(r1, text="Clear crop", command=self._clear_crop).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(r1, text="Remove last ignore", command=lambda: self._pop("ignore")).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Clear ignore", command=lambda: self._clear("ignore")).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Remove last text", command=lambda: self._pop("text")).pack(side=tk.LEFT, padx=2)
        ttk.Button(r1, text="Clear text", command=lambda: self._clear("text")).pack(side=tk.LEFT, padx=2)
        r2 = ttk.Frame(mf)
        r2.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(r2, text="Crop x,y,w,h:").pack(side=tk.LEFT)
        self.crop_vars = [tk.StringVar() for _ in range(4)]
        for v in self.crop_vars:
            e = ttk.Entry(r2, textvariable=v, width=6)
            e.pack(side=tk.LEFT, padx=2)
            e.bind("<Return>", lambda _e: self._crop_entry_apply())
            e.bind("<FocusOut>", lambda _e: self._crop_entry_apply())
        self.rect_info = tk.StringVar()
        ttk.Label(r2, textvariable=self.rect_info, foreground="#666666").pack(side=tk.LEFT, padx=12)

        bf = ttk.Frame(top)
        bf.pack(fill=tk.X, pady=(6, 0))
        self.progress_var = tk.StringVar(value="")
        ttk.Label(bf, textvariable=self.progress_var, foreground="#666666").pack(side=tk.LEFT)
        ttk.Button(bf, text="Cancel", command=self._cancel).pack(side=tk.RIGHT)
        ttk.Button(bf, text="OK", command=self._ok).pack(side=tk.RIGHT, padx=6)

        self._drag: Optional[Dict] = None

    # ------------------------------------------------------------ デコード
    def _start_decode(self) -> None:
        def worker() -> None:
            cap = cv2.VideoCapture(self.video_path)
            idx = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    th = cv2.resize(bgr, (self.cw, self.ch), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                    with self._lock:
                        self._thumbs[idx] = np.ascontiguousarray(th)
                idx += 1
                self._decoded = idx
            cap.release()
            with self._lock:
                if idx > 0:
                    self.n = idx     # 実際のフレーム数（メタデータより正確）
                self._done = True
        threading.Thread(target=worker, daemon=True).start()
        self.after(200, self._poll_decode)

    def _poll_decode(self) -> None:
        if self._done:
            self.progress_var.set(f"decoded {self.n} frames")
            self.end = min(self.end, self.n - 1)
            self.start = min(self.start, self.end)
            self._refresh_all()
            return
        self.progress_var.set(f"decoding thumbnails... {self._decoded}")
        # 表示中のフレームがデコード済みになったら描き直す
        self._refresh_previews()
        self.after(300, self._poll_decode)

    def _get_thumb(self, idx: int) -> Optional[np.ndarray]:
        with self._lock:
            th = self._thumbs.get(idx)
        if th is not None:
            return th
        if not self._done and idx > self._decoded:
            return None
        # キャッシュに無い（間引き対象 or 未デコード）フレームはシークして取得
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self._cap.read()
        if not ok:
            return None
        th = np.ascontiguousarray(cv2.resize(bgr, (self.cw, self.ch), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        with self._lock:
            self._thumbs[idx] = th
        return th

    # ------------------------------------------------------------ 描画
    def _refresh_all(self) -> None:
        self.start_var.set(str(self.start))
        self.end_var.set(str(self.end))
        self.start_lab.set(f"{self.start / self.fps:.3f} s")
        self.end_lab.set(f"{self.end / self.fps:.3f} s")
        self.count_var.set(f"{self.end - self.start + 1} frames / {self.n} total, {self.fps:.3f} fps, {self.W}x{self.H}")
        if self.crop:
            for v, val in zip(self.crop_vars, self.crop):
                v.set(str(val))
        else:
            for v in self.crop_vars:
                v.set("")
        self.rect_info.set(f"ignore: {len(self.ignore_rects)}  text: {len(self.text_rects)}")
        self._draw_bar()
        self._refresh_previews()

    def _refresh_previews(self) -> None:
        for which, idx in (("start", self.start), ("end", self.end)):
            c = self._canvases[which]
            th = self._get_thumb(idx)
            c.delete("all")
            if th is not None:
                self._photos[which] = ImageTk.PhotoImage(Image.fromarray(th))
                c.create_image(0, 0, image=self._photos[which], anchor=tk.NW)
            else:
                c.create_text(self.cw // 2, self.ch // 2, text=f"frame {idx}\n(decoding...)", fill="#aaaaaa", justify=tk.CENTER)
            c.create_text(4, 4, text=f"#{idx}", fill="#ffff00", anchor=tk.NW, font=("TkDefaultFont", 9, "bold"))
            self._draw_rects(c)

    def _draw_rects(self, c: tk.Canvas) -> None:
        s = self.scale
        for (x, y, w, h) in self.ignore_rects:
            c.create_rectangle(x * s, y * s, (x + w) * s, (y + h) * s, outline="#ff4040", width=2)
        for (x, y, w, h) in self.text_rects:
            c.create_rectangle(x * s, y * s, (x + w) * s, (y + h) * s, outline="#40d0ff", width=2)
        if self.crop:
            x, y, w, h = self.crop
            # クロップ外を暗くする
            for (a, b, cx, d) in ((0, 0, self.cw, y * s), (0, (y + h) * s, self.cw, self.ch),
                                  (0, y * s, x * s, (y + h) * s), ((x + w) * s, y * s, self.cw, (y + h) * s)):
                if cx > a and d > b:
                    c.create_rectangle(a, b, cx, d, fill="#000000", stipple="gray50", outline="")
            c.create_rectangle(x * s, y * s, (x + w) * s, (y + h) * s, outline="#ffe040", width=2)
            for hx, hy in self._crop_handles():
                c.create_rectangle(hx - 4, hy - 4, hx + 4, hy + 4, fill="#ffe040", outline="")
        if self._drag and self._drag.get("preview"):
            x, y, w, h = self._drag["preview"]
            col = {"crop": "#ffe040", "ignore": "#ff4040", "text": "#40d0ff"}[self._drag["mode"]]
            c.create_rectangle(x * s, y * s, (x + w) * s, (y + h) * s, outline=col, width=2, dash=(4, 2))

    def _crop_handles(self) -> List[Tuple[float, float]]:
        if not self.crop:
            return []
        x, y, w, h = self.crop
        s = self.scale
        xs = (x * s, (x + w / 2) * s, (x + w) * s)
        ys = (y * s, (y + h / 2) * s, (y + h) * s)
        return [(hx, hy) for hy in ys for hx in xs if not (hx == xs[1] and hy == ys[1])]

    def _draw_bar(self) -> None:
        b = self.bar
        b.delete("all")
        w = b.winfo_width()
        if w < 20:
            return
        pad = 8
        bw = w - 2 * pad
        b.create_rectangle(pad, 14, pad + bw, 20, fill="#b0b0b0", outline="")
        xs = pad + bw * self.start / max(1, self.n - 1)
        xe = pad + bw * self.end / max(1, self.n - 1)
        b.create_rectangle(xs, 14, xe, 20, fill="#4a90e2", outline="")
        for x, col in ((xs, "#2060c0"), (xe, "#2060c0")):
            b.create_polygon(x - 7, 4, x + 7, 4, x, 14, fill=col, outline="")
            b.create_rectangle(x - 2, 4, x + 2, 30, fill=col, outline="")
        b.create_text(pad, 32, text="0", anchor=tk.SW, font=("TkDefaultFont", 8), fill="#666666")
        b.create_text(pad + bw, 32, text=str(self.n - 1), anchor=tk.SE, font=("TkDefaultFont", 8), fill="#666666")

    # ------------------------------------------------------------ 範囲バー操作
    def _bar_frame(self, x: float) -> int:
        w = self.bar.winfo_width()
        pad = 8
        t = (x - pad) / max(1, w - 2 * pad)
        return int(round(max(0.0, min(1.0, t)) * (self.n - 1)))

    def _bar_press(self, e) -> None:
        f = self._bar_frame(e.x)
        self._bar_active = "start" if abs(f - self.start) <= abs(f - self.end) else "end"
        self._set(self._bar_active, f)

    def _bar_drag(self, e) -> None:
        if self._bar_active:
            self._set(self._bar_active, self._bar_frame(e.x))

    def _set(self, which: str, f: int) -> None:
        f = max(0, min(self.n - 1, int(f)))
        if which == "start":
            self.start = f
            if self.end < f:
                self.end = f
        else:
            self.end = f
            if self.start > f:
                self.start = f
        self._refresh_all()

    def _step(self, which: str, d: int) -> None:
        self._set(which, (self.start if which == "start" else self.end) + d)

    def _entry_apply(self, which: str) -> None:
        var = self.start_var if which == "start" else self.end_var
        try:
            self._set(which, int(float(var.get())))
        except ValueError:
            self._refresh_all()

    # ------------------------------------------------------------ 矩形操作（プレビュー上）
    def _to_frame(self, e) -> Tuple[float, float]:
        return max(0.0, min(self.W, e.x / self.scale)), max(0.0, min(self.H, e.y / self.scale))

    def _hit_crop_handle(self, e) -> Optional[int]:
        for i, (hx, hy) in enumerate(self._crop_handles()):
            if abs(e.x - hx) <= 6 and abs(e.y - hy) <= 6:
                return i
        return None

    def _on_move(self, e, which: str) -> None:
        c = self._canvases[which]
        inside = False
        if self.crop:
            fx, fy = self._to_frame(e)
            x, y, w, h = self.crop
            inside = x <= fx <= x + w and y <= fy <= y + h
        if self.mode_var.get() == "crop" and (self._hit_crop_handle(e) is not None or inside):
            c.configure(cursor="fleur")
        else:
            c.configure(cursor="crosshair")

    def _on_press(self, e, which: str) -> None:
        mode = self.mode_var.get()
        fx, fy = self._to_frame(e)
        if mode == "crop" and self.crop:
            hi = self._hit_crop_handle(e)
            x, y, w, h = self.crop
            if hi is not None:
                # ハンドル: 0..7 = 左上, 上, 右上, 左, 右, 左下, 下, 右下
                self._drag = {"mode": "crop", "handle": hi, "orig": (x, y, x + w, y + h), "preview": None}
                return
            if x <= fx <= x + w and y <= fy <= y + h:
                # 内側をドラッグ: 移動
                self._drag = {"mode": "crop", "move": (fx, fy), "orig": (x, y, x + w, y + h), "preview": None}
                return
        self._drag = {"mode": mode, "x0": fx, "y0": fy, "preview": None}

    def _apply_aspect(self, x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float, float, float]:
        ar = ASPECTS.get(self.aspect_var.get())
        if ar is None:
            return x0, y0, x1, y1
        w = abs(x1 - x0)
        h = w / ar
        y1 = y0 + h if y1 >= y0 else y0 - h
        return x0, y0, x1, y1

    def _on_drag(self, e, which: str) -> None:
        if not self._drag:
            return
        fx, fy = self._to_frame(e)
        d = self._drag
        if d["mode"] == "crop" and "move" in d:
            x0, y0, x1, y1 = d["orig"]
            dx, dy = fx - d["move"][0], fy - d["move"][1]
            dx = max(-x0, min(self.W - x1, dx))
            dy = max(-y0, min(self.H - y1, dy))
            d["preview"] = _norm_rect(x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        elif d["mode"] == "crop" and "handle" in d:
            x0, y0, x1, y1 = d["orig"]
            hi = d["handle"]
            if hi in (0, 3, 5):
                x0 = fx
            if hi in (2, 4, 7):
                x1 = fx
            if hi in (0, 1, 2):
                y0 = fy
            if hi in (5, 6, 7):
                y1 = fy
            ar = ASPECTS.get(self.aspect_var.get())
            if ar is not None:
                y1 = y0 + abs(x1 - x0) / ar
            d["preview"] = _norm_rect(x0, y0, x1, y1)
        else:
            x0, y0, x1, y1 = d["x0"], d["y0"], fx, fy
            if d["mode"] == "crop":
                x0, y0, x1, y1 = self._apply_aspect(x0, y0, x1, y1)
            d["preview"] = _norm_rect(x0, y0, x1, y1)
        self._refresh_previews()

    def _on_release(self, e, which: str) -> None:
        if not self._drag:
            return
        d = self._drag
        self._drag = None
        r = d.get("preview")
        if r is None or r[2] < 4 or r[3] < 4:
            self._refresh_previews()
            return
        x, y, w, h = r
        x, y = max(0, x), max(0, y)
        w, h = min(self.W - x, w), min(self.H - y, h)
        r = (x, y, w, h)
        if d["mode"] == "crop":
            self.crop = r
        elif d["mode"] == "ignore":
            self.ignore_rects.append(r)
        else:
            self.text_rects.append(r)
        self._refresh_all()

    def _crop_entry_apply(self) -> None:
        try:
            vals = [int(float(v.get())) for v in self.crop_vars]
        except ValueError:
            self._refresh_all()
            return
        x, y, w, h = vals
        if w > 0 and h > 0:
            self.crop = (max(0, x), max(0, y), min(self.W - max(0, x), w), min(self.H - max(0, y), h))
        self._refresh_all()

    def _clear_crop(self) -> None:
        self.crop = None
        self._refresh_all()

    def _pop(self, kind: str) -> None:
        lst = self.ignore_rects if kind == "ignore" else self.text_rects
        if lst:
            lst.pop()
        self._refresh_all()

    def _clear(self, kind: str) -> None:
        (self.ignore_rects if kind == "ignore" else self.text_rects).clear()
        self._refresh_all()

    # ------------------------------------------------------------ 終了
    def _to_cropped(self, rects: List[Rect]) -> List[Rect]:
        if not self.crop:
            return list(rects)
        cx, cy, cw, ch = self.crop
        out: List[Rect] = []
        for (x, y, w, h) in rects:
            x0, y0 = max(x, cx), max(y, cy)
            x1, y1 = min(x + w, cx + cw), min(y + h, cy + ch)
            if x1 - x0 > 0 and y1 - y0 > 0:
                out.append((x0 - cx, y0 - cy, x1 - x0, y1 - y0))
        return out

    def _ok(self) -> None:
        self.result = {
            "start_frame": self.start, "end_frame": self.end, "crop": self.crop,
            "ignore_rects": self._to_cropped(self.ignore_rects), "text_rects": self._to_cropped(self.text_rects),
            "fps": self.fps, "n_frames": self.n,
        }
        self._close()

    def _cancel(self) -> None:
        self.result = None
        self._close()

    def _close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
        self.grab_release()
        self.destroy()


def ask_trim_crop(master, video_path: str, initial: Optional[Dict] = None) -> Optional[Dict]:
    dlg = TrimCropDialog(master, video_path, initial)
    master.wait_window(dlg)
    return dlg.result


if __name__ == "__main__":  # 単体テスト用: python trim_dialog.py video.mp4
    import sys
    root = tk.Tk()
    root.withdraw()
    print(ask_trim_crop(root, sys.argv[1]))
