#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI for stitch_candidates.py and video_strip_reconstruct.py

必須: pip install pillow numpy
推奨: pip install opencv-python
"""

from __future__ import annotations

import io
import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Optional, Callable

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SCRIPT_DIR = Path(__file__).resolve().parent


def _initial_dir(path_str: str) -> Optional[str]:
    """Return existing directory from a path string for use as initialdir, or None."""
    if not path_str:
        return None
    p = Path(path_str)
    # If it's a glob like "frames/*.png", get the parent
    if "*" in path_str or "?" in path_str:
        p = p.parent
    if p.is_dir():
        return str(p)
    if p.parent.is_dir():
        return str(p.parent)
    return None


# ============================================================
# stdout capture → log panel bridge
# ============================================================

class _StdoutRedirector(io.TextIOBase):
    """Captures writes to sys.stdout and routes them to a callback on the main thread."""

    def __init__(self, callback: Callable[[str], None], root: tk.Tk):
        self._cb = callback
        self._root = root

    def write(self, s: str) -> int:
        if s:
            self._root.after(0, self._cb, s)
        return len(s)

    def flush(self):
        pass


def run_with_capture(func: Callable, argv: List[str],
                     callback: Callable[[str], None],
                     on_done: Callable[[Optional[str]], None],
                     root: tk.Tk) -> threading.Thread:
    """Run *func(argv)* in a background thread, capturing stdout/stderr to *callback*.
    Calls *on_done(error_or_none)* on the main thread when finished."""

    def _worker():
        redirector = _StdoutRedirector(callback, root)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = redirector  # type: ignore[assignment]
        sys.stderr = redirector  # type: ignore[assignment]
        error: Optional[str] = None
        try:
            func(argv)
        except SystemExit as e:
            if e.code and str(e.code) != "0":
                error = str(e.code)
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            root.after(0, on_done, error)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# ============================================================
# Helper widgets
# ============================================================

class FileListWidget(ttk.Frame):
    """Listbox with add/remove/reorder for image files."""

    def __init__(self, master, on_select: Optional[Callable[[str], None]] = None, **kw):
        super().__init__(master, **kw)
        self._on_select_cb = on_select
        self._build()

    def _build(self):
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btn_frame, text="Add Files...", command=self._add).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Add Folder...", command=self._add_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove", command=self._remove).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear", command=self._clear).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Up", command=self._up).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Down", command=self._down).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reverse", command=self._reverse).pack(side=tk.LEFT, padx=2)

        list_frame = ttk.Frame(self)
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))
        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=6)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    def _add(self):
        # Start from directory of last item in list
        last = self.listbox.get(tk.END) if self.listbox.size() > 0 else ""
        files = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("All", "*.*")],
            initialdir=_initial_dir(last)
        )
        for f in files:
            self.listbox.insert(tk.END, f)

    def _add_folder(self):
        last = self.listbox.get(tk.END) if self.listbox.size() > 0 else ""
        d = filedialog.askdirectory(initialdir=_initial_dir(last))
        if not d:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in exts:
                self.listbox.insert(tk.END, str(p))

    def _remove(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)

    def _clear(self):
        self.listbox.delete(0, tk.END)

    def _up(self):
        sel = self.listbox.curselection()
        if not sel or sel[0] == 0:
            return
        for i in sel:
            txt = self.listbox.get(i)
            self.listbox.delete(i)
            self.listbox.insert(i - 1, txt)
            self.listbox.selection_set(i - 1)

    def _down(self):
        sel = self.listbox.curselection()
        if not sel or sel[-1] == self.listbox.size() - 1:
            return
        for i in reversed(sel):
            txt = self.listbox.get(i)
            self.listbox.delete(i)
            self.listbox.insert(i + 1, txt)
            self.listbox.selection_set(i + 1)

    def _reverse(self):
        items = list(self.listbox.get(0, tk.END))
        self.listbox.delete(0, tk.END)
        for item in reversed(items):
            self.listbox.insert(tk.END, item)

    def _on_listbox_select(self, _event):
        if self._on_select_cb:
            sel = self.listbox.curselection()
            if sel:
                self._on_select_cb(self.listbox.get(sel[0]))

    def get_files(self) -> List[str]:
        return list(self.listbox.get(0, tk.END))


class LabeledEntry(ttk.Frame):
    """Label + Entry in one row."""

    def __init__(self, master, label: str, default: str = "", width: int = 20, tooltip: str = "", **kw):
        super().__init__(master, **kw)
        self.label = ttk.Label(self, text=label, width=22, anchor=tk.W)
        self.label.pack(side=tk.LEFT)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(side=tk.LEFT, padx=(4, 0), fill=tk.X, expand=True)
        if tooltip:
            self._tooltip_text = tooltip
            self.entry.bind("<Enter>", self._show_tip)
            self.entry.bind("<Leave>", self._hide_tip)
            self._tip_win: Optional[tk.Toplevel] = None

    def get(self) -> str:
        return self.var.get().strip()

    def _show_tip(self, event):
        if self._tip_win:
            return
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 25
        self._tip_win = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=self._tooltip_text, background="#ffffe0",
                       relief=tk.SOLID, borderwidth=1, font=("TkDefaultFont", 9))
        lbl.pack()

    def _hide_tip(self, _event):
        if self._tip_win:
            self._tip_win.destroy()
            self._tip_win = None


class ImagePreview(ttk.Frame):
    """Shows a preview of an image file, scaled to fit."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._photo = None
        self._canvas.bind("<Configure>", self._on_resize)
        self._current_path: Optional[str] = None

    def show(self, path: str):
        if not HAS_PIL:
            return
        self._current_path = path
        self._render()

    def clear(self):
        self._current_path = None
        self._canvas.delete("all")

    def _on_resize(self, _event):
        if self._current_path:
            self._render()

    def _render(self):
        if not self._current_path or not os.path.isfile(self._current_path):
            return
        try:
            img = Image.open(self._current_path)
        except Exception:
            return
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        img.thumbnail((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER)


# ============================================================
# Output log panel (shared)
# ============================================================

class LogPanel(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.text = tk.Text(self, wrap=tk.WORD, height=12, state=tk.DISABLED,
                            bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        sb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def append(self, msg: str):
        self.text.configure(state=tk.NORMAL)
        self.text.insert(tk.END, msg)
        self.text.see(tk.END)
        self.text.configure(state=tk.DISABLED)

    def clear(self):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.configure(state=tk.DISABLED)


# ============================================================
# Stitch Candidates Tab
# ============================================================

class StitchTab(ttk.Frame):
    def __init__(self, master, log: LogPanel, preview: ImagePreview, app: "App", **kw):
        super().__init__(master, **kw)
        self.log = log
        self.preview = preview
        self.app = app
        self._running = False
        self._build()

    def _build(self):
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Left: parameters ---
        left = ttk.Frame(pw)
        pw.add(left, weight=1)

        canvas = tk.Canvas(left, highlightthickness=0)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        self._param_frame = ttk.Frame(canvas)
        self._param_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._param_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        pf = self._param_frame

        # Run button (top)
        btn_frame = ttk.Frame(pf)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 8))
        self.run_btn = ttk.Button(btn_frame, text="Run Stitch", command=self._run)
        self.run_btn.pack(side=tk.LEFT)

        # Mode
        f = ttk.LabelFrame(pf, text="Mode")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.mode_var = tk.StringVar(value="v")
        for val, txt in [("v", "Vertical"), ("h", "Horizontal"), ("snake", "Snake/Zigzag")]:
            ttk.Radiobutton(f, text=txt, variable=self.mode_var, value=val).pack(side=tk.LEFT, padx=6)
        self.cols_entry = LabeledEntry(f, "Columns (snake):", "4", width=5)
        self.cols_entry.pack(side=tk.LEFT, padx=6)

        # Output
        f = ttk.LabelFrame(pf, text="Output")
        f.pack(fill=tk.X, padx=4, pady=2)
        of = ttk.Frame(f)
        of.pack(fill=tk.X)
        self.out_var = tk.StringVar(value="out_candidates")
        ttk.Label(of, text="Output dir:").pack(side=tk.LEFT)
        ttk.Entry(of, textvariable=self.out_var, width=30).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(of, text="Browse...", command=self._browse_out).pack(side=tk.LEFT)

        # Overlap
        f = ttk.LabelFrame(pf, text="Overlap")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.overlap_entry = LabeledEntry(f, "Overlap (px):", "80,120", tooltip="Comma-separated pixel values")
        self.overlap_entry.pack(fill=tk.X, padx=4)
        self.overlap_pct_entry = LabeledEntry(f, "Overlap (%):", "", tooltip="Ratio 0.01-0.95, comma-separated")
        self.overlap_pct_entry.pack(fill=tk.X, padx=4)

        # Overlap scan
        f2 = ttk.LabelFrame(pf, text="Overlap Scan")
        f2.pack(fill=tk.X, padx=4, pady=2)
        self.overlap_scan_entry = LabeledEntry(f2, "Scan (min,max,step):", "", tooltip="e.g. 50,150,5")
        self.overlap_scan_entry.pack(fill=tk.X, padx=4)
        self.overlap_auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f2, text="Auto scan (3-stage)", variable=self.overlap_auto_var).pack(anchor=tk.W, padx=4)
        self.topn_entry = LabeledEntry(f2, "Top N:", "6", width=5)
        self.topn_entry.pack(fill=tk.X, padx=4)

        # Matching
        f = ttk.LabelFrame(pf, text="Matching")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.band_entry = LabeledEntry(f, "Band (px):", "20,30,50", tooltip="Comma-separated")
        self.band_entry.pack(fill=tk.X, padx=4)
        self.search_entry = LabeledEntry(f, "Search range (px):", "5,10,20", tooltip="Comma-separated")
        self.search_entry.pack(fill=tk.X, padx=4)
        self.exclude_entry = LabeledEntry(f, "Exclude methods:", "", tooltip="e.g. ssim")
        self.exclude_entry.pack(fill=tk.X, padx=4)

        # Thresholds
        f = ttk.LabelFrame(pf, text="Thresholds")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.min_overlap_entry = LabeledEntry(f, "Min overlap ratio:", "0.3")
        self.min_overlap_entry.pack(fill=tk.X, padx=4)
        self.min_boundary_entry = LabeledEntry(f, "Min boundary score:", "0.3")
        self.min_boundary_entry.pack(fill=tk.X, padx=4)

        # Ignore
        f = ttk.LabelFrame(pf, text="Ignore Regions")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.ignore_entry = LabeledEntry(f, "Ignore (px):", "", tooltip="top,right|left|X,w,h  (multiple: semicolon)")
        self.ignore_entry.pack(fill=tk.X, padx=4)
        self.ignore_pct_entry = LabeledEntry(f, "Ignore (%):", "", tooltip="top,right|left|X,w,h  (multiple: semicolon)")
        self.ignore_pct_entry.pack(fill=tk.X, padx=4)

        # Refine
        f = ttk.LabelFrame(pf, text="Refine Mode")
        f.pack(fill=tk.X, padx=4, pady=2)
        rf = ttk.Frame(f)
        rf.pack(fill=tk.X, padx=4)
        self.refine_var = tk.StringVar(value="")
        ttk.Label(rf, text="Refine from:").pack(side=tk.LEFT)
        ttk.Entry(rf, textvariable=self.refine_var, width=30).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(rf, text="Browse...", command=self._browse_refine).pack(side=tk.LEFT)
        self.refine_delta_entry = LabeledEntry(f, "Refine delta:", "2", width=5)
        self.refine_delta_entry.pack(fill=tk.X, padx=4)

        # --- Right: file list ---
        right = ttk.Frame(pw)
        pw.add(right, weight=1)
        ttk.Label(right, text="Input Images (ordered):").pack(anchor=tk.W)
        self.file_list = FileListWidget(right, on_select=self._on_image_select)
        self.file_list.pack(fill=tk.BOTH, expand=True)

    def _on_image_select(self, path: str):
        self.preview.show(path)

    def _browse_out(self):
        d = filedialog.askdirectory(initialdir=_initial_dir(self.out_var.get()))
        if d:
            self.out_var.set(d)

    def _browse_refine(self):
        f = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.bmp"), ("All", "*.*")],
            initialdir=_initial_dir(self.refine_var.get()) or _initial_dir(self.out_var.get())
        )
        if f:
            self.refine_var.set(f)

    def _build_argv(self) -> List[str]:
        """Build argv list for stitch_candidates.main(argv)."""
        argv: List[str] = []
        argv += ["-m", self.mode_var.get()]
        argv += ["-o", self.out_var.get()]

        if self.mode_var.get() == "snake" and self.cols_entry.get():
            argv += ["--cols", self.cols_entry.get()]

        if self.overlap_entry.get():
            argv += ["--overlap", self.overlap_entry.get()]
        if self.overlap_pct_entry.get():
            argv += ["--overlap-pct", self.overlap_pct_entry.get()]
        if self.overlap_scan_entry.get():
            argv += ["--overlap-scan", self.overlap_scan_entry.get()]
        if self.overlap_auto_var.get():
            argv += ["--overlap-auto"]
        if self.topn_entry.get():
            argv += ["--top-n", self.topn_entry.get()]

        if self.band_entry.get():
            argv += ["--band", self.band_entry.get()]
        if self.search_entry.get():
            argv += ["--search", self.search_entry.get()]
        if self.exclude_entry.get():
            argv += ["--exclude-method", self.exclude_entry.get()]

        if self.min_overlap_entry.get():
            argv += ["--min-overlap-ratio", self.min_overlap_entry.get()]
        if self.min_boundary_entry.get():
            argv += ["--min-boundary-score", self.min_boundary_entry.get()]

        for region in self.ignore_entry.get().split(";"):
            region = region.strip()
            if region:
                argv += ["--ignore", region]
        for region in self.ignore_pct_entry.get().split(";"):
            region = region.strip()
            if region:
                argv += ["--ignore-pct", region]

        if self.refine_var.get():
            argv += ["--refine-from", self.refine_var.get()]
        if self.refine_delta_entry.get():
            argv += ["--refine-delta", self.refine_delta_entry.get()]

        argv += self.file_list.get_files()
        return argv

    def _run(self):
        if self._running:
            return
        files = self.file_list.get_files()
        if not files:
            messagebox.showwarning("Warning", "Input images not selected.")
            return

        argv = self._build_argv()
        self.log.clear()
        self.log.append(f"[stitch_candidates] argv = {argv}\n\n")
        self._running = True
        self.run_btn.configure(state=tk.DISABLED)

        from stitch_candidates import main as sc_main
        run_with_capture(sc_main, argv, self.log.append, self._on_done, self.app)

    def _on_done(self, error: Optional[str]):
        self._running = False
        self.run_btn.configure(state=tk.NORMAL)
        if error:
            self.log.append(f"\n--- Error ---\n{error}\n")
        else:
            self.log.append("\n--- Finished ---\n")
            self._load_output_browser()

    def _load_output_browser(self):
        out_dir = Path(self.out_var.get())
        if not out_dir.is_absolute():
            out_dir = SCRIPT_DIR / out_dir
        if out_dir.exists():
            self.app.output_tab.load_dir(str(out_dir))
            self.app.notebook.select(self.app.output_tab)


# ============================================================
# Video Reconstruct Tab
# ============================================================

class VideoTab(ttk.Frame):
    def __init__(self, master, log: LogPanel, preview: ImagePreview, app: "App", **kw):
        super().__init__(master, **kw)
        self.log = log
        self.preview = preview
        self.app = app
        self._running = False
        self._build()

    def _build(self):
        canvas = tk.Canvas(self, highlightthickness=0)
        vsb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        self._param_frame = ttk.Frame(canvas)
        self._param_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._param_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        pf = self._param_frame

        # Run button (top)
        btn_frame = ttk.Frame(pf)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 8))
        self.run_btn = ttk.Button(btn_frame, text="Run Reconstruct", command=self._run)
        self.run_btn.pack(side=tk.LEFT)

        # --- Input ---
        f = ttk.LabelFrame(pf, text="Input")
        f.pack(fill=tk.X, padx=4, pady=2)

        vf = ttk.Frame(f)
        vf.pack(fill=tk.X, padx=4, pady=2)
        self.video_var = tk.StringVar()
        ttk.Label(vf, text="Video:").pack(side=tk.LEFT)
        ttk.Entry(vf, textvariable=self.video_var, width=40).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(vf, text="Browse...", command=self._browse_video).pack(side=tk.LEFT)

        ff = ttk.Frame(f)
        ff.pack(fill=tk.X, padx=4, pady=2)
        self.frames_var = tk.StringVar()
        ttk.Label(ff, text="Frames glob:").pack(side=tk.LEFT)
        ttk.Entry(ff, textvariable=self.frames_var, width=40).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(ff, text="Browse folder...", command=self._browse_frames_dir).pack(side=tk.LEFT)

        ef = ttk.Frame(f)
        ef.pack(fill=tk.X, padx=4, pady=2)
        self.fps_entry = LabeledEntry(ef, "FPS:", "2.0", width=8)
        self.fps_entry.pack(side=tk.LEFT)
        self.start_entry = LabeledEntry(ef, "Start (s):", "", width=8)
        self.start_entry.pack(side=tk.LEFT, padx=(12, 0))
        self.dur_entry = LabeledEntry(ef, "Duration (s):", "", width=8)
        self.dur_entry.pack(side=tk.LEFT, padx=(12, 0))
        self.deinterlace_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ef, text="Deinterlace", variable=self.deinterlace_var).pack(side=tk.LEFT, padx=(12, 0))

        # --- Output ---
        f = ttk.LabelFrame(pf, text="Output")
        f.pack(fill=tk.X, padx=4, pady=2)
        of = ttk.Frame(f)
        of.pack(fill=tk.X, padx=4)
        self.out_var = tk.StringVar(value="recon_out")
        ttk.Label(of, text="Output dir:").pack(side=tk.LEFT)
        ttk.Entry(of, textvariable=self.out_var, width=30).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(of, text="Browse...", command=self._browse_out).pack(side=tk.LEFT)

        # --- Scroll axis & Strip ---
        f = ttk.LabelFrame(pf, text="Strip / Scroll")
        f.pack(fill=tk.X, padx=4, pady=2)

        sf = ttk.Frame(f)
        sf.pack(fill=tk.X, padx=4)
        ttk.Label(sf, text="Scroll axis:").pack(side=tk.LEFT)
        self.scroll_axis_var = tk.StringVar(value="vertical")
        ttk.Radiobutton(sf, text="Vertical", variable=self.scroll_axis_var, value="vertical").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(sf, text="Horizontal", variable=self.scroll_axis_var, value="horizontal").pack(side=tk.LEFT, padx=6)

        ttk.Label(sf, text="  Direction:").pack(side=tk.LEFT)
        self.scroll_dir_var = tk.StringVar(value="both")
        for v in ["both", "up", "down", "left", "right"]:
            ttk.Radiobutton(sf, text=v, variable=self.scroll_dir_var, value=v).pack(side=tk.LEFT, padx=3)

        s2 = ttk.Frame(f)
        s2.pack(fill=tk.X, padx=4, pady=2)
        self.strip_y_entry = LabeledEntry(s2, "Strip Y (px):", "0", width=8)
        self.strip_y_entry.pack(side=tk.LEFT)
        self.strip_h_entry = LabeledEntry(s2, "Strip H (px):", "100", width=8)
        self.strip_h_entry.pack(side=tk.LEFT, padx=(12, 0))
        self.strip_x_entry = LabeledEntry(s2, "Strip X (px):", "", width=8)
        self.strip_x_entry.pack(side=tk.LEFT, padx=(12, 0))
        self.strip_w_entry = LabeledEntry(s2, "Strip W (px):", "", width=8)
        self.strip_w_entry.pack(side=tk.LEFT, padx=(12, 0))

        # --- Matching ---
        f = ttk.LabelFrame(pf, text="Matching")
        f.pack(fill=tk.X, padx=4, pady=2)
        mf = ttk.Frame(f)
        mf.pack(fill=tk.X, padx=4)
        ttk.Label(mf, text="Method:", width=22, anchor=tk.W).pack(side=tk.LEFT)
        _MATCH_METHODS = ["phase", "phase_gray", "ncc_gray", "ncc_edge", "template", "robust",
                          "optical_flow", "optical_flow_fb", "optical_flow_lk"]
        self.match_method_vars: dict[str, tk.BooleanVar] = {}
        for m in _MATCH_METHODS:
            var = tk.BooleanVar(value=(m == "phase"))
            self.match_method_vars[m] = var
            ttk.Checkbutton(mf, text=m, variable=var).pack(side=tk.LEFT, padx=3)
        self.edge_thr_entry = LabeledEntry(f, "Edge threshold:", "0.35", tooltip="Comma-separated")
        self.edge_thr_entry.pack(fill=tk.X, padx=4)
        self.min_peak_entry = LabeledEntry(f, "Min peak:", "0.0")
        self.min_peak_entry.pack(fill=tk.X, padx=4)
        self.dy_search_entry = LabeledEntry(f, "dy search range:", "40")
        self.dy_search_entry.pack(fill=tk.X, padx=4)
        self.dy_region_entry = LabeledEntry(f, "dy region (y,h):", "", tooltip="e.g. 0,200")
        self.dy_region_entry.pack(fill=tk.X, padx=4)
        self.uniform_dy_entry = LabeledEntry(f, "Uniform dy:", "", tooltip="e.g. -50")
        self.uniform_dy_entry.pack(fill=tk.X, padx=4)
        self.no_edges_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="No edges (use raw grayscale)", variable=self.no_edges_var).pack(anchor=tk.W, padx=4)

        # --- Template ---
        f = ttk.LabelFrame(pf, text="Template Matching")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.template_region_entry = LabeledEntry(f, "Template region (y,h,x,w):", "")
        self.template_region_entry.pack(fill=tk.X, padx=4)
        self.suggest_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Suggest templates", variable=self.suggest_var).pack(anchor=tk.W, padx=4)
        self.suggest_n_entry = LabeledEntry(f, "Suggest N:", "15", width=5)
        self.suggest_n_entry.pack(fill=tk.X, padx=4)

        # --- Jigsaw ---
        f = ttk.LabelFrame(pf, text="Jigsaw Mode")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.jigsaw_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Enable jigsaw", variable=self.jigsaw_var).pack(anchor=tk.W, padx=4)
        self.base_frame_entry = LabeledEntry(f, "Base frame:", "", tooltip="-1=last, -2=second-to-last")
        self.base_frame_entry.pack(fill=tk.X, padx=4)
        self.auto_exclude_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Auto-exclude text from dy estimation", variable=self.auto_exclude_var).pack(anchor=tk.W, padx=4)

        # --- Keyframe ---
        f = ttk.LabelFrame(pf, text="Keyframe Mode")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.keyframe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Enable keyframe mode", variable=self.keyframe_var).pack(anchor=tk.W, padx=4)
        self.min_quality_entry = LabeledEntry(f, "Min quality:", "0.3")
        self.min_quality_entry.pack(fill=tk.X, padx=4)
        self.max_gap_entry = LabeledEntry(f, "Max keyframe gap:", "5")
        self.max_gap_entry.pack(fill=tk.X, padx=4)
        self.allow_ignore_edges_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Allow ignore at edges", variable=self.allow_ignore_edges_var).pack(anchor=tk.W, padx=4)
        ttk.Label(f, text="Stitch method:").pack(anchor=tk.W, padx=4)
        self.kf_stitch_var = tk.StringVar(value="matching")
        sf2 = ttk.Frame(f)
        sf2.pack(anchor=tk.W, padx=4)
        ttk.Radiobutton(sf2, text="matching", variable=self.kf_stitch_var, value="matching").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(sf2, text="position", variable=self.kf_stitch_var, value="position").pack(side=tk.LEFT, padx=6)

        # --- Static background ---
        f = ttk.LabelFrame(pf, text="Static Background")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.static_bg_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Enable static-bg", variable=self.static_bg_var).pack(anchor=tk.W, padx=4)
        ttk.Label(f, text="Method:").pack(anchor=tk.W, padx=4)
        self.static_method_var = tk.StringVar(value="median")
        sm = ttk.Frame(f)
        sm.pack(anchor=tk.W, padx=4)
        ttk.Radiobutton(sm, text="median", variable=self.static_method_var, value="median").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(sm, text="min_edge", variable=self.static_method_var, value="min_edge").pack(side=tk.LEFT, padx=6)

        # --- Ignore ---
        f = ttk.LabelFrame(pf, text="Ignore Regions")
        f.pack(fill=tk.X, padx=4, pady=2)
        self.ignore_entry = LabeledEntry(f, "Ignore (px):", "", tooltip="top,right|left|X,w,h  (multiple: semicolon)")
        self.ignore_entry.pack(fill=tk.X, padx=4)
        self.ignore_pct_entry = LabeledEntry(f, "Ignore (%):", "")
        self.ignore_pct_entry.pack(fill=tk.X, padx=4)

    def _browse_video(self):
        f = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov *.webm"), ("All", "*.*")],
            initialdir=_initial_dir(self.video_var.get())
        )
        if f:
            self.video_var.set(f)

    def _browse_frames_dir(self):
        d = filedialog.askdirectory(initialdir=_initial_dir(self.frames_var.get()))
        if d:
            self.frames_var.set(str(Path(d) / "*.png"))

    def _browse_out(self):
        d = filedialog.askdirectory(initialdir=_initial_dir(self.out_var.get()))
        if d:
            self.out_var.set(d)

    def _build_argv(self) -> List[str]:
        """Build argv list for video_strip_reconstruct.main(argv)."""
        argv: List[str] = []

        if self.video_var.get():
            argv += ["--video", self.video_var.get()]
        if self.frames_var.get():
            argv += ["--frames", self.frames_var.get()]

        argv += ["--out", self.out_var.get()]

        if self.fps_entry.get():
            argv += ["--fps", self.fps_entry.get()]
        if self.start_entry.get():
            argv += ["--start", self.start_entry.get()]
        if self.dur_entry.get():
            argv += ["--dur", self.dur_entry.get()]
        if self.deinterlace_var.get():
            argv += ["--deinterlace"]

        argv += ["--scroll-axis", self.scroll_axis_var.get()]
        argv += ["--scroll-dir", self.scroll_dir_var.get()]

        if self.strip_y_entry.get():
            argv += ["--strip-y", self.strip_y_entry.get()]
        if self.strip_h_entry.get():
            argv += ["--strip-h", self.strip_h_entry.get()]
        if self.strip_x_entry.get():
            argv += ["--strip-x", self.strip_x_entry.get()]
        if self.strip_w_entry.get():
            argv += ["--strip-w", self.strip_w_entry.get()]

        selected_methods = [m for m, v in self.match_method_vars.items() if v.get()]
        if selected_methods:
            argv += ["--match-method", ",".join(selected_methods)]
        if self.edge_thr_entry.get():
            argv += ["--edge-thr", self.edge_thr_entry.get()]
        if self.min_peak_entry.get():
            argv += ["--min-peak", self.min_peak_entry.get()]
        if self.dy_search_entry.get():
            argv += ["--dy-search", self.dy_search_entry.get()]
        if self.dy_region_entry.get():
            argv += ["--dy-region", self.dy_region_entry.get()]
        if self.uniform_dy_entry.get():
            argv += ["--uniform-dy", self.uniform_dy_entry.get()]
        if self.no_edges_var.get():
            argv += ["--no-edges"]

        if self.template_region_entry.get():
            argv += ["--template-region", self.template_region_entry.get()]
        if self.suggest_var.get():
            argv += ["--suggest-templates"]
        if self.suggest_n_entry.get():
            argv += ["--suggest-n", self.suggest_n_entry.get()]

        if self.jigsaw_var.get():
            argv += ["--jigsaw"]
        else:
            argv += ["--no-jigsaw"]

        if self.base_frame_entry.get():
            argv += ["--base-frame", self.base_frame_entry.get()]
        if self.auto_exclude_var.get():
            argv += ["--auto-exclude-text"]

        if self.keyframe_var.get():
            argv += ["--keyframe-mode"]
        else:
            argv += ["--no-keyframe-mode"]
        if self.min_quality_entry.get():
            argv += ["--min-quality", self.min_quality_entry.get()]
        if self.max_gap_entry.get():
            argv += ["--max-keyframe-gap", self.max_gap_entry.get()]
        if self.allow_ignore_edges_var.get():
            argv += ["--allow-ignore-at-edges"]
        argv += ["--keyframe-stitch-method", self.kf_stitch_var.get()]

        if self.static_bg_var.get():
            argv += ["--static-bg"]
            argv += ["--static-method", self.static_method_var.get()]

        for region in self.ignore_entry.get().split(";"):
            region = region.strip()
            if region:
                argv += ["--ignore", region]
        for region in self.ignore_pct_entry.get().split(";"):
            region = region.strip()
            if region:
                argv += ["--ignore-pct", region]

        return argv

    def _run(self):
        if self._running:
            return
        if not self.video_var.get() and not self.frames_var.get():
            messagebox.showwarning("Warning", "Video or frames glob not specified.")
            return

        argv = self._build_argv()
        self.log.clear()
        self.log.append(f"[video_strip_reconstruct] argv = {argv}\n\n")
        self._running = True
        self.run_btn.configure(state=tk.DISABLED)

        from video_strip_reconstruct import main as vsr_main
        run_with_capture(vsr_main, argv, self.log.append, self._on_done, self.app)

    def _on_done(self, error: Optional[str]):
        self._running = False
        self.run_btn.configure(state=tk.NORMAL)
        if error:
            self.log.append(f"\n--- Error ---\n{error}\n")
        else:
            self.log.append("\n--- Finished ---\n")
            self._load_output_browser()

    def _load_output_browser(self):
        out_dir = Path(self.out_var.get())
        if not out_dir.is_absolute():
            out_dir = SCRIPT_DIR / out_dir
        if out_dir.exists():
            self.app.output_tab.load_dir(str(out_dir))
            self.app.notebook.select(self.app.output_tab)


# ============================================================
# Output Browser Tab
# ============================================================

class OutputBrowserTab(ttk.Frame):
    """Browse output directory images."""

    def __init__(self, master, preview: ImagePreview, **kw):
        super().__init__(master, **kw)
        self.preview = preview
        self._images: List[str] = []
        self._idx = 0
        self._build()

    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=4, pady=4)
        self.dir_var = tk.StringVar()
        ttk.Label(top, text="Directory:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.dir_var, width=40).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(top, text="Browse...", command=self._browse).pack(side=tk.LEFT)
        ttk.Button(top, text="Load", command=self._load).pack(side=tk.LEFT, padx=4)

        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.listbox = tk.Listbox(list_frame, height=20)
        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)
        self.listbox.bind("<Delete>", lambda _e: self._delete_selected())

        nav = ttk.Frame(self)
        nav.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(nav, text="<< Prev", command=self._prev).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next >>", command=self._next).pack(side=tk.LEFT, padx=(4, 0))
        self.info_label = ttk.Label(nav, text="")
        self.info_label.pack(side=tk.LEFT, padx=12)
        ttk.Button(nav, text="Open in Explorer", command=self._open_explorer).pack(side=tk.RIGHT)
        ttk.Button(nav, text="Delete", command=self._delete_selected).pack(side=tk.RIGHT, padx=(0, 8))

    def _browse(self):
        d = filedialog.askdirectory(initialdir=_initial_dir(self.dir_var.get()))
        if d:
            self.dir_var.set(d)
            self._load()

    def _load(self):
        d = self.dir_var.get()
        if not d or not os.path.isdir(d):
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
        self._images = sorted(
            [str(p) for p in Path(d).iterdir() if p.suffix.lower() in exts],
            key=lambda x: os.path.getmtime(x), reverse=True
        )
        self.listbox.delete(0, tk.END)
        for img in self._images:
            self.listbox.insert(tk.END, Path(img).name)
        if self._images:
            self._idx = 0
            self.listbox.selection_set(0)
            self._show_current()

    def _on_select(self, _event):
        sel = self.listbox.curselection()
        if sel:
            self._idx = sel[0]
            self._show_current()

    def _show_current(self):
        if 0 <= self._idx < len(self._images):
            self.preview.show(self._images[self._idx])
            self.info_label.configure(text=f"{self._idx + 1}/{len(self._images)}  {Path(self._images[self._idx]).name}")

    def _prev(self):
        if self._idx > 0:
            self._idx -= 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self._idx)
            self.listbox.see(self._idx)
            self._show_current()

    def _next(self):
        if self._idx < len(self._images) - 1:
            self._idx += 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self._idx)
            self.listbox.see(self._idx)
            self._show_current()

    def load_dir(self, path: str):
        """Programmatically set directory and load."""
        self.dir_var.set(path)
        self._load()

    def _delete_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        paths = [self._images[i] for i in sel]
        names = [Path(p).name for p in paths]
        if len(names) == 1:
            msg = f"{names[0]} をごみ箱に移動しますか？"
        else:
            msg = f"{len(names)} 件のファイルをごみ箱に移動しますか？\n" + "\n".join(names[:10])
            if len(names) > 10:
                msg += f"\n... 他 {len(names) - 10} 件"
        if not messagebox.askyesno("Delete", msg):
            return
        try:
            from send2trash import send2trash
            for p in paths:
                send2trash(p)
        except Exception as e:
            messagebox.showerror("Error", f"削除に失敗しました:\n{e}")
            return
        # Update list: remove deleted entries and adjust index
        for i in reversed(sel):
            self.listbox.delete(i)
            del self._images[i]
        if self._images:
            self._idx = min(sel[0], len(self._images) - 1)
            self.listbox.selection_set(self._idx)
            self.listbox.see(self._idx)
            self._show_current()
        else:
            self._idx = 0
            self.info_label.configure(text="")
            self.preview.clear()

    def _open_explorer(self):
        d = self.dir_var.get()
        if d and os.path.isdir(d):
            os.startfile(d)


# ============================================================
# Main Application
# ============================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stitch Candidates GUI")
        self.geometry("1200x800")
        self.minsize(900, 600)
        self._build()

    def _build(self):
        main_pw = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_pw.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_pw)
        main_pw.add(self.notebook, weight=3)

        bottom = ttk.PanedWindow(main_pw, orient=tk.HORIZONTAL)
        main_pw.add(bottom, weight=2)

        self.log = LogPanel(bottom)
        bottom.add(self.log, weight=1)

        self.preview = ImagePreview(bottom)
        bottom.add(self.preview, weight=1)

        self.stitch_tab = StitchTab(self.notebook, self.log, self.preview, self)
        self.notebook.add(self.stitch_tab, text="  Stitch Candidates  ")

        self.video_tab = VideoTab(self.notebook, self.log, self.preview, self)
        self.notebook.add(self.video_tab, text="  Video Reconstruct  ")

        self.output_tab = OutputBrowserTab(self.notebook, self.preview)
        self.notebook.add(self.output_tab, text="  Output Browser  ")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
