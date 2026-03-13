"""
YOLOv8 Real-Time Object Detection with Interactive GUI
======================================================
Requirements:
    pip install ultralytics opencv-python Pillow tkinter numpy

Usage:
    python yolov8_detector.py

Place your YOLOv8 .pt model file in the same directory,
or let the app download yolov8n.pt / yolov8s.pt automatically.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import time
import numpy as np
from PIL import Image, ImageTk
from collections import defaultdict, deque
import os

# ── Try importing ultralytics ──────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    messagebox.showerror(
        "Missing dependency",
        "ultralytics is not installed.\n\nRun:  pip install ultralytics",
    )
    raise SystemExit

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
PALETTE = [
    (255, 56,  56),  (255, 157,  151), (255, 112,  31),
    (255, 178,  29),  (207, 210,  49),  (72, 249,  10),
    (146, 204,  23),  (61, 219, 134),   (26, 147,  52),
    (0,  212, 187),  (44, 153, 168),  (0,  194, 255),
    (52,  69, 147),  (100,  115, 255),  (0,  24, 236),
    (132,  56, 255),  (82,   0, 133),  (203,  56, 255),
    (255,  49, 203),  (255,   4, 130),
]


def color_for(class_id: int):
    return PALETTE[class_id % len(PALETTE)]


# ─────────────────────────────────────────────────────────────────────────────
class DetectorApp(tk.Tk):
    """Main application window."""

    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__()
        self.title("🔍  YOLOv8 Object Detector")
        self.configure(bg="#0d0f1a")
        self.resizable(True, True)
        self.minsize(1100, 680)

        # ── state ─────────────────────────────────────────────────────────────
        self.model        = None
        self.model_path   = tk.StringVar(value="yolov8n.pt")
        self.conf_thresh  = tk.DoubleVar(value=0.45)
        self.iou_thresh   = tk.DoubleVar(value=0.50)
        self.device_var   = tk.StringVar(value="cpu")
        self.source_var   = tk.StringVar(value="Webcam 0")
        self.show_labels  = tk.BooleanVar(value=True)
        self.show_conf    = tk.BooleanVar(value=True)
        self.show_fps     = tk.BooleanVar(value=True)
        self.half_prec    = tk.BooleanVar(value=False)

        self._running     = False
        self._cap         = None
        self._thread      = None
        self._frame_buf   = None
        self._lock        = threading.Lock()

        # FPS ring buffer
        self._fps_times   = deque(maxlen=30)
        self._fps_display = 0.0

        # Detection counts (last frame)
        self._det_counts  = defaultdict(int)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)

        # ── left: video canvas ────────────────────────────────────────────────
        left = tk.Frame(self, bg="#0d0f1a")
        left.grid(row=0, column=0, sticky="nsew", padx=(14, 6), pady=14)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # title bar
        title_bar = tk.Frame(left, bg="#0d0f1a")
        title_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            title_bar, text="⬡  YOLOv8  LIVE DETECTOR",
            font=("Courier New", 14, "bold"),
            fg="#00e5ff", bg="#0d0f1a"
        ).pack(side="left")
        self._fps_lbl = tk.Label(
            title_bar, text="FPS: --",
            font=("Courier New", 12), fg="#b0bec5", bg="#0d0f1a"
        )
        self._fps_lbl.pack(side="right")

        # canvas
        self._canvas = tk.Canvas(
            left, bg="#060810", highlightthickness=2,
            highlightbackground="#1e2a3a", width=860, height=540
        )
        self._canvas.grid(row=1, column=0, sticky="nsew")
        self._canvas.bind("<Configure>", lambda e: self._redraw_placeholder())
        self._redraw_placeholder()

        # status bar
        self._status_var = tk.StringVar(value="Ready — load a model to begin.")
        tk.Label(
            left, textvariable=self._status_var,
            font=("Courier New", 9), fg="#546e7a", bg="#0d0f1a",
            anchor="w"
        ).grid(row=2, column=0, sticky="ew", pady=(4, 0))

        # ── right: control panel ──────────────────────────────────────────────
        right = tk.Frame(self, bg="#111420", bd=0)
        right.grid(row=0, column=1, sticky="ns", padx=(0, 14), pady=14)
        right.configure(width=280)

        self._build_panel(right)

    # ── control panel ─────────────────────────────────────────────────────────
    def _build_panel(self, parent):
        pad = {"padx": 14, "pady": 5}

        # ── section: model ────────────────────────────────────────────────────
        self._section(parent, "⬡  MODEL")

        tk.Label(parent, text="Model file (.pt):", **self._lbl_cfg()).pack(anchor="w", **pad)
        mf = tk.Frame(parent, bg="#111420")
        mf.pack(fill="x", **pad)
        self._model_entry = ttk.Entry(mf, textvariable=self.model_path, width=18)
        self._model_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(mf, text="Browse", width=7,
                   command=self._browse_model).pack(side="right", padx=(4, 0))

        tk.Label(parent, text="Device:", **self._lbl_cfg()).pack(anchor="w", **pad)
        dev_frame = tk.Frame(parent, bg="#111420")
        dev_frame.pack(fill="x", padx=14, pady=2)
        for d in ("cpu", "cuda:0", "mps"):
            tk.Radiobutton(
                dev_frame, text=d, variable=self.device_var, value=d,
                bg="#111420", fg="#cfd8dc", selectcolor="#0d0f1a",
                activebackground="#111420", font=("Courier New", 9)
            ).pack(side="left", padx=4)

        ttk.Checkbutton(parent, text="Half precision (FP16 / GPU)",
                        variable=self.half_prec).pack(anchor="w", **pad)

        ttk.Button(parent, text="⬡  Load Model",
                   command=self._load_model).pack(fill="x", padx=14, pady=(6, 2))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # ── section: source ───────────────────────────────────────────────────
        self._section(parent, "▶  SOURCE")

        sources = [f"Webcam {i}" for i in range(4)] + ["Video File…", "Image File…"]
        tk.Label(parent, text="Input source:", **self._lbl_cfg()).pack(anchor="w", **pad)
        self._src_combo = ttk.Combobox(
            parent, textvariable=self.source_var,
            values=sources, state="readonly", width=22
        )
        self._src_combo.pack(fill="x", padx=14, pady=2)
        self._src_combo.bind("<<ComboboxSelected>>", self._on_source_change)

        self._file_lbl = tk.Label(
            parent, text="", wraplength=250,
            font=("Courier New", 8), fg="#546e7a", bg="#111420"
        )
        self._file_lbl.pack(anchor="w", padx=14)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # ── section: thresholds ───────────────────────────────────────────────
        self._section(parent, "⚙  THRESHOLDS")

        self._make_slider(parent, "Confidence", self.conf_thresh, 0.05, 0.95)
        self._make_slider(parent, "IoU overlap", self.iou_thresh, 0.10, 0.90)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # ── section: display ──────────────────────────────────────────────────
        self._section(parent, "🖥  DISPLAY")

        ttk.Checkbutton(parent, text="Show labels",
                        variable=self.show_labels).pack(anchor="w", **pad)
        ttk.Checkbutton(parent, text="Show confidence",
                        variable=self.show_conf).pack(anchor="w", **pad)
        ttk.Checkbutton(parent, text="Show FPS overlay",
                        variable=self.show_fps).pack(anchor="w", **pad)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # ── start / stop ──────────────────────────────────────────────────────
        self._start_btn = ttk.Button(
            parent, text="▶  START", command=self._start)
        self._start_btn.pack(fill="x", padx=14, pady=3)

        self._stop_btn = ttk.Button(
            parent, text="⏹  STOP", command=self._stop, state="disabled")
        self._stop_btn.pack(fill="x", padx=14, pady=3)

        ttk.Button(parent, text="📷  Screenshot",
                   command=self._screenshot).pack(fill="x", padx=14, pady=3)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # ── detection list ────────────────────────────────────────────────────
        self._section(parent, "📋  DETECTIONS")

        list_frame = tk.Frame(parent, bg="#111420")
        list_frame.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self._det_list = tk.Listbox(
            list_frame, bg="#0d0f1a", fg="#00e5ff",
            font=("Courier New", 9), selectbackground="#1e3a5f",
            highlightthickness=0, bd=0,
            yscrollcommand=scrollbar.set
        )
        self._det_list.pack(fill="both", expand=True)
        scrollbar.config(command=self._det_list.yview)

        self._apply_style()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _section(self, parent, text):
        tk.Label(
            parent, text=text,
            font=("Courier New", 10, "bold"),
            fg="#00e5ff", bg="#111420"
        ).pack(anchor="w", padx=14, pady=(10, 2))

    def _lbl_cfg(self):
        return dict(font=("Courier New", 9), fg="#b0bec5", bg="#111420")

    def _make_slider(self, parent, label, var, from_, to):
        row = tk.Frame(parent, bg="#111420")
        row.pack(fill="x", padx=14, pady=3)
        tk.Label(row, text=f"{label}:", **self._lbl_cfg(), width=14,
                 anchor="w").pack(side="left")
        val_lbl = tk.Label(row, text=f"{var.get():.2f}",
                           font=("Courier New", 9), fg="#00e5ff", bg="#111420", width=5)
        val_lbl.pack(side="right")
        s = ttk.Scale(
            parent, from_=from_, to=to, variable=var, orient="horizontal",
            command=lambda v, lbl=val_lbl, vr=var: lbl.config(
                text=f"{float(v):.2f}")
        )
        s.pack(fill="x", padx=14, pady=(0, 2))

    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton",
                        background="#1b2a40", foreground="#e0f7fa",
                        font=("Courier New", 9, "bold"),
                        borderwidth=0, focusthickness=0, padding=6)
        style.map("TButton",
                  background=[("active", "#00acc1"), ("pressed", "#006064")],
                  foreground=[("active", "#ffffff")])
        style.configure("TEntry", fieldbackground="#1a1f2e",
                        foreground="#cfd8dc", insertcolor="#00e5ff")
        style.configure("TCombobox", fieldbackground="#1a1f2e",
                        foreground="#cfd8dc", selectbackground="#1b2a40")
        style.configure("TCheckbutton", background="#111420",
                        foreground="#b0bec5", font=("Courier New", 9))
        style.configure("TScale", background="#111420", troughcolor="#1a1f2e",
                        sliderlength=14)
        style.configure("TSeparator", background="#1e2a3a")

    def _redraw_placeholder(self):
        self._canvas.delete("placeholder")
        w = self._canvas.winfo_width()  or 860
        h = self._canvas.winfo_height() or 540
        self._canvas.create_text(
            w // 2, h // 2,
            text="[ NO SIGNAL ]",
            font=("Courier New", 20, "bold"),
            fill="#1e3a4a", tags="placeholder"
        )

    # ── model & source ────────────────────────────────────────────────────────
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLOv8 .pt file",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def _load_model(self):
        path = self.model_path.get().strip()
        self._set_status(f"Loading model: {os.path.basename(path)} …")
        self.update_idletasks()
        try:
            self.model = YOLO(path)
            # warm-up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            self._set_status(
                f"✓ Model loaded: {os.path.basename(path)}  "
                f"| classes: {len(self.model.names)}"
            )
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self._set_status("Model load FAILED.")

    def _on_source_change(self, _=None):
        if self.source_var.get() in ("Video File…", "Image File…"):
            path = filedialog.askopenfilename(
                filetypes=[
                    ("Media", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp"),
                    ("All", "*.*")
                ]
            )
            if path:
                self._file_lbl.config(text=os.path.basename(path))
                self.source_var.set(path)
            else:
                self.source_var.set("Webcam 0")
        else:
            self._file_lbl.config(text="")

    # ── start / stop ──────────────────────────────────────────────────────────
    def _start(self):
        if self.model is None:
            messagebox.showwarning("No Model", "Please load a YOLOv8 model first.")
            return

        src = self.source_var.get()
        if src.startswith("Webcam"):
            cam_id = int(src.split()[-1])
            cap_src = cam_id
        else:
            cap_src = src

        self._cap = cv2.VideoCapture(cap_src)
        if not self._cap.isOpened():
            messagebox.showerror("Source Error", f"Cannot open: {cap_src}")
            return

        # Camera optimisation flags
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(cap_src, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_FPS, 60)

        self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_status("Running …")

        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        self._poll_frame()

    def _stop(self):
        self._running = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        if self._cap:
            self._cap.release()
            self._cap = None
        self._set_status("Stopped.")
        self._redraw_placeholder()

    def _on_close(self):
        self._running = False
        if self._cap:
            self._cap.release()
        self.destroy()

    # ── inference loop (background thread) ───────────────────────────────────
    def _inference_loop(self):
        device = self.device_var.get()
        half   = self.half_prec.get() and device != "cpu"

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if not ret:
                # loop video files
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t0 = time.perf_counter()

            results = self.model(
                frame,
                conf    = self.conf_thresh.get(),
                iou     = self.iou_thresh.get(),
                device  = device,
                half    = half,
                verbose = False,
                imgsz   = 640,
            )[0]

            t1 = time.perf_counter()
            self._fps_times.append(1.0 / max(t1 - t0, 1e-6))
            self._fps_display = sum(self._fps_times) / len(self._fps_times)

            # Draw
            annotated = self._draw(frame, results)

            with self._lock:
                self._frame_buf = annotated

    # ── drawing ───────────────────────────────────────────────────────────────
    def _draw(self, frame, results):
        img = frame.copy()
        counts = defaultdict(int)

        boxes = results.boxes
        if boxes is not None and len(boxes):
            for box in boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                label   = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color   = color_for(cls_id)

                counts[label] += 1

                # box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # label pill
                if self.show_labels.get():
                    text = label
                    if self.show_conf.get():
                        text += f" {conf:.0%}"
                    (tw, th), bl = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    pill_y1 = max(y1 - th - 8, 0)
                    cv2.rectangle(
                        img,
                        (x1, pill_y1),
                        (x1 + tw + 6, pill_y1 + th + 6),
                        color, -1
                    )
                    cv2.putText(
                        img, text,
                        (x1 + 3, pill_y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA
                    )

        # FPS overlay
        if self.show_fps.get():
            fps_text = f"FPS {self._fps_display:.1f}"
            cv2.putText(
                img, fps_text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 229, 255), 2, cv2.LINE_AA
            )

        # object count overlay
        total = sum(counts.values())
        cv2.putText(
            img, f"Objects: {total}", (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (200, 200, 200), 1, cv2.LINE_AA
        )

        self._det_counts = counts
        return img

    # ── GUI poll / render ─────────────────────────────────────────────────────
    def _poll_frame(self):
        if not self._running:
            return

        with self._lock:
            frame = self._frame_buf

        if frame is not None:
            self._render_frame(frame)
            self._fps_lbl.config(text=f"FPS: {self._fps_display:.1f}")
            self._update_det_list()

        self.after(15, self._poll_frame)   # ~66 Hz poll, actual FPS limited by model

    def _render_frame(self, frame):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        fh, fw = frame.shape[:2]
        scale  = min(cw / fw, ch / fh)
        nw, nh = int(fw * scale), int(fh * scale)

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(Image.fromarray(rgb))

        self._canvas.delete("all")
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2
        self._canvas.create_image(x0, y0, anchor="nw", image=img)
        self._canvas._img = img   # prevent GC

    def _update_det_list(self):
        self._det_list.delete(0, "end")
        for label, cnt in sorted(self._det_counts.items(),
                                  key=lambda x: -x[1]):
            self._det_list.insert("end", f"  {label:<20} × {cnt}")

    def _set_status(self, msg):
        self._status_var.set(msg)

    # ── screenshot ────────────────────────────────────────────────────────────
    def _screenshot(self):
        with self._lock:
            frame = self._frame_buf
        if frame is None:
            messagebox.showinfo("Screenshot", "No frame to capture yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
            initialfile="yolo_detection.png"
        )
        if path:
            cv2.imwrite(path, frame)
            messagebox.showinfo("Saved", f"Screenshot saved to:\n{path}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = DetectorApp()
    app.mainloop()
