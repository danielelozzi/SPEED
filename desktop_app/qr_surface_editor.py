# desktop_app/qr_surface_editor.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from pathlib import Path
import logging
import numpy as np

class QRSurfaceEditor(tk.Toplevel):
    """
    Una finestra per definire una superficie per l'arricchimento mappando
    i dati dei QR code agli angoli della superficie.
    """
    def __init__(self, parent, video_path: Path):
        super().__init__(parent)
        self.title("Define Surface from QR Codes")
        self.geometry("1100x850")
        self.transient(parent)
        self.grab_set()

        if not video_path or not video_path.exists():
            messagebox.showerror("Error", f"Video file not found:\n{video_path}", parent=self)
            self.destroy()
            return

        self.cap = cv2.VideoCapture(str(video_path))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30

        # --- QR Code Detector Setup ---
        self.qr_detector = cv2.QRCodeDetector()

        # --- State ---
        self.current_frame_idx = 0
        self.detected_qrs = {}  # Cache for detected QR codes {frame_idx: {data: center_point}}
        self.is_playing = False
        self.is_updating_slider = False

        # --- Results ---
        self.aoi_name = None
        self.result = None
        self.result_type = 'qr_surface'
        self.corner_mappings = {
            'tl': tk.StringVar(value="Not Set"), 'tr': tk.StringVar(value="Not Set"),
            'br': tk.StringVar(value="Not Set"), 'bl': tk.StringVar(value="Not Set")
        }

        # --- GUI ---
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_frame = tk.Frame(main_pane)
        main_pane.add(video_frame, stretch="always")
        self.video_canvas = tk.Canvas(video_frame)
        self.video_canvas.pack(pady=5, expand=True)

        controls_panel = tk.Frame(main_pane, width=300)
        controls_panel.pack_propagate(False)
        main_pane.add(controls_panel)

        video_controls_frame = tk.Frame(self)
        video_controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.play_pause_btn = tk.Button(video_controls_frame, text="▶ Play", command=self.toggle_play)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        self.frame_scale = ttk.Scale(video_controls_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(video_controls_frame, text=f"Frame: 0 / {self.total_frames}", width=20)
        self.time_label.pack(side=tk.RIGHT, padx=5)

        qr_frame = tk.LabelFrame(controls_panel, text="QR Code to Corner Mapping", padx=10, pady=10)
        qr_frame.pack(fill=tk.X, pady=10)

        tk.Label(qr_frame, text="Detected QR Data on this frame:").pack(anchor='w')
        self.detected_qr_listbox = tk.Listbox(qr_frame, height=8)
        self.detected_qr_listbox.pack(fill=tk.X, expand=True, pady=5)

        mapping_grid = tk.Frame(qr_frame)
        mapping_grid.pack(fill=tk.X, pady=10)
        positions = {'Top-Left': 'tl', 'Top-Right': 'tr', 'Bottom-Right': 'br', 'Bottom-Left': 'bl'}
        for i, (label, key) in enumerate(positions.items()):
            row, col = divmod(i, 2)
            f = tk.Frame(mapping_grid)
            f.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            tk.Button(f, text=f"Set {label}", command=lambda k=key: self.set_corner_from_selection(k)).pack(side=tk.LEFT)
            tk.Label(f, textvariable=self.corner_mappings[key], width=8).pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self, text="Find a frame where all 4 QR codes are visible.")
        self.status_label.pack(pady=5)

        self.save_button = tk.Button(self, text="Save Surface Definition", command=self.save_and_close, font=('Helvetica', 10, 'bold'))
        self.save_button.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, lambda: self.update_frame(0))

    def on_close(self):
        self.is_playing = False
        self.cap.release()
        self.destroy()

    def seek_frame(self, frame_idx_str):
        if self.is_updating_slider: return
        if self.is_playing: self.toggle_play()
        self.update_frame(int(float(frame_idx_str)))

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="❚❚ Pause" if self.is_playing else "▶ Play")
        if self.is_playing: self.play_video()

    def play_video(self):
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            self.update_frame(self.current_frame_idx + 1)
            self.after(int(1000 / self.fps), self.play_video)
        else:
            self.is_playing = False
            self.play_pause_btn.config(text="▶ Play")

    def update_frame(self, frame_idx):
        self.current_frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        self.is_updating_slider = True
        self.frame_scale.set(self.current_frame_idx)
        self.is_updating_slider = False
        self.time_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret: self.detect_and_draw_qrs(frame)

    def detect_and_draw_qrs(self, frame):
        if self.current_frame_idx not in self.detected_qrs:
            ok, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(frame)
            self.detected_qrs[self.current_frame_idx] = (decoded_info, points) if ok else ([], None)
        
        decoded_info, points = self.detected_qrs[self.current_frame_idx]

        self.detected_qr_listbox.delete(0, tk.END)
        if points is not None:
            for i, info in enumerate(decoded_info):
                if info:
                    self.detected_qr_listbox.insert(tk.END, info)
                    qr_points = points[i].astype(int)
                    cv2.polylines(frame, [qr_points], True, (0, 255, 0), 2)
                    center_x = int(np.mean(qr_points[:, 0]))
                    cv2.putText(frame, info, (center_x, qr_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((960, 720))
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.config(width=img.width, height=img.height)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def set_corner_from_selection(self, corner_key):
        selection = self.detected_qr_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a QR code from the list first.", parent=self)
            return
        qr_data = self.detected_qr_listbox.get(selection[0])
        for key, var in self.corner_mappings.items():
            if var.get() == qr_data and key != corner_key:
                messagebox.showerror("Duplicate Assignment", f"QR data '{qr_data}' is already assigned to {key.upper()}.", parent=self)
                return
        self.corner_mappings[corner_key].set(qr_data)

    def save_and_close(self):
        aoi_name = simpledialog.askstring("Surface Name", "Enter a unique name for this surface:", parent=self)
        if not aoi_name: return

        mappings = {}
        all_corners_set = True
        for key, var in self.corner_mappings.items():
            val = var.get()
            if val == "Not Set":
                all_corners_set = False
                break
            mappings[key] = val

        if not all_corners_set:
            messagebox.showerror("Incomplete Mapping", "All four corners must be assigned to a QR code.", parent=self)
            return

        self.aoi_name = aoi_name
        self.result = mappings
        self.result_type = 'qr_surface'
        logging.info(f"Saved QR surface '{self.aoi_name}' with mappings: {self.result}")
        self.on_close()