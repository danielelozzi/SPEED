# desktop_app/marker_surface_editor.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from pathlib import Path
import logging

class MarkerSurfaceEditor(tk.Toplevel):
    """
    A window for defining a surface for enrichment by mapping ArUco markers
    to the corners of the surface.
    """
    def __init__(self, parent, video_path: Path):
        super().__init__(parent)
        self.title("Define Surface from Markers")
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

        # --- ArUco Setup ---
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except AttributeError:
            messagebox.showerror("OpenCV Error", "cv2.aruco module not found. Please ensure you have 'opencv-contrib-python' installed (`pip install opencv-contrib-python`).", parent=self)
            self.destroy()
            return

        # --- State ---
        self.current_frame_idx = 0
        self.detected_markers = {}  # Cache for detected markers {frame_idx: (corners, ids)}
        self.is_playing = False
        self.is_updating_slider = False

        # --- Results ---
        self.aoi_name = None
        self.result = None
        self.result_type = 'marker_surface'
        self.corner_mappings = {
            'tl': tk.StringVar(value="Not Set"),
            'tr': tk.StringVar(value="Not Set"),
            'br': tk.StringVar(value="Not Set"),
            'bl': tk.StringVar(value="Not Set")
        }

        # --- GUI ---
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side: Video
        video_frame = tk.Frame(main_pane)
        main_pane.add(video_frame, stretch="always")
        self.video_canvas = tk.Canvas(video_frame)
        self.video_canvas.pack(pady=5, expand=True)

        # Right side: Controls
        controls_panel = tk.Frame(main_pane, width=300)
        controls_panel.pack_propagate(False)
        main_pane.add(controls_panel)

        # --- Video Controls ---
        video_controls_frame = tk.Frame(self)
        video_controls_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.play_pause_btn = tk.Button(video_controls_frame, text="▶ Play", command=self.toggle_play)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        self.frame_scale = ttk.Scale(video_controls_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(video_controls_frame, text=f"Frame: 0 / {self.total_frames}", width=20)
        self.time_label.pack(side=tk.RIGHT, padx=5)

        # --- Marker Controls (in right panel) ---
        marker_frame = tk.LabelFrame(controls_panel, text="Marker-to-Corner Mapping", padx=10, pady=10)
        marker_frame.pack(fill=tk.X, pady=10)

        tk.Label(marker_frame, text="Detected Marker IDs on this frame:").pack(anchor='w')
        self.detected_ids_listbox = tk.Listbox(marker_frame, height=8)
        self.detected_ids_listbox.pack(fill=tk.X, expand=True, pady=5)

        mapping_grid = tk.Frame(marker_frame)
        mapping_grid.pack(fill=tk.X, pady=10)
        positions = {'Top-Left': 'tl', 'Top-Right': 'tr', 'Bottom-Right': 'br', 'Bottom-Left': 'bl'}
        for i, (label, key) in enumerate(positions.items()):
            row, col = divmod(i, 2)
            f = tk.Frame(mapping_grid)
            f.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            tk.Button(f, text=f"Set {label}", command=lambda k=key: self.set_corner_from_selection(k)).pack(side=tk.LEFT)
            tk.Label(f, textvariable=self.corner_mappings[key], width=8).pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self, text="Use the video controls to find a frame where all 4 markers are visible.")
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
        if self.is_playing:
            self.play_video()

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
        if ret:
            self.detect_and_draw_markers(frame)

    def detect_and_draw_markers(self, frame):
        # Use cache if available
        if self.current_frame_idx in self.detected_markers:
            corners, ids, _ = self.detected_markers[self.current_frame_idx]
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            self.detected_markers[self.current_frame_idx] = (corners, ids, rejected)

        # Update listbox
        self.detected_ids_listbox.delete(0, tk.END)
        if ids is not None:
            for marker_id in sorted(ids.flatten()):
                self.detected_ids_listbox.insert(tk.END, f"ID: {marker_id}")

        # Draw on frame
        if corners:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Highlight mapped corners
        if ids is not None:
            flat_ids = ids.flatten().tolist()
            for key, var in self.corner_mappings.items():
                try:
                    marker_id = int(var.get())
                    if marker_id in flat_ids:
                        idx = flat_ids.index(marker_id)
                        marker_corners = corners[idx][0]
                        # Draw a thicker rectangle around the mapped marker
                        cv2.polylines(frame, [marker_corners.astype(int)], True, (0, 255, 255), 4)
                except (ValueError, IndexError):
                    continue

        # Display frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.thumbnail((960, 720))
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.config(width=img.width, height=img.height)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def set_corner_from_selection(self, corner_key):
        """Assigns the selected marker ID from the listbox to a corner."""
        selection = self.detected_ids_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a marker ID from the list first.", parent=self)
            return

        selected_text = self.detected_ids_listbox.get(selection[0])
        marker_id = selected_text.split(":")[-1].strip()

        # Check for duplicates
        for key, var in self.corner_mappings.items():
            if var.get() == marker_id and key != corner_key:
                messagebox.showerror("Duplicate Assignment", f"Marker ID {marker_id} is already assigned to {key.upper()}. Each corner must have a unique marker.", parent=self)
                return

        self.corner_mappings[corner_key].set(marker_id)
        logging.info(f"Assigned marker ID {marker_id} to corner {corner_key.upper()}")
        self.update_frame(self.current_frame_idx) # Redraw to show highlight

    def save_and_close(self):
        """Validates the mappings and saves the result."""
        aoi_name = simpledialog.askstring("Surface Name", "Enter a unique name for this surface:", parent=self)
        if not aoi_name:
            return

        mappings = {}
        all_corners_set = True
        for key, var in self.corner_mappings.items():
            try:
                mappings[key] = int(var.get())
            except ValueError:
                all_corners_set = False
                break

        if not all_corners_set:
            messagebox.showerror("Incomplete Mapping", "All four corners (TL, TR, BR, BL) must be assigned to a marker ID.", parent=self)
            return

        if len(set(mappings.values())) != 4:
            messagebox.showerror("Duplicate IDs", "Each corner must be assigned a unique marker ID.", parent=self)
            return

        self.aoi_name = aoi_name
        self.result = mappings
        self.result_type = 'marker_surface'
        logging.info(f"Saved marker surface '{self.aoi_name}' with mappings: {self.result}")
        self.on_close()