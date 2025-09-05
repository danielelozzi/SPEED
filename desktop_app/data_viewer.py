# desktop_app/data_viewer.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import pandas as pd
import json
import pydicom
import cv2
import numpy as np
from PIL import Image, ImageTk
import logging

# Importa le funzioni di conversione e le utility di disegno
from src.speed_analyzer.analysis_modules.bids_converter import load_from_bids
from src.speed_analyzer.analysis_modules.dicom_converter import load_from_dicom
from src.speed_analyzer.analysis_modules.video_generator import (
    _draw_pupil_plot, _draw_generic_plot, PUPIL_PLOT_HISTORY, PUPIL_PLOT_WIDTH,
    PUPIL_PLOT_HEIGHT, FRAG_PLOT_HISTORY, FRAG_PLOT_WIDTH, FRAG_PLOT_HEIGHT,
    FRAG_LINE_COLOR, FRAG_BG_COLOR, GAZE_COLOR, GAZE_RADIUS, GAZE_THICKNESS
)

def _overlay_transparent(background, overlay, x, y):
    """
    Sovrappone un'immagine (overlay) con canale alpha su uno sfondo.
    """
    background_width = background.shape[1]
    background_height = background.shape[0]
    h, w = overlay.shape[0], overlay.shape[1]

    if x >= background_width or y >= background_height:
        return background

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]
    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        return background

    alpha = overlay[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    b, g, r = overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2]
    bgr = np.dstack([b, g, r])

    background_subsection = background[y:y+h, x:x+w]
    composite = bgr * alpha + background_subsection * (1.0 - alpha)
    background[y:y+h, x:x+w] = composite
    return background

class DataViewerWindow(tk.Toplevel):
    """
    Finestra avanzata per visualizzare dati BIDS/DICOM/Un-enriched con 
    overlay video, tabelle e metadati.
    """
    def __init__(self, parent, defined_aois=None):
        super().__init__(parent)
        self.title("Interactive Data Viewer")
        self.geometry("1280x900")
        self.transient(parent)
        self.grab_set()

        self.data_folder = None
        self.dataframe_cache = {}
        self.cap = None
        self.sync_data = pd.DataFrame()
        self.is_playing = False
        self.total_frames = 0
        self.fps = 30
        self.is_updating_slider = False

        self.defined_aois = defined_aois if defined_aois else []
        self.aoi_positions_per_frame = {}
        self.aoi_colors = {aoi['name']: tuple(np.random.randint(100, 256, 3).tolist()) for aoi in self.defined_aois}

        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)
        
        # --- MODIFICA: Aggiunto pulsante per Un-enriched ---
        tk.Button(top_frame, text="Load Un-enriched Folder...", command=self.load_unenriched_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load BIDS Directory...", command=self.load_bids).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load DICOM File...", command=self.load_dicom).pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(top_frame, text="Please load a data source to begin.")
        self.status_label.pack(side=tk.LEFT, padx=10)

        left_panel = tk.Frame(main_frame, width=350, relief=tk.SUNKEN, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        metadata_frame = tk.LabelFrame(left_panel, text="File Metadata")
        metadata_frame.pack(fill=tk.X, padx=5, pady=5)
        self.metadata_tree = ttk.Treeview(metadata_frame, columns=("Property", "Value"), show="headings", height=8)
        self.metadata_tree.heading("Property", text="Property")
        self.metadata_tree.heading("Value", text="Value")
        self.metadata_tree.column("Property", width=120)
        self.metadata_tree.pack(fill=tk.X, expand=True)

        overlay_controls_frame = tk.LabelFrame(left_panel, text="Video Overlay Controls")
        overlay_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        self.overlay_vars = {
            "gaze": tk.BooleanVar(value=True), "pupil_plot": tk.BooleanVar(value=True),
            "frag_plot": tk.BooleanVar(value=True), "aois": tk.BooleanVar(value=True),
            "gaze_path": tk.BooleanVar(value=True), "heatmap": tk.BooleanVar(value=False)
        }
        for key, text in {"gaze": "Show Gaze Point", "gaze_path": "Show Gaze Path", 
                           "pupil_plot": "Show Pupil Plot", "frag_plot": "Show Fragmentation Plot", 
                           "aois": "Show Defined AOIs"}.items():
            tk.Checkbutton(overlay_controls_frame, text=text, variable=self.overlay_vars[key], 
                           command=self.update_current_frame_display).pack(anchor='w')
        
        heatmap_frame = tk.Frame(overlay_controls_frame)
        heatmap_frame.pack(fill=tk.X, pady=(5,0))
        tk.Checkbutton(heatmap_frame, text="Dynamic Gaze Heatmap", variable=self.overlay_vars["heatmap"], 
                       command=self.update_current_frame_display).pack(anchor='w')
        
        slider_frame = tk.Frame(heatmap_frame)
        slider_frame.pack(fill=tk.X, padx=(18, 0))
        self.heatmap_window_var = tk.DoubleVar(value=2.0)
        self.heatmap_label_var = tk.StringVar(value=f"{self.heatmap_window_var.get():.1f} s")
        tk.Label(slider_frame, text="Window:").pack(side=tk.LEFT)
        ttk.Scale(slider_frame, from_=0.5, to=10.0, variable=self.heatmap_window_var, 
                  orient=tk.HORIZONTAL, command=self.on_heatmap_slider_change).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        tk.Label(slider_frame, textvariable=self.heatmap_label_var, width=5).pack(side=tk.RIGHT)

        right_panel = tk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.video_tab = ttk.Frame(self.notebook)
        self.table_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.video_tab, text="Video Player")
        self.notebook.add(self.table_tab, text="Data Tables")
        
        self.video_canvas = tk.Canvas(self.video_tab, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        video_controls = tk.Frame(self.video_tab)
        video_controls.pack(fill=tk.X, pady=5)
        self.play_pause_btn = tk.Button(video_controls, text="▶ Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_pause_btn.pack(side=tk.LEFT)
        self.frame_scale = ttk.Scale(video_controls, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_frame, state=tk.DISABLED)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(video_controls, text="Frame: 0 / 0", width=20)
        self.time_label.pack(side=tk.RIGHT)

        table_controls = tk.Frame(self.table_tab)
        table_controls.pack(fill=tk.X, pady=5)
        tk.Label(table_controls, text="Select table to view:").pack(side=tk.LEFT)
        self.table_selector = ttk.Combobox(table_controls, state="readonly")
        self.table_selector.pack(side=tk.LEFT, padx=5)
        self.table_selector.bind("<<ComboboxSelected>>", self.display_table)
        self.table_tree = ttk.Treeview(self.table_tab, show="headings")
        self.table_tree.pack(fill=tk.BOTH, expand=True)
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_heatmap_slider_change(self, value):
        self.heatmap_label_var.set(f"{float(value):.1f} s")
        if self.overlay_vars["heatmap"].get():
            self.update_current_frame_display()

    def on_close(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.destroy()

    def _prepare_sync_data(self):
        if not self.data_folder: return
        try:
            wt = pd.read_csv(self.data_folder / 'world_timestamps.csv').sort_values('timestamp [ns]')
            gaze = pd.read_csv(self.data_folder / 'gaze.csv').sort_values('timestamp [ns]')
            pupil = pd.read_csv(self.data_folder / '3d_eye_states.csv').sort_values('timestamp [ns]')

            wt['frame'] = wt.index
            
            merged = pd.merge_asof(gaze, pupil, on='timestamp [ns]', direction='nearest', tolerance=pd.Timedelta('50ms').value)
            self.sync_data = pd.merge_asof(wt, merged, on='timestamp [ns]', direction='nearest', tolerance=pd.Timedelta('50ms').value)
            
            self.sync_data.sort_values('frame', inplace=True)
            time_sec = (self.sync_data['timestamp [ns]'] - self.sync_data['timestamp [ns]'].min()) / 1e9
            dx = self.sync_data['gaze x [px]'].diff()
            dy = self.sync_data['gaze y [px]'].diff()
            dt = time_sec.diff()
            self.sync_data['fragmentation'] = np.sqrt(dx**2 + dy**2) / dt
            
            logging.info("Data successfully synchronized for video playback.")
        except Exception as e:
            messagebox.showerror("Data Sync Error", f"Could not sync data for video playback: {e}", parent=self)
            self.sync_data = pd.DataFrame()
            
    def _prepare_aoi_positions(self):
        if not self.defined_aois or self.total_frames == 0:
            return

        logging.info("Pre-calculating AOI positions for all frames...")
        all_frames = np.arange(self.total_frames)
        self.aoi_positions_per_frame = {}

        for aoi in self.defined_aois:
            aoi_name = aoi['name']
            aoi_type = aoi['type']
            aoi_data = aoi['data']
            
            coords = np.zeros((self.total_frames, 4))

            if aoi_type == 'static':
                c = aoi_data
                coords[:] = [c['x1'], c['y1'], c['x2'], c['y2']]

            elif aoi_type == 'dynamic_manual':
                keyframes = {int(k): v for k, v in aoi_data.items()}
                kf_frames = np.array(list(keyframes.keys()))
                kf_coords = np.array(list(keyframes.values()))
                
                for i in range(4):
                    coords[:, i] = np.interp(all_frames, kf_frames, kf_coords[:, i])
            
            self.aoi_positions_per_frame[aoi_name] = coords
        logging.info("AOI positions pre-calculation complete.")

    def load_unenriched_folder(self):
        folder_path = filedialog.askdirectory(title="Select the un-enriched data folder")
        if not folder_path: return

        try:
            self.data_folder = Path(folder_path)
            self.status_label.config(text=f"Loaded: {self.data_folder.name}")
            self._populate_all_views()
            
            video_file = next(self.data_folder.glob('*.mp4'))
            self._load_video_and_data(video_file)
        except StopIteration:
            messagebox.showerror("Video Error", f"No .mp4 file found in folder:\n{self.data_folder}", parent=self)
        except Exception as e:
            messagebox.showerror("Error Loading Data", str(e), parent=self)

    def load_bids(self):
        bids_root_path = filedialog.askdirectory(title="Select the root BIDS directory (containing sub-...)")
        if not bids_root_path: return

        subject_id = simpledialog.askstring("Input", "Enter Subject ID (e.g., 01):", parent=self)
        if not subject_id: return
        session_id = simpledialog.askstring("Input", "Enter Session ID (e.g., 01):", parent=self)
        if not session_id: return
        task_name = simpledialog.askstring("Input", "Enter Task Name (e.g., visualsearch):", parent=self)
        if not task_name: return

        try:
            self.data_folder = load_from_bids(Path(bids_root_path), subject_id, session_id, task_name)
            self.status_label.config(text=f"Loaded BIDS: sub-{subject_id}_ses-{session_id}")
            self._populate_all_views()
            
            json_path = Path(bids_root_path) / f"sub-{subject_id}" / f"ses-{session_id}" / "eyetrack" / f"sub-{subject_id}_ses-{session_id}_task-{task_name}_eyetrack.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                for key, value in metadata.items():
                    self.metadata_tree.insert("", "end", values=(key, str(value)))
            
            video_file = next(self.data_folder.glob('*.mp4'), None)
            if video_file: self._load_video_and_data(video_file)

        except Exception as e:
            messagebox.showerror("Error Loading BIDS", str(e), parent=self)

    def load_dicom(self):
        dicom_path = filedialog.askopenfilename(title="Select DICOM File", filetypes=[("DICOM files", "*.dcm")])
        if not dicom_path: return
        
        try:
            self.data_folder = load_from_dicom(Path(dicom_path))
            ds = pydicom.dcmread(dicom_path, force=True)
            self.status_label.config(text=f"Loaded DICOM: {Path(dicom_path).name}")
            self._populate_all_views()

            for tag in ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'SamplingFrequency']:
                if hasattr(ds, tag):
                    self.metadata_tree.insert("", "end", values=(tag, str(ds[tag].value)))
            
            messagebox.showinfo("Info", "DICOM data loaded. Video player is disabled for this format.", parent=self)

        except Exception as e:
            messagebox.showerror("Error Loading DICOM", str(e), parent=self)

    def _populate_all_views(self):
        self._clear_views()
        if not self.data_folder: return
        
        csv_files = sorted([p.name for p in self.data_folder.glob("*.csv")])
        self.table_selector['values'] = csv_files
        if csv_files:
            self.table_selector.set(csv_files[0])
            self.display_table()

    def _load_video_and_data(self, video_path):
        self._prepare_sync_data()
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        
        self._prepare_aoi_positions()
        
        self.frame_scale.config(to=self.total_frames - 1, state=tk.NORMAL)
        self.play_pause_btn.config(state=tk.NORMAL)
        self.update_frame(0)

    def seek_frame(self, frame_idx_str):
        if self.is_updating_slider: return
        if self.is_playing: self.toggle_play()
        self.update_frame(int(float(frame_idx_str)))

    def update_frame(self, frame_idx):
        if not self.cap: return
        self.current_frame_idx = int(frame_idx)
        
        self.is_updating_slider = True
        self.frame_scale.set(self.current_frame_idx)
        self.is_updating_slider = False
        
        self.time_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.draw_overlays(frame)

    def draw_overlays(self, frame):
        if self.sync_data.empty:
            self.update_current_frame_display(frame)
            return

        if self.overlay_vars["heatmap"].get():
            window_size_frames = int(self.heatmap_window_var.get() * self.fps)
            start_frame = max(0, self.current_frame_idx - window_size_frames)
            
            heatmap_data = self.sync_data.iloc[start_frame:self.current_frame_idx]
            gaze_points = heatmap_data[['gaze x [px]', 'gaze y [px]']].dropna().astype(int).values
            
            if len(gaze_points) > 1:
                intensity_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                for x, y in gaze_points:
                    cv2.circle(intensity_map, (x, y), radius=25, color=50, thickness=-1)
                
                intensity_map = cv2.blur(intensity_map, (91, 91))
                heatmap_color = cv2.applyColorMap(intensity_map, cv2.COLORMAP_JET)
                heatmap_rgba = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
                heatmap_rgba[:, :, 3] = intensity_map
                frame = _overlay_transparent(frame, heatmap_rgba, 0, 0)
        
        frame_data_row = self.sync_data.iloc[min(self.current_frame_idx, len(self.sync_data)-1)]
        if self.overlay_vars["aois"].get():
            for name, positions in self.aoi_positions_per_frame.items():
                coords = positions[self.current_frame_idx].astype(int)
                color = self.aoi_colors.get(name, (255, 0, 255))
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                cv2.putText(frame, name, (coords[0], coords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.overlay_vars["gaze"].get() and pd.notna(frame_data_row['gaze x [px]']):
            px, py = int(frame_data_row['gaze x [px]']), int(frame_data_row['gaze y [px]'])
            cv2.circle(frame, (px, py), GAZE_RADIUS, GAZE_COLOR, GAZE_THICKNESS, cv2.LINE_AA)

        # --- NUOVO: Logica per disegnare la scia dello sguardo ---
        if self.overlay_vars["gaze_path"].get():
            path_length = 10
            start_idx = max(0, self.current_frame_idx - path_length)
            path_data = self.sync_data.iloc[start_idx:self.current_frame_idx]
            gaze_points = path_data[['gaze x [px]', 'gaze y [px]']].dropna().astype(int).values

            if len(gaze_points) > 1:
                for i in range(1, len(gaze_points)):
                    # Applica un gradiente di spessore per l'effetto di dissolvenza
                    thickness = int(np.ceil((i / len(gaze_points)) * (GAZE_THICKNESS + 1)))
                    cv2.line(frame, tuple(gaze_points[i-1]), tuple(gaze_points[i]), GAZE_COLOR, thickness, cv2.LINE_AA)
        # --- FINE BLOCCO ---
        
        history_data = self.sync_data.iloc[max(0, self.current_frame_idx - PUPIL_PLOT_HISTORY):self.current_frame_idx]
        
        if self.overlay_vars["pupil_plot"].get():
            pupil_data = {
                "Left": history_data['pupil diameter left [mm]'].tolist(),
                "Mean": history_data[['pupil diameter left [mm]', 'pupil diameter right [mm]']].mean(axis=1).tolist()
            }
            _draw_pupil_plot(frame, pupil_data, 2, 8, PUPIL_PLOT_WIDTH, PUPIL_PLOT_HEIGHT, (frame.shape[1] - PUPIL_PLOT_WIDTH - 10, 10))

        if self.overlay_vars["frag_plot"].get():
            frag_data = history_data['fragmentation'].tolist()
            y_pos = (PUPIL_PLOT_HEIGHT + 20) if self.overlay_vars["pupil_plot"].get() else 10
            _draw_generic_plot(frame, frag_data, 0, 3000, FRAG_PLOT_WIDTH, FRAG_PLOT_HEIGHT, (frame.shape[1] - FRAG_PLOT_WIDTH - 10, y_pos), "Fragmentation", FRAG_LINE_COLOR, FRAG_BG_COLOR)

        self.update_current_frame_display(frame)

    def update_current_frame_display(self, frame=None):
        if frame is None: 
            if not self.cap or not self.cap.isOpened(): return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, current_frame = self.cap.read()
            if not ret: return
            self.draw_overlays(current_frame)
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        
        canvas_w, canvas_h = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
        if canvas_w > 1 and canvas_h > 1:
            img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

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
            
    def display_table(self, event=None):
        file_name = self.table_selector.get()
        if not file_name: return
        
        self.table_tree.delete(*self.table_tree.get_children())
        if hasattr(self.table_tree, "scroll_x"):
            self.table_tree.scroll_x.destroy()
            self.table_tree.scroll_y.destroy()

        try:
            df = pd.read_csv(self.data_folder / file_name)
            
            self.table_tree["columns"] = list(df.columns)
            for col in df.columns:
                self.table_tree.heading(col, text=col)
                self.table_tree.column(col, width=120)
            
            for index, row in df.head(100).iterrows():
                self.table_tree.insert("", "end", values=list(row.fillna('N/A')))
        except Exception as e:
            messagebox.showerror("Error Reading Table", str(e), parent=self)

    def _clear_views(self):
        self.metadata_tree.delete(*self.metadata_tree.get_children())
        self.table_tree.delete(*self.table_tree.get_children())
        self.table_selector['values'] = []
        self.table_selector.set('')
        self.dataframe_cache = {}
        self.video_canvas.delete("all")
        self.play_pause_btn.config(state=tk.DISABLED)
        self.frame_scale.config(state=tk.DISABLED)