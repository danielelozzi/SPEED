# desktop_app/data_viewer.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
from pathlib import Path
import pandas as pd
import json
import pydicom
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import tempfile
import logging

# Importa le funzioni di conversione e le utility di disegno
from src.speed_analyzer.analysis_modules.bids_converter import load_from_bids
from src.speed_analyzer.analysis_modules.dicom_converter import load_from_dicom
from src.speed_analyzer.analysis_modules.video_generator import (
    _draw_pupil_plot, _draw_generic_plot, PUPIL_PLOT_HISTORY, PUPIL_PLOT_WIDTH,
    PUPIL_PLOT_HEIGHT, FRAG_PLOT_HISTORY, FRAG_PLOT_WIDTH, FRAG_PLOT_HEIGHT,
    FRAG_LINE_COLOR, FRAG_BG_COLOR, GAZE_COLOR, GAZE_RADIUS, GAZE_THICKNESS,
    _overlay_transparent
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

        # Dizionario dei modelli YOLO specifico per questa finestra
        self.YOLO_MODELS = {
            'detect': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
            'detect_custom': ['yolov8s-world.pt', 'yolov8s-worldv2.pt', 'yolov8m-world.pt', 'yolov8x-world.pt', 'yolov8x-worldv2.pt'],
            'segment': ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt'],
            'pose': ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt', 'yolov8x-pose-p6.pt']
        }
        self.data_folder = None
        self.dataframe_cache = {}
        self.cap = None
        # NUOVO: Cache per i dati YOLO
        self.yolo_detections_df = pd.DataFrame()
        self.sync_data = pd.DataFrame()
        self.is_playing = False
        self.total_frames = 0
        self.fps = 30
        self.is_updating_slider = False
        self.yolo_model = None
        self.is_yolo_live = False

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
            "gaze": tk.BooleanVar(value=True), "pupil_plot": tk.BooleanVar(value=False),
            "frag_plot": tk.BooleanVar(value=False), "aois": tk.BooleanVar(value=True), "gaze_path": tk.BooleanVar(value=True), 
            "heatmap": tk.BooleanVar(value=False), 
            "segmentation": tk.BooleanVar(value=True), # NUOVO
            "pose": tk.BooleanVar(value=True) # NUOVO
        }
        for key, text in {"gaze": "Show Gaze Point", "gaze_path": "Show Gaze Path", 
                           "pupil_plot": "Show Pupil Plot", "frag_plot": "Show Fragmentation Plot", 
                           "aois": "Show Defined AOIs", "segmentation": "Show Segmentation", 
                           "pose": "Show Pose Estimation"}.items():
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

        # --- NUOVO: Interfaccia di controllo YOLO ---
        yolo_live_frame = tk.LabelFrame(left_panel, text="YOLO Live Analysis")
        yolo_live_frame.pack(fill=tk.X, padx=5, pady=5)

        self.yolo_task_var = tk.StringVar(value='detect')
        self.yolo_model_var = tk.StringVar()
        self.yolo_classes_var = tk.StringVar()

        tk.Label(yolo_live_frame, text="Task:").pack(anchor='w')
        self.yolo_task_combo = ttk.Combobox(yolo_live_frame, textvariable=self.yolo_task_var, values=list(self.YOLO_MODELS.keys()), state='readonly', width=25)
        self.yolo_task_combo.pack(pady=(0,5), fill=tk.X)
        self.yolo_task_combo.bind('<<ComboboxSelected>>', self.update_yolo_model_options)

        tk.Label(yolo_live_frame, text="Model:").pack(anchor='w')
        self.yolo_model_combo = ttk.Combobox(yolo_live_frame, textvariable=self.yolo_model_var, state='readonly', width=25)
        self.yolo_model_combo.pack(fill=tk.X)

        tk.Label(yolo_live_frame, text="Custom Classes:").pack(anchor='w', pady=(5,0))
        self.yolo_classes_entry = tk.Entry(yolo_live_frame, textvariable=self.yolo_classes_var, width=28, state=tk.DISABLED)
        self.yolo_classes_entry.pack(fill=tk.X)
        self.add_placeholder(self.yolo_classes_entry, "person, car, dog")

        self.run_yolo_btn = tk.Button(yolo_live_frame, text="Run Live YOLO", command=self.run_live_yolo, state=tk.DISABLED)
        self.run_yolo_btn.pack(pady=5, fill=tk.X)

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
        self.update_yolo_model_options()

    def on_heatmap_slider_change(self, value):
        self.heatmap_label_var.set(f"{float(value):.1f} s")
        if self.overlay_vars["heatmap"].get():
            self.update_current_frame_display()

    def on_close(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        self.destroy()

    def add_placeholder(self, entry_widget, placeholder_text):
        """Aggiunge un testo di placeholder a un widget Entry di Tkinter."""
        entry_widget.insert(0, placeholder_text)
        entry_widget.config(fg='grey')

        def on_focus_in(event):
            if entry_widget.cget('fg') == 'grey':
                entry_widget.delete(0, "end")
                entry_widget.config(fg='black')

        def on_focus_out(event):
            if not entry_widget.get():
                entry_widget.insert(0, placeholder_text)
                entry_widget.config(fg='grey')

        entry_widget.bind("<FocusIn>", on_focus_in)
        entry_widget.bind("<FocusOut>", on_focus_out)

    def update_yolo_model_options(self, event=None):
        selected_task = self.yolo_task_var.get()
        available_models = self.YOLO_MODELS.get(selected_task, [])
        self.yolo_model_combo['values'] = available_models
        if available_models:
            self.yolo_model_var.set(available_models[0])
        else:
            self.yolo_model_var.set('')
        is_custom_detect_task = (selected_task == 'detect_custom')
        self.yolo_classes_entry.config(state=tk.NORMAL if is_custom_detect_task else tk.DISABLED)

    def run_live_yolo(self):
        task = self.yolo_task_var.get()
        model_name = self.yolo_model_var.get()
        custom_classes_str = self.yolo_classes_var.get().strip()
        custom_classes = [cls.strip() for cls in custom_classes_str.split(',') if cls.strip()] if custom_classes_str and custom_classes_str != "person, car, dog" else None

        self.status_label.config(text=f"Loading YOLO model: {model_name}...")
        self.update()

        try:
            if task == 'detect_custom' and custom_classes:
                base_model = YOLO(model_name)
                base_model.set_classes(custom_classes)
                temp_dir = Path(tempfile.gettempdir())
                custom_model_path = temp_dir / f"yolo_world_custom_viewer_{'_'.join(custom_classes)}.pt"
                base_model.save(custom_model_path)
                self.yolo_model = YOLO(custom_model_path)
            else:
                self.yolo_model = YOLO(model_name)
            
            self.is_yolo_live = True
            self.status_label.config(text=f"YOLO model '{model_name}' loaded. Live analysis is ON.")
            self.update_frame(self.current_frame_idx) # Force refresh
        except Exception as e:
            self.is_yolo_live = False
            self.yolo_model = None
            messagebox.showerror("YOLO Error", f"Failed to load model: {e}", parent=self)

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

        # --- NUOVO: Caricamento dei dati di detection YOLO ---
        # Cerca il file nella cartella di output, che si presume sia la parente della cartella dati
        potential_output_dir = self.data_folder.parent
        yolo_cache_path = potential_output_dir / 'yolo_detections_cache.csv'
        if yolo_cache_path.exists():
            try:
                self.yolo_detections_df = pd.read_csv(yolo_cache_path)
                logging.info(f"Loaded YOLO detection data from: {yolo_cache_path}")
            except Exception as e:
                logging.error(f"Failed to load YOLO cache file: {e}")
                self.yolo_detections_df = pd.DataFrame()
        
        self._prepare_aoi_positions()
        
        self.frame_scale.config(to=self.total_frames - 1, state=tk.NORMAL)
        self.play_pause_btn.config(state=tk.NORMAL)
        self.run_yolo_btn.config(state=tk.NORMAL)
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
        # Esegui l'inferenza YOLO live se il modello è attivo
        if hasattr(self, 'is_yolo_live') and self.is_yolo_live and hasattr(self, 'yolo_model') and self.yolo_model is not None:
            try:
                # Usiamo track per coerenza, anche se qui non sfruttiamo il tracking tra frame
                results = self.yolo_model.track(frame, persist=False, verbose=False)
                frame = results[0].plot() # Sovrascrivi il frame con quello annotato da YOLO
            except Exception as e:
                print(f"Errore durante l'inferenza YOLO live: {e}")

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
        
        # --- MODIFICATO: Disegno dei risultati YOLO (Segmentazione e Posa) ---
        if not self.yolo_detections_df.empty:
            detections_for_frame = self.yolo_detections_df[self.yolo_detections_df['frame_idx'] == self.current_frame_idx]
            
            if not detections_for_frame.empty:
                # Disegna le maschere di segmentazione
                if self.overlay_vars["segmentation"].get() and 'mask_contours' in detections_for_frame.columns:
                    overlay_mask = np.zeros_like(frame, dtype=np.uint8)
                    for _, det in detections_for_frame.iterrows():
                        if pd.notna(det['mask_contours']):
                            try:
                                # La colonna ora contiene una stringa JSON di un singolo contorno
                                contour = json.loads(det['mask_contours'])
                                color = self.aoi_colors.get(det.get('track_id', 0), (0, 255, 0)) # Colore casuale per maschera
                                contour_np = np.array(contour, dtype=np.int32)
                                cv2.fillPoly(overlay_mask, [contour_np], color)
                            except (json.JSONDecodeError, TypeError):
                                continue # Ignora se il formato non è corretto
                    frame = cv2.addWeighted(frame, 1.0, overlay_mask, 0.4, 0)

                # Disegna gli scheletri della posa
                if self.overlay_vars["pose"].get() and 'keypoints' in detections_for_frame.columns:
                    # Definisci le connessioni dello scheletro (es. COCO)
                    skeleton_connections = [
                        (0, 1), (0, 2), (1, 3), (2, 4), # Testa
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Corpo
                        (11, 12), (5, 11), (6, 12), # Spalle
                        (11, 13), (13, 15), (12, 14), (14, 16) # Braccia
                    ]
                    for _, det in detections_for_frame.iterrows():
                        if pd.notna(det['keypoints']):
                            try:
                                keypoints = np.array(json.loads(det['keypoints']))
                                for i, (x, y, conf) in enumerate(keypoints):
                                    if conf > 0.5: cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1) # Disegna solo se confident
                                for start_idx, end_idx in skeleton_connections:
                                    if start_idx < len(keypoints) and end_idx < len(keypoints) and keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                                        start_point = tuple(keypoints[start_idx][:2])
                                        end_point = tuple(keypoints[end_idx][:2])
                                        cv2.line(frame, start_point, end_point, (0, 255, 0), 1)
                            except (json.JSONDecodeError, TypeError, IndexError):
                                continue

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
        self.run_yolo_btn.config(state=tk.DISABLED)