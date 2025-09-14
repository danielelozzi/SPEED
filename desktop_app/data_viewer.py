# desktop_app/data_viewer.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, Toplevel
import json
from pathlib import Path
import pandas as pd
import json
import pydicom
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from moviepy import VideoFileClip, AudioFileClip
import tempfile
import threading
import pygame
import torch
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

# --- NUOVO: Gestione centralizzata dei modelli ---
project_root = Path(__file__).resolve().parent.parent
MODELS_DIR = project_root / 'models'
MODELS_DIR.mkdir(exist_ok=True)

def _get_yolo_device():
    """Determina il dispositivo ottimale per l'inferenza YOLO (CUDA, MPS o CPU)."""
    if torch.cuda.is_available():
        logging.info("CUDA GPU detected. Using 'cuda' for YOLO.")
        return 'cuda'
    # Controlla la disponibilità di MPS e se è stato compilato correttamente
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logging.warning("MPS is available but not built. Falling back to CPU.")
            return 'cpu'
        logging.info("Apple MPS detected. Using 'mps' for YOLO.")
        return 'mps'
    else:
        logging.info("No compatible GPU detected. Using 'cpu' for YOLO.")
        return 'cpu'

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
        self.root = parent
        self.grab_set()

        # Dizionario dei modelli YOLO specifico per questa finestra
        self.YOLO_MODELS = {
            # Detection Models
            'detect_v11': ['yolov11n.pt', 'yolov11s.pt', 'yolov11m.pt', 'yolov11l.pt', 'yolov11x.pt'],
            'detect_v10': ['yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt', 'yolov10b.pt', 'yolov10l.pt', 'yolov10x.pt'],
            'detect_v9': ['yolov9c.pt', 'yolov9e.pt'],
            'detect_v8': ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
            'detect_v5': ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt'],
            'detect_v3': ['yolov3-tiny.pt', 'yolov3.pt'],
            'detect_nas': ['yolo_nas_s.pt', 'yolo_nas_m.pt', 'yolo_nas_l.pt'],
            'detect_world': ['yolov8s-world.pt', 'yolov8m-world.pt', 'yolov8l.pt', 'yolov8x-world.pt'],

            # Segmentation Models
            'segment_v8': ['yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l.pt', 'yolov8x-seg.pt'],
            'segment_sam': ['sam_b.pt', 'sam_l.pt'],
            'segment_fastsam': ['FastSAM-s.pt', 'FastSAM-x.pt'],
            'segment_mobilesam': ['mobile_sam.pt'],

            # Pose Estimation Models
            'pose_v8': ['yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt']
        }
        self.data_folder = None
        self.dataframe_cache = {}
        self.yolo_model_cache = {}
        self.cap = None
        
        # --- MODIFICA: Unificazione gestione dati ---
        self.events_df = pd.DataFrame(columns=['name', 'timestamp [ns]', 'selected', 'source', 'recording id'])
        self.world_ts = pd.DataFrame()
        self.yolo_detections_df = pd.DataFrame()
        self.sync_data = pd.DataFrame()
        self.saved_df = None
        self.saved_yolo_df = None
        self.selected_event_index = None
        self.dragged_event_index = None
        
        self.is_playing = False
        self.total_frames = 0
        self.fps = 30
        
        # --- NUOVO: Gestione Audio ---
        self.audio_clip = None
        self.audio_thread = None
        self.is_muted = tk.BooleanVar(value=True)

        self.video_path = None
        self.is_updating_slider = False
        self.yolo_model = None
        self.is_yolo_live = False
        self.yolo_class_filter = set()
        self.yolo_id_filter = set()

        self.defined_aois = defined_aois if defined_aois else []
        self.aoi_positions_per_frame = {}
        self.aoi_colors = {aoi['name']: tuple(np.random.randint(100, 256, 3).tolist()) for aoi in self.defined_aois}

        # --- MODIFICA: Layout con PanedWindow ---
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Pannello Sinistro (Video e Tabelle) ---
        left_pane_content = tk.Frame(main_pane)
        main_pane.add(left_pane_content, stretch="always")

        self.notebook = ttk.Notebook(left_pane_content)
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

        # --- NUOVO: Pulsante Mute/Unmute ---
        self.mute_btn = tk.Button(video_controls, text="🔇 Unmute", command=self.toggle_mute)
        self.mute_btn.pack(side=tk.LEFT, padx=5)

        self.frame_scale = ttk.Scale(video_controls, from_=0, to=100, orient=tk.HORIZONTAL, command=self.seek_frame, state=tk.DISABLED)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(video_controls, text="Frame: 0 / 0", width=20)
        self.time_label.pack(side=tk.RIGHT)

        self.timeline_canvas = tk.Canvas(self.video_tab, height=80, bg='lightgrey')
        self.timeline_canvas.pack(fill=tk.X, padx=10, pady=5)
        self.timeline_canvas.bind("<Button-1>", self.handle_timeline_click)
        self.timeline_canvas.bind("<B1-Motion>", self.handle_timeline_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.handle_timeline_release)

        table_controls = tk.Frame(self.table_tab)
        table_controls.pack(fill=tk.X, pady=5)
        tk.Label(table_controls, text="Select table to view:").pack(side=tk.LEFT)
        self.table_selector = ttk.Combobox(table_controls, state="readonly")
        self.table_selector.pack(side=tk.LEFT, padx=5)
        self.table_selector.bind("<<ComboboxSelected>>", self.display_table)
        self.table_tree = ttk.Treeview(self.table_tab, show="headings")
        self.table_tree.pack(fill=tk.BOTH, expand=True)

        # --- Pannello Destro (Controlli) ---
        right_panel = tk.Frame(main_pane, width=350, relief=tk.SUNKEN, borderwidth=1)
        right_panel.pack_propagate(False)
        main_pane.add(right_panel)

        load_frame = tk.LabelFrame(right_panel, text="Data Loading")
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(load_frame, text="Load Un-enriched Folder...", command=self.load_unenriched_folder).pack(fill=tk.X, pady=2)
        tk.Button(load_frame, text="Load BIDS Directory...", command=self.load_bids).pack(fill=tk.X, pady=2)
        tk.Button(load_frame, text="Load DICOM File...", command=self.load_dicom).pack(fill=tk.X, pady=2)
        tk.Button(load_frame, text="Load Video Only...", command=self.load_video_only).pack(fill=tk.X, pady=2)
        self.status_label = tk.Label(load_frame, text="Please load a data source.", wraplength=330)
        self.status_label.pack(pady=5)

        event_frame = tk.LabelFrame(right_panel, text="Event Management")
        event_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(event_frame, text="Add Event at Frame", command=self.add_event_at_frame, bg='#c8e6c9').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(event_frame, text="Remove Selected", command=self.remove_selected_event, bg='#ffcdd2').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        overlay_controls_frame = tk.LabelFrame(right_panel, text="Video Overlay Controls")
        overlay_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        self.overlay_vars = {
            "gaze": tk.BooleanVar(value=True), "pupil_plot": tk.BooleanVar(value=False),
            "frag_plot": tk.BooleanVar(value=False), "aois": tk.BooleanVar(value=True), "gaze_path": tk.BooleanVar(value=True), 
            "heatmap": tk.BooleanVar(value=False)
        }
        for key, text in {"gaze": "Show Gaze Point", "gaze_path": "Show Gaze Path", 
                           "pupil_plot": "Show Pupil Plot", "frag_plot": "Show Fragmentation Plot", 
                           "aois": "Show Defined AOIs"}.items():
            tk.Checkbutton(overlay_controls_frame, text=text, variable=self.overlay_vars[key], 
                           command=self.update_current_frame_display).pack(anchor='w')
        
        yolo_run_frame = tk.LabelFrame(right_panel, text="YOLO Analysis")
        yolo_run_frame.pack(fill=tk.X, padx=5, pady=5)
        self.yolo_model_vars = {'detect': tk.StringVar(), 'segment': tk.StringVar(), 'pose': tk.StringVar()}
        self.yolo_model_combos = {}
        for task in ['detect', 'segment', 'pose']:
            tk.Label(yolo_run_frame, text=f"{task.capitalize()} Model:").pack(anchor='w', pady=(5,0))
            combo = ttk.Combobox(yolo_run_frame, textvariable=self.yolo_model_vars[task], state='readonly')
            combo.pack(fill=tk.X)
            self.yolo_model_combos[task] = combo
        self.run_yolo_btn = tk.Button(yolo_run_frame, text="Run YOLO on Video", command=self.run_yolo_analysis, state=tk.DISABLED)
        self.run_yolo_btn.pack(pady=5, fill=tk.X)
        if YOLO is None: self.run_yolo_btn.config(text="YOLO not installed", state=tk.DISABLED)

        yolo_filter_frame = tk.LabelFrame(right_panel, text="YOLO Object Filter")
        yolo_filter_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # --- MODIFICA: Aggiunta colonna per checkbox ---
        self.yolo_filter_tree = ttk.Treeview(yolo_filter_frame, columns=("#1"), show="tree headings")
        self.yolo_filter_tree.heading("#0", text="Object")
        self.yolo_filter_tree.heading("#1", text="Show")
        self.yolo_filter_tree.column("#1", width=50, anchor='center')
        self.yolo_filter_tree.pack(fill=tk.BOTH, expand=True)
        self.yolo_filter_tree.bind('<Button-1>', self.on_yolo_filter_click) # Associazione evento

        metadata_frame = tk.LabelFrame(right_panel, text="File Metadata")
        metadata_frame.pack(fill=tk.X, padx=5, pady=5)
        self.metadata_tree = ttk.Treeview(metadata_frame, columns=("Property", "Value"), show="headings", height=5)
        self.metadata_tree.heading("Property", text="Property"); self.metadata_tree.heading("Value", text="Value")
        self.metadata_tree.column("Property", width=120)
        self.metadata_tree.pack(fill=tk.X, expand=True)

        export_frame = tk.Frame(right_panel)
        export_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        tk.Button(export_frame, text="Export Video...", command=self.export_video_dialog, bg='#ffcc80').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(export_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_yolo_model_options()
        # Inizializza pygame mixer
        pygame.mixer.init()

    def on_heatmap_slider_change(self, value):
        self.heatmap_label_var.set(f"{float(value):.1f} s")
        if self.overlay_vars["heatmap"].get():
            self.update_current_frame_display()

    def on_close(self):
        self.is_playing = False
        if self.audio_thread and self.audio_thread.is_alive():
            pygame.mixer.music.stop()
        if self.cap:
            self.saved_df = self.events_df # Salva lo stato corrente in caso di chiusura
            self.cap.release()
        self.destroy()

    def _load_audio(self):
        """Carica la traccia audio dal file video in memoria."""
        if not self.video_path:
            self.audio_clip = None
            return
        try:
            video = VideoFileClip(str(self.video_path))
            if video.audio:
                self.audio_clip = video.audio
                logging.info("Traccia audio caricata con successo.")
            else:
                self.audio_clip = None
                logging.warning("Il video non contiene una traccia audio.")
        except Exception as e:
            self.audio_clip = None
            logging.error(f"Impossibile caricare l'audio: {e}")
        finally:
            self.update_mute_button_text()

    def toggle_mute(self):
        self.is_muted.set(not self.is_muted.get())
        self.update_mute_button_text()
        if not self.is_muted.get() and self.is_playing:
            self.play_audio()

    def update_mute_button_text(self):
        self.mute_btn.config(text="🔇 Unmute" if self.is_muted.get() else "🔊 Mute")
        if not self.audio_clip:
            self.mute_btn.config(state=tk.DISABLED, text="No Audio")

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
        """Popola i combobox dei modelli YOLO con le opzioni corrette."""
        for task, combo in self.yolo_model_combos.items():
            # --- NUOVA LOGICA: Raccoglie tutti i modelli per un task base (es. 'detect') ---
            all_task_models = []
            for model_key, model_list in self.YOLO_MODELS.items():
                if model_key.startswith(task):
                    all_task_models.extend(model_list)
            
            # Rimuovi duplicati e ordina, aggiungendo l'opzione vuota
            task_models = [""] + sorted(list(set(all_task_models)))
            combo['values'] = task_models
            if self.yolo_model_vars[task].get() not in task_models:
                self.yolo_model_vars[task].set(task_models[0])
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

    def load_video_only(self, video_path_str=None):
        """Carica un singolo file video senza una struttura di cartelle."""
        if not video_path_str:
            video_path_str = filedialog.askopenfilename(
                title="Select a video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
                parent=self
            )

        if not video_path_str: return

        try:
            self.video_path = Path(video_path_str)
            self.data_folder = self.video_path.parent
            self.status_label.config(text=f"Loaded: {self.video_path.name}")
            self._reset_and_load_data()
            video_file = self.video_path
            self._populate_all_views() # Carica tabelle
            self._load_video_and_data(video_file) # Carica video e dati sincronizzati

        except Exception as e:
            messagebox.showerror("Error Loading Video", str(e), parent=self)

    def load_unenriched_folder(self):
        folder_path = filedialog.askdirectory(title="Select the un-enriched data folder")
        if not folder_path: return

        try:
            self.data_folder = Path(folder_path)
            self._reset_and_load_data()
            self.status_label.config(text=f"Loaded: {self.data_folder.name}")
            self._populate_all_views()
            
            video_file = next(self.data_folder.glob('*.mp4'), None)
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
            self._reset_and_load_data()
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
            self._reset_and_load_data()
            ds = pydicom.dcmread(dicom_path, force=True)
            self.status_label.config(text=f"Loaded DICOM: {Path(dicom_path).name}")
            self._populate_all_views()

            for tag in ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'SamplingFrequency']:
                if hasattr(ds, tag):
                    self.metadata_tree.insert("", "end", values=(tag, str(ds[tag].value)))
            
            messagebox.showinfo("Info", "DICOM data loaded. Video player is disabled for this format.", parent=self)

        except Exception as e:
            messagebox.showerror("Error Loading DICOM", str(e), parent=self)

    def _reset_and_load_data(self):
        """Resetta lo stato e carica i dati base come events.csv."""
        self.events_df = pd.DataFrame(columns=['name', 'timestamp [ns]', 'selected', 'source', 'recording id'])
        self.yolo_detections_df = pd.DataFrame()
        self.yolo_filter_tree.delete(*self.yolo_filter_tree.get_children())
        self.selected_event_index = None
        self.is_playing = False

        if not self.data_folder: return

        # Carica events.csv se esiste
        events_path = self.data_folder / 'events.csv'
        if events_path.exists():
            try:
                self.events_df = pd.read_csv(events_path)
                if 'selected' not in self.events_df.columns:
                    self.events_df['selected'] = True
            except Exception as e:
                messagebox.showerror("Error", f"Could not read events.csv:\n{e}", parent=self)

        # Carica world_timestamps.csv se esiste
        world_ts_path = self.data_folder / 'world_timestamps.csv'
        if world_ts_path.exists():
            try:
                self.world_ts = pd.read_csv(world_ts_path)
                if 'frame' not in self.world_ts.columns:
                    self.world_ts['frame'] = self.world_ts.index
            except Exception:
                self.world_ts = pd.DataFrame()


    def _populate_all_views(self):
        self._clear_views()
        if not self.data_folder: return
        
        csv_files = sorted([p.name for p in self.data_folder.glob("*.csv")])
        self.table_selector['values'] = csv_files
        if csv_files:
            self.table_selector.set(csv_files[0])
            self.display_table()

    def _load_video_and_data(self, video_path):
        if not video_path or not video_path.exists():
            self.play_pause_btn.config(state=tk.DISABLED)
            self.run_yolo_btn.config(state=tk.DISABLED)
            self.audio_clip = None
            self.update_mute_button_text()
            return

        self.video_path = video_path
        self._prepare_sync_data()
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30

        # Carica l'audio
        self._load_audio()

        if self.world_ts.empty:
            timestamps_ns = (np.arange(self.total_frames) * (1e9 / self.fps)).astype('int64')
            self.world_ts = pd.DataFrame({'timestamp [ns]': timestamps_ns, 'frame': np.arange(self.total_frames)})
        # Cerca il file nella cartella di output, che si presume sia la parente della cartella dati
        potential_output_dir = self.data_folder.parent
        yolo_cache_path = potential_output_dir / 'yolo_detections_cache.csv'
        if yolo_cache_path.exists():
            try:
                self.yolo_detections_df = pd.read_csv(yolo_cache_path)
                logging.info(f"Loaded YOLO detection data from: {yolo_cache_path}")
                self.process_yolo_data()
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
        self.draw_timeline()

    def draw_overlays(self, frame):
        if self.sync_data.empty and self.yolo_detections_df.empty:
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
        self.draw_yolo_overlays(frame)
        
        # Disegna overlay Eventi
        if not self.world_ts.empty and self.current_frame_idx < len(self.world_ts):
            current_ts = self.world_ts.iloc[self.current_frame_idx]['timestamp [ns]']
            self.draw_event_overlay(frame, current_ts)

        if self.sync_data.empty:
            self.update_current_frame_display(frame)
            return

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

    # --- NUOVE FUNZIONI DALL'EDITOR ---
    def draw_yolo_overlays(self, frame):
        if self.yolo_detections_df.empty: return

        detections_for_frame = self.yolo_detections_df[self.yolo_detections_df['frame_idx'] == self.current_frame_idx]

        # --- MODIFICA: Applica i filtri per nascondere gli elementi deselezionati ---
        # Se il set di filtri non è vuoto, significa che alcuni elementi sono stati deselezionati.
        # Quindi, mostriamo solo gli elementi che sono ancora nei nostri set di filtri.
        # Se un set di filtri è vuoto, significa che tutti gli elementi di quel tipo sono selezionati, quindi non applichiamo quel filtro.
        if self.yolo_class_filter: # Se non è vuoto, applica il filtro
            detections_for_frame = detections_for_frame[detections_for_frame['class_name'].isin(self.yolo_class_filter)].copy()
        
        if self.yolo_id_filter: # Se non è vuoto, applica il filtro
            detections_for_frame = detections_for_frame[detections_for_frame['track_id'].isin(self.yolo_id_filter)].copy()
        # --- FINE MODIFICA ---

        # --- NUOVO: Crea un'immagine di overlay per le maschere ---
        overlay_mask = frame.copy()
        if detections_for_frame.empty: return

        for _, det in detections_for_frame.iterrows():
            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
            color = (0, 255, 255) # Ciano per i box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = f"{det.get('class_name', 'Obj')}:{int(det['track_id'])}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if 'mask_contours' in det and pd.notna(det['mask_contours']):
                try:
                    contour = np.array(json.loads(det['mask_contours'])).astype(np.int32)
                    # Disegna la maschera sull'immagine di overlay separata
                    cv2.fillPoly(overlay_mask, [contour], (0, 255, 0)) # Verde per le maschere
                except Exception: pass

            if 'keypoints' in det and pd.notna(det['keypoints']) and det['keypoints'] != '[]':
                try:
                    keypoints = np.array(json.loads(det['keypoints']))
                    for x, y, conf in keypoints:
                        if conf > 0.5: cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1) # Rosso per i keypoint
                except Exception: pass

        # --- NUOVO: Applica l'overlay delle maschere una sola volta alla fine ---
        cv2.addWeighted(overlay_mask, 0.3, frame, 0.7, 0, dst=frame)

    def draw_event_overlay(self, frame, current_ts):
        if self.events_df.empty: return
        active_events = self.events_df[self.events_df['timestamp [ns]'] <= current_ts]
        if not active_events.empty:
            current_event_name = active_events.iloc[-1]['name']
            cv2.putText(frame, current_event_name, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_timeline(self):
        self.timeline_canvas.delete("all")
        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1: return
        
        color_map = {'default': 'red', 'optional': 'purple', 'manual': 'green'}
        for index, event in self.events_df.iterrows():
            frame_idx = self.get_frame_from_ts(event['timestamp [ns]'])
            x_pos = (frame_idx / self.total_frames) * canvas_width
            base_color = color_map.get(event.get('source', 'manual'), 'black')
            final_color = "blue" if index == self.selected_event_index else base_color
            self.timeline_canvas.create_line(x_pos, 10, x_pos, 50, fill=final_color, width=3 if final_color=="blue" else 2, tags=f"event_{index}")
            self.timeline_canvas.create_text(x_pos, 60, text=event['name'], anchor=tk.N, fill=final_color, tags=f"event_{index}")

        cursor_x = (self.current_frame_idx / self.total_frames) * canvas_width
        self.timeline_canvas.create_line(cursor_x, 0, cursor_x, 80, fill='dark green', width=2)

    def get_frame_from_ts(self, ts):
        if self.world_ts.empty: return 0
        match_index = (self.world_ts['timestamp [ns]'] - ts).abs().idxmin()
        return self.world_ts.loc[match_index, 'frame']

    def get_event_at_pos(self, x):
        canvas_width = self.timeline_canvas.winfo_width()
        for index, event in self.events_df.iterrows():
            if abs(x - ((self.get_frame_from_ts(event['timestamp [ns]']) / self.total_frames) * canvas_width)) < 5:
                return index
        return None

    def handle_timeline_click(self, event):
        clicked_event = self.get_event_at_pos(event.x)
        self.selected_event_index = clicked_event
        self.dragged_event_index = clicked_event
        if clicked_event is None:
            self.update_frame((event.x / self.timeline_canvas.winfo_width()) * self.total_frames)
        self.draw_timeline()

    def handle_timeline_drag(self, event):
        if self.dragged_event_index is not None:
            canvas_width = self.timeline_canvas.winfo_width()
            new_frame = max(0, min(int((event.x / canvas_width) * self.total_frames), self.total_frames - 1))
            if new_frame < len(self.world_ts):
                self.events_df.loc[self.dragged_event_index, 'timestamp [ns]'] = self.world_ts.iloc[new_frame]['timestamp [ns]']
                self.update_frame(new_frame)
            
    def handle_timeline_release(self, event):
        if self.dragged_event_index is not None:
            self.dragged_event_index = None
            self.events_df.sort_values('timestamp [ns]', inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.selected_event_index = None
            self.draw_timeline()

    def add_event_at_frame(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please load a video first.", parent=self)
            return
        name = simpledialog.askstring("Add Event", "Enter event name:", parent=self)
        if name and self.current_frame_idx < len(self.world_ts):
            ts = self.world_ts.iloc[self.current_frame_idx]['timestamp [ns]']
            new_row = {'name': name, 'timestamp [ns]': ts, 'selected': True, 'source': 'manual', 'recording id': 'rec_001'}
            self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
            self.events_df.sort_values('timestamp [ns]', inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.draw_timeline()

    def remove_selected_event(self):
        if self.selected_event_index is None:
            messagebox.showinfo("Info", "Click on an event on the timeline to select it first.", parent=self)
            return
        event_name = self.events_df.loc[self.selected_event_index, 'name']
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to remove the event '{event_name}'?", parent=self):
            self.events_df.drop(self.selected_event_index, inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.selected_event_index = None
            self.draw_timeline()

    def save_and_close(self):
        self.is_playing = False
        cols_to_save = ['name', 'timestamp [ns]', 'recording id', 'selected', 'source']
        self.saved_df = self.events_df[[col for col in cols_to_save if col in self.events_df.columns]]

        # --- NUOVO: Salva il DataFrame YOLO filtrato ---
        if not self.yolo_detections_df.empty:
            shown_instances = self.yolo_detections_df['class_name'].isin(self.yolo_class_filter or self.yolo_detections_df['class_name'].unique()) & \
                              self.yolo_detections_df['track_id'].isin(self.yolo_id_filter or self.yolo_detections_df['track_id'].unique())
            self.saved_yolo_df = self.yolo_detections_df[shown_instances].copy()

        self.on_close()
    # --- FINE NUOVE FUNZIONI ---

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
        if self.is_playing:
            self.play_pause_btn.config(text="❚❚ Pause")
            self.play_video()
            self.play_audio()
        else:
            self.play_pause_btn.config(text="▶ Play")
            if self.audio_thread and self.audio_thread.is_alive():
                pygame.mixer.music.stop()

    def play_audio(self):
        """Riproduce l'audio in un thread separato."""
        if self.is_playing and self.audio_clip and not self.is_muted.get():
            
            def audio_worker():
                try:
                    start_time = self.current_frame_idx / self.fps
                    temp_audio_file = self.audio_clip.subclip(start_time).to_soundarray(fps=44100)
                    pygame.mixer.music.load(pygame.sndarray.make_sound(temp_audio_file))
                    pygame.mixer.music.play()
                except Exception as e:
                    logging.error(f"Errore durante la riproduzione audio: {e}")

            self.audio_thread = threading.Thread(target=audio_worker, daemon=True)
            self.audio_thread.start()

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

    # --- NUOVE FUNZIONI YOLO ---
    def run_yolo_analysis(self):
        if YOLO is None: return
        
        selected_models = {task: name for task, name in self.yolo_model_vars.items() if name.get()}
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one YOLO model to run.", parent=self)
            return

        try:
            self.yolo_models = {}
            for task, model_name_var in selected_models.items(): # model_name_var is a StringVar
                model_path = MODELS_DIR / model_name_var.get()
                self.yolo_models[task] = YOLO(model_path)
                logging.info(f"Loaded {task} model: {model_name_var.get()}")
        except Exception as e:
            logging.error(f"Failed to load/download one or more YOLO models: {e}")
            messagebox.showerror("YOLO Error", f"Failed to load model: {e}", parent=self)
            return

        self.run_yolo_btn.config(text="Analyzing...", state=tk.DISABLED)
        self.root.update_idletasks()

        import threading
        threading.Thread(target=self._yolo_thread_worker, daemon=True).start()

    def _yolo_thread_worker(self):
        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        detections = []
        effective_device = _get_yolo_device()
        
        pbar_desc = f"YOLO Tracking on {effective_device.upper()}"

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            try:
                # Esegui tutti i modelli selezionati
                all_results = {}
                for task, model in self.yolo_models.items():
                    # --- NUOVO: Logica per forzare la CPU sui modelli di posa su MPS ---
                    device_for_task = effective_device
                    if task == 'pose' and effective_device == 'mps':
                        logging.warning("Pose model on Apple MPS detected. Forcing CPU to avoid known bugs.")
                        device_for_task = 'cpu'
                    all_results[task] = model.track(frame, persist=True, verbose=False, device=device_for_task)

            except Exception as e:
                # Fallback alla CPU se l'accelerazione GPU/MPS fallisce
                if effective_device != 'cpu':
                    logging.warning(f"Inference on '{effective_device}' failed: {e}. Falling back to CPU.")
                    effective_device = 'cpu'
                    pbar_desc = f"YOLO Tracking on {effective_device.upper()} (Fallback)"
                    for task, model in self.yolo_models.items():
                        all_results[task] = model.track(frame, persist=True, verbose=False, device=effective_device)
                else:
                    raise e # Se fallisce anche sulla CPU, l'errore è più grave

            # --- NUOVA LOGICA DI UNIONE RISULTATI ---
            frame_detections = {}

            for task, results in all_results.items():
                res = results[0]
                if res.boxes is None or res.boxes.id is None: continue

                for i, box in enumerate(res.boxes):
                    track_id = int(box.id[0])
                    
                    if track_id not in frame_detections:
                        frame_detections[track_id] = {'frame_idx': frame_idx, 'track_id': track_id}

                    if 'class_id' not in frame_detections[track_id]:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_models[task].names[class_id]
                        xyxy = box.xyxy[0].cpu().numpy()
                        frame_detections[track_id].update({
                            'class_id': class_id, 'class_name': class_name,
                            'x1': xyxy[0], 'y1': xyxy[1], 'x2': xyxy[2], 'y2': xyxy[3]
                        })

                    if task == 'segment' and res.masks and i < len(res.masks.xy):
                        frame_detections[track_id]['mask_contours'] = json.dumps(res.masks.xy[i].tolist())

                    if task == 'pose' and res.keypoints and i < len(res.keypoints.xy):
                        kpts_xy = res.keypoints.xy[i].cpu().numpy()
                        kpts_conf_tensor = res.keypoints.conf[i] if res.keypoints.conf is not None else torch.ones(len(kpts_xy))
                        kpts_conf = kpts_conf_tensor.cpu().numpy()[:, None]
                        kpts_with_conf = np.hstack((kpts_xy, kpts_conf))
                        frame_detections[track_id]['keypoints'] = json.dumps(kpts_with_conf.tolist())

            detections.extend(frame_detections.values())
            # --- FINE NUOVA LOGICA ---

        cap.release()
        self.yolo_detections_df = pd.DataFrame(detections)
        # --- CORREZIONE: Controlla se la finestra esiste ancora prima di aggiornare la UI ---
        if self.winfo_exists():
            self.after(0, self.process_yolo_data)
            self.after(0, lambda: self.run_yolo_btn.config(text="Run YOLO on Video", state=tk.NORMAL))
        # --- FINE CORREZIONE ---

    def process_yolo_data(self):
        if self.yolo_detections_df.empty: return
        self.detected_yolo_items = self.yolo_detections_df.groupby('class_name')['track_id'].unique().apply(list).to_dict()
        self.yolo_filter_tree.delete(*self.yolo_filter_tree.get_children())
        for class_name, ids in sorted(self.detected_yolo_items.items()):
            # --- MODIFICA: Inserimento con valore per checkbox ---
            class_node = self.yolo_filter_tree.insert("", "end", text=f"Class: {class_name}", open=True, values=("☑",), tags=('class', class_name))
            for track_id in sorted(ids):
                self.yolo_filter_tree.insert(class_node, "end", text=f"  ID: {track_id}", values=("☑",), tags=('id', track_id))
        self.update_frame(self.current_frame_idx)

    def on_yolo_filter_click(self, event):
        # --- MODIFICA: Logica per gestire il click sulla colonna checkbox ---
        item_id = self.yolo_filter_tree.identify_row(event.y)
        column = self.yolo_filter_tree.identify_column(event.x)
        
        # Agisci solo se si clicca sulla colonna "Show"
        if not item_id or column != "#1":
            return

        current_val = self.yolo_filter_tree.item(item_id, 'values')[0]
        new_val = "☐" if current_val == "☑" else "☑"
        self.yolo_filter_tree.set(item_id, column, new_val)

        tags = self.yolo_filter_tree.item(item_id, 'tags')
        if tags and tags[0] == 'class':
            # Se si clicca su una classe, aggiorna tutti i suoi figli (ID)
            for child_id in self.yolo_filter_tree.get_children(item_id):
                self.yolo_filter_tree.set(child_id, column, new_val)
        
        self.update_yolo_filters_and_redraw()

    def on_yolo_filter_click_old(self, event):
        item_id = self.yolo_filter_tree.identify_row(event.y)
        if not item_id: return
        is_checked = not self.yolo_filter_tree.item(item_id, 'values')[0]
        self.yolo_filter_tree.item(item_id, values=(is_checked,))
        tags = self.yolo_filter_tree.item(item_id, 'tags')
        if tags and tags[0] == 'class':
            for child_id in self.yolo_filter_tree.get_children(item_id):
                self.yolo_filter_tree.item(child_id, values=(is_checked,))
        
        # Aggiorna i filtri e ridisegna
        self.update_yolo_filters_and_redraw()

    def update_yolo_filters_and_redraw(self):
        """Aggiorna i set di filtri in base allo stato delle checkbox e ridisegna il frame."""
        self.yolo_class_filter = set()
        self.yolo_id_filter = set()
        all_classes = set()
        all_ids = set()

        for class_node in self.yolo_filter_tree.get_children(''):
            class_name = self.yolo_filter_tree.item(class_node, 'tags')[1]
            all_classes.add(class_name)
            # --- MODIFICA: Controlla il testo del checkbox ---
            if self.yolo_filter_tree.item(class_node, 'values')[0] == "☑":
                self.yolo_class_filter.add(class_name)
                for id_node in self.yolo_filter_tree.get_children(class_node):
                    track_id = int(self.yolo_filter_tree.item(id_node, 'tags')[1] )
                    all_ids.add(track_id)
                    if self.yolo_filter_tree.item(id_node, 'values')[0] == "☑":
                        self.yolo_id_filter.add(track_id)

        # Se tutti gli elementi sono selezionati, il set di filtri dovrebbe essere vuoto per non filtrare nulla.
        if len(self.yolo_class_filter) == len(all_classes):
            self.yolo_class_filter.clear()
        if len(self.yolo_id_filter) == len(all_ids):
            self.yolo_id_filter.clear()

        self.update_frame(self.current_frame_idx)

    def export_video_dialog(self):
        dialog = Toplevel(self); dialog.title("Export Video Options"); dialog.geometry("400x150"); dialog.transient(self); dialog.grab_set()
        tk.Label(dialog, text="Configure video export:", pady=10).pack()
        include_audio_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dialog, text="Include Audio from Original Video", variable=include_audio_var).pack(pady=5)
        def on_export():
            output_path = filedialog.asksaveasfilename(title="Save Video As", defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
            if not output_path: return
            dialog.destroy()
            self.export_video(output_path, include_audio_var.get())
        tk.Button(dialog, text="Export", command=on_export, font=('Helvetica', 10, 'bold')).pack(pady=10)

    def export_video(self, output_path, include_audio):
        temp_video_path = Path(output_path).with_suffix('.temp.mp4')
        original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); original_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (original_w, original_h))
        
        progress_win = Toplevel(self); progress_win.title("Exporting Video"); progress_win.geometry("300x80")
        tk.Label(progress_win, text="Export in progress...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_win, orient='horizontal', length=280, mode='determinate', maximum=self.total_frames)
        progress_bar.pack(pady=5)

        try:
            for frame_idx in range(self.total_frames):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if not ret: break
                self.draw_yolo_overlays(frame)
                if not self.world_ts.empty and frame_idx < len(self.world_ts):
                    current_ts = self.world_ts.iloc[frame_idx]['timestamp [ns]']
                    self.draw_event_overlay(frame, current_ts)
                writer.write(frame)
                progress_bar['value'] = frame_idx + 1
                progress_win.update_idletasks()
        finally:
            writer.release()
            progress_win.destroy()

        if include_audio and self.video_path:
            try:
                video_clip = VideoFileClip(str(temp_video_path)); audio_clip = AudioFileClip(str(self.video_path))
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac', logger=None)
                final_clip.close(); audio_clip.close(); video_clip.close()
                temp_video_path.unlink()
                messagebox.showinfo("Success", f"Video exported with audio to:\n{output_path}", parent=self)
            except Exception as e:
                temp_video_path.rename(output_path)
                messagebox.showwarning("Audio Error", f"Could not add audio: {e}\nVideo saved without audio.", parent=self)
        else:
            temp_video_path.rename(output_path)
            messagebox.showinfo("Success", f"Video exported without audio to:\n{output_path}", parent=self)