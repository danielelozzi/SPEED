import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import traceback
import shutil
import json
import pandas as pd
import cv2
import logging
import time
import sys
import webbrowser
import threading
from PIL import Image, ImageTk
from src.speed_analyzer.analysis_modules.bids_converter import convert_to_bids, load_from_bids
from ultralytics import YOLO
from src.speed_analyzer.analysis_modules.dicom_converter import convert_to_dicom, load_from_dicom 
from desktop_app.data_viewer import DataViewerWindow

# Importazione della libreria LSL
try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    StreamInfo = None
    StreamOutlet = None

# --- NUOVO: Gestione centralizzata dei modelli ---
project_root = Path(__file__).resolve().parent.parent
MODELS_DIR = project_root / 'models'
MODELS_DIR.mkdir(exist_ok=True)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from desktop_app.interactive_video_editor import InteractiveVideoEditor
from desktop_app.aoi_editor import AoiEditor
from desktop_app.manual_aoi_editor import ManualAoiEditor
from desktop_app.marker_surface_editor import MarkerSurfaceEditor
from desktop_app.realtime_qr_aoi_editor import RealtimeQRAoiEditor
from desktop_app.qr_surface_editor import QRSurfaceEditor
from device_converter_window import DeviceConverterWindow
from src.speed_analyzer.nsi_calculator import NsiCalculatorWindow
from src.speed_analyzer import run_full_analysis
from src.speed_analyzer import _prepare_working_directory # Importazione diretta
from src.speed_analyzer.analysis_modules import video_generator # Importazione diretta
from src.speed_analyzer.analysis_modules import speed_script_events # NUOVO: Import per la generazione dei grafici
from src.speed_analyzer.analysis_modules.realtime_analyzer import RealtimeNeonAnalyzer


# --- NUOVA CLASSE PER GESTIRE IL PONTE LSL ---
class LSLBridge:
    """
    Gestisce la creazione e l'invio di dati attraverso stream LSL
    in un thread separato.
    """
    def __init__(self, analyzer: RealtimeNeonAnalyzer):
        if not StreamOutlet:
            raise ImportError("Libreria `pylsl` non trovata. Impossibile avviare lo streaming LSL.")
        
        self.analyzer = analyzer
        self.is_running = False
        self.thread = threading.Thread(target=self._run_bridge, daemon=True)
        
        # Gli outlet verranno creati all'interno del thread
        self.outlet_gaze = None
        self.outlet_video = None
        self.outlet_event = None
        
        print("LSL Bridge inizializzato.")

    def start(self):
        self.is_running = True
        self.thread.start()
        print("LSL Bridge thread avviato.")

    def stop(self):
        self.is_running = False
        print("Fermando LSL Bridge...")

    def push_event(self, event_name: str):
        if self.outlet_event and self.is_running:
            try:
                self.outlet_event.push_sample([event_name])
                print(f"--> Inviato evento LSL: {event_name}")
            except Exception as e:
                print(f"Errore durante l'invio dell'evento LSL: {e}")

    def _run_bridge(self):
        """
        Ciclo principale che recupera dati dall'analizzatore e li invia a LSL.
        """
        # --- Setup Stream Gaze ---
        info_gaze = StreamInfo('PupilNeonGaze', 'Gaze', 3, 120, 'float32', 'PupilNeonGazeID1')
        info_gaze.desc().append_child_value("manufacturer", "Pupil Labs")
        channels_gaze = info_gaze.desc().append_child("channels")
        channels_gaze.append_child("channel").append_child_value("label", "x_position_px")
        channels_gaze.append_child("channel").append_child_value("label", "y_position_px")
        channels_gaze.append_child("channel").append_child_value("label", "pupil_diameter_mm")
        self.outlet_gaze = StreamOutlet(info_gaze)

        # --- Setup Stream Eventi ---
        info_event = StreamInfo('GUI_Events', 'Markers', 1, 0, 'string', 'GUI_EventID1')
        self.outlet_event = StreamOutlet(info_event)

        print("Stream LSL (Gaze, Events) creati. In attesa di dati video...")

        while self.is_running:
            # Usa gli ultimi dati già recuperati dal thread principale della GUI
            gaze_datum = self.analyzer.last_gaze
            scene_frame = self.analyzer.last_scene_frame

            if gaze_datum:
                sample = [
                    gaze_datum.x, gaze_datum.y,
                    gaze_datum.pupil_diameter_mm if hasattr(gaze_datum, 'pupil_diameter_mm') else 0.0
                ]
                self.outlet_gaze.push_sample(sample, gaze_datum.timestamp_unix_seconds)

            if scene_frame:
                frame_image, frame_ts = scene_frame
                
                if self.outlet_video is None: # Inizializza al primo frame
                    h, w, channels = frame_image.shape
                    video_channel_count = h * w * channels
                    info_video = StreamInfo('PupilNeonVideo', 'Video', video_channel_count, 30, 'uint8', 'PupilNeonVideoID1')
                    info_video.desc().append_child_value("width", str(w))
                    info_video.desc().append_child_value("height", str(h))
                    info_video.desc().append_child_value("color_format", "RGB")
                    self.outlet_video = StreamOutlet(info_video)
                    print(f"Video stream LSL inizializzato con dimensioni {w}x{h}.")

                self.outlet_video.push_sample(frame_image.flatten(), frame_ts)

            time.sleep(1 / 200) # Dormi per un breve periodo per non sovraccaricare la CPU
        
        print("Thread LSL Bridge terminato.")

class RealtimeDisplayWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Real-time Neon Stream")
        self.geometry("1280x950")

        self.analyzer = None 
        self.is_running = False
        self.is_paused_for_drawing = False
        
        # --- NUOVO: Filtri per la visualizzazione YOLO ---
        self.yolo_class_filter = set() # Classi da mostrare (se vuoto, mostra tutto)
        self.yolo_id_filter = set()    # ID da mostrare (se vuoto, mostra tutto)
        self.detected_yolo_items = {} # Cache per {class_name: [id1, id2]}

        # --- NUOVO: Variabile per il ponte LSL ---
        self.lsl_bridge = None

        self.canvas = tk.Canvas(self, width=1280, height=720, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.temp_rect_id = None
        
        # --- NUOVO: PanedWindow per controlli e status ---
        bottom_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        bottom_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        main_control_frame = tk.Frame(bottom_pane, pady=10)
        bottom_pane.add(main_control_frame, stretch="always")

        status_and_yolo_frame = tk.Frame(bottom_pane, padx=10)
        bottom_pane.add(status_and_yolo_frame)

        # --- MODIFICA: Logica di connessione e avvio in due fasi ---
        connect_frame = tk.Frame(main_control_frame)
        connect_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        self.connect_button = tk.Button(connect_frame, text="Connect to Device", command=self.connect_to_device, font=('Helvetica', 10, 'bold'), bg='#a5d6a7')
        self.connect_button.pack(fill=tk.X, expand=True)
        self.start_analysis_button = tk.Button(connect_frame, text="Start Analysis", command=self.start_stream, font=('Helvetica', 10, 'bold'), bg='#c8e6c9', state=tk.DISABLED)
        self.start_analysis_button.pack(fill=tk.X, expand=True, pady=(5,0))

        record_frame = tk.LabelFrame(main_control_frame, text="Controls", padx=10, pady=10)
        record_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        
        self.record_button = tk.Button(record_frame, text="Start Recording", command=self.toggle_recording, font=('Helvetica', 10, 'bold'), bg='#c8e6c9', state=tk.DISABLED)
        self.record_button.pack(pady=(0,5))
        
        # --- NUOVO: Checkbox per l'audio nella registrazione ---
        self.include_audio_var = tk.BooleanVar(value=True)
        tk.Checkbutton(record_frame, text="Include Audio", variable=self.include_audio_var).pack(pady=(0,5))

        self.event_name_entry = tk.Entry(record_frame, width=25)
        self.event_name_entry.pack(pady=5)
        self.event_name_entry.insert(0, "New Event")
        
        self.add_event_button = tk.Button(record_frame, text="Add Event", command=self.add_event, state=tk.DISABLED)
        self.add_event_button.pack(pady=5)
        
        # --- NUOVA CHECKBOX PER LSL ---
        self.lsl_stream_var = tk.BooleanVar(value=False)
        lsl_check = tk.Checkbutton(record_frame, text="Enable LSL Stream", variable=self.lsl_stream_var)
        lsl_check.pack(pady=5)
        if StreamInfo is None:
            lsl_check.config(text="Enable LSL Stream (pylsl not found)", state=tk.DISABLED)

        aoi_frame = tk.LabelFrame(main_control_frame, text="AOI Management", padx=10, pady=10)
        aoi_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        self.aoi_listbox = tk.Listbox(aoi_frame, height=4)
        self.aoi_listbox.pack(side=tk.LEFT, fill=tk.Y)
        aoi_btn_frame = tk.Frame(aoi_frame)
        aoi_btn_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        tk.Button(aoi_btn_frame, text="Add AOI", command=self.prepare_to_draw_aoi).pack()
        tk.Button(aoi_btn_frame, text="Remove", command=self.remove_selected_aoi).pack()
        tk.Button(aoi_btn_frame, text="Add QR AOI", command=self.add_qr_aoi_dialog).pack()

        # --- MODIFICA: Frame per i controlli YOLO in tempo reale ---
        self.yolo_config_frame = tk.LabelFrame(main_control_frame, text="YOLO Real-Time Controls", padx=10, pady=10)
        self.yolo_config_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        
        # --- MODIFICA: Selezione Multi-Modello per Real-time ---
        self.rt_yolo_model_vars = {
            'detect': tk.StringVar(), 'segment': tk.StringVar(), 'pose': tk.StringVar(),
            'reid': tk.StringVar(), 'detect_world': tk.StringVar(), 'obb': tk.StringVar()
        }
        self.rt_yolo_model_combos = {}

        for task, label in [('detect', 'Detection Model:'), ('segment', 'Segmentation Model:'), 
                            ('pose', 'Pose Model:'), ('obb', 'OBB Model:'), ('reid', 'Re-ID Model:'), 
                            ('detect_world', 'World Model:')]:
            tk.Label(self.yolo_config_frame, text=label).pack(anchor='w')
            combo = ttk.Combobox(self.yolo_config_frame, textvariable=self.rt_yolo_model_vars[task], state='disabled', width=25)
            combo.pack(fill=tk.X, pady=(0, 2))
            self.rt_yolo_model_combos[task] = combo

        tk.Label(self.yolo_config_frame, text="Custom Classes (for -world models):").pack(anchor='w', pady=(5,0))
        self.yolo_classes_var = tk.StringVar()
        self.yolo_classes_entry = tk.Entry(self.yolo_config_frame, textvariable=self.yolo_classes_var, width=28, state=tk.DISABLED)
        self.yolo_classes_entry.pack()
        SpeedApp.add_placeholder(self.yolo_classes_entry, "person, car, dog")

        tk.Label(self.yolo_config_frame, text="Custom Tracker Config (.yaml):").pack(anchor='w', pady=(5,0))
        self.rt_yolo_tracker_config_var = tk.StringVar()
        tracker_frame = tk.Frame(self.yolo_config_frame)
        tracker_frame.pack(fill=tk.X)
        tk.Entry(tracker_frame, textvariable=self.rt_yolo_tracker_config_var, state=tk.DISABLED).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.rt_tracker_browse_btn = tk.Button(tracker_frame, text="...", command=lambda: self.select_file(self.rt_yolo_tracker_config_var, [("YAML files", "*.yaml")]), state=tk.DISABLED, width=3)
        self.rt_tracker_browse_btn.pack(side=tk.RIGHT)

        
        vis_options_frame = tk.LabelFrame(main_control_frame, text="Visual Options", padx=10, pady=10)
        vis_options_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.overlay_vars = {
            "show_yolo": tk.BooleanVar(value=True),
            "show_pupil": tk.BooleanVar(value=True),
            "show_frag": tk.BooleanVar(value=True),
            "show_blink": tk.BooleanVar(value=True),
            "show_aois": tk.BooleanVar(value=True),
            "show_heatmap": tk.BooleanVar(value=False),
            "show_gaze_point": tk.BooleanVar(value=True),
            "show_gaze_path": tk.BooleanVar(value=True)
        }
        tk.Checkbutton(vis_options_frame, text="YOLO", variable=self.overlay_vars["show_yolo"]).pack(anchor='w')
        tk.Checkbutton(vis_options_frame, text="Gaze Point", variable=self.overlay_vars["show_gaze_point"]).pack(anchor='w')
        tk.Checkbutton(vis_options_frame, text="Gaze Path", variable=self.overlay_vars["show_gaze_path"]).pack(anchor='w')
        tk.Checkbutton(vis_options_frame, text="Pupil Plot", variable=self.overlay_vars["show_pupil"]).pack(anchor='w')
        tk.Checkbutton(vis_options_frame, text="Frag. Plot", variable=self.overlay_vars["show_frag"]).pack(anchor='w')
        tk.Checkbutton(vis_options_frame, text="Blink", variable=self.overlay_vars["show_blink"]).pack(anchor='w')
        tk.Checkbutton(vis_options_frame, text="AOIs", variable=self.overlay_vars["show_aois"]).pack(anchor='w')
        tk.Checkbutton(vis_options_frame, text="Heatmap", variable=self.overlay_vars["show_heatmap"]).pack(anchor='w')

        # --- NUOVO: Spostato lo status label e aggiunto il treeview per i filtri YOLO ---
        self.status_label = tk.Label(status_and_yolo_frame, text="Ready. Configure and press 'Connect'.", font=('Helvetica', 10), wraplength=400, justify=tk.LEFT)
        self.status_label.pack(side=tk.TOP, fill=tk.X, pady=(5,10))

        yolo_filter_frame = tk.LabelFrame(status_and_yolo_frame, text="YOLO Object Filter")
        yolo_filter_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.yolo_filter_tree = ttk.Treeview(yolo_filter_frame, show="tree")
        self.yolo_filter_tree.pack(fill=tk.BOTH, expand=True)
        self.yolo_filter_tree.bind('<Button-1>', self.on_yolo_filter_click)
        # --- FINE NUOVI WIDGET ---

        self.update_yolo_model_options() # Imposta i valori iniziali
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_yolo_model_options(self, event=None):
        """Popola i combobox dei modelli YOLO con le opzioni corrette."""
        for task, combo in self.rt_yolo_model_combos.items():
            models_for_task = [""] + YOLO_MODELS.get(task, [])
            combo['values'] = models_for_task
            # Imposta un default o lascia vuoto
            if self.rt_yolo_model_vars[task].get() not in models_for_task:
                self.rt_yolo_model_vars[task].set(models_for_task[0])

        # Abilita/disabilita l'entry per le classi custom in base alla selezione del world model
        is_custom_detect_task = bool(self.rt_yolo_model_vars['detect_world'].get())
        self.yolo_classes_entry.config(state=tk.NORMAL if is_custom_detect_task else tk.DISABLED)

    def connect_to_device(self):
        """Tenta di connettersi al dispositivo Neon."""
        self.status_label.config(text="Initializing and connecting...")
        self.connect_button.config(state=tk.DISABLED)

        # Istanzia l'analizzatore senza modello YOLO per ora
        self.analyzer = RealtimeNeonAnalyzer()

        if self.analyzer.connect():
            self.status_label.config(text="Device connected. Configure analysis and press 'Start Analysis'.")
            # Abilita i controlli per la fase successiva
            self.start_analysis_button.config(state=tk.NORMAL)
            for combo in self.rt_yolo_model_combos.values():
                combo.config(state='readonly')
            self.yolo_classes_entry.config(state='normal')
            self.rt_yolo_tracker_config_var.get()
            self.rt_tracker_browse_btn.config(state=tk.NORMAL)
            self.update_yolo_model_options() # Aggiorna stato entry classi custom
        else:
            self.status_label.config(text="Failed to connect. Please check device.")
            self.connect_button.config(state=tk.NORMAL)
            self.analyzer = None # Resetta l'analizzatore

    def start_stream(self):
        """Avvia il loop di analisi e streaming video."""
        if not self.analyzer or not self.analyzer.device:
            messagebox.showerror("Error", "Device not connected. Please connect first.", parent=self)
            return

        self.start_analysis_button.config(state=tk.DISABLED)

        # --- MODIFICA: Recupera tutti i modelli selezionati ---
        yolo_models_to_run = {task: str(MODELS_DIR / var.get()) for task, var in self.rt_yolo_model_vars.items() if var.get()}
        
        custom_classes_str = self.yolo_classes_var.get().strip()
        yolo_custom_classes = None
        if custom_classes_str and custom_classes_str != "person, car, dog": # Ignora il placeholder
            yolo_custom_classes = [cls.strip() for cls in custom_classes_str.split(',') if cls.strip()]
        yolo_tracker_config = self.rt_yolo_tracker_config_var.get() or None

        # Ora inizializza il modello YOLO nell'analizzatore esistente
        self.analyzer.initialize_yolo_models(
            yolo_models=yolo_models_to_run,
            custom_classes=yolo_custom_classes,
            tracker_config_path=yolo_tracker_config)
        
        self.thread = threading.Thread(target=self.stream_loop, daemon=True)
        self.thread.start()

    def stream_loop(self):
        self.is_running = True
        
        if self.lsl_stream_var.get():
            try:
                self.lsl_bridge = LSLBridge(self.analyzer)
                self.lsl_bridge.start()
                self.status_label.config(text="Streaming... (LSL Enabled)")
            except ImportError as e:
                messagebox.showerror("LSL Error", str(e), parent=self)
                self.status_label.config(text="Streaming... (LSL FAILED)")
        else:
             self.status_label.config(text="Streaming...")
        
        # Disabilita i controlli di configurazione YOLO una volta avviato lo stream
        for combo in self.rt_yolo_model_combos.values():
            combo.config(state=tk.DISABLED)
        self.yolo_classes_entry.config(state=tk.DISABLED)
        self.rt_tracker_browse_btn.config(state=tk.DISABLED)

        self.record_button.config(state=tk.NORMAL)

        while self.is_running and self.analyzer:
            if not self.is_paused_for_drawing:
                overlay_settings = {key: var.get() for key, var in self.overlay_vars.items()}
                
                # --- NUOVO: Applica i filtri all'analizzatore ---
                self.analyzer.set_yolo_filters(
                    class_filter=self.yolo_class_filter,
                    id_filter=self.yolo_id_filter
                )

                frame, detected_objects = self.analyzer.process_and_visualize(**overlay_settings)
                
                # --- NUOVO: Aggiorna la lista dei filtri se sono stati trovati nuovi oggetti ---
                if detected_objects:
                    self.update_yolo_filter_tree(detected_objects)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                self.photo = ImageTk.PhotoImage(image=img)
                self.canvas.after(0, self.update_canvas)
            time.sleep(1/60)

    def update_yolo_filter_tree(self, detected_objects):
        """Aggiorna il treeview con le classi e gli ID rilevati."""
        for class_name, track_id in detected_objects:
            if class_name not in self.detected_yolo_items:
                self.detected_yolo_items[class_name] = set()
                # Aggiunge il nodo genitore per la nuova classe
                class_node = self.yolo_filter_tree.insert("", "end", text=f"Class: {class_name}", open=True, tags=('class', class_name))
                self.yolo_filter_tree.item(class_node, values=(True,)) # True = checked

            if track_id not in self.detected_yolo_items[class_name]:
                self.detected_yolo_items[class_name].add(track_id)
                # Trova il nodo genitore e aggiunge il figlio per l'ID
                class_node_id = [item for item in self.yolo_filter_tree.get_children('') if self.yolo_filter_tree.item(item, 'tags')[1] == class_name][0]
                id_node = self.yolo_filter_tree.insert(class_node_id, "end", text=f"  ID: {track_id}", tags=('id', track_id))
                self.yolo_filter_tree.item(id_node, values=(True,)) # True = checked

    def on_yolo_filter_click(self, event):
        """Gestisce il click per attivare/disattivare i filtri."""
        item_id = self.yolo_filter_tree.identify_row(event.y)
        if not item_id: return

        # Ottieni lo stato corrente e invertilo
        is_checked = not self.yolo_filter_tree.item(item_id, 'values')[0]
        self.yolo_filter_tree.item(item_id, values=(is_checked,))

        # --- NUOVO: Logica a cascata ---
        # Se un nodo 'classe' viene cliccato, aggiorna tutti i suoi figli (ID)
        tags = self.yolo_filter_tree.item(item_id, 'tags')
        if tags and tags[0] == 'class':
            for child_id in self.yolo_filter_tree.get_children(item_id):
                self.yolo_filter_tree.item(child_id, values=(is_checked,))
        # --- FINE NUOVA LOGICA ---
        
        # Aggiorna i set di filtri
        self.yolo_class_filter = {self.yolo_filter_tree.item(i, 'tags')[1] for i in self.yolo_filter_tree.get_children('') if self.yolo_filter_tree.item(i, 'values')[0] is True}
        
        self.yolo_id_filter = set()
        for class_node in self.yolo_filter_tree.get_children(''):
            if self.yolo_filter_tree.item(class_node, 'values')[0]: # Se la classe è attiva
                for id_node in self.yolo_filter_tree.get_children(class_node):
                    if self.yolo_filter_tree.item(id_node, 'values')[0]: # Se l'ID è attivo
                        self.yolo_id_filter.add(int(self.yolo_filter_tree.item(id_node, 'tags')[1]))

        # Se tutti sono checkati, il filtro è vuoto (mostra tutto)
        if len(self.yolo_class_filter) == len(self.detected_yolo_items):
            self.yolo_class_filter.clear()
        
        total_ids = sum(len(ids) for ids in self.detected_yolo_items.values())
        if len(self.yolo_id_filter) == total_ids:
            self.yolo_id_filter.clear()

    def update_canvas(self):
        try:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        except tk.TclError:
            pass

    def toggle_recording(self):
        if not self.analyzer.is_recording:
            folder_path = filedialog.askdirectory(title="Select Folder for Real-time Recording")
            if not folder_path: return
            # --- MODIFICA: Passa l'opzione per includere l'audio ---
            include_audio = self.include_audio_var.get()
            if self.analyzer.start_recording(folder_path, include_audio=include_audio):
                self.record_button.config(text="Stop Recording", bg='#ffcdd2'); self.add_event_button.config(state=tk.NORMAL)
                self.status_label.config(text=f"REC ● | Saving to: {folder_path}")
        else:
            self.analyzer.stop_recording()
            self.record_button.config(text="Start Recording", bg='#c8e6c9'); self.add_event_button.config(state=tk.DISABLED)
            self.status_label.config(text="Streaming...")

    def add_event(self):
        event_name = self.event_name_entry.get()
        if event_name:
            # Invia l'evento sia alla registrazione interna sia a LSL
            if self.analyzer.is_recording:
                self.analyzer.add_event(event_name)
            
            # --- NUOVO: Invia evento a LSL se attivo ---
            if self.lsl_bridge:
                self.lsl_bridge.push_event(event_name)

            self.event_name_entry.delete(0, tk.END); self.event_name_entry.insert(0, "New Event")
        else:
            messagebox.showwarning("Input Error", "Please enter an event name.", parent=self)
    
    def prepare_to_draw_aoi(self):
        self.is_paused_for_drawing = True
        self.status_label.config(text="DRAW AOI: Click and drag on the video to define the area.")
        
    def on_canvas_press(self, event):
        if not self.is_paused_for_drawing: return
        self.start_x, self.start_y = event.x, event.y
        self.temp_rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='magenta', width=2)

    def on_canvas_drag(self, event):
        if not self.is_paused_for_drawing or not self.temp_rect_id: return
        self.canvas.coords(self.temp_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_canvas_release(self, event):
        if not self.is_paused_for_drawing or not self.temp_rect_id: return
        
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        aoi_name = simpledialog.askstring("AOI Name", "Enter a unique name for this AOI:", parent=self)
        if aoi_name:
            self.analyzer.add_static_aoi(aoi_name, [x1, y1, x2, y2])
            self.update_aoi_listbox()

        self.canvas.delete(self.temp_rect_id)
        self.temp_rect_id = None
        self.is_paused_for_drawing = False
        self.status_label.config(text="Streaming...")

    def add_qr_aoi_dialog(self):
        """Apre una finestra di dialogo per aggiungere una AOI basata su QR code."""
        if not self.analyzer or self.analyzer.last_scene_frame is None:
            messagebox.showwarning("No Stream", "Cannot add QR AOI. Please start the stream and pause on a clear frame.", parent=self)
            return

        # Metti in pausa lo stream se è in esecuzione
        was_running = self.is_running and not self.is_paused_for_drawing
        if was_running:
            self.is_paused_for_drawing = True

        editor = RealtimeQRAoiEditor(self, self.analyzer, self.analyzer.last_scene_frame.image)
        self.wait_window(editor)

        if editor.result:
            self.analyzer.add_qr_aoi(editor.aoi_name, editor.qr_data_list)
            self.update_aoi_listbox()
        self.is_paused_for_drawing = False # Riprendi lo stream

    def update_aoi_listbox(self):
        self.aoi_listbox.delete(0, tk.END)
        if not self.analyzer: return
        
        for aoi in self.analyzer.static_aois: # AOI Statiche
            self.aoi_listbox.insert(tk.END, f"{aoi['name']} (static)")
        for aoi in self.analyzer.qr_aois: # AOI QR
            self.aoi_listbox.insert(tk.END, f"{aoi['name']} (qr)")
            
    def remove_selected_aoi(self):
        selected_indices = self.aoi_listbox.curselection()
        if not selected_indices: return
        
        full_name = self.aoi_listbox.get(selected_indices[0])
        aoi_name = full_name.split(' (')[0]

        # Prova a rimuovere da entrambe le liste (statica e qr)
        self.analyzer.remove_static_aoi(aoi_name)
        self.analyzer.remove_qr_aoi(aoi_name)
        self.update_aoi_listbox()
            
    def on_close(self):
        # --- MODIFICATO: Ferma anche il ponte LSL ---
        if self.lsl_bridge:
            self.lsl_bridge.stop()
        self.is_running = False
        if self.analyzer: self.analyzer.close()
        self.destroy()

class EventManagerWindow(tk.Toplevel):
    """
    Una finestra per visualizzare, selezionare, modificare, aggiungere, unire e rimuovere eventi
    in una vista tabellare.
    """
    def __init__(self, parent, events_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Event Manager (Table View)")
        self.geometry("800x600")
        self.transient(parent)
        self.grab_set()

        self.events_df = events_df.copy()
        if 'selected' not in self.events_df.columns:
            self.events_df['selected'] = True
        self.events_df.sort_values('timestamp [ns]', inplace=True)
        self.saved_df = None

        frame = tk.Frame(self, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        cols = ("Selected", "Event Name", "Timestamp (s)", "Source")
        self.tree = ttk.Treeview(frame, columns=cols, show='headings', selectmode='extended')
        for col in cols:
            self.tree.heading(col, text=col)
        self.tree.column("Selected", width=80, anchor=tk.CENTER)
        self.tree.column("Event Name", width=350)
        self.tree.column("Timestamp (s)", width=150, anchor=tk.CENTER)
        self.tree.column("Source", width=100, anchor=tk.CENTER)

        self.populate_tree()
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Button-1>", self.on_click)
        self.tree.bind('<<TreeviewSelect>>', self.on_selection_change)

        button_frame = tk.Frame(self, pady=10)
        button_frame.pack(fill=tk.X)

        tk.Button(button_frame, text="Sort by Time", command=self.sort_events).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Add Event", command=self.add_event).pack(side=tk.LEFT, padx=5)
        self.merge_button = tk.Button(button_frame, text="Merge Selected", command=self.merge_events, state=tk.DISABLED)
        self.merge_button.pack(side=tk.LEFT, padx=5)
        self.remove_button = tk.Button(button_frame, text="Remove Selected", command=self.remove_selected_events, state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for index, row in self.events_df.iterrows():
            selected_text = "Yes" if row.get('selected', True) else "No"
            timestamp_sec = row['timestamp [ns]'] / 1e9
            source = row.get('source', 'manual')
            self.tree.insert("", "end", iid=str(index), values=(selected_text, row['name'], f"{timestamp_sec:.4f}", source))

    def sort_events(self):
        self.events_df.sort_values('timestamp [ns]', inplace=True)
        self.populate_tree()

    def on_selection_change(self, event):
        num_selected = len(self.tree.selection())
        self.remove_button.config(state=tk.NORMAL if num_selected > 0 else tk.DISABLED)
        self.merge_button.config(state=tk.NORMAL if num_selected >= 2 else tk.DISABLED)

    def on_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        column = self.tree.identify_column(event.x)
        if region == "cell" and column == "#1":
            item_id = self.tree.identify_row(event.y)
            if item_id:
                df_index = int(item_id)
                self.events_df.loc[df_index, 'selected'] = not self.events_df.loc[df_index, 'selected']
                self.populate_tree()

    def on_double_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell": return
        column_id = self.tree.identify_column(event.x)
        item_id = self.tree.identify_row(event.y)
        df_index = int(item_id)
        
        if column_id not in ("#2", "#3"): return

        x, y, width, height = self.tree.bbox(item_id, column_id)
        entry = ttk.Entry(self.tree)
        current_values = self.tree.item(item_id, 'values')
        col_index = int(column_id.replace('#', '')) - 1
        entry.insert(0, current_values[col_index])
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def save_edit(evt):
            new_value = entry.get()
            try:
                if column_id == "#2":
                    self.events_df.loc[df_index, 'name'] = new_value
                elif column_id == "#3":
                    new_timestamp_sec = float(new_value)
                    self.events_df.loc[df_index, 'timestamp [ns]'] = int(new_timestamp_sec * 1e9)
                self.populate_tree()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for the timestamp.")
            finally:
                entry.destroy()
        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def add_event(self):
        name = simpledialog.askstring("Add Event", "Enter the new event name:", parent=self)
        if not name: return
        ts_str = simpledialog.askstring("Add Event", f"Enter timestamp in seconds for '{name}':", parent=self)
        if not ts_str: return
        try:
            ts_sec = float(ts_str)
            new_row = {'name': name, 'timestamp [ns]': int(ts_sec * 1e9), 'selected': True, 'source': 'manual', 'recording id': 'rec_001'}
            self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
            self.sort_events()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for the timestamp.")

    def merge_events(self):
        selected_items = self.tree.selection()
        if len(selected_items) < 2: return
        new_name = simpledialog.askstring("Merge Events", "Enter the name for the new merged event:", parent=self)
        if not new_name: return
        indices = [int(item_id) for item_id in selected_items]
        selected_df = self.events_df.loc[indices]
        first_timestamp_ns = selected_df['timestamp [ns]'].min()
        new_row = {'name': new_name, 'timestamp [ns]': first_timestamp_ns, 'selected': True, 'source': 'manual', 'recording id': 'rec_001'}
        self.events_df.drop(indices, inplace=True)
        self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
        self.sort_events()

    def remove_selected_events(self):
        selected_items = self.tree.selection()
        if not selected_items: return
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to remove {len(selected_items)} event(s)?"):
            indices_to_drop = [int(item_id) for item_id in selected_items]
            self.events_df.drop(indices_to_drop, inplace=True)
            self.populate_tree()

    def save_and_close(self):
        self.saved_df = self.events_df
        self.destroy()

# --- MODIFICA: Ristrutturata la costante dei modelli per task ---
YOLO_MODELS = {
    'detect': [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        'yolov9c.pt', 'yolov9e.pt', 'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt',
        'yolov5n.pt', 'yolov5s.pt', 'yolov3.pt', 'yolo_nas_s.pt'
    ],
    'segment': [
        'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt',
        'sam_b.pt', 'FastSAM-s.pt', 'mobile_sam.pt'
    ],
    'pose': [
        'yolov8n-pose.pt', 'yolov8s-pose.pt', 'yolov8m-pose.pt', 'yolov8l-pose.pt', 'yolov8x-pose.pt'
    ],
    'obb': ['yolov8n-obb.pt', 'yolov8s-obb.pt', 'yolov8m-obb.pt', 'yolov8l-obb.pt', 'yolov8x-obb.pt'],
    'detect_world': ['yolov8s-world.pt', 'yolov8m-world.pt', 'yolov8l-world.pt', 'yolov8x-world.pt'],    
    'reid': [
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'
    ]
}

# --- NUOVO: Lista dei modelli di classificazione ufficiali ---
OFFICIAL_YOLO_CLS_MODELS = [
    'yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt', 'yolov8l-cls.pt', 'yolov8x-cls.pt',
    'yolov5n-cls.pt', 'yolov5s-cls.pt', 'yolov5m-cls.pt', 'yolov5l-cls.pt', 'yolov5x-cls.pt'
]

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v5.3.3")
        # --- MODIFICA: Avvia a schermo intero ---
        self.root.state('zoomed')

        self.raw_dir_var = tk.StringVar()
        # --- FINE MODIFICA ---
        self.unenriched_dir_var = tk.StringVar()
        self.enriched_dir_var = tk.StringVar()
        self.external_event_file_var = tk.StringVar()
        self.viv_events_df = pd.DataFrame()
        self.events_df = pd.DataFrame()
        self.plot_vars = {}
        self.video_vars = {}
        self.world_timestamps_df = pd.DataFrame()
        self.analysis_completed = False # NUOVO: Flag per tracciare il completamento dell'analisi
        self.concatenated_video_path = None
        
        self.user_defined_aois = []

        # --- MODIFICA: Contenitore principale per centrare i contenuti ---
        main_container = tk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        # --- FINE MODIFICA ---

        self.canvas = tk.Canvas(main_container, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview) # Rimane qui per il layout
        self.h_scrollbar = ttk.Scrollbar(main_container, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        # --- MODIFICA: Usa grid per il layout del canvas e delle scrollbar ---
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        # --- FINE MODIFICA ---

        # --- MODIFICA: Associa lo scroll del mouse al canvas per un comportamento corretto ---
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        # --- FINE MODIFICA ---
        
        # --- MODIFICA: Layout a colonne per un uso migliore dello spazio landscape ---
        main_frame = self.scrollable_frame
        main_frame.grid_columnconfigure(0, weight=1, uniform="group1")
        main_frame.grid_columnconfigure(1, weight=1, uniform="group1")

        # --- COLONNA SINISTRA: Setup, Input, Real-time ---
        left_column = tk.Frame(main_frame)
        left_column.grid(row=0, column=0, sticky="nsew", padx=(5, 2), pady=5)

        setup_frame = tk.LabelFrame(left_column, text="1. Project Setup", padx=5, pady=5)
        setup_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)
        name_frame = tk.Frame(setup_frame); name_frame.pack(fill=tk.X, pady=2)
        tk.Label(name_frame, text="Participant Name:", width=15, anchor='w').pack(side=tk.LEFT)
        self.participant_name_var = tk.StringVar(); self.participant_name_var.trace_add("write", self.update_output_dir_default)
        tk.Entry(name_frame, textvariable=self.participant_name_var).pack(fill=tk.X, expand=True)
        output_frame = tk.Frame(setup_frame); output_frame.pack(fill=tk.X, pady=2)
        tk.Label(output_frame, text="Output Folder:", width=15, anchor='w').pack(side=tk.LEFT)
        self.output_dir_entry = tk.Entry(output_frame); self.output_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(output_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.RIGHT)

        folders_frame = tk.LabelFrame(left_column, text="2. Input Data", padx=5, pady=5)
        folders_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)

        load_buttons_frame = tk.Frame(folders_frame)
        load_buttons_frame.pack(fill=tk.X, pady=5, side=tk.TOP)
        tk.Button(load_buttons_frame, text="Load from BIDS Directory...", command=self.load_bids_data, bg="#E0E0E0").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        tk.Button(load_buttons_frame, text="Load from DICOM File...", command=self.load_dicom_data, bg="#E0E0E0").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5) # <-- NUOVO BOTTONE
         
        self.unenriched_dir_var.trace_add("write", lambda *args: self.load_data_for_editors())
        self.enriched_dir_var.trace_add("write", lambda *args: self.update_aoi_list_display())

        raw_frame = tk.Frame(folders_frame); raw_frame.pack(fill=tk.X, pady=2)
        tk.Label(raw_frame, text="RAW Data Folder:", width=18, anchor='w').pack(side=tk.LEFT)
        tk.Entry(raw_frame, textvariable=self.raw_dir_var).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(raw_frame, text="Browse...", command=lambda: self.select_folder(self.raw_dir_var, "Select RAW Data Folder")).pack(side=tk.RIGHT)
        
        unenriched_frame = tk.Frame(folders_frame); unenriched_frame.pack(fill=tk.X, pady=2)
        tk.Label(unenriched_frame, text="Un-enriched Data Folder:", width=18, anchor='w').pack(side=tk.LEFT)
        tk.Entry(unenriched_frame, textvariable=self.unenriched_dir_var).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(unenriched_frame, text="Browse...", command=lambda: self.select_folder(self.unenriched_dir_var, "Select Un-enriched Data Folder")).pack(side=tk.RIGHT)
        
        enriched_frame = tk.LabelFrame(folders_frame, text="Enriched Data Folders (Optional):", padx=5, pady=5)
        enriched_frame.pack(fill=tk.X, pady=2)
        
        self.enriched_listbox = tk.Listbox(enriched_frame, height=3)
        self.enriched_listbox.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        
        enriched_btn_frame = tk.Frame(enriched_frame)
        enriched_btn_frame.pack(side=tk.RIGHT, fill=tk.Y)
        tk.Button(enriched_btn_frame, text="Add...", command=self.add_enriched_dir).pack(pady=2)
        tk.Button(enriched_btn_frame, text="Remove", command=self.remove_enriched_dir).pack(pady=2)
        self.enriched_dir_paths = [] # Lista per memorizzare i percorsi
        
        aoi_frame = tk.LabelFrame(left_column, text="2.1 Area of Interest (AOI) Management", padx=5, pady=5)
        aoi_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)

        self.aoi_listbox = tk.Listbox(aoi_frame, height=4)
        self.aoi_listbox.pack(fill=tk.X, expand=True, pady=5)
        
        aoi_button_frame = tk.Frame(aoi_frame)
        aoi_button_frame.pack(fill=tk.X)
        self.add_aoi_btn = tk.Button(aoi_button_frame, text="Add New AOI...", command=self.open_aoi_editor, state=tk.DISABLED)
        self.add_aoi_btn.pack(side=tk.LEFT)
        self.remove_aoi_btn = tk.Button(aoi_button_frame, text="Remove Selected AOI", command=self.remove_selected_aoi, state=tk.DISABLED)
        self.remove_aoi_btn.pack(side=tk.LEFT, padx=10)
        self.aoi_listbox.bind('<<ListboxSelect>>', self.on_aoi_select)

        # --- NUOVO: Sezione per la creazione del Video-in-Video ---
        viv_frame = tk.LabelFrame(left_column, text="2.2 Video-in-Video Setup (Optional)", padx=5, pady=5)
        viv_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)
        viv_info_label = tk.Label(viv_frame, text="This feature replaces the scene video with screen recordings synchronized to events.", justify=tk.LEFT)
        viv_info_label.pack(anchor='w')

        # --- MODIFICA: Flusso in due passaggi per il Video-in-Video ---
        viv_buttons_frame = tk.Frame(viv_frame)
        viv_buttons_frame.pack(fill=tk.X, pady=5)

        self.map_viv_btn = tk.Button(viv_buttons_frame, text="1. Map Events to Videos...", command=self.open_viv_editor, state=tk.DISABLED)
        self.map_viv_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.create_viv_btn = tk.Button(viv_buttons_frame, text="2. Create new external.mp4", command=self.run_video_in_video_creation, state=tk.DISABLED)
        self.create_viv_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        # --- FINE MODIFICA ---

        self.viv_status_label = tk.Label(viv_frame, text="Status: Using original scene video.", fg="grey")
        self.viv_status_label.pack(anchor='w', pady=(0, 5))

        event_frame = tk.LabelFrame(left_column, text="2.5 Event Management", padx=5, pady=5)
        event_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)
        ext_event_file_frame = tk.Frame(event_frame)
        ext_event_file_frame.pack(fill=tk.X, pady=2)
        tk.Label(ext_event_file_frame, text="Optional Events File:", width=18, anchor='w').pack(side=tk.LEFT)
        self.external_event_file_var.trace_add("write", lambda *args: self.load_data_for_editors())
        tk.Entry(ext_event_file_frame, textvariable=self.external_event_file_var).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(ext_event_file_frame, text="Browse...", command=self.select_event_file).pack(side=tk.RIGHT)
        event_buttons_frame = tk.Frame(event_frame, pady=5)
        event_buttons_frame.pack(fill=tk.X)
        self.event_summary_label = tk.Label(event_buttons_frame, text="Load data to manage events.")
        self.event_summary_label.pack(side=tk.LEFT, pady=5)
        self.edit_video_btn = tk.Button(event_buttons_frame, text="Edit on Video", command=self.open_event_manager_video, state=tk.DISABLED)
        self.edit_video_btn.pack(side=tk.RIGHT, padx=5)
        self.edit_video_only_btn = tk.Button(event_buttons_frame, text="Open Video in Editor...", command=self.open_video_editor_standalone)
        self.edit_video_only_btn.pack(side=tk.RIGHT, padx=5)
        self.edit_events_btn = tk.Button(event_buttons_frame, text="Edit in Table", command=self.open_event_manager_table, state=tk.DISABLED)
        self.edit_events_btn.pack(side=tk.RIGHT, padx=5)

        realtime_frame = tk.LabelFrame(left_column, text="3. Real-time Analysis", padx=10, pady=10)
        realtime_frame.pack(pady=(3, 15), ipadx=2, ipady=2, fill=tk.X)
        tk.Button(realtime_frame, text="START REAL-TIME STREAM", command=self.start_realtime_stream, font=('Helvetica', 10, 'bold'), bg='#a5d6a7').pack(pady=5, fill=tk.X)

        # --- COLONNA DESTRA: Analisi, Filtri, Classificazione, Export, Generazione ---
        right_column = tk.Frame(main_frame)
        right_column.grid(row=0, column=1, sticky="nsew", padx=(2, 5), pady=5)

        analysis_frame = tk.LabelFrame(right_column, text="4. Run Full Analysis", padx=5, pady=5)
        analysis_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)
        self.yolo_var = tk.BooleanVar(value=True)
        yolo_checkbutton = tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (Required for Dynamic AOI, GPU Recommended)", variable=self.yolo_var, command=self.toggle_yolo_options, wraplength=350, justify=tk.LEFT)
        yolo_checkbutton.pack(anchor='w')

        # --- MODIFICA: Controlli YOLO Multi-Task ---
        yolo_options_frame = tk.Frame(analysis_frame)
        yolo_options_frame.pack(fill=tk.X, padx=20, pady=5)

        self.yolo_model_vars = {
            'detect': tk.StringVar(),
            'segment': tk.StringVar(),
            'pose': tk.StringVar(),
            'obb': tk.StringVar(),
            'detect_world': tk.StringVar(),
            'reid': tk.StringVar()
        }
        self.yolo_model_combos = {}
        
        for task, label in [('detect', 'Detection Model:'), ('segment', 'Segmentation Model:'), ('pose', 'Pose Model:'), ('obb', 'OBB Model:'), ('reid', 'Re-ID Model:'), ('detect_world', 'World Model (for custom classes):')]:
            model_frame = tk.Frame(yolo_options_frame)
            model_frame.pack(fill=tk.X, pady=2)
            tk.Label(model_frame, text=label, width=22, anchor='w').pack(side=tk.LEFT)
            combo = ttk.Combobox(model_frame, textvariable=self.yolo_model_vars[task], state='readonly')
            combo.pack(fill=tk.X, expand=True)
            self.yolo_model_combos[task] = combo

        classes_frame = tk.Frame(yolo_options_frame)
        classes_frame.pack(fill=tk.X, pady=2)
        tk.Label(classes_frame, text="Custom Classes (for World Model):", width=22, anchor='w').pack(side=tk.LEFT)
        self.yolo_classes_var = tk.StringVar()
        self.yolo_classes_entry = tk.Entry(classes_frame, textvariable=self.yolo_classes_var)
        self.yolo_classes_entry.pack(fill=tk.X, expand=True)
        self.add_placeholder(self.yolo_classes_entry, "person, car, dog")

        # --- NUOVO: Selezione del file di configurazione del tracker ---
        tracker_frame = tk.Frame(yolo_options_frame)
        tracker_frame.pack(fill=tk.X, pady=2)
        tk.Label(tracker_frame, text="Custom Tracker Config (.yaml):", width=22, anchor='w').pack(side=tk.LEFT)
        self.yolo_tracker_config_var = tk.StringVar()
        # --- NUOVO: Imposta il valore predefinito per il tracker ---
        default_tracker_path = project_root / 'desktop_app' / 'default_yaml.yaml'
        if default_tracker_path.exists():
            self.yolo_tracker_config_var.set(str(default_tracker_path))
        tk.Entry(tracker_frame, textvariable=self.yolo_tracker_config_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Button(tracker_frame, text="Browse...", command=lambda: self.select_file(self.yolo_tracker_config_var, [("YAML files", "*.yaml")])).pack(side=tk.RIGHT)
        
        # Imposta lo stato iniziale
        self.toggle_yolo_options()
        # --- FINE MODIFICA ---
        
        tk.Button(analysis_frame, text="RUN FULL ANALYSIS", command=self.run_full_analysis_wrapper, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5, fill=tk.X)

        # --- NUOVO: Frame per il filtraggio dei risultati YOLO ---
        yolo_filter_frame = tk.LabelFrame(right_column, text="5. YOLO Results & Filtering", padx=5, pady=5)
        yolo_filter_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)

        yolo_filter_controls = tk.Frame(yolo_filter_frame)
        yolo_filter_controls.pack(fill=tk.X, pady=5)
        tk.Button(yolo_filter_controls, text="Load/Refresh YOLO Results", command=self.load_yolo_results_for_filtering, font=('Helvetica', 10, 'bold'), bg='#ffcc80').pack(side=tk.LEFT)

        self.yolo_filter_notebook = ttk.Notebook(yolo_filter_frame)
        self.yolo_filter_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        self.yolo_filter_trees = {}
        self.yolo_detections_df = pd.DataFrame()
        self.yolo_class_filter = set()
        self.yolo_id_filter = set()

        for task in ['detection', 'segmentation', 'pose', 'obb']:
            tab = ttk.Frame(self.yolo_filter_notebook)
            tree = ttk.Treeview(tab, columns=("#1"), show="tree headings")
            tree.heading("#0", text="Object")
            tree.heading("#1", text="Show")
            tree.column("#1", width=50, anchor='center')
            tree.pack(fill=tk.BOTH, expand=True)
            tree.bind('<Button-1>', self.on_yolo_filter_click)
            self.yolo_filter_trees[task] = {'tab': tab, 'tree': tree}
        # --- FINE NUOVO FRAME ---
        
        # --- NUOVO: Frame per la classificazione delle detection ---
        post_analysis_frame = tk.LabelFrame(right_column, text="6. Post-Analysis Tools", padx=5, pady=5)
        post_analysis_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)

        self.nsi_button = tk.Button(post_analysis_frame, text="Calculate Normalized Switching Index (NSI)...", command=self.open_nsi_calculator, state=tk.DISABLED)
        self.nsi_button.pack(fill=tk.X, pady=5)
        nsi_info_label = tk.Label(post_analysis_frame, text="Requires at least 2 defined AOIs. Enabled after 'Run Full Analysis'.", fg="grey", font=('Helvetica', 8))
        nsi_info_label.pack(anchor='w')


        yolo_classify_frame = tk.LabelFrame(right_column, text="6. Classify Detections", padx=5, pady=5)
        yolo_classify_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)

        # --- MODIFICA: Sostituzione dell'Entry con un Combobox + opzione custom ---
        classify_combo_frame = tk.Frame(yolo_classify_frame)
        classify_combo_frame.pack(fill=tk.X, pady=2)
        tk.Label(classify_combo_frame, text="Classification Model:", width=16, anchor='w').pack(side=tk.LEFT)
        self.yolo_classify_model_combo_var = tk.StringVar()
        self.yolo_classify_combo = ttk.Combobox(classify_combo_frame, textvariable=self.yolo_classify_model_combo_var, state='readonly')
        self.yolo_classify_combo.pack(fill=tk.X, expand=True, side=tk.LEFT)
        self.yolo_classify_combo.bind('<<ComboboxSelected>>', self.on_classify_model_selected)

        # Frame per il percorso custom, inizialmente nascosto
        self.classify_custom_path_frame = tk.Frame(yolo_classify_frame)
        # Non fare il pack() qui, verrà gestito da on_classify_model_selected

        tk.Label(self.classify_custom_path_frame, text="Custom Model Path:", width=16, anchor='w').pack(side=tk.LEFT)
        self.yolo_classify_model_var = tk.StringVar()
        self.yolo_classify_custom_entry = tk.Entry(self.classify_custom_path_frame, textvariable=self.yolo_classify_model_var)
        self.yolo_classify_custom_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(self.classify_custom_path_frame, text="Browse...", command=self.select_classification_model).pack(side=tk.RIGHT)
        # --- FINE MODIFICA ---

        tk.Button(yolo_classify_frame, text="RUN CLASSIFICATION ON FILTERED DETECTIONS", command=self.run_classification_on_detections, font=('Helvetica', 10, 'bold'), bg='#80deea').pack(pady=5, fill=tk.X)
        
        # Popola il combobox all'avvio
        self.update_classification_model_options()
        # Imposta lo stato iniziale della UI
        self.on_classify_model_selected()

        # --- FINE NUOVO FRAME ---

        bids_frame = tk.LabelFrame(right_column, text="7. Data Export & Tools", padx=5, pady=5)
        bids_frame.pack(pady=3, ipadx=2, ipady=2, fill=tk.X)
        export_buttons_frame = tk.Frame(bids_frame)
        export_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        tk.Button(export_buttons_frame, text="CONVERT TO BIDS FORMAT", command=self.run_bids_conversion, font=('Helvetica', 10, 'bold'), bg='#FFD54F').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5), pady=5)
        tk.Button(export_buttons_frame, text="CONVERT TO DICOM FORMAT", command=self.run_dicom_conversion, font=('Helvetica', 10, 'bold'), bg='#FFD54F').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5) # <-- NUOVO BOTTONE
        tk.Button(export_buttons_frame, text="Open Data Viewer...", command=self.open_data_viewer, font=('Helvetica', 10, 'bold'), bg='#AED581').pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)

        # --- NUOVO PULSANTE PER IL CONVERTITORE DI DISPOSITIVI ---
        device_converter_frame = tk.Frame(bids_frame)
        device_converter_frame.pack(fill=tk.X)
        tk.Button(device_converter_frame, text="Device Converter...", command=self.open_device_converter,
                  font=('Helvetica', 10, 'bold'), bg='#B2EBF2').pack(expand=True, fill=tk.X, pady=(5, 0))
        # --- FINE NUOVO PULSANTE ---

        notebook = ttk.Notebook(right_column)
        notebook.pack(fill=tk.X, expand=True, pady=5, padx=0)
        plot_tab = tk.Frame(notebook); notebook.add(plot_tab, text='8. Generate Plots')
        video_tab = tk.Frame(notebook); notebook.add(video_tab, text='9. Generate Videos')
        yolo_tab = tk.Frame(notebook); notebook.add(yolo_tab, text='10. YOLO Stats')
        self.setup_plot_tab(plot_tab)
        self.setup_video_tab(video_tab)
        self.setup_yolo_tab(yolo_tab)
        
        footer_frame = tk.Frame(root, pady=5)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        footer_label_part1 = tk.Label(footer_frame, text="Developed by Dr. Daniele Lozzi & the ", font=('Helvetica', 9))
        footer_label_part1.pack(side=tk.LEFT, padx=(10, 0))

        footer_link = tk.Label(footer_frame, text="LabSCoC team", font=('Helvetica', 9, 'underline'), fg="blue", cursor="hand2")
        footer_link.pack(side=tk.LEFT)
        footer_link.bind("<Button-1>", lambda e: self.open_github())
        
        footer_label_part2 = tk.Label(footer_frame, text=" at the University of L'Aquila.", font=('Helvetica', 9))
        footer_label_part2.pack(side=tk.LEFT)
        
        self.update_aoi_list_display()

    @staticmethod
    def add_placeholder(entry_widget, placeholder_text):
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


    def toggle_yolo_options(self):
        """Abilita o disabilita tutti i widget delle opzioni YOLO in base al checkbutton principale."""
        is_enabled = self.yolo_var.get()
        combo_state = 'readonly' if is_enabled else 'disabled'
        entry_state = 'normal' if is_enabled else 'disabled'

        for combo in self.yolo_model_combos.values():
            combo.config(state=combo_state)
        self.yolo_classes_entry.config(state=entry_state)

        self.update_yolo_model_options()

    def update_yolo_model_options(self, event=None):
        """Popola i combobox dei modelli YOLO con le opzioni corrette."""
        for task, combo in self.yolo_model_combos.items():
            models_for_task = [""] + YOLO_MODELS.get(task, []) # Aggiungi opzione vuota
            combo['values'] = models_for_task
            # Imposta un default o lascia vuoto
            if self.yolo_model_vars[task].get() not in models_for_task:
                self.yolo_model_vars[task].set(models_for_task[0])

    def update_classification_model_options(self):
        """Scansiona la cartella dei modelli, aggiunge i modelli ufficiali e popola il combobox."""
        try:
            # 1. Prendi i modelli locali
            local_cls_models = {f.name for f in MODELS_DIR.glob('*-cls.pt')}
            
            # 2. Unisci con i modelli ufficiali (il set gestisce i duplicati)
            all_models = sorted(list(local_cls_models.union(set(OFFICIAL_YOLO_CLS_MODELS))))

        except Exception as e:
            logging.error(f"Could not scan for classification models in {MODELS_DIR}: {e}")
            # In caso di errore, usa solo la lista ufficiale
            all_models = sorted(OFFICIAL_YOLO_CLS_MODELS)
        
        # 3. Aggiungi l'opzione custom
        options = all_models + ["Custom..."]
        
        self.yolo_classify_combo['values'] = options
        if self.yolo_classify_model_combo_var.get() not in options:
            self.yolo_classify_model_combo_var.set(options[0] if options else "")

    def on_classify_model_selected(self, event=None):
        """Mostra o nasconde il campo per il percorso custom in base alla selezione."""
        if self.yolo_classify_model_combo_var.get() == "Custom...":
            self.classify_custom_path_frame.pack(fill=tk.X, pady=2, before=self.yolo_classify_combo.master.master.winfo_children()[-1]) # Inserisce prima del bottone RUN
        else:
            self.classify_custom_path_frame.pack_forget()
            # Pulisce il campo custom per evitare confusione
            self.yolo_classify_model_var.set("")




    def open_data_viewer(self):
        viewer_window = DataViewerWindow(self.root, defined_aois=self.user_defined_aois)

    def load_video_only_viewer(self):
        """
        Apre un file dialog per selezionare un video e lo carica nel DataViewer.
        """
        video_path_str = filedialog.askopenfilename(
            title="Select a video file to view",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if not video_path_str:
            return

        # Apre il viewer e poi chiama il metodo per caricare il video specifico
        viewer = DataViewerWindow(self.root, defined_aois=self.user_defined_aois)
        viewer.load_video_only(video_path_str)

    def open_device_converter(self):
        """
        Apre la finestra per la conversione di dati da altri dispositivi (es. Tobii).
        """
        DeviceConverterWindow(self.root)
    
    def run_dicom_conversion(self):
        unenriched_path_str = self.unenriched_dir_var.get()
        if not unenriched_path_str or not Path(unenriched_path_str).is_dir():
            messagebox.showerror("Error", "La cartella 'Un-enriched' è obbligatoria per la conversione DICOM.")
            return

        patient_name = self.participant_name_var.get().strip()
        if not patient_name:
            messagebox.showerror("Error", "Il nome del partecipante è obbligatorio.")
            return
        
        output_dicom_path = filedialog.asksaveasfilename(
            title="Salva file DICOM",
            defaultextension=".dcm",
            filetypes=[("DICOM files", "*.dcm")]
        )
        if not output_dicom_path:
            return

        try:
            messagebox.showinfo("In Corso", "Avvio della conversione in formato DICOM...")
            patient_info = {"name": patient_name, "id": patient_name}
            convert_to_dicom(
                unenriched_dir=Path(unenriched_path_str),
                output_dicom_path=Path(output_dicom_path),
                patient_info=patient_info
            )
            messagebox.showinfo("Successo", f"Conversione DICOM completata!\nFile salvato in: {output_dicom_path}")
        except Exception as e:
            logging.error(f"Errore durante la conversione DICOM: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Errore di Conversione", f"Si è verificato un errore: {e}\n\nControlla i log per i dettagli.")

    def load_dicom_data(self):
        dicom_path_str = filedialog.askopenfilename(
            title="Seleziona un file DICOM Waveform",
            filetypes=[("DICOM files", "*.dcm"), ("All files", "*.*")]
        )
        if not dicom_path_str:
            return

        try:
            messagebox.showinfo("In Corso", "Caricamento e conversione del file DICOM...")
            temp_unenriched_path = load_from_dicom(Path(dicom_path_str))
            
            # Leggi il nome del paziente dal file DICOM per popolare la GUI
            ds = pydicom.dcmread(dicom_path_str)
            patient_name = str(ds.PatientName) if 'PatientName' in ds else "dicom_patient"

            self.unenriched_dir_var.set(str(temp_unenriched_path))
            self.participant_name_var.set(patient_name)
            
            messagebox.showinfo("Successo", "Dati DICOM caricati e convertiti con successo per l'analisi in SPEED.")
        except Exception as e:
            logging.error(f"Errore durante il caricamento da DICOM: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Errore di Caricamento", f"Si è verificato un errore: {e}\n\nControlla i log per i dettagli.")
    
    def run_bids_conversion(self):
        unenriched_path_str = self.unenriched_dir_var.get()
        if not unenriched_path_str or not Path(unenriched_path_str).is_dir():
            messagebox.showerror("Error", "La cartella 'Un-enriched' è obbligatoria per la conversione BIDS.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("BIDS Conversion Setup")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Inserisci i metadati per la conversione BIDS:", pady=10).pack()

        tk.Label(dialog, text="Subject ID (es. 01):").pack()
        subject_entry = tk.Entry(dialog)
        subject_entry.pack(padx=20, fill=tk.X)
        subject_entry.insert(0, self.participant_name_var.get().replace("participant_", ""))

        tk.Label(dialog, text="Session ID (es. 01):").pack()
        session_entry = tk.Entry(dialog)
        session_entry.pack(padx=20, fill=tk.X)
        session_entry.insert(0, "01")

        tk.Label(dialog, text="Task Name (es. reading):").pack()
        task_entry = tk.Entry(dialog)
        task_entry.pack(padx=20, fill=tk.X)
        task_entry.insert(0, "eyetracking")
        
        def on_convert():
            subject_id = subject_entry.get().strip()
            session_id = session_entry.get().strip()
            task_name = task_entry.get().strip()

            if not all([subject_id, session_id, task_name]):
                messagebox.showwarning("Input Mancante", "Tutti i campi sono obbligatori.", parent=dialog)
                return

            output_bids_dir = filedialog.askdirectory(title="Seleziona la cartella di output per i dati BIDS")
            if not output_bids_dir:
                return
            
            dialog.destroy()
            
            try:
                messagebox.showinfo("In Corso", "Avvio della conversione in formato BIDS...")
                convert_to_bids(
                    unenriched_dir=Path(unenriched_path_str),
                    output_bids_dir=Path(output_bids_dir),
                    subject_id=subject_id,
                    session_id=session_id,
                    task_name=task_name
                )
                messagebox.showinfo("Successo", f"Conversione BIDS completata con successo!\nDati salvati in: {output_bids_dir}")
            except Exception as e:
                logging.error(f"Errore durante la conversione BIDS: {e}\n{traceback.format_exc()}")
                messagebox.showerror("Errore di Conversione", f"Si è verificato un errore: {e}\n\nControlla i log per i dettagli.")

        tk.Button(dialog, text="Avvia Conversione", command=on_convert, font=('Helvetica', 10, 'bold')).pack(pady=20)

    def add_enriched_dir(self):
        dir_path = filedialog.askdirectory(title="Select Enriched Data Folder")
        if dir_path and dir_path not in self.enriched_dir_paths:
            self.enriched_dir_paths.append(dir_path)
            self.enriched_listbox.insert(tk.END, Path(dir_path).name) # Mostra solo il nome della cartella
            self.load_data_for_editors() # --- CORREZIONE: Ricarica i dati quando si aggiunge una cartella enriched

    def remove_enriched_dir(self):
        selected_indices = self.enriched_listbox.curselection()
        if not selected_indices:
            return
        
        # Rimuovi in ordine inverso per evitare problemi con gli indici
        for index in sorted(selected_indices, reverse=True):
            self.enriched_listbox.delete(index)
            del self.enriched_dir_paths[index]
        self.update_aoi_list_display()

    def start_realtime_stream(self):
        RealtimeDisplayWindow(self.root)

    def open_github(self):
        webbrowser.open_new(r"https://github.com/danielelozzi/SPEED")
        
    def _on_mousewheel(self, event):
        # --- MODIFICA: Gestione cross-platform dello scroll ---
        if sys.platform.startswith('linux'):
            if event.num == 4: self.canvas.yview_scroll(-1, "units")
            elif event.num == 5: self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def setup_plot_tab(self, parent_tab):
        plot_options_frame = tk.LabelFrame(parent_tab, text="Plot Options", padx=10, pady=10)
        plot_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plot_types = {"path_plots": "Path Plots", "heatmaps": "Density Heatmaps", "histograms": "Duration Histograms", "pupillometry": "Pupillometry", "advanced_timeseries": "Advanced Time Series", "fragmentation": "Gaze Fragmentation Plot"}
        for key, text in plot_types.items():
            self.plot_vars[key] = tk.BooleanVar(value=True); tk.Checkbutton(plot_options_frame, text=text, variable=self.plot_vars[key]).pack(anchor='w')
        tk.Button(parent_tab, text="GENERATE SELECTED PLOTS", command=self.run_plot_generation, font=('Helvetica', 10, 'bold'), bg='#90caf9').pack(pady=10)

    def setup_video_tab(self, parent_tab):
        video_options_frame = tk.LabelFrame(parent_tab, text="Video Options", padx=10, pady=10)
        video_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_opts = {"trim_to_events": "Trim video to selected events only", "crop_and_correct_perspective": "Crop & Correct Perspective", "overlay_yolo": "Overlay YOLO detections", "overlay_gaze": "Overlay gaze point", "overlay_gaze_path": "Overlay gaze path (trail)", "overlay_pupil_plot": "Overlay pupillometry plot", "overlay_fragmentation_plot": "Overlay gaze fragmentation plot", "overlay_event_text": "Overlay event name text", "overlay_on_surface_text": "Overlay Enriched Area / AOI text", "include_internal_cam": "Include internal camera (PiP)", "overlay_dynamic_heatmap": "Overlay Dynamic Gaze Heatmap"}
        for key, text in video_opts.items():
            self.video_vars[key] = tk.BooleanVar(value=False); tk.Checkbutton(video_options_frame, text=text, variable=self.video_vars[key]).pack(anchor='w')

        heatmap_video_frame = tk.Frame(video_options_frame)
        heatmap_video_frame.pack(fill=tk.X, padx=(20, 0), pady=(5,0))
        self.heatmap_video_window_var = tk.DoubleVar(value=2.0)
        tk.Label(heatmap_video_frame, text="Heatmap Window (seconds):").pack(side=tk.LEFT)
        ttk.Scale(heatmap_video_frame, from_=0.5, to=10.0, variable=self.heatmap_video_window_var, orient=tk.HORIZONTAL).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        
        self.video_vars['overlay_gaze'].set(True); self.video_vars['overlay_gaze_path'].set(True); self.video_vars['overlay_event_text'].set(True)
        tk.Label(video_options_frame, text="\nOutput Filename:").pack(anchor='w')
        self.video_filename_var = tk.StringVar(value="video_output_1.mp4")
        tk.Entry(video_options_frame, textvariable=self.video_filename_var).pack(fill=tk.X, pady=5)
        tk.Button(parent_tab, text="GENERATE VIDEO", command=self.run_video_generation, font=('Helvetica', 10, 'bold'), bg='#ef9a9a').pack(pady=10)
    def setup_yolo_tab(self, parent_tab):
        tk.Button(parent_tab, text="Load/Refresh YOLO Results", command=self.load_yolo_results, font=('Helvetica', 10, 'bold'), bg='#ffcc80').pack(pady=10)
        
        # --- MODIFICA: Usa un PanedWindow per ridimensionare i riquadri ---
        yolo_pane = tk.PanedWindow(parent_tab, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        yolo_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        class_frame = tk.LabelFrame(yolo_pane, text="Results per Class", padx=10, pady=10)
        yolo_pane.add(class_frame, stretch="always")
        self.class_treeview = ttk.Treeview(class_frame, show='headings')
        self.class_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        instance_frame = tk.LabelFrame(yolo_pane, text="Results per Instance (Filterable)", padx=10, pady=10)
        yolo_pane.add(instance_frame, stretch="always")
        
        # --- NUOVO: Aggiunto Treeview con checkbox per il filtraggio ---
        self.instance_treeview = ttk.Treeview(instance_frame, show='headings', columns=("Show", "Instance", "Fixations", "Avg Pupil", "Norm. Fixations"))
        self.instance_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.instance_treeview.heading("Show", text="Show")
        self.instance_treeview.column("Show", width=50, anchor='center')
        self.instance_treeview.bind('<Button-1>', self.on_instance_filter_click)
        self.instance_stats_df = pd.DataFrame() # Cache per i dati
        
    def select_folder(self, var, title):
        dir_path = filedialog.askdirectory(title=title)
        if dir_path: var.set(dir_path)

    def select_event_file(self):
        filepath = filedialog.askopenfilename(title="Select Custom Event File", filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.external_event_file_var.set(filepath)

    def select_file(self, var, filetypes):
        """Funzione generica per selezionare un file e impostare una StringVar."""
        filepath = filedialog.askopenfilename(title="Select File", filetypes=filetypes)
        if filepath: var.set(filepath)

    def load_data_for_editors(self):
        self.update_aoi_list_display()
        unenriched_path_str = self.unenriched_dir_var.get()
        ext_event_file_str = self.external_event_file_var.get()

        event_dfs = []

        # --- MODIFICA: Logica di caricamento eventi migliorata ---
        # 1. Prova a caricare gli eventi dalle cartelle "Enriched" se presenti.
        #    Questo è utile se gli eventi sono specifici per un'analisi arricchita.
        if self.enriched_dir_paths:
            for enriched_path_str in self.enriched_dir_paths:
                events_path = Path(enriched_path_str) / 'events.csv'
                if events_path.exists():
                    try:
                        df = pd.read_csv(events_path)
                        df['source'] = Path(enriched_path_str).name # Usa il nome della cartella come sorgente
                        event_dfs.append(df)
                        logging.info(f"Loaded events from enriched folder: {events_path}")
                    except Exception as e:
                        messagebox.showerror("Error", f"Could not read events.csv from {enriched_path_str}:\n{e}")

        # 2. Se non sono stati trovati eventi nelle cartelle enriched, carica dalla cartella "Un-enriched".
        if not event_dfs and unenriched_path_str:
            events_path = Path(unenriched_path_str) / 'events.csv'
            if events_path.exists():
                try:
                    df = pd.read_csv(events_path)
                    df['source'] = 'un-enriched'
                    event_dfs.append(df)
                    logging.info(f"Loaded events from un-enriched folder: {events_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not read events.csv from {unenriched_path_str}:\n{e}")

        if ext_event_file_str:
            optional_events_path = Path(ext_event_file_str)
            if optional_events_path.exists():
                try:
                    df_optional = pd.read_csv(optional_events_path)
                    df_optional['source'] = 'optional'
                    event_dfs.append(df_optional)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not read optional events file:\n{e}")
        # --- FINE MODIFICA ---

        if event_dfs:
            self.events_df = pd.concat(event_dfs, ignore_index=True).sort_values('timestamp [ns]').reset_index(drop=True)
            if 'selected' not in self.events_df.columns:
                self.events_df['selected'] = True
        else:
            self.events_df = pd.DataFrame()

        if unenriched_path_str:
            world_ts_path = Path(unenriched_path_str) / 'world_timestamps.csv'
            if world_ts_path.exists():
                try:
                    self.world_timestamps_df = pd.read_csv(world_ts_path)
                    if 'frame' not in self.world_timestamps_df.columns:
                        self.world_timestamps_df['frame'] = self.world_timestamps_df.index
                except Exception:
                    self.world_timestamps_df = pd.DataFrame()
            else:
                self.world_timestamps_df = pd.DataFrame()
        else:
            self.world_timestamps_df = pd.DataFrame()
            
        self.update_event_summary_display()
        self.on_aoi_select(None) # Aggiorna lo stato dei pulsanti dipendenti, come ViV
   

    def update_event_summary_display(self):
        if not self.events_df.empty:
            selected_count = self.events_df['selected'].sum() if 'selected' in self.events_df.columns else len(self.events_df)
            self.event_summary_label.config(text=f"{selected_count} of {len(self.events_df)} events loaded.")
            self.edit_events_btn.config(state=tk.NORMAL)
        else:
            self.event_summary_label.config(text="Load data to manage events.")
            self.edit_events_btn.config(state=tk.DISABLED)

        video_path = None
        if self.unenriched_dir_var.get():
            try:
                video_path = next(Path(self.unenriched_dir_var.get()).glob('*.mp4'))
            except StopIteration:
                video_path = None

        self.edit_video_btn.config(state=tk.NORMAL if video_path and video_path.exists() else tk.DISABLED)

    def open_event_manager_table(self):
        if self.events_df.empty:
            messagebox.showwarning("Warning", "No events loaded to edit.")
            return
        manager = EventManagerWindow(self.root, self.events_df)
        self.root.wait_window(manager)
        if manager.saved_df is not None:
            self.events_df = manager.saved_df.reset_index(drop=True)
            self.update_event_summary_display()
            logging.info("Event list updated via table editor.")

    def open_event_manager_video(self):
        video_path = next(Path(self.unenriched_dir_var.get()).glob('*.mp4'))
        # --- MODIFICA: Passa anche i risultati YOLO se disponibili ---
        yolo_results_path = Path(self.output_dir_entry.get()) / 'yolo_detections_cache.csv'
        
        # Se abbiamo già un yolo_df filtrato in memoria, usiamo quello. Altrimenti, carichiamo dal file.
        yolo_df = self.yolo_detections_df if hasattr(self, 'yolo_detections_df') and not self.yolo_detections_df.empty else None
        if yolo_df is None and yolo_results_path.exists():
            yolo_df = pd.read_csv(yolo_results_path)

        manager = InteractiveVideoEditor(self.root, video_path, self.events_df, self.world_timestamps_df, yolo_df)
        self.root.wait_window(manager)
        
        if manager.saved_df is not None:
            self.events_df = manager.saved_df.reset_index(drop=True)
            self.update_event_summary_display()
            logging.info("Event list updated via video editor.")
        if manager.saved_yolo_df is not None:
            self.yolo_detections_df = manager.saved_yolo_df
            logging.info(f"YOLO detections updated via video editor. {len(self.yolo_detections_df)} detections remaining.")

    def open_video_editor_standalone(self):
        """
        Apre l'editor video interattivo con solo un file video,
        senza dati di eye-tracking pre-caricati.
        """
        video_path_str = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if not video_path_str:
            return

        editor = InteractiveVideoEditor(self.root, Path(video_path_str))
        self.root.wait_window(editor)

        if editor.saved_df is not None and not editor.saved_df.empty:
            save_path = filedialog.asksaveasfilename(
                title="Save new events to CSV",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")]
            )
            if save_path:
                editor.saved_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Events saved successfully to:\n{save_path}")

    def update_aoi_list_display(self):
        self.aoi_listbox.delete(0, tk.END)
        for aoi in self.user_defined_aois:
            self.aoi_listbox.insert(tk.END, f"{aoi['name']} ({aoi['type']})")
        
        unenriched_ok = Path(self.unenriched_dir_var.get()).is_dir()
        self.add_aoi_btn.config(state=tk.NORMAL if unenriched_ok else tk.DISABLED)
        self.on_aoi_select(None)

    def on_aoi_select(self, event):
        # --- NUOVO: Abilita il pulsante ViV se ci sono AOI (che genereranno dati enriched) ---
        # CORREZIONE: Il pulsante ViV deve essere abilitato se gli eventi sono stati caricati,
        # non solo se ci sono AOI o cartelle enriched. La presenza di eventi è il vero requisito.
        can_map_viv = Path(self.unenriched_dir_var.get()).is_dir() and not self.events_df.empty
        self.map_viv_btn.config(state=tk.NORMAL if can_map_viv else tk.DISABLED)
        # --- FINE ---
        self.remove_aoi_btn.config(state=tk.NORMAL if self.aoi_listbox.curselection() else tk.DISABLED)

    def remove_selected_aoi(self):
        selected_indices = self.aoi_listbox.curselection()
        if not selected_indices:
            return
        
        if messagebox.askyesno("Confirm", "Are you sure you want to remove the selected AOI?"):
            for index in sorted(selected_indices, reverse=True):
                del self.user_defined_aois[index]
            self.update_aoi_list_display()
            self.on_aoi_select(None) # Aggiorna lo stato del pulsante ViV

    def open_aoi_editor(self):
        choice_dialog = tk.Toplevel(self.root)
        choice_dialog.title("Choose AOI Definition Method")
        choice_dialog.geometry("450x200")
        choice_dialog.transient(self.root)
        choice_dialog.grab_set()
        
        tk.Label(choice_dialog, text="How would you like to define the Area of Interest?", pady=10).pack()
        
        aoi_mode = tk.StringVar(value="static")
        
        ttk.Radiobutton(choice_dialog, text="Static AOI (Fixed Rectangle)", variable=aoi_mode, value="static").pack(anchor='w', padx=20)
        ttk.Radiobutton(choice_dialog, text="Dynamic AOI (Automatic Object Tracking)", variable=aoi_mode, value="dynamic_auto").pack(anchor='w', padx=20)
        ttk.Radiobutton(choice_dialog, text="Dynamic AOI (Manual Keyframes)", variable=aoi_mode, value="dynamic_manual").pack(anchor='w', padx=20)
        ttk.Radiobutton(choice_dialog, text="Surface from Markers (for Enrichment)", variable=aoi_mode, value="marker_surface").pack(anchor='w', padx=20)
        ttk.Radiobutton(choice_dialog, text="Surface from QR Codes (for Enrichment)", variable=aoi_mode, value="qr_surface").pack(anchor='w', padx=20) # NUOVO

        def on_proceed():
            mode = aoi_mode.get()
            choice_dialog.destroy()
            self.launch_specific_aoi_editor(mode)

        tk.Button(choice_dialog, text="Proceed", command=on_proceed, font=('Helvetica', 10, 'bold')).pack(pady=20)

    def launch_specific_aoi_editor(self, mode):
        try:
            video_path = next(Path(self.unenriched_dir_var.get()).glob('*.mp4'))
            analysis_output_path = Path(self.output_dir_entry.get()) if self.output_dir_entry.get() else None
        except StopIteration:
            messagebox.showerror("Error", "No .mp4 video file found in the Un-enriched folder.")
            return

        editor = None
        if mode == 'static' or mode == 'dynamic_auto':
            editor = AoiEditor(self.root, video_path, analysis_output_path=analysis_output_path)
            editor.mode_var.set(mode)
            editor.update_ui_for_mode()
        elif mode == 'dynamic_manual':
            editor = ManualAoiEditor(self.root, video_path)
        elif mode == 'marker_surface':
            editor = MarkerSurfaceEditor(self.root, video_path)
        elif mode == 'qr_surface': # NUOVO
            editor = QRSurfaceEditor(self.root, video_path)

        if editor:
            self.root.wait_window(editor)
            
            new_aoi = None
            if isinstance(editor, AoiEditor):
                if editor.result is not None:
                    new_aoi = {
                        'name': editor.aoi_name, 
                        'type': editor.result_type, 
                        'data': editor.result
                    }
            elif isinstance(editor, ManualAoiEditor):
                 if editor.saved_keyframes:
                    new_aoi = {
                        'name': editor.aoi_name, 
                        'type': 'dynamic_manual', 
                        'data': editor.saved_keyframes
                    }
            elif isinstance(editor, MarkerSurfaceEditor):
                if editor.result is not None:
                    new_aoi = {
                        'name': editor.aoi_name,
                        'type': 'marker_surface',
                        'data': editor.result
                    }
            elif isinstance(editor, QRSurfaceEditor): # NUOVO
                if editor.result is not None:
                    new_aoi = {
                        'name': editor.aoi_name,
                        'type': 'qr_surface',
                        'data': editor.result
                    }

            if new_aoi:
                if any(aoi['name'] == new_aoi['name'] for aoi in self.user_defined_aois):
                    messagebox.showerror("Error", f"An AOI with the name '{new_aoi['name']}' already exists. Please use a unique name.")
                else:
                    self.user_defined_aois.append(new_aoi)
                    logging.info(f"Added new AOI: {new_aoi}")
            
            self.update_aoi_list_display()

    def run_full_analysis_wrapper(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not (output_dir and subj_name and self.raw_dir_var.get() and self.unenriched_dir_var.get()):
            messagebox.showerror("Error", "Participant Name, Output Folder, RAW, and Un-enriched folders are mandatory.")
            return
        
        # Disabilita il pulsante NSI all'inizio di una nuova analisi
        self.nsi_button.config(state=tk.DISABLED)
        
        try:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            final_events_df = self.events_df.copy()
            if not final_events_df.empty:
                cols_to_save = ['name', 'timestamp [ns]', 'recording id', 'selected']
                df_to_save = final_events_df[[col for col in cols_to_save if col in final_events_df.columns]]
                modified_events_path = output_dir_path / 'modified_events.csv'
                df_to_save.to_csv(modified_events_path, index=False)
                logging.info(f"Final user-modified event list saved to: {modified_events_path}")

            messagebox.showinfo("In Progress", "Starting full analysis...")
            
            enriched_path_to_use = self.enriched_dir_var.get() or None
            if self.user_defined_aois:
                enriched_path_to_use = None
                logging.info("User-defined AOIs are present. Ignoring 'Enriched Data Folder' and generating new enriched data.")

            # --- MODIFICA: Recupera i modelli selezionati per ogni task ---
            yolo_models_to_run = {task: var.get() for task, var in self.yolo_model_vars.items() if var.get()}
            custom_classes_str = self.yolo_classes_var.get().strip()
            yolo_custom_classes = None
            if custom_classes_str and custom_classes_str != "person, car, dog": # Ignora il placeholder
                yolo_custom_classes = [cls.strip() for cls in custom_classes_str.split(',') if cls.strip()]
            yolo_tracker_config = self.yolo_tracker_config_var.get() or None
            # --- FINE MODIFICA ---
            
            # --- NUOVO: Passa il DataFrame YOLO filtrato se esiste ---
            yolo_df_to_use = None
            # --- MODIFICA: Usa il video concatenato se esiste ---
            if hasattr(self, 'yolo_detections_df') and not self.yolo_detections_df.empty:
                yolo_cache_path = output_dir_path / 'yolo_detections_cache.csv'
                if yolo_cache_path.exists():
                    full_yolo_df = pd.read_csv(yolo_cache_path)
                    # Mantieni solo le istanze (track_id) che sono presenti nel dataframe filtrato dalla GUI,
                    # ma usa le colonne del file di cache completo per evitare errori di colonne mancanti.
                    yolo_df_to_use = full_yolo_df[full_yolo_df['track_id'].isin(self.yolo_detections_df['track_id'])]
                else:
                    # Se il file di cache non esiste, usa il dataframe in memoria così com'è.
                    yolo_df_to_use = self.yolo_detections_df
            
            # --- NUOVO: Passa il percorso del video concatenato se esiste ---
            concatenated_video_path_str = str(
                self.concatenated_video_path) if self.concatenated_video_path else None
            # --- FINE ---

            run_full_analysis(
                raw_data_path=self.raw_dir_var.get(),
                unenriched_data_path=self.unenriched_dir_var.get(),
                enriched_data_paths=self.enriched_dir_paths if not self.user_defined_aois else None,
                output_path=output_dir,
                subject_name=subj_name,
                events_df=final_events_df,
                yolo_models=yolo_models_to_run if self.yolo_var.get() else None,
                defined_aois=self.user_defined_aois,
                yolo_detections_df=yolo_df_to_use, # Passa il df filtrato se disponibile
                tracker_config_path=yolo_tracker_config, # Passa il file di config del tracker
                concatenated_video_path=concatenated_video_path_str, # Passa il percorso del video ViV
                generate_video=False # Non generare il video da qui
            )

            messagebox.showinfo("Success", f"Full analysis completed.\nResults in: {output_dir}")
        except Exception as e:
            self.analysis_completed = False # L'analisi è fallita
            logging.error(f"Full Analysis Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Analysis Error", f"An error occurred: {e}\n\nSee log for details.")
        else:
            # Se l'analisi ha successo, aggiorna lo stato
            self.analysis_completed = True
            self.update_post_analysis_buttons_state()

    def update_output_dir_default(self, *args):
        # Disabilita i pulsanti post-analisi se il nome del partecipante cambia
        self.analysis_completed = False
        self.update_post_analysis_buttons_state()
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(project_root / f'analysis_results_{subj_name}'))

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(title="Select Output Folder")
        if dir_path: self.output_dir_entry.delete(0, tk.END); self.output_dir_entry.insert(0, dir_path)

    def update_post_analysis_buttons_state(self):
        """Abilita o disabilita i pulsanti degli strumenti post-analisi."""
        # Logica per il pulsante NSI
        can_run_nsi = self.analysis_completed and len(self.user_defined_aois) >= 2
        self.nsi_button.config(state=tk.NORMAL if can_run_nsi else tk.DISABLED)

    def open_nsi_calculator(self):
        """Apre la finestra per il calcolo dell'NSI."""
        output_dir = self.output_dir_entry.get().strip()
        if not output_dir or not Path(output_dir).is_dir():
            messagebox.showerror("Errore", "La cartella di output dell'analisi non è valida o non esiste.", parent=self)
            return
        
        NsiCalculatorWindow(self, Path(output_dir), self.user_defined_aois)

    def _get_common_paths(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not output_dir or not subj_name:
            messagebox.showerror("Error", "Please enter participant name and output folder.")
            return None
        
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        return Path(output_dir), subj_name

    def run_plot_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths

        config_path = output_dir_path / 'config.json'
        if not config_path.exists():
            messagebox.showerror("Error", "Analysis not run yet. Please run 'RUN FULL ANALYSIS' first to generate necessary configuration.")
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            un_enriched_mode = config.get("unenriched_mode", False)
            
            plot_selections = {key: var.get() for key, var in self.plot_vars.items()}

            messagebox.showinfo("In Progress", "Starting plot generation...")
            
            speed_script_events.generate_plots_on_demand(
                output_dir_str=str(output_dir_path),
                subj_name=subj_name,
                plot_selections=plot_selections,
                un_enriched_mode=un_enriched_mode
            )
            
            messagebox.showinfo("Success", f"Plot generation complete!\nPlots saved in: {output_dir_path / 'plots'}")
        except Exception as e:
            logging.error(f"Plot Generation Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Plot Generation Error", f"An error occurred: {e}\n\nSee log for details.")
    def run_video_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths:
            return
        output_dir_path, subj_name = common_paths

        raw_dir_str = self.raw_dir_var.get()
        unenriched_dir_str = self.unenriched_dir_var.get()

        if not all([raw_dir_str, unenriched_dir_str]):
            messagebox.showerror("Error", "RAW and Un-enriched folders are mandatory for video generation.")
            return

        try:
            video_options = {key: var.get() for key, var in self.video_vars.items()}
            video_options['output_filename'] = self.video_filename_var.get()
            video_options['heatmap_window_seconds'] = self.heatmap_video_window_var.get()
            
            # Passa i filtri YOLO al generatore video
            video_options['yolo_class_filter'] = self.yolo_class_filter
            video_options['yolo_id_filter'] = self.yolo_id_filter

            final_events_df = self.events_df.copy()
            if not final_events_df.empty and 'selected' in final_events_df.columns:
                final_events_df = final_events_df[final_events_df['selected']]
            
            working_dir = _prepare_working_directory(
                output_dir=output_dir_path,
                raw_dir=Path(raw_dir_str),
                unenriched_dir=Path(unenriched_dir_str),
                enriched_dirs=self.enriched_dir_paths,
                events_df=final_events_df,
                concatenated_video_path=self.concatenated_video_path
            )

            un_enriched_mode = not bool(self.enriched_dir_paths)
            selected_event_names = final_events_df['name'].tolist() if not final_events_df.empty else []

            messagebox.showinfo("In Progress", "Starting video generation...")
            video_generator.create_custom_video(
                data_dir=working_dir, output_dir=output_dir_path, subj_name=subj_name,
                options=video_options, un_enriched_mode=un_enriched_mode,
                selected_events=selected_event_names
            )
            messagebox.showinfo("Success", f"Video generation complete!\nFile saved in: {output_dir_path}")
        except Exception as e:
            logging.error(f"Video Generation Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Video Generation Error", f"An error occurred: {e}\n\nSee log for details.")

    def open_viv_editor(self):
        """
        Apre l'editor per mappare gli eventi ai file video.
        """
        from desktop_app.video_in_video_editor import VideoInVideoEditor
        if self.events_df.empty:
            messagebox.showerror("Error", "Events must be loaded to create a video-in-video. Please load a dataset.", parent=self.root)
            return
        
        # Passa una copia degli eventi correnti all'editor
        df_for_editor = self.events_df.copy()

        # Se esiste già una mappatura, uniscila.
        if not self.viv_events_df.empty:
            # Colonne da unire dalla mappatura esistente
            map_cols = ['timestamp [ns]', 'video_path']
            # Rimuovi le vecchie colonne di mappatura dal df principale per evitare duplicati
            df_for_editor = df_for_editor.drop(columns=['video_path'], errors='ignore')
            # Unisci la nuova mappatura
            df_for_editor = pd.merge(df_for_editor, self.viv_events_df[map_cols].drop_duplicates(subset=['timestamp [ns]']), on='timestamp [ns]', how='left')

        editor = VideoInVideoEditor(self.root, df_for_editor)
        self.root.wait_window(editor)

        # Il risultato dell'editor è il nuovo DataFrame di mappatura, che salviamo separatamente.
        if editor.result is not None and not editor.result.empty:
            self.viv_events_df = editor.result
            self.create_viv_btn.config(state=tk.NORMAL) # Abilita il pulsante di creazione
            self.viv_status_label.config(text="Status: Event mapping is ready. Press 'Create' to generate the video.", fg="blue")
            logging.info("Video-in-video event mapping updated.")
        elif editor.result is not None: # L'utente ha salvato una mappatura vuota
            # Se l'utente annulla o non mappa nulla, resetta la mappatura.
            self.viv_events_df = pd.DataFrame()
            self.create_viv_btn.config(state=tk.DISABLED)
            self.viv_status_label.config(text="Status: Using original scene video.", fg="grey")
            logging.info("Video-in-video mapping cancelled or no valid mappings provided.")

    def run_video_in_video_creation(self):
        """
        Crea il video concatenato usando la mappatura definita.
        """
        try:
            if self.viv_events_df.empty:
                messagebox.showerror("Error", "No event-to-video mapping found. Please map events first.", parent=self.root)
                return
    
            unenriched_dir_path = Path(self.unenriched_dir_var.get())
            if not unenriched_dir_path.is_dir():
                messagebox.showerror("Error", "Un-enriched folder not set or not found.", parent=self.root)
                return
    
            output_dir = Path(self.output_dir_entry.get())
            # 1. Crea una directory di lavoro temporanea
            working_dir = output_dir / "SPEED_viv_workspace"
            working_dir.mkdir(exist_ok=True, parents=True)
    
            # 2. Copia i file necessari nella directory temporanea
            files_to_copy = ['events.csv', 'gaze.csv']
            for filename in files_to_copy:
                source_path = unenriched_dir_path / filename
                if source_path.exists():
                    shutil.copy(source_path, working_dir / filename)
                else:
                    messagebox.showerror("Error", f"'{filename}' not found in un-enriched folder. This file is required to generate the video.", parent=self.root)
                    shutil.rmtree(working_dir)
                    return
    
            messagebox.showinfo("In Progress", "Creating new scene video from media mapping. This may take a moment...", parent=self.root)
            
            # 3. Chiama la funzione di generazione video
            new_video_path = video_generator.generate_concatenated_video(
                data_dir=working_dir,
                viv_events_df=self.viv_events_df
            )
            
            # 4. Esegui il backup del video originale e sostituiscilo con quello nuovo
            original_video_path = unenriched_dir_path / 'external.mp4'
            if not original_video_path.exists():
                # Se non esiste, prova a cercare un qualsiasi .mp4 come fallback
                try:
                    original_video_path = next(unenriched_dir_path.glob('*.mp4'))
                except StopIteration:
                    raise FileNotFoundError("No original .mp4 video found in the un-enriched folder to replace.")

            timestamp = time.strftime('%Y%m%d-%H%M%S')
            backup_path = original_video_path.with_name(f"{original_video_path.stem}-old-{timestamp}.mp4")
            original_video_path.rename(backup_path)
            logging.info(f"Original video backed up to: {backup_path}")
    
            # 5. Sposta il nuovo video e rinominalo
            final_video_path = unenriched_dir_path / 'external.mp4'
            shutil.move(new_video_path, final_video_path)
            logging.info(f"New concatenated video created at: {final_video_path}")
    
            # 6. Pulisci la cartella di lavoro temporanea
            shutil.rmtree(working_dir)

            # Memorizza il percorso del nuovo video e aggiorna la UI
            self.concatenated_video_path = final_video_path

            self.viv_status_label.config(text=f"Status: New 'external.mp4' created in un-enriched folder.", fg="green")
            self.create_viv_btn.config(state=tk.DISABLED) # Disabilita dopo la creazione per evitare duplicati
            messagebox.showinfo("Success", f"New scene video 'external.mp4' created successfully in:\n{unenriched_dir_path}")

        except Exception as e:
            logging.error(f"Video-in-Video Generation Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Video-in-Video Error", f"An error occurred: {e}\n\nSee log for details.")

    def select_classification_model(self):
        """Apre un file dialog per selezionare un modello di classificazione .pt."""
        filepath = filedialog.askopenfilename(
            title="Select YOLO Classification Model",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")]
        )
        if filepath:
            self.yolo_classify_model_var.set(filepath)

    def run_classification_on_detections(self):
        """Esegue la classificazione sulle detection filtrate."""
        # --- MODIFICA: Ottieni il percorso del modello dal combobox o dal campo custom ---
        selected_model = self.yolo_classify_model_combo_var.get()
        if selected_model == "Custom...":
            model_path_str = self.yolo_classify_model_var.get()
        elif selected_model in OFFICIAL_YOLO_CLS_MODELS:
            # Usa solo il nome del modello. Ultralytics lo scaricherà se non esiste.
            model_path_str = selected_model
        else:
            # È un modello locale, usa il percorso completo.
            model_path_str = str(MODELS_DIR / selected_model) if selected_model else ""
        # --- FINE MODIFICA ---

        # --- MODIFICA: Controllo più robusto ---
        if not model_path_str or (selected_model == "Custom..." and not Path(model_path_str).exists()):
            messagebox.showerror("Error", "Please select a YOLO classification model first.", parent=self.root)
            return

        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, _ = common_paths

        yolo_cache_path = output_dir_path / 'yolo_detections_cache.csv'
        if not yolo_cache_path.exists():
            messagebox.showerror("Error", "yolo_detections_cache.csv not found. Run Analysis with YOLO object detection first.", parent=self.root)
            return

        unenriched_dir_str = self.unenriched_dir_var.get()
        if not unenriched_dir_str:
            messagebox.showerror("Error", "Un-enriched folder is mandatory to get the video.", parent=self.root)
            return
        
        try:
            video_path = next(Path(unenriched_dir_str).glob('*.mp4'))
        except StopIteration:
            messagebox.showerror("Error", "No .mp4 video file found in the Un-enriched folder.", parent=self.root)
            return

        try:
            messagebox.showinfo("In Progress", "Starting classification on detected objects. This may take a while...", parent=self.root)
            
            # Carica i dati e il modello
            detections_df = pd.read_csv(yolo_cache_path)
            # L'istanziazione di YOLO qui gestirà il download automatico se model_path_str è solo un nome (es. 'yolov8n-cls.pt')
            # e non un percorso completo.
            classify_model = YOLO(model_path_str)
            
            cap = cv2.VideoCapture(str(video_path))

            # Applica i filtri dalla GUI
            self.update_yolo_filters()
            filtered_df = detections_df.copy()
            if self.yolo_class_filter:
                filtered_df = filtered_df[filtered_df['class_name'].isin(self.yolo_class_filter)]
            if self.yolo_id_filter:
                filtered_df = filtered_df[filtered_df['track_id'].isin(self.yolo_id_filter)]

            # Esegui classificazione
            classification_results = video_generator.classify_detections(cap, filtered_df, classify_model)
            cap.release()

            # Unisci e salva i risultati
            if classification_results:
                results_df = pd.DataFrame(classification_results)
                detections_df = detections_df.merge(results_df, on=['frame_idx', 'track_id'], how='left')
                detections_df.to_csv(yolo_cache_path, index=False) # Sovrascrivi il cache con i nuovi dati
                results_df.to_csv(output_dir_path / 'yolo_classification_results.csv', index=False)
                messagebox.showinfo("Success", f"Classification complete. Results saved and merged into cache.", parent=self.root)
                self.load_yolo_results() # Ricarica i risultati nella GUI
            else:
                messagebox.showinfo("Info", "No objects were classified.", parent=self.root)

        except Exception as e:
            logging.error(f"Error during classification: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Classification Error", f"An error occurred: {e}", parent=self.root)

    def load_yolo_results_for_filtering(self):
        """Carica i risultati YOLO e popola i treeview per il filtraggio."""
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, _ = common_paths

        yolo_cache_path = output_dir_path / 'yolo_detections_cache.csv'
        if not yolo_cache_path.exists():
            messagebox.showinfo("Info", "yolo_detections_cache.csv not found. Run Analysis with YOLO enabled first.", parent=self.root)
            return

        try:
            self.yolo_detections_df = pd.read_csv(yolo_cache_path)
            self._reset_yolo_filter_ui()

            for task, group_df in self.yolo_detections_df.groupby('task'):
                task_name_map = {'detect': 'detection', 'segment': 'segmentation', 'pose': 'pose', 'obb': 'obb'}
                task_key = task_name_map.get(task)
                if task_key not in self.yolo_filter_trees: continue

                tree = self.yolo_filter_trees[task_key]['tree']
                tab = self.yolo_filter_trees[task_key]['tab']
                self.yolo_filter_notebook.add(tab, text=task.capitalize())

                detected_items = group_df.groupby('class_name')['track_id'].unique().apply(list).to_dict()
                for class_name, ids in sorted(detected_items.items()):
                    class_node = tree.insert("", "end", text=f"Class: {class_name}", open=True, values=("☑",), tags=('class', class_name))
                    for track_id in sorted(ids):
                        tree.insert(class_node, "end", text=f"  ID: {track_id}", values=("☑",), tags=('id', track_id))
            
            self.update_yolo_filters() # Imposta i filtri iniziali (tutto selezionato)

        except Exception as e:
            logging.error(f"Could not load or process YOLO results: {e}")
            messagebox.showerror("Error", f"Could not load or process YOLO results: {e}", parent=self.root)

    def on_yolo_filter_click(self, event):
        tree = event.widget
        item_id = tree.identify_row(event.y)
        column = tree.identify_column(event.x)
        if not item_id or column != "#1": return

        current_val = tree.item(item_id, 'values')[0]
        new_val = "☐" if current_val == "☑" else "☑"
        tree.set(item_id, column, new_val)

        tags = tree.item(item_id, 'tags')
        if tags and tags[0] == 'class':
            for child_id in tree.get_children(item_id):
                tree.set(child_id, column, new_val)
        
        self.update_yolo_filters()

    def _reset_yolo_filter_ui(self):
        for task_key, data in self.yolo_filter_trees.items():
            data['tree'].delete(*data['tree'].get_children())
            try:
                self.yolo_filter_notebook.hide(data['tab'])
            except tk.TclError: pass

    def load_yolo_results(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, _ = common_paths
        try:
            try:
                class_csv = output_dir_path / 'stats_per_class.csv'
                if class_csv.exists(): 
                    df = pd.read_csv(class_csv)
                    self._populate_treeview(self.class_treeview, df)
                else: self._clear_treeview(self.class_treeview); messagebox.showinfo("Info", "stats_per_class.csv not found. Run Analysis with YOLO enabled.", parent=self.root)
                
                instance_csv = output_dir_path / 'stats_per_instance.csv'
                if instance_csv.exists():
                    self.instance_stats_df = pd.read_csv(instance_csv)
                    # --- NUOVO: Controlla se ci sono risultati di classificazione ---
                    classification_results_csv = output_dir_path / 'yolo_classification_results.csv'
                    if classification_results_csv.exists():
                        class_df = pd.read_csv(classification_results_csv)
                        # Aggrega per ottenere la classe più probabile per istanza
                        top_class_per_instance = class_df.loc[class_df.groupby('track_id')['confidence'].idxmax()]
                        self.instance_stats_df = self.instance_stats_df.merge(top_class_per_instance[['track_id', 'classification_class', 'confidence']], on='track_id', how='left')
                    self._populate_instance_treeview()
                else:
                    self._clear_treeview(self.instance_treeview); self.instance_stats_df = pd.DataFrame(); messagebox.showinfo("Info", "stats_per_instance.csv not found. Run Analysis with YOLO enabled.", parent=self.root) # Mostra il messaggio ma non procedere oltre
            except pd.errors.EmptyDataError:
                self._clear_treeview(self.class_treeview); self._clear_treeview(self.instance_treeview)
                messagebox.showinfo("Info", "YOLO results files are empty. No objects were detected or fixated upon.", parent=self.root)
        except Exception as e:
            logging.error(f"Could not read YOLO results: {e}")
            messagebox.showerror("Read Error", f"Could not read YOLO results: {e}", parent=self.root)

    def _populate_instance_treeview(self, df_to_show=None):
        """Popola il treeview delle istanze, con checkbox."""
        self._clear_treeview(self.instance_treeview)
        df = df_to_show if df_to_show is not None else self.instance_stats_df
        
        # --- MODIFICA: Aggiungi colonne per la classificazione se esistono ---
        base_cols = ["Show", "Instance", "Fixations", "Avg Pupil", "Norm. Fixations"]
        extra_cols = []
        if 'classification_class' in df.columns: extra_cols.extend(["Classification", "Confidence"])
        cols = tuple(base_cols + extra_cols)
        self.instance_treeview['columns'] = cols
        
        for col in cols:
            self.instance_treeview.heading(col, text=col.replace('_', ' ').title())
        self.instance_treeview.column("Show", width=50, anchor='center')
        # --- FINE CORREZIONE ---
        if df.empty:
            return

        for _, row in df.iterrows():
            base_values = [f"{row.get(c, ''):.3f}" if isinstance(row.get(c), float) else row.get(c, '') for c in ['instance_name', 'n_fixations', 'avg_pupil_diameter_mm', 'normalized_fixation_count']]
            extra_values = []
            if 'classification_class' in df.columns:
                conf_val = f"{row.get('confidence', 0.0):.2f}" if pd.notna(row.get('confidence')) else ''
                extra_values.extend([row.get('classification_class', ''), conf_val])

            values = ["☑"] + base_values + extra_values
            self.instance_treeview.insert("", "end", values=values, tags=(row['instance_name'],))

    def on_instance_filter_click(self, event):
        """Gestisce il click sulla colonna 'Show' per filtrare i risultati."""
        region = self.instance_treeview.identify("region", event.x, event.y)
        column = self.instance_treeview.identify_column(event.x)
        if region != "cell" or column != "#1": return

        item_id = self.instance_treeview.identify_row(event.y)
        current_val = self.instance_treeview.item(item_id, 'values')[0]
        new_val = "☐" if current_val == "☑" else "☑"
        self.instance_treeview.set(item_id, column, new_val)
        
        shown_instances = {self.instance_treeview.item(i, 'tags')[0] for i in self.instance_treeview.get_children() if self.instance_treeview.item(i, 'values')[0] == "☑"}
        filtered_df = self.instance_stats_df[self.instance_stats_df['instance_name'].isin(shown_instances)]
        # Potresti usare filtered_df per aggiornare altri grafici o analisi in futuro.

    def update_yolo_filters(self):
        """Aggiorna i set di filtri globali in base allo stato dei treeview."""
        self.yolo_class_filter.clear()
        self.yolo_id_filter.clear()
        
        for task_key, data in self.yolo_filter_trees.items():
            tree = data['tree']
            for class_node in tree.get_children(''):
                if tree.item(class_node, 'values')[0] == "☑":
                    self.yolo_class_filter.add(tree.item(class_node, 'tags')[1])
                    for id_node in tree.get_children(class_node):
                        if tree.item(id_node, 'values')[0] == "☑":
                            self.yolo_id_filter.add(int(tree.item(id_node, 'tags')[1]))

        if not self.yolo_detections_df.empty:
            if len(self.yolo_class_filter) == len(self.yolo_detections_df['class_name'].unique()): self.yolo_class_filter.clear()
            if len(self.yolo_id_filter) == len(self.yolo_detections_df['track_id'].unique()): self.yolo_id_filter.clear()
        logging.info(f"YOLO filters updated. Classes: {len(self.yolo_class_filter) or 'All'}, IDs: {len(self.yolo_id_filter) or 'All'}")

    def _populate_treeview(self, treeview, dataframe):
        treeview.delete(*treeview.get_children())
        if dataframe.empty:
            treeview["columns"] = (" ")
            treeview.heading("#0", text="")
            return
            
        treeview["columns"] = list(dataframe.columns)
        for col in dataframe.columns:
            treeview.heading(col, text=col)
            treeview.column(col, width=150, anchor='center')
        for index, row in dataframe.iterrows():
            treeview.insert("", "end", values=list(row))

    def _clear_treeview(self, treeview):
        treeview.delete(*treeview.get_children())
        # --- CORREZIONE: Gestione specifica per il treeview delle istanze ---
        if treeview == self.instance_treeview:
            # Rileva le colonne dinamicamente come in _populate_instance_treeview
            base_cols = ["Show", "Instance", "Fixations", "Avg Pupil", "Norm. Fixations"]
            extra_cols = []
            if 'classification_class' in self.instance_stats_df.columns: extra_cols.extend(["Classification", "Confidence"])
            cols = tuple(base_cols + extra_cols)
            treeview['columns'] = cols
            for col in cols:
                treeview.heading(col, text=col.replace('_', ' ').title())
            treeview.column("Show", width=50, anchor='center')
        else:
            treeview["columns"] = (" ")
            treeview.heading("#0", text="") # Usa #0 per la colonna di default
            
    def load_bids_data(self):
        bids_root_path = filedialog.askdirectory(title="Select the root BIDS directory (containing sub-...)")
        if not bids_root_path:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("BIDS Data Selection")
        dialog.geometry("400x250")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Specify the data to load from BIDS:", pady=10).pack()

        tk.Label(dialog, text="Subject ID (es. 01):").pack()
        subject_entry = tk.Entry(dialog)
        subject_entry.pack(padx=20, fill=tk.X)

        tk.Label(dialog, text="Session ID (es. 01):").pack()
        session_entry = tk.Entry(dialog)
        session_entry.pack(padx=20, fill=tk.X)

        tk.Label(dialog, text="Task Name (es. reading):").pack()
        task_entry = tk.Entry(dialog)
        task_entry.pack(padx=20, fill=tk.X)
        
        def on_load():
            subject_id = subject_entry.get().strip()
            session_id = session_entry.get().strip()
            task_name = task_entry.get().strip()

            if not all([subject_id, session_id, task_name]):
                messagebox.showwarning("Input Missing", "All fields are required.", parent=dialog)
                return
            
            dialog.destroy()
            
            try:
                messagebox.showinfo("In Progress", "Loading and converting BIDS data...")
                temp_unenriched_path = load_from_bids(
                    bids_dir=Path(bids_root_path),
                    subject_id=subject_id,
                    session_id=session_id,
                    task_name=task_name
                )
                
                # Popola automaticamente il campo "Un-enriched" con la cartella temporanea
                self.unenriched_dir_var.set(str(temp_unenriched_path))
                # Imposta anche il nome del partecipante e la cartella di output di default
                self.participant_name_var.set(f"sub-{subject_id}_ses-{session_id}")
                
                messagebox.showinfo("Success", "BIDS data loaded successfully and converted for SPEED analysis.\nThe 'Un-enriched' path has been set automatically.")

            except Exception as e:
                logging.error(f"Error loading from BIDS: {e}\n{traceback.format_exc()}")
                messagebox.showerror("Loading Error", f"An error occurred: {e}\n\nCheck logs for details.")

        tk.Button(dialog, text="Load Data", command=on_load, font=('Helvetica', 10, 'bold')).pack(pady=20)
        

if __name__ == '__main__':
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"speed_log_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    logging.info("Application started.")
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()
    logging.info("Application closed.")