# interactive_video_editor.py
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
from pathlib import Path
import pandas as pd
import cv2
from PIL import Image, ImageTk
import numpy as np
import json
from moviepy import VideoFileClip, AudioFileClip
import logging
import torch
from tqdm import tqdm
import threading
import pygame

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

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
class InteractiveVideoEditor(tk.Toplevel):
    """
    Una finestra di editor video interattiva per aggiungere, rimuovere e modificare
    eventi e visualizzare/filtrare risultati YOLO.
    """
    def __init__(self, parent, video_path, events_df: pd.DataFrame = None, world_timestamps_df: pd.DataFrame = None, yolo_df: pd.DataFrame = None):
        super().__init__(parent)
        self.title("Interactive Video Event Editor")
        self.geometry("1400x850")
        self.transient(parent)
        self.grab_set()
        self.root = parent
        self.video_path = video_path

        if not video_path or not video_path.exists():
            messagebox.showerror("Error", f"Video file not found:\n{video_path}", parent=self)
            self.destroy()
            return
            
        if events_df is None:
            self.events_df = pd.DataFrame(columns=['name', 'timestamp [ns]', 'selected', 'source', 'recording id'])
        else:
            self.events_df = events_df.copy().reset_index(drop=True)
            if 'timestamp [ns]' not in self.events_df.columns and not self.events_df.empty:
                messagebox.showerror("Error", "Events DataFrame must have a 'timestamp [ns]' column.", parent=self)
                self.destroy()
                return
            
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        self.original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if world_timestamps_df is None:
            timestamps_ns = (np.arange(self.total_frames) * (1e9 / self.fps)).astype('int64')
            self.world_ts = pd.DataFrame({'timestamp [ns]': timestamps_ns, 'frame': np.arange(self.total_frames)})
        else:
            self.world_ts = world_timestamps_df

        self.saved_df = None
        self.saved_yolo_df = None
        self.is_playing = False
        self.current_frame_idx = 0
        
        # --- NUOVO: Gestione Audio ---
        self.audio_clip = None
        self.audio_thread = None
        self.is_muted = tk.BooleanVar(value=True)
        self._load_audio()
        
        # --- CORREZIONE: Aggiunto flag per prevenire loop ---
        self.is_updating_slider = False

        # --- NUOVO: Gestione YOLO ---
        self.yolo_df = yolo_df if yolo_df is not None else pd.DataFrame()
        self.yolo_model = None
        self.yolo_class_filter = set()
        self.yolo_id_filter = set() # Set di ID da mostrare
        self.detected_yolo_items = {} # Cache per {class_name: [id1, id2]}
        self.yolo_models = {} # Dizionario per i modelli caricati {task: model}
        
        # --- Layout Principale con PanedWindow ---
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        main_frame = tk.Frame(main_pane)
        main_pane.add(main_frame, stretch="always")

        self.video_label = tk.Label(main_frame)
        self.video_label.pack(pady=10)

        self.timeline_canvas = tk.Canvas(main_frame, height=80, bg='lightgrey')
        self.timeline_canvas.pack(fill=tk.X, padx=10, side=tk.BOTTOM)
        self.timeline_canvas.bind("<Button-1>", self.handle_timeline_click)
        self.timeline_canvas.bind("<B1-Motion>", self.handle_timeline_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.handle_timeline_release)
        self.dragged_event_index = None

        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, padx=10, side=tk.BOTTOM)
        self.play_pause_btn = tk.Button(controls_frame, text="▶ Play", command=self.toggle_play, width=10)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)

        # --- NUOVO: Pulsante Mute/Unmute ---
        self.mute_btn = tk.Button(controls_frame, text="🔇 Unmute", command=self.toggle_mute)
        self.mute_btn.pack(side=tk.LEFT, padx=5)
        self.update_mute_button_text()
        
        self.frame_scale = ttk.Scale(controls_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(controls_frame, text="Frame: 0 / 0", width=20)
        self.time_label.pack(side=tk.RIGHT, padx=5)

        action_frame = tk.Frame(main_frame)
        action_frame.pack(side=tk.BOTTOM, pady=10)
        tk.Button(action_frame, text="Add Event at Current Frame", command=self.add_event_at_frame, bg='#c8e6c9').pack(side=tk.LEFT, padx=10)
        tk.Button(action_frame, text="Remove Selected Event", command=self.remove_selected_event, bg='#ffcdd2').pack(side=tk.LEFT, padx=10)
        tk.Button(action_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        tk.Button(action_frame, text="Export Video...", command=self.export_video_dialog, bg='#ffcc80').pack(side=tk.RIGHT, padx=10)
        
        # --- NUOVO: Pannello di controllo YOLO a destra ---
        yolo_panel = tk.Frame(main_pane, width=350, relief=tk.SUNKEN, borderwidth=1)
        yolo_panel.pack_propagate(False)
        main_pane.add(yolo_panel)

        # --- MODIFICA: Spostato il pulsante in un frame dedicato per chiarezza ---
        video_management_frame = tk.LabelFrame(yolo_panel, text="Video Management")
        video_management_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(video_management_frame, text="Load New Video...", command=self.load_new_video).pack(fill=tk.X, pady=5)
        # --- FINE MODIFICA ---

        yolo_run_frame = tk.LabelFrame(yolo_panel, text="YOLO Analysis")
        yolo_run_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # --- MODIFICA: Selezione Multi-Modello ---
        self.yolo_model_vars = {
            'detect': tk.StringVar(),
            'segment': tk.StringVar(),
            'pose': tk.StringVar()
        }
        self.yolo_model_combos = {}

        for task in ['detect', 'segment', 'pose']:
            tk.Label(yolo_run_frame, text=f"{task.capitalize()} Model:").pack(anchor='w', pady=(5,0))
            combo = ttk.Combobox(yolo_run_frame, textvariable=self.yolo_model_vars[task], state='readonly')
            combo.pack(fill=tk.X)
            self.yolo_model_combos[task] = combo

        self.update_yolo_model_options()
        # --- FINE MODIFICA ---
        
        self.run_yolo_btn = tk.Button(yolo_run_frame, text="Run YOLO on Video", command=self.run_yolo_analysis)
        self.run_yolo_btn.pack(pady=5, fill=tk.X)
        
        if YOLO is None:
            self.run_yolo_btn.config(text="YOLO not installed", state=tk.DISABLED)

        yolo_filter_frame = tk.LabelFrame(yolo_panel, text="YOLO Object Filter")
        yolo_filter_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- MODIFICA: Aggiunta colonna per checkbox ---
        self.yolo_filter_tree = ttk.Treeview(yolo_filter_frame, columns=("#1"), show="tree headings")
        self.yolo_filter_tree.heading("#0", text="Object")
        self.yolo_filter_tree.heading("#1", text="Show")
        self.yolo_filter_tree.column("#1", width=50, anchor='center')
        self.yolo_filter_tree.pack(fill=tk.BOTH, expand=True)
        self.yolo_filter_tree.bind('<Button-1>', self.on_yolo_filter_click) # Associazione evento

        self.selected_event_index = None
        if not self.yolo_df.empty:
            self.process_yolo_data()

        self.update_frame(self.current_frame_idx)
        self.after(100, lambda: self.draw_timeline())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_audio(self):
        """Carica la traccia audio dal file video in memoria."""
        try:
            pygame.mixer.init()
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

    def toggle_mute(self):
        """Attiva o disattiva l'audio."""
        self.is_muted.set(not self.is_muted.get())
        self.update_mute_button_text()
        if not self.is_muted.get() and self.is_playing:
            # Se si riattiva l'audio durante la riproduzione, riavvia l'audio dal punto giusto
            self.play_audio()

    def update_mute_button_text(self):
        """Aggiorna il testo del pulsante mute."""
        if self.is_muted.get():
            self.mute_btn.config(text="🔇 Unmute")
        else:
            self.mute_btn.config(text="🔊 Mute")
        
        if not self.audio_clip:
            self.mute_btn.config(state=tk.DISABLED, text="No Audio")


    def load_new_video(self):
        """Permette di caricare un nuovo video all'interno dell'editor esistente."""
        new_video_path_str = filedialog.askopenfilename(
            title="Select a new video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
            parent=self
        )
        if not new_video_path_str:
            return

        # Rilascia le risorse del video corrente
        if self.cap:
            self.cap.release()

        # Resetta lo stato dell'editor
        self.video_path = Path(new_video_path_str)
        self.events_df = pd.DataFrame(columns=['name', 'timestamp [ns]', 'selected', 'source', 'recording id'])
        self.yolo_df = pd.DataFrame()
        self.yolo_models = {}
        self.detected_yolo_items = {}
        self.yolo_class_filter.clear()
        self.yolo_id_filter.clear()
        self.selected_event_index = None
        self.is_playing = False

        # Ricarica i componenti video
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        timestamps_ns = (np.arange(self.total_frames) * (1e9 / self.fps)).astype('int64')
        self.world_ts = pd.DataFrame({'timestamp [ns]': timestamps_ns, 'frame': np.arange(self.total_frames)})
        self.frame_scale.config(to=self.total_frames - 1)

        # Ricarica l'audio per il nuovo video
        self._load_audio()
        self.update_mute_button_text()
        # Pulisci e aggiorna la UI
        self.yolo_filter_tree.delete(*self.yolo_filter_tree.get_children())
        self.update_frame(0)
        
    def update_yolo_model_options(self, event=None):
        """Aggiorna i modelli disponibili in base al task."""
        # Usa la costante globale definita in GUI.py
        from desktop_app.GUI import YOLO_MODELS

        for task, combo in self.yolo_model_combos.items():
            # --- NUOVA LOGICA: Raccoglie tutti i modelli per un task base (es. 'detect') ---
            all_task_models = []
            for model_key, model_list in YOLO_MODELS.items():
                if model_key.startswith(task):
                    all_task_models.extend(model_list)
            
            # Rimuovi duplicati e ordina, aggiungendo l'opzione vuota
            task_models = [""] + sorted(list(set(all_task_models)))
            combo['values'] = task_models
            self.yolo_model_vars[task].set(task_models[0]) # Seleziona l'opzione vuota

    def on_close(self):
        self.is_playing = False
        if self.audio_thread and self.audio_thread.is_alive():
            pygame.mixer.music.stop()
        self.cap.release()
        self.destroy()

    def seek_frame(self, frame_idx_str):
        # --- CORREZIONE: Controlla il flag ---
        if self.is_updating_slider or self.is_playing:
            return
        self.update_frame(int(float(frame_idx_str)))

    def update_frame(self, frame_idx):
        self.current_frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        
        # --- CORREZIONE: Usa il flag ---
        self.is_updating_slider = True
        self.frame_scale.set(self.current_frame_idx)
        self.is_updating_slider = False
        
        self.time_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((900, 650))
            # Converti in BGR per OpenCV
            frame_to_draw = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # --- NUOVO: Disegna overlay YOLO ---
            self.draw_yolo_overlays(frame_to_draw)

            # --- NUOVO: Disegna overlay Eventi ---
            current_ts = self.world_ts.iloc[self.current_frame_idx]['timestamp [ns]']
            self.draw_event_overlay(frame_to_draw, current_ts)


            # Riconverti per Tkinter
            final_img = Image.fromarray(cv2.cvtColor(frame_to_draw, cv2.COLOR_BGR2RGB))
            self.photo = ImageTk.PhotoImage(image=final_img)
            self.video_label.config(image=self.photo)

            self.draw_timeline()

    def draw_event_overlay(self, frame, current_ts):
        """Disegna il nome dell'evento corrente sul frame."""
        if self.events_df.empty:
            return
        
        active_events = self.events_df[self.events_df['timestamp [ns]'] <= current_ts]
        if not active_events.empty:
            current_event_name = active_events.iloc[-1]['name']
            
            font_scale = 0.7
            font_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(current_event_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            img_h, img_w, _ = frame.shape
            text_pos = (15, 30)
            
            cv2.putText(frame, current_event_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def draw_yolo_overlays(self, frame):
        """Disegna bounding box, maschere o pose sul frame."""
        if self.yolo_df.empty:
            return

        detections_for_frame = self.yolo_df[self.yolo_df['frame_idx'] == self.current_frame_idx].copy()

        # --- MODIFICA: Applica i filtri per nascondere gli elementi deselezionati ---
        # Se il set di filtri non è vuoto, significa che alcuni elementi sono stati deselezionati.
        # Quindi, mostriamo solo gli elementi che sono ancora nei nostri set di filtri.
        # Se un set di filtri è vuoto, significa che tutti gli elementi di quel tipo sono selezionati, quindi non applichiamo quel filtro.
        if self.yolo_class_filter: # Se non è vuoto, applica il filtro
            detections_for_frame = detections_for_frame[detections_for_frame['class_name'].isin(self.yolo_class_filter)]
        
        if self.yolo_id_filter: # Se non è vuoto, applica il filtro
            detections_for_frame = detections_for_frame[detections_for_frame['track_id'].isin(self.yolo_id_filter)]
        # --- FINE MODIFICA ---
        if detections_for_frame.empty: return

        # Scala le coordinate alle dimensioni del thumbnail
        thumb_h, thumb_w, _ = frame.shape
        scale_w = thumb_w / self.original_w
        scale_h = thumb_h / self.original_h

        # --- NUOVO: Crea un'immagine di overlay per le maschere ---
        overlay_mask = frame.copy()

        for _, det in detections_for_frame.iterrows():
            # Disegna Bounding Box (sempre)
            x1, y1 = int(det['x1'] * scale_w), int(det['y1'] * scale_h)
            x2, y2 = int(det['x2'] * scale_w), int(det['y2'] * scale_h)
            color = (0, 255, 255) # Ciano per i box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = f"{det.get('class_name', 'Obj')}:{int(det['track_id'])}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Disegna Maschere di Segmentazione
            if 'mask_contours' in det and pd.notna(det['mask_contours']):
                try:
                    contour = np.array(json.loads(det['mask_contours'])).astype(np.int32)
                    contour[:, 0] = (contour[:, 0] * scale_w).astype(np.int32)
                    contour[:, 1] = (contour[:, 1] * scale_h).astype(np.int32)
                    # Disegna la maschera sull'immagine di overlay separata
                    cv2.fillPoly(overlay_mask, [contour], (0, 255, 0)) # Verde per le maschere
                except Exception:
                    pass

            # Disegna Scheletri di Posa
            if 'keypoints' in det and pd.notna(det['keypoints']) and det['keypoints'] != '[]':
                try:
                    keypoints = np.array(json.loads(det['keypoints']))
                    keypoints[:, 0] = (keypoints[:, 0] * scale_w).astype(np.int32)
                    keypoints[:, 1] = (keypoints[:, 1] * scale_h).astype(np.int32)
                    # Semplice disegno dei punti
                    for x, y, conf in keypoints:
                        if conf > 0.5:
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1) # Rosso per i keypoint
                except Exception:
                    pass
        
        # --- NUOVO: Applica l'overlay delle maschere una sola volta alla fine ---
        cv2.addWeighted(overlay_mask, 0.3, frame, 0.7, 0, dst=frame)


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
                    # Crea un file audio temporaneo per la parte di audio da riprodurre
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

    def get_frame_from_ts(self, ts):
        if self.world_ts.empty: return 0
        # Trova l'indice più vicino nel dataframe world_ts
        match_index = (self.world_ts['timestamp [ns]'] - ts).abs().idxmin()
        return self.world_ts.loc[match_index, 'frame']

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
            # Se l'audio è in riproduzione, riavvialo dalla nuova posizione
            if self.is_playing:
                self.play_audio()
        self.draw_timeline()

    def handle_timeline_drag(self, event):
        if self.dragged_event_index is not None:
            canvas_width = self.timeline_canvas.winfo_width()
            new_frame = max(0, min(int((event.x / canvas_width) * self.total_frames), self.total_frames - 1))
            if new_frame < len(self.world_ts):
                self.events_df.loc[self.dragged_event_index, 'timestamp [ns]'] = self.world_ts.iloc[new_frame]['timestamp [ns]']
                # Se l'audio è in riproduzione, riavvialo
                if self.is_playing:
                    self.play_audio()
                self.update_frame(new_frame)
            
    def handle_timeline_release(self, event):
        if self.dragged_event_index is not None:
            self.dragged_event_index = None
            self.events_df.sort_values('timestamp [ns]', inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.selected_event_index = None
            self.draw_timeline()

    def add_event_at_frame(self):
        name = simpledialog.askstring("Add Event", "Enter event name:", parent=self)
        if name and self.current_frame_idx < len(self.world_ts):
            ts = self.world_ts.iloc[self.current_frame_idx]['timestamp [ns]']
            new_row = {'name': name, 'timestamp [ns]': ts, 'selected': True, 'source': 'manual', 'recording id': 'rec_001'}
            self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
            self.events_df.sort_values('timestamp [ns]', inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.draw_timeline()

    def run_yolo_analysis(self):
        if YOLO is None: return
        
        selected_models = {task: name for task, name in self.yolo_model_vars.items() if name.get()}
        if not selected_models:
            messagebox.showerror("Error", "Please select at least one YOLO model to run.", parent=self)
            return

        try:
            self.yolo_models.clear()
            for task, model_name in selected_models.items():
                model_path = MODELS_DIR / model_name.get()
                self.yolo_models[task] = YOLO(model_path)
                logging.info(f"Loaded {task} model: {model_name.get()}")
        except Exception as e:
            logging.error(f"Failed to load/download one or more YOLO models: {e}")
            messagebox.showerror("YOLO Error", f"Failed to load model: {e}", parent=self)
            return

        self.run_yolo_btn.config(text="Analyzing...", state=tk.DISABLED)
        self.root.update_idletasks()

        # Esegui l'analisi in un thread per non bloccare la GUI
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
            # Mappa per unire i risultati basata su track_id
            frame_detections = {}

            # Processa ogni task in modo indipendente
            for task, results in all_results.items():
                res = results[0] # Prendi il primo risultato del batch
                if res.boxes is None or res.boxes.id is None: continue

                for i, box in enumerate(res.boxes):
                    track_id = int(box.id[0])
                    
                    # Inizializza il dizionario per questo ID se non esiste
                    if track_id not in frame_detections:
                        frame_detections[track_id] = {'frame_idx': frame_idx, 'track_id': track_id}

                    # Aggiungi dati comuni (box, classe) se disponibili
                    if 'class_id' not in frame_detections[track_id]:
                        class_id = int(box.cls[0])
                        class_name = self.yolo_models[task].names[class_id]
                        xyxy = box.xyxy[0].cpu().numpy()
                        frame_detections[track_id].update({
                            'class_id': class_id, 'class_name': class_name,
                            'x1': xyxy[0], 'y1': xyxy[1], 'x2': xyxy[2], 'y2': xyxy[3]
                        })

                    # Aggiungi dati di segmentazione
                    if task == 'segment' and res.masks and i < len(res.masks.xy):
                        frame_detections[track_id]['mask_contours'] = json.dumps(res.masks.xy[i].tolist())

                    # Aggiungi dati di posa
                    if task == 'pose' and res.keypoints and i < len(res.keypoints.xy):
                        kpts_xy = res.keypoints.xy[i].cpu().numpy()
                        kpts_conf_tensor = res.keypoints.conf[i] if res.keypoints.conf is not None else torch.ones(len(kpts_xy))
                        kpts_conf = kpts_conf_tensor.cpu().numpy()[:, None]
                        kpts_with_conf = np.hstack((kpts_xy, kpts_conf))
                        frame_detections[track_id]['keypoints'] = json.dumps(kpts_with_conf.tolist())

            # Aggiungi tutte le rilevazioni del frame alla lista principale
            detections.extend(frame_detections.values())
            # --- FINE NUOVA LOGICA ---

        cap.release()
        self.yolo_df = pd.DataFrame(detections)
        # --- CORREZIONE: Controlla se la finestra esiste ancora prima di aggiornare la UI ---
        if self.winfo_exists():
            self.after(0, self.process_yolo_data)
            self.after(0, lambda: self.run_yolo_btn.config(text="Run YOLO on Video", state=tk.NORMAL))
        # --- FINE CORREZIONE ---


    def process_yolo_data(self):
        """Popola il treeview dei filtri e aggiorna la visualizzazione."""
        if self.yolo_df.empty: return

        self.detected_yolo_items = self.yolo_df.groupby('class_name')['track_id'].unique().apply(list).to_dict()

        # Pulisci il treeview precedente
        self.yolo_filter_tree.delete(*self.yolo_filter_tree.get_children())

        # Popola con i nuovi filtri
        # --- MODIFICA: Inserimento con valore per checkbox ---
        for class_name, ids in sorted(self.detected_yolo_items.items()):
            class_node = self.yolo_filter_tree.insert("", "end", text=f"Class: {class_name}", open=True, values=("☑",), tags=('class', class_name))
            for track_id in sorted(ids):
                self.yolo_filter_tree.insert(class_node, "end", text=f"  ID: {track_id}", values=("☑",), tags=('id', track_id))

        self.update_frame(self.current_frame_idx)

    def on_yolo_filter_click(self, event):
        """Gestisce il click per attivare/disattivare i filtri."""
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
                    track_id = int(self.yolo_filter_tree.item(id_node, 'tags')[1])
                    all_ids.add(track_id)
                    if self.yolo_filter_tree.item(id_node, 'values')[0] == "☑":
                        self.yolo_id_filter.add(track_id)

        # Se tutti gli elementi sono selezionati, il set di filtri dovrebbe essere vuoto per non filtrare nulla.
        if len(self.yolo_class_filter) == len(all_classes):
            self.yolo_class_filter.clear()
        if len(self.yolo_id_filter) == len(all_ids):
            self.yolo_id_filter.clear()

        self.update_frame(self.current_frame_idx)

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
        if not self.yolo_df.empty:
            shown_instances = self.yolo_df['class_name'].isin(self.yolo_class_filter or self.yolo_df['class_name'].unique()) & \
                              self.yolo_df['track_id'].isin(self.yolo_id_filter or self.yolo_df['track_id'].unique())
            self.saved_yolo_df = self.yolo_df[shown_instances].copy()

        self.on_close()

    def export_video_dialog(self):
        """Apre una finestra di dialogo per le opzioni di esportazione video."""
        dialog = tk.Toplevel(self)
        dialog.title("Export Video Options")
        dialog.geometry("400x150")
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(dialog, text="Configure video export:", pady=10).pack()

        include_audio_var = tk.BooleanVar(value=True)
        tk.Checkbutton(dialog, text="Include Audio from Original Video", variable=include_audio_var).pack(pady=5)

        def on_export():
            output_path = filedialog.asksaveasfilename(
                title="Save Video As",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")]
            )
            if not output_path:
                return
            
            dialog.destroy()
            self.export_video(output_path, include_audio_var.get())

        tk.Button(dialog, text="Export", command=on_export, font=('Helvetica', 10, 'bold')).pack(pady=10)

    def export_video(self, output_path, include_audio):
        """Esporta il video con gli overlay correnti."""
        temp_video_path = Path(output_path).with_suffix('.temp.mp4')
        
        # Prepara lo scrittore video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Usa le dimensioni originali del video per la massima qualità
        original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(temp_video_path), fourcc, self.fps, (original_w, original_h))

        # Finestra di progresso
        progress_win = tk.Toplevel(self)
        progress_win.title("Exporting Video")
        progress_win.geometry("300x80")
        tk.Label(progress_win, text="Export in progress...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_win, orient='horizontal', length=280, mode='determinate', maximum=self.total_frames)
        progress_bar.pack(pady=5)

        try:
            for frame_idx in range(self.total_frames):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Disegna overlay sul frame originale (non sul thumbnail)
                # Per fare ciò, dobbiamo scalare gli overlay alle dimensioni originali.
                # In questo caso, draw_yolo_overlays già lo fa internamente.
                # Per semplicità, qui riutilizziamo la logica di disegno del thumbnail
                # ma su un frame a piena risoluzione.
                
                # 1. Disegna overlay YOLO
                self.draw_yolo_overlays(frame)
                
                # 2. Disegna overlay Eventi
                current_ts = self.world_ts.iloc[frame_idx]['timestamp [ns]']
                self.draw_event_overlay(frame, current_ts)

                writer.write(frame)
                progress_bar['value'] = frame_idx + 1
                progress_win.update_idletasks()

        finally:
            writer.release()
            progress_win.destroy()

        # Aggiungi audio se richiesto
        if include_audio:
            try:
                video_clip = VideoFileClip(str(temp_video_path))
                audio_clip = AudioFileClip(str(self.video_path))
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(str(output_path), codec='libx264', audio_codec='aac', logger=None)
                final_clip.close()
                audio_clip.close()
                video_clip.close()
                temp_video_path.unlink() # Rimuovi il file temporaneo
                messagebox.showinfo("Success", f"Video exported successfully with audio to:\n{output_path}", parent=self)
            except Exception as e:
                temp_video_path.rename(output_path)
                messagebox.showwarning("Audio Error", f"Could not add audio: {e}\nVideo saved without audio.", parent=self)
        else:
            temp_video_path.rename(output_path)
            messagebox.showinfo("Success", f"Video exported successfully without audio to:\n{output_path}", parent=self)