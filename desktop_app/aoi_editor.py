# desktop_app/aoi_editor.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import pygame
from moviepy import VideoFileClip
from pathlib import Path
import pandas as pd
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class AoiEditor(tk.Toplevel):
    """
    Una finestra per definire un'Area di Interesse (AOI) in tre modalità:
    1. Statica: Un rettangolo fisso.
    2. Dinamica (Auto): Tracciando un oggetto rilevato da YOLO.
    """
    def __init__(self, parent, video_path, analysis_output_path=None):
        super().__init__(parent)
        self.title("Define Area of Interest (AOI)")
        self.geometry("1100x850")
        self.transient(parent)
        self.grab_set()
        self.analysis_output_path = analysis_output_path

        if not video_path or not video_path.exists():
            messagebox.showerror("Error", f"Video file not found:\n{video_path}", parent=self)
            self.destroy()
            return
        
        self.cap = cv2.VideoCapture(str(video_path))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_path = video_path
        
        # Risultati
        self.result = None 
        self.result_type = None
        self.aoi_name = None # NUOVO: Nome per l'AOI

        # Stato del disegno/selezione
        self.rect = None
        self.start_x = None
        self.start_y = None
        # Inizializza con le colonne per prevenire KeyError
        self.yolo_detections_df = pd.DataFrame(columns=['frame_idx', 'track_id', 'class_name', 'task', 'x1', 'y1', 'x2', 'y2'])
        self.selected_track_id = None
        self.is_playing = False
        self.current_frame_idx = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_updating_slider = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30

        # Audio
        self.audio_clip = None
        self.temp_audio_file_path = None
        self.is_muted = tk.BooleanVar(value=True)

        # --- GUI Setup ---
        self.mode_var = tk.StringVar(value="static")
        
        top_frame = tk.Frame(self, pady=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Radiobutton(top_frame, text="Static AOI", variable=self.mode_var, value="static", command=self.update_ui_for_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(top_frame, text="Dynamic AOI (Tracking)", variable=self.mode_var, value="dynamic_auto", command=self.update_ui_for_mode).pack(side=tk.LEFT, padx=10)

        # --- NUOVO: PanedWindow per video e filtri YOLO ---
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        video_frame = tk.Frame(main_pane)
        main_pane.add(video_frame, stretch="always", minsize=400)

        self.video_canvas = tk.Canvas(video_frame)
        self.video_canvas.pack(pady=5, expand=True)
        self.video_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # --- NUOVO: Pannello destro per i filtri YOLO ---
        self.yolo_filter_panel = tk.Frame(main_pane, width=350)
        self.yolo_filter_panel.pack_propagate(False)
        main_pane.add(self.yolo_filter_panel)
        self.setup_yolo_filter_ui()

        # Controlli Video
        controls_frame = tk.Frame(self)
        controls_frame.pack(fill=tk.X, padx=10)
        self.play_pause_btn = tk.Button(controls_frame, text="▶ Play", command=self.toggle_play)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        self.mute_btn = tk.Button(controls_frame, text="🔇 Unmute", command=self.toggle_mute)
        self.mute_btn.pack(side=tk.LEFT, padx=5)
        self.frame_scale = ttk.Scale(controls_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(controls_frame, text=f"Frame: 0 / {self.total_frames}", width=20)
        self.time_label.pack(side=tk.RIGHT, padx=5)

        self.status_label = tk.Label(self, text="Draw a rectangle for the static AOI.")
        self.status_label.pack(pady=5)

        self.save_button = tk.Button(self, text="Save AOI", command=self.save_and_close, font=('Helvetica', 10, 'bold'))
        self.save_button.pack(pady=10)

        self._load_audio()
        self.update_frame(0)
        self.after(100, self.update_ui_for_mode) # Ritarda per permettere alla UI di disegnarsi
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _load_audio(self):
        try:
            pygame.mixer.init()
            video = VideoFileClip(str(self.video_path))
            if video.audio:
                self.audio_clip = video.audio
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    self.temp_audio_file_path = Path(f.name)
                self.audio_clip.write_audiofile(str(self.temp_audio_file_path), logger=None)
                pygame.mixer.music.load(str(self.temp_audio_file_path))
            else:
                self.audio_clip = None
        except Exception as e:
            self.audio_clip = None
            logging.error(f"Could not load audio: {e}")
        finally:
            self.update_mute_button_text()

    def on_close(self):
        self.is_playing = False
        pygame.mixer.music.stop()
        self.cap.release()
        if self.temp_audio_file_path and self.temp_audio_file_path.exists():
            try:
                self.temp_audio_file_path.unlink()
            except OSError as e:
                logging.error(f"Error removing temp audio file: {e}")
        self.destroy()

    def seek_frame(self, frame_idx_str):
        if self.is_updating_slider: return
        if self.is_playing: self.toggle_play()
        self.update_frame(int(float(frame_idx_str)))

    def update_frame(self, frame_idx, redraw_overlays=True):
        self.current_frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        
        self.is_updating_slider = True
        self.frame_scale.set(self.current_frame_idx)
        self.is_updating_slider = False
        self.time_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.original_frame = frame
            self.display_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.display_image.thumbnail((960, 720))
            
            self.img_display_width, self.img_display_height = self.display_image.size
            self.scale_x = self.video_width / self.img_display_width
            self.scale_y = self.video_height / self.img_display_height

            self.photo = ImageTk.PhotoImage(image=self.display_image)
            self.video_canvas.config(width=self.img_display_width, height=self.img_display_height)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            # Se il video finisce, ferma la riproduzione
            if self.is_playing:
                self.toggle_play()
            return # Esce per evitare di disegnare su un frame non valido
        
        # Disegna gli overlay dopo aver disegnato il frame
        if self.mode_var.get() == "dynamic_auto" and redraw_overlays:
            self.draw_yolo_overlays()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_btn.config(text="❚❚ Pause")
            self.play_video()
            if not self.is_muted.get():
                self.after(10, self.play_audio) # Aggiungi un piccolo ritardo
        else:
            self.play_pause_btn.config(text="▶ Play")
            pygame.mixer.music.stop()

    def play_video(self):
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            self.update_frame(self.current_frame_idx + 1)
            self.after(int(1000 / self.fps), self.play_video)
        else:
            self.is_playing = False
            self.play_pause_btn.config(text="▶ Play")

    def play_audio(self):
        if self.is_playing and self.audio_clip and not self.is_muted.get():
            start_time = self.current_frame_idx / self.fps
            try:
                pygame.mixer.music.play(start=start_time)
            except pygame.error as e:
                # A volte pygame non è pronto, specialmente dopo un seek. Ignora l'errore.
                logging.warning(f"Pygame audio error (ignorable): {e}")

    def toggle_mute(self):
        self.is_muted.set(not self.is_muted.get())
        self.update_mute_button_text()
        if self.is_playing:
            if self.is_muted.get(): pygame.mixer.music.stop()
            else: self.play_audio()

    def update_mute_button_text(self):
        self.mute_btn.config(text="🔇 Unmute" if self.is_muted.get() else "🔊 Mute")
        if not self.audio_clip: self.mute_btn.config(state=tk.DISABLED, text="No Audio")

    def update_ui_for_mode(self):
        """Aggiorna la UI in base alla modalità selezionata (statica o dinamica)."""
        mode = self.mode_var.get()
        self.rect = None
        self.selected_track_id = None
        
        self.update_frame(self.current_frame_idx) # Ridisegna il frame corrente

        if mode == "static":
            self.yolo_filter_panel.pack_forget() # Nasconde il pannello YOLO
            self.status_label.config(text="Draw a rectangle on the image to define the static AOI.")
            self.save_button.config(state=tk.NORMAL)
        elif mode == "dynamic_auto":
            self.yolo_filter_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0)) # Mostra il pannello
            self.status_label.config(text="Loading object data...")
            self.save_button.config(state=tk.DISABLED)
            self.after(100, self.load_and_display_yolo_results)

    def setup_yolo_filter_ui(self):
        """Crea i widget per il pannello di filtro YOLO."""
        # PanedWindow verticale per dividere classificazione e segmentazione
        right_pane = tk.PanedWindow(self.yolo_filter_panel, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        right_pane.pack(fill=tk.BOTH, expand=True)

        # Frame per gli oggetti rilevati (classificazione)
        detection_frame = tk.LabelFrame(right_pane, text="Classification (for Tracking)")
        right_pane.add(detection_frame, stretch="always")
        self.detection_tree = ttk.Treeview(detection_frame, columns=("#1"), show="tree headings")
        self.detection_tree.heading("#0", text="Object")
        self.detection_tree.heading("#1", text="Show")
        self.detection_tree.column("#1", width=50, anchor='center')
        self.detection_tree.pack(fill=tk.BOTH, expand=True)
        self.detection_tree.bind('<Button-1>', self.on_yolo_tree_click)
        
        # Frame per gli oggetti segmentati
        segmentation_frame = tk.LabelFrame(right_pane, text="Segmentation (for Info)")
        right_pane.add(segmentation_frame, stretch="always")
        self.segmentation_tree = ttk.Treeview(segmentation_frame, columns=("#1"), show="tree headings")
        self.segmentation_tree.heading("#0", text="Object")
        self.segmentation_tree.heading("#1", text="Show")
        self.segmentation_tree.column("#1", width=50, anchor='center')
        self.segmentation_tree.pack(fill=tk.BOTH, expand=True)
        self.segmentation_tree.bind('<Button-1>', self.on_yolo_tree_click)

        # Mappa per accedere facilmente ai treeview
        self.yolo_filter_trees = {'detect': self.detection_tree, 'segment': self.segmentation_tree}
        # Set per i filtri di visibilità
        self.yolo_class_filter = set()
        self.yolo_id_filter = set()


    def load_and_display_yolo_results(self):
        """Carica i risultati da yolo_detections_cache.csv o esegue una nuova analisi."""
        yolo_cache_path = None
        if self.analysis_output_path:
            yolo_cache_path = self.analysis_output_path / 'yolo_detections_cache.csv'

        if yolo_cache_path and yolo_cache_path.exists():
            logging.info(f"Loading YOLO detections from cache: {yolo_cache_path}")
            temp_df = pd.read_csv(yolo_cache_path)
            # Validate cache file
            if 'task' in temp_df.columns and 'frame_idx' in temp_df.columns:
                self.yolo_detections_df = temp_df
            else:
                logging.warning("YOLO cache is outdated. Running detection on the first frame as fallback.")
                self.run_fallback_detection()
        else:
            logging.warning("YOLO cache not found. Running detection on the first frame as fallback.")
            self.run_fallback_detection()

        self.populate_yolo_filter_trees()

    def run_fallback_detection(self):
        """Esegue una rilevazione YOLO di base se il file di cache non esiste."""
        if YOLO is None:
            messagebox.showerror("Error", "YOLO (ultralytics) is not installed and no cache file was found.", parent=self)
            self.mode_var.set("static"); self.update_ui_for_mode()
            return

        try:
            model = YOLO('yolov8n.pt')
            results = model.track(self.original_frame, persist=True, verbose=False)
            detections = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    detections.append({
                        'frame_idx': 0, 'track_id': track_id, 'task': 'detect',
                        'class_name': model.names[class_id],
                        'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]
                    })
            self.yolo_detections_df = pd.DataFrame(detections)
        except Exception as e:
            logging.error(f"YOLO detection failed: {e}")
            messagebox.showerror("YOLO Error", f"Object detection failed: {e}", parent=self)

    def populate_yolo_filter_trees(self):
        """Popola i treeview con i dati YOLO caricati."""
        # Pulisci entrambi i treeview
        for tree in self.yolo_filter_trees.values():
            tree.delete(*tree.get_children())

        if self.yolo_detections_df.empty:
            self.status_label.config(text="No objects detected. Try Static AOI.")
            return

        # Itera sui task ('detect', 'segment', etc.) e popola il treeview corretto
        for task, group_df in self.yolo_detections_df.groupby('task'):
            # Trova il treeview corrispondente al task
            tree_to_populate = self.yolo_filter_trees.get(task)

            if tree_to_populate:
                # Raggruppa per classe e poi per ID
                detected_items = group_df.groupby('class_name')['track_id'].unique().apply(list).to_dict()
                
                # Popola il treeview
                for class_name, ids in sorted(detected_items.items()):
                    class_node = tree_to_populate.insert("", "end", text=f"Class: {class_name}", open=True, values=("☑",), tags=('class', class_name))
                    for track_id in sorted(ids):
                        tree_to_populate.insert(class_node, "end", text=f"  ID: {track_id}", values=("☑",), tags=('id', track_id))

        self.update_yolo_filters_and_redraw()
        self.status_label.config(text="Click on an object in the list or on the video to select it.")

    def on_button_press(self, event):
        mode = self.mode_var.get()
        if mode == "static":
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.video_canvas.delete(self.rect)
            self.rect = self.video_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)
        elif mode == "dynamic_auto":
            self.select_object_at_pos(event.x, event.y)

    def on_mouse_drag(self, event):
        if self.mode_var.get() == "static":
            cur_x, cur_y = (event.x, event.y)
            self.video_canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_yolo_tree_click(self, event):
        """Gestisce il click su un oggetto nel treeview."""
        tree = event.widget
        item_id = tree.identify_row(event.y)
        if not item_id: return

        column = tree.identify_column(event.x)
        tags = tree.item(item_id, 'tags')
        if not tags: return

        # Se si interagisce con i filtri, ferma la riproduzione
        if self.is_playing: self.toggle_play()

        # Se si clicca sulla colonna della checkbox, si gestisce la visibilità
        if column == "#1":
            current_val = tree.item(item_id, 'values')[0]
            new_val = "☐" if current_val == "☑" else "☑"
            tree.set(item_id, column, new_val)

            # Logica a cascata: se si clicca su una classe, aggiorna tutti i figli
            if tags[0] == 'class':
                for child_id in tree.get_children(item_id):
                    tree.set(child_id, column, new_val)
            
            self.update_yolo_filters_and_redraw()

        # Se si clicca sul nome dell'oggetto (colonna #0), si seleziona per il tracking
        elif column == "#0":
            if tags[0] == 'id':
                track_id = int(tags[1])
                self.select_object_by_id(track_id)
            elif tags[0] == 'class':
                # Potremmo implementare la selezione di tutti gli ID di una classe qui se voluto
                pass

    def on_button_release(self, event):
        pass

    def select_object_at_pos(self, x, y):
        """Seleziona un oggetto in base alla posizione del click sulla canvas."""
        click_x_orig = x * self.scale_x
        click_y_orig = y * self.scale_y

        found_track_id = None
        detections_on_frame = self.yolo_detections_df[self.yolo_detections_df['frame_idx'] == self.current_frame_idx]
        for _, det in detections_on_frame.iterrows():
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            if x1 <= click_x_orig <= x2 and y1 <= click_y_orig <= y2:
                found_track_id = det['track_id']
                break

        if found_track_id is not None:
            self.select_object_by_id(found_track_id)

    def select_object_by_id(self, track_id):
        """Evidenzia l'oggetto selezionato (sia per ID che per posizione)."""
        self.selected_track_id = track_id
        self.update_yolo_filters_and_redraw() # Ridisegna per mostrare la selezione
        self.status_label.config(text=f"Selected object with Track ID: {self.selected_track_id}. Click Save.")
        self.save_button.config(state=tk.NORMAL)

    def update_yolo_filters_and_redraw(self):
        """Aggiorna i set di filtri e ridisegna il canvas del video."""
        self.yolo_class_filter.clear()
        self.yolo_id_filter.clear()

        # Itera su entrambi i treeview per raccogliere i filtri
        for task_key, tree in self.yolo_filter_trees.items():
            for class_node in tree.get_children(''):
                # Aggiungi la classe al filtro se la sua checkbox è attiva
                if tree.item(class_node, 'values')[0] == "☑":
                    self.yolo_class_filter.add(tree.item(class_node, 'tags')[1])
                
                # Controlla gli ID figli
                for id_node in tree.get_children(class_node):
                    if tree.item(id_node, 'values')[0] == "☑":
                        self.yolo_id_filter.add(int(tree.item(id_node, 'tags')[1]))

        # Se tutti gli elementi sono selezionati, i set di filtri dovrebbero essere vuoti per non filtrare nulla
        if not self.yolo_detections_df.empty:
            if len(self.yolo_class_filter) == len(self.yolo_detections_df['class_name'].unique()):
                self.yolo_class_filter.clear()
            if len(self.yolo_id_filter) == len(self.yolo_detections_df['track_id'].unique()):
                self.yolo_id_filter.clear()

        self.update_frame(self.current_frame_idx) # Ridisegna il frame completo

    def draw_yolo_overlays(self):
        """Disegna tutti i bounding box, evidenziando quello selezionato."""
        if self.yolo_detections_df.empty:
            return
            
        self.video_canvas.delete("object_box")
        detections_on_current_frame = self.yolo_detections_df[self.yolo_detections_df['frame_idx'] == self.current_frame_idx]

        for _, det in detections_on_current_frame.iterrows():
            # Applica i filtri di visibilità
            if self.yolo_class_filter and det['class_name'] not in self.yolo_class_filter:
                continue
            if self.yolo_id_filter and det['track_id'] not in self.yolo_id_filter:
                continue

            display_box = (det['x1'] / self.scale_x, det['y1'] / self.scale_y,
                           det['x2'] / self.scale_x, det['y2'] / self.scale_y)

            color = 'magenta' if det['track_id'] == self.selected_track_id else 'cyan'
            label = f"{det['class_name']}_{int(det['track_id'])}"
            
            self.video_canvas.create_rectangle(display_box, outline=color, width=3 if color=='magenta' else 2, tags="object_box")
            self.video_canvas.create_text(display_box[0], display_box[1] - 10, text=label, fill=color, anchor='sw', tags="object_box")

    def save_and_close(self):
        mode = self.mode_var.get()
        
        # --- MODIFICATO: Chiedi il nome dell'AOI ---
        aoi_name = simpledialog.askstring("AOI Name", "Enter a unique name for this AOI:", parent=self)
        if not aoi_name:
            return # L'utente ha annullato

        self.aoi_name = aoi_name
        
        if mode == "static":
            if not self.rect:
                messagebox.showwarning("Warning", "Please draw a rectangle on the video first.", parent=self)
                return
            
            coords = self.video_canvas.coords(self.rect)
            original_coords = {
                "x1": int(min(coords[0], coords[2]) * self.scale_x),
                "y1": int(min(coords[1], coords[3]) * self.scale_y),
                "x2": int(max(coords[0], coords[2]) * self.scale_x),
                "y2": int(max(coords[1], coords[3]) * self.scale_y)
            }
            self.result = original_coords
            self.result_type = 'static'

        elif mode == "dynamic_auto":
            if self.selected_track_id is None:
                messagebox.showwarning("Warning", "Please select an object to track first.", parent=self)
                return
            self.result = self.selected_track_id
            self.result_type = 'dynamic_auto'
        else:
            # Caso in cui nessuna modalità valida è selezionata (non dovrebbe accadere)
            messagebox.showerror("Error", "Invalid AOI mode selected.", parent=self)
            return
        
        self.on_close()