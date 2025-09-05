# src/speed_analyzer/analysis_modules/realtime_analyzer.py
import cv2
import numpy as np
import time
import pandas as pd
from pathlib import Path
import threading
from pupil_labs.realtime_api.simple import discover_one_device
from ultralytics import YOLO
from .video_generator import (_draw_pupil_plot, _draw_generic_plot, 
                               FRAG_PLOT_WIDTH, FRAG_PLOT_HEIGHT, FRAG_LINE_COLOR, FRAG_BG_COLOR, 
                               BLINK_TEXT_COLOR, EVENT_TEXT_COLOR, EVENT_BG_COLOR, FRAG_PLOT_HISTORY)

# --- NUOVA COSTANTE PER IL GRAFICO DEI BLINK ---
BLINK_PLOT_HISTORY = 200
BLINK_PLOT_WIDTH = 350
BLINK_PLOT_HEIGHT = 100 # Altezza ridotta per un grafico binario
BLINK_PLOT_BG_COLOR = (80, 80, 80)
BLINK_PLOT_LINE_COLOR = (255, 100, 255) # Lilla

# --- NUOVA FUNZIONE HELPER PER LA TRASPARENZA ---
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
# --- FINE FUNZIONE HELPER ---


class RealtimeNeonAnalyzer:
    """
    Gestisce la connessione, l'acquisizione dati e l'analisi in tempo reale
    da un dispositivo Pupil Labs Neon.
    """
    def __init__(self, model_path='yolov8n.pt'):
        print("Initializing Real-time Neon Analyzer...")
        self.device = None
        self.last_gaze = None
        self.last_scene_frame = None
        self.last_eye_frame = None
        
        try:
            self.yolo_model = YOLO(model_path)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

        # Dati per grafici e overlay
        self.pupil_data = {"Left": [], "Right": [], "Mean": []}
        self.fragmentation_data = []
        self.blink_data = []
        self.gaze_path_history = []
        self.gaze_history_heatmap = []
        self.is_blinking = False
        self.blink_off_counter = 0
        self.last_gazed_object = "N/A"
        self.last_gazed_aoi = "N/A"
        self.last_event_name = ""

        # Gestione AOI
        self.static_aois = []
        self.aoi_colors = {}

        # Attributi di registrazione
        self.is_recording = False
        self.recording_thread = None
        self.output_folder = None
        self.video_writers = {}
        self.gaze_data_list = []
        self.events_list = []
        self.recording_start_time_unix = None

    def connect(self, mock_device=None):
        if mock_device:
            print("Connecting to Mock Neon Device for testing.")
            self.device = mock_device
            return True
        try:
            print("Searching for Neon device on the network...")
            self.device = discover_one_device(max_search_duration_seconds=10)
            if self.device:
                print(f"Connected to device: {self.device.phone_name} @ {self.device.ip_address}")
                return True
            else:
                print("No device found.")
                return False
        except Exception as e:
            print(f"Failed to connect to device: {e}")
            return False

    def get_latest_frames_and_gaze(self):
        if not self.device: return None, None, None
        self.last_scene_frame = self.device.receive_scene_video_frame()
        self.last_eye_frame = self.device.receive_eyes_video_frame()
        self.last_gaze = self.device.receive_gaze_datum()
        return self.last_scene_frame, self.last_eye_frame, self.last_gaze
        
    def get_gazed_object(self, scene_img, gaze):
        if self.yolo_model is None or scene_img is None or gaze is None: return "N/A", scene_img
        results = self.yolo_model.track(scene_img, persist=True, verbose=False)
        gaze_point = (int(gaze.x), int(gaze.y))
        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            if x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2:
                class_id = int(box.cls[0])
                return self.yolo_model.names[class_id], annotated_frame
        return "No object", annotated_frame

    def start_recording(self, output_dir: str = "./realtime_recording"):
        if self.is_recording:
            print("Recording is already in progress.")
            return False
        self.output_folder = Path(output_dir)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.gaze_data_list, self.events_list = [], []
        
        scene_frame, eye_frame, _ = self.get_latest_frames_and_gaze()
        if scene_frame is None: return False
            
        scene_h, scene_w, _ = scene_frame.image.shape
        eye_h, eye_w, _ = eye_frame.image.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writers['external'] = cv2.VideoWriter(str(self.output_folder / 'external.mp4'), fourcc, 30.0, (scene_w, scene_h))
        self.video_writers['internal'] = cv2.VideoWriter(str(self.output_folder / 'internal.mp4'), fourcc, 30.0, (eye_w, eye_h))
        
        self.is_recording = True
        self.recording_start_time_unix = time.time()
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        
        self.add_event("begin.recording")
        print(f"Recording started. Saving data to: {self.output_folder.resolve()}")
        return True

    def stop_recording(self):
        if not self.is_recording: return
        self.is_recording = False
        if self.recording_thread: self.recording_thread.join()
        for writer in self.video_writers.values(): writer.release()
        self.video_writers = {}
        
        gaze_df = pd.DataFrame(self.gaze_data_list)
        gaze_df.to_csv(self.output_folder / 'gaze_data.csv', index=False)
        
        if self.events_list: pd.DataFrame(self.events_list).to_csv(self.output_folder / 'events.csv', index=False)

        if self.static_aois:
            self._analyze_gaze_in_aois(gaze_df)

        self.last_event_name = ""
        print(f"Recording stopped. All files saved in {self.output_folder.resolve()}")

    def add_event(self, event_name: str):
        if not self.is_recording:
            print("Cannot add event, recording is not active.")
            return
        event_time_ns = int((time.time() - self.recording_start_time_unix) * 1e9)
        self.events_list.append({'name': event_name, 'timestamp [ns]': event_time_ns, 'recording id': 'realtime_rec'})
        self.last_event_name = event_name
        print(f"Event '{event_name}' added at timestamp {event_time_ns}.")

    def _recording_loop(self):
        while self.is_recording:
            scene, eye, gaze = self.get_latest_frames_and_gaze()
            if scene: self.video_writers['external'].write(cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR))
            if eye: self.video_writers['internal'].write(cv2.cvtColor(eye.image, cv2.COLOR_RGB2BGR))
            if gaze: self.gaze_data_list.append({
                'timestamp [ns]': int((gaze.timestamp_unix_seconds - self.recording_start_time_unix) * 1e9),
                'gaze x [px]': gaze.x, 'gaze y [px]': gaze.y,
                'pupil_diameter [mm]': gaze.pupil_diameter_mm if hasattr(gaze, 'pupil_diameter_mm') else None
            })
            time.sleep(1/60)

    def add_static_aoi(self, name, rect):
        for aoi in self.static_aois:
            if aoi['name'] == name:
                aoi['rect'] = rect
                print(f"AOI '{name}' updated.")
                return
        self.static_aois.append({'name': name, 'rect': rect})
        self.aoi_colors[name] = tuple(np.random.randint(100, 256, 3).tolist())
        print(f"AOI '{name}' added.")

    def remove_static_aoi(self, name):
        self.static_aois = [aoi for aoi in self.static_aois if aoi['name'] != name]
        if name in self.aoi_colors:
            del self.aoi_colors[name]
        print(f"AOI '{name}' removed.")
        
    def _analyze_gaze_in_aois(self, gaze_df):
        if gaze_df.empty or not self.static_aois:
            return
            
        aoi_results = []
        total_gaze_points = len(gaze_df.dropna(subset=['gaze x [px]', 'gaze y [px]']))
        
        for aoi in self.static_aois:
            name, (x1, y1, x2, y2) = aoi['name'], aoi['rect']
            gaze_in_aoi = gaze_df[
                (gaze_df['gaze x [px]'] >= x1) & (gaze_df['gaze x [px]'] <= x2) &
                (gaze_df['gaze y [px]'] >= y1) & (gaze_df['gaze y [px]'] <= y2)
            ]
            gaze_count = len(gaze_in_aoi)
            percentage = (gaze_count / total_gaze_points * 100) if total_gaze_points > 0 else 0
            
            aoi_results.append({
                'aoi_name': name,
                'gaze_points_count': gaze_count,
                'gaze_time_percentage': round(percentage, 2)
            })
        
        results_df = pd.DataFrame(aoi_results)
        results_path = self.output_folder / 'gaze_in_aoi_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"AOI analysis complete. Results saved to {results_path}")

    def process_and_visualize(self, show_yolo=True, show_pupil=True, show_frag=True, show_blink=True, show_aois=True, show_heatmap=False, show_gaze_path=True):
        if not self.is_recording: self.get_latest_frames_and_gaze()
        if self.last_scene_frame is None: return np.zeros((720, 1280, 3), dtype=np.uint8)

        scene_img, scene_ts = self.last_scene_frame.image, self.last_scene_frame.timestamp_unix_seconds

        # --- MODIFICA CHIAVE: Logica Heatmap con Trasparenza ---
        if show_heatmap and self.last_gaze:
            heatmap_window_size = 60  # Finestra fissa di 60 frame (circa 2 secondi)
            
            self.gaze_history_heatmap.append((int(self.last_gaze.x), int(self.last_gaze.y)))
            if len(self.gaze_history_heatmap) > heatmap_window_size:
                self.gaze_history_heatmap.pop(0)

            if len(self.gaze_history_heatmap) > 1:
                # 1. Crea mappa di intensità in scala di grigi
                intensity_map = np.zeros((scene_img.shape[0], scene_img.shape[1]), dtype=np.uint8)
                for point in self.gaze_history_heatmap:
                    cv2.circle(intensity_map, point, radius=25, color=50, thickness=-1)
                
                intensity_map = cv2.blur(intensity_map, (91, 91))
                
                # 2. Applica la colormap
                heatmap_color = cv2.applyColorMap(intensity_map, cv2.COLORMAP_JET)
                
                # 3. Usa la mappa di intensità come canale Alpha
                heatmap_rgba = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
                heatmap_rgba[:, :, 3] = intensity_map
                
                # 4. Sovrapponi l'immagine con trasparenza
                scene_img = _overlay_transparent(scene_img, heatmap_rgba, 0, 0)
        else:
            self.gaze_history_heatmap.clear()
        # --- FINE MODIFICA CHIAVE ---
        
        self.last_gazed_aoi = "N/A"
        if self.last_gaze and show_aois:
            for aoi in self.static_aois:
                name, (x1, y1, x2, y2) = aoi['name'], aoi['rect']
                if x1 <= self.last_gaze.x <= x2 and y1 <= self.last_gaze.y <= y2:
                    self.last_gazed_aoi = name
                    break
        
        if show_yolo and self.last_gaze:
            self.last_gazed_object, scene_img_yolo = self.get_gazed_object(scene_img.copy(), self.last_gaze)
            # Usa l'immagine annotata da YOLO solo se YOLO è attivo
            scene_img = scene_img_yolo
        
        if self.last_gaze:
            gaze = self.last_gaze
            # Disegna il punto di sguardo sopra la heatmap e gli altri overlay
            cv2.circle(scene_img, (int(gaze.x), int(gaze.y)), 20, (0, 0, 255), 2)

            # --- NUOVO: Disegna il percorso dello sguardo con dissolvenza ---
            if show_gaze_path and len(self.gaze_path_history) > 1:
                # Itera attraverso la cronologia per disegnare le linee
                for i in range(1, len(self.gaze_path_history)):
                    # Calcola l'intensità del colore in base alla posizione nella cronologia
                    # Le linee più vecchie (indice più basso) saranno più scure
                    intensity = i / len(self.gaze_path_history)
                    color = (0, 0, int(255 * intensity)) # Rosso che si dissolve al nero
                    cv2.line(scene_img, self.gaze_path_history[i-1], self.gaze_path_history[i], color, 2)
            # --- FINE BLOCCO ---

            pupil_val = gaze.pupil_diameter_mm if hasattr(gaze, 'pupil_diameter_mm') and gaze.pupil_diameter_mm > 0 else None
            if pupil_val:
                self.pupil_data["Mean"].append(pupil_val)
                self.is_blinking = False
                self.blink_off_counter = 0
            else:
                self.blink_off_counter += 1
                if self.blink_off_counter > 3: self.is_blinking = True
            
            self.blink_data.append(1 if self.is_blinking else 0)
            
            # Aggiorna la cronologia del percorso dello sguardo
            self.gaze_path_history.append((int(gaze.x), int(gaze.y)))

            # Calcolo della frammentazione (usa la nuova cronologia)
            if len(self.gaze_path_history) > 1:
                p1 = self.gaze_path_history[-2]
                p2 = self.gaze_path_history[-1]
                dist = np.linalg.norm(np.array(p1) - np.array(p2))
                time_delta = scene_ts - getattr(self, '_last_frag_ts', scene_ts)
                self.fragmentation_data.append(dist / time_delta if time_delta > 0 else 0)
            self._last_frag_ts = scene_ts
        if len(self.pupil_data["Mean"]) > BLINK_PLOT_HISTORY: self.pupil_data["Mean"].pop(0)
        if len(self.fragmentation_data) > FRAG_PLOT_HISTORY: self.fragmentation_data.pop(0)
        if len(self.blink_data) > BLINK_PLOT_HISTORY: self.blink_data.pop(0)
        if len(self.gaze_path_history) > 10: self.gaze_path_history.pop(0)

        y_pos = 10
        if show_pupil: 
            _draw_pupil_plot(scene_img, {"Mean": self.pupil_data["Mean"]}, 2, 8, 350, 150, (scene_img.shape[1] - 360, y_pos))
            y_pos += 150 + 10
        if show_frag:
            _draw_generic_plot(scene_img, self.fragmentation_data, 0, 3000, FRAG_PLOT_WIDTH, FRAG_PLOT_HEIGHT, (scene_img.shape[1] - 360, y_pos), "Fragmentation (px/s)", FRAG_LINE_COLOR, FRAG_BG_COLOR)
            y_pos += FRAG_PLOT_HEIGHT + 10
        
        if show_blink:
            _draw_generic_plot(scene_img, self.blink_data, 0, 1.1, BLINK_PLOT_WIDTH, BLINK_PLOT_HEIGHT, (scene_img.shape[1] - 360, y_pos), "Blink Detection", BLINK_PLOT_LINE_COLOR, BLINK_PLOT_BG_COLOR)
            y_pos += BLINK_PLOT_HEIGHT + 10

        if show_aois:
            cv2.putText(scene_img, f"Gazing at AOI: {self.last_gazed_aoi}", (scene_img.shape[1] - 360, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if show_blink and self.is_blinking: cv2.putText(scene_img, "BLINK", (scene_img.shape[1] - 150, scene_img.shape[0] - 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, BLINK_TEXT_COLOR, 2)
        if show_yolo: cv2.putText(scene_img, f"Gazing at Object: {self.last_gazed_object}", (20, scene_img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if show_aois:
            for aoi in self.static_aois:
                name, (x1, y1, x2, y2) = aoi['name'], aoi['rect']
                color = self.aoi_colors.get(name, (255, 0, 255))
                cv2.rectangle(scene_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(scene_img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if self.last_event_name:
            font_scale = 1.0
            font_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(self.last_event_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            img_h, img_w, _ = scene_img.shape
            padding = 15
            
            rect_start = (img_w - text_w - padding * 2, img_h - text_h - padding * 2 - baseline)
            rect_end = (img_w - padding, img_h - padding)
            text_pos = (img_w - text_w - int(padding*1.5), img_h - int(padding*1.5) - baseline//2)

            cv2.rectangle(scene_img, rect_start, rect_end, EVENT_BG_COLOR, -1)
            cv2.putText(scene_img, self.last_event_name, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, EVENT_TEXT_COLOR, font_thickness)

        if self.last_eye_frame:
            eye_img, _ = self.last_eye_frame
            cv2.rectangle(scene_img, (10, 10), (410, 210), (0,0,0), -1)
            scene_img[10:210, 10:410] = cv2.resize(eye_img, (400, 200))

        return scene_img

    def close(self):
        if self.is_recording: self.stop_recording()
        if self.device: self.device.close()
        print("Connection closed.")