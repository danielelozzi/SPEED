# src/speed_analyzer/analysis_modules/realtime_analyzer.py
import cv2
import numpy as np
import time
from pupil_labs.real_time_api import Device, Network, receive_gaze_data
from ultralytics import YOLO

# Importa le funzioni di disegno che possiamo riutilizzare
from .video_generator import _draw_pupil_plot, _draw_generic_plot, PUPIL_COLORS, FRAG_LINE_COLOR, FRAG_BG_COLOR

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
        
        # Inizializza YOLO
        try:
            self.yolo_model = YOLO(model_path)
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

        # Dati per i plot
        self.pupil_data = {"Left": [], "Right": []}
        self.gaze_speed_data = []

    def connect(self, mock_device=None):
        """
        Cerca e si connette a un dispositivo Neon.
        Se viene fornito un mock_device, usa quello per i test.
        """
        if mock_device:
            print("Connecting to Mock Neon Device for testing.")
            self.device = mock_device
            return True

        """Cerca e si connette al primo dispositivo Neon disponibile sulla rete."""
        try:
            print("Searching for Neon device on the network...")
            self.device = Device.discover()
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
        """Ottiene i frame più recenti e i dati dello sguardo."""
        if not self.device:
            return None, None, None

        self.last_scene_frame = self.device.receive_scene_video_frame()
        self.last_eye_frame = self.device.receive_eyes_video_frame()
        self.last_gaze = self.device.receive_gaze_datum()

        return self.last_scene_frame, self.last_eye_frame, self.last_gaze

    def get_gazed_object(self):
        """
        Restituisce il nome della classe dell'oggetto guardato, classificato da YOLO.
        """
        if self.yolo_model is None or self.last_scene_frame is None or self.last_gaze is None:
            return "N/A"

        scene_img, _ = self.last_scene_frame
        gaze = self.last_gaze
        
        # Esegui YOLO sul frame
        results = self.yolo_model.track(scene_img, persist=True, verbose=False)
        
        gaze_point = (int(gaze.x), int(gaze.y))

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            if x1 <= gaze_point[0] <= x2 and y1 <= gaze_point[1] <= y2:
                class_id = int(box.cls[0])
                return self.yolo_model.names[class_id]
        
        return "No object detected"

    def process_and_visualize(self):
        """
        Processa il frame della scena, aggiunge overlay e lo restituisce.
        """
        scene_frame, eye_frame, gaze = self.get_latest_frames_and_gaze()

        if scene_frame is None:
            # Ritorna un'immagine nera se non c'è connessione
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        scene_img, scene_ts = scene_frame
        
        # Esegui YOLO
        if self.yolo_model:
            results = self.yolo_model.track(scene_img, persist=True, verbose=False)
            scene_img = results[0].plot() # YOLO disegna i bounding box

        # Disegna il punto di sguardo
        if gaze:
            gaze_point = (int(gaze.x), int(gaze.y))
            cv2.circle(scene_img, gaze_point, 20, (0, 0, 255), 2)
            
            # Aggiorna dati per i grafici
            if 'pupil_diameter_mm' in gaze:
                # Nota: l'API v3 unifica i diametri, qui simuliamo una divisione
                self.pupil_data["Left"].append(gaze.pupil_diameter_mm)
                if len(self.pupil_data["Left"]) > 200: self.pupil_data["Left"].pop(0)

        # Disegna i grafici (riutilizzando la logica esistente)
        _draw_pupil_plot(scene_img, self.pupil_data, 2, 8, 350, 150, (scene_img.shape[1] - 360, 10))

        # Aggiungi il video dell'occhio (PiP)
        if eye_frame:
            eye_img, _ = eye_frame
            pip_h, pip_w = 200, 400
            scene_img[10:10+pip_h, 10:10+pip_w] = cv2.resize(eye_img, (pip_w, pip_h))

        return scene_img

    def close(self):
        """Chiude la connessione con il dispositivo."""
        if self.device:
            self.device.close()
            print("Connection closed.")