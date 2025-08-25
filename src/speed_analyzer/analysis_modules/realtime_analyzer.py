import cv2
import numpy as np
import time
from pupil_labs.realtime_api.simple import discover_one_device
from ultralytics import YOLO
from .video_generator import _draw_pupil_plot

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

        self.pupil_data = {"Left": [], "Right": []}

    def connect(self, mock_device=None):
        """
        Cerca e si connette a un dispositivo Neon.
        Se viene fornito un mock_device, usa quello per i test.
        """
        if mock_device:
            print("Connecting to Mock Neon Device for testing.")
            self.device = mock_device
            return True

        try:
            print("Searching for Neon device on the network...")
            # CORREZIONE 2: Usa discover_one_device per una connessione più stabile
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
        """Ottiene i frame più recenti e i dati dello sguardo."""
        if not self.device:
            return None, None, None

        # I metodi per ricevere i dati sono corretti
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

        # L'unpacking della tupla è corretto
        scene_img, _ = self.last_scene_frame
        gaze = self.last_gaze
        
        results = self.yolo_model.track(scene_img, persist=True, verbose=False)
        gaze_point = (int(gaze.x), int(gaze.y))

        # La logica di hit-testing è corretta
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
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        scene_img, _ = scene_frame
        
        if self.yolo_model:
            results = self.yolo_model.track(scene_img, persist=True, verbose=False)
            scene_img = results[0].plot()

        if gaze:
            gaze_point = (int(gaze.x), int(gaze.y))
            cv2.circle(scene_img, gaze_point, 20, (0, 0, 255), 2)
            
            # CORREZIONE 3: Controlla se l'attributo esiste con hasattr()
            if hasattr(gaze, 'pupil_diameter_mm'):
                self.pupil_data["Left"].append(gaze.pupil_diameter_mm)
                # NOTA: L'API non distingue più tra occhio destro e sinistro per il diametro.
                # Per semplicità, inseriamo lo stesso dato in entrambi per il grafico.
                self.pupil_data["Right"].append(gaze.pupil_diameter_mm) 
                if len(self.pupil_data["Left"]) > 200:
                    self.pupil_data["Left"].pop(0)
                    self.pupil_data["Right"].pop(0)

        _draw_pupil_plot(scene_img, self.pupil_data, 2, 8, 350, 150, (scene_img.shape[1] - 360, 10))

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