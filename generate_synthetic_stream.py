# generate_synthetic_stream.py
import cv2
import numpy as np
import time
import threading
from collections import namedtuple
import qrcode
from PIL import Image

# Definiamo delle tuple simili a quelle usate dall'API di Pupil Labs
GazeDatum = namedtuple('GazeDatum', ['x', 'y', 'pupil_diameter_mm', 'timestamp_unix_seconds'])
SceneFrame = namedtuple('SceneFrame', ['image', 'timestamp_unix_seconds'])
EyesFrame = namedtuple('EyesFrame', ['image', 'timestamp_unix_seconds'])

class MockNeonDevice:
    # --- NUOVA FUNZIONE STATICA PER I QR CODE ---
    @staticmethod
    def _create_qr_code_image(data: str, size: int = 40):
        qr = qrcode.QRCode(
            version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10, border=2,
        )
        qr.add_data(data)
        qr.make(fit=True)
        img_pil = qr.make_image(fill_color="black", back_color="white").convert('RGB')
        img_pil = img_pil.resize((size, size), Image.Resampling.NEAREST)
        return np.array(img_pil)
    # --- FINE NUOVA FUNZIONE ---

    """
    Un simulatore che imita l'API di un dispositivo Pupil Labs Neon per
    generare un flusso di dati sintetici in tempo reale.
    """
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        
        # Stato della simulazione
        self.object1_pos = (0, 0)
        self.object2_pos = (0, 0)
        self.gaze_target = self.object1_pos

        # --- NUOVO: Genera le immagini dei QR code una sola volta ---
        self.qr_size = 40
        self.qr_tl_img = self._create_qr_code_image("TL", size=self.qr_size)
        self.qr_tr_img = self._create_qr_code_image("TR", size=self.qr_size)
        self.qr_bl_img = self._create_qr_code_image("BL", size=self.qr_size)
        self.qr_br_img = self._create_qr_code_image("BR", size=self.qr_size)

    def _generate_scene_frame(self):
        """Genera un singolo frame della scena con oggetti in movimento."""
        frame = np.full((self.height, self.width, 3), (25, 25, 25), dtype=np.uint8)
        
        # Oggetto 1 (cerchio rosso) - si muove in cerchio
        angle1 = self.frame_count / self.fps
        self.object1_pos = (
            int(self.width / 2 + np.sin(angle1) * 200),
            int(self.height / 2 + np.cos(angle1) * 200)
        )
        cv2.circle(frame, self.object1_pos, 40, (0, 0, 255), -1)

        # Oggetto 2 (quadrato verde) - si muove in orizzontale
        self.object2_pos = (
            int(self.width * 0.75 + np.sin(self.frame_count / (self.fps / 2)) * 150),
            int(self.height * 0.75)
        )
        cv2.rectangle(frame, (self.object2_pos[0]-30, self.object2_pos[1]-30),
                      (self.object2_pos[0]+30, self.object2_pos[1]+30), (0, 255, 0), -1)
        
        # --- NUOVO: Disegna i QR code in modo intermittente attorno all'oggetto 1 ---
        # Appaiono per 10 secondi, scompaiono per 5 secondi
        if (self.frame_count % (self.fps * 15)) < (self.fps * 10):
            surface_w, surface_h = 250, 200
            center_x, center_y = self.object1_pos
            
            tl = (int(center_x - surface_w/2), int(center_y - surface_h/2))
            br = (int(center_x + surface_w/2), int(center_y + surface_h/2))

            def overlay_image(background, overlay, pos):
                x, y = pos; h, w, _ = overlay.shape
                if x >= 0 and y >= 0 and y+h < background.shape[0] and x+w < background.shape[1]:
                    background[y:y+h, x:x+w] = overlay

            overlay_image(frame, self.qr_tl_img, (tl[0], tl[1]))
            overlay_image(frame, self.qr_tr_img, (br[0] - self.qr_size, tl[1]))
            overlay_image(frame, self.qr_bl_img, (tl[0], br[1] - self.qr_size))
            overlay_image(frame, self.qr_br_img, (br[0] - self.qr_size, br[1] - self.qr_size))
        # --- FINE NUOVA LOGICA ---

        # Cambia il target dello sguardo ogni 5 secondi
        if self.frame_count % (self.fps * 5) == 0:
            self.gaze_target = self.object2_pos if np.random.rand() > 0.5 else self.object1_pos

        return frame

    def _generate_eyes_frame(self):
        """Genera un singolo frame della camera degli occhi."""
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        pupil_radius = int(20 + np.sin(self.frame_count / 10) * 8)
        # Simula due occhi
        cv2.circle(frame, (100, 100), 40, (200, 200, 200), -1)
        cv2.circle(frame, (100, 100), pupil_radius, (10, 10, 10), -1)
        cv2.circle(frame, (300, 100), 40, (200, 200, 200), -1)
        cv2.circle(frame, (300, 100), pupil_radius, (10, 10, 10), -1)
        return frame

    def _generate_gaze_datum(self):
        """Genera un singolo dato di sguardo che segue un oggetto."""
        noise_x = np.random.randint(-10, 11)
        noise_y = np.random.randint(-10, 11)
        
        gaze_x = self.gaze_target[0] + noise_x
        gaze_y = self.gaze_target[1] + noise_y
        
        # Simula il diametro pupillare in mm
        pupil_diameter = 4.5 + np.sin(self.frame_count / 15) * 1.5
        
        return GazeDatum(x=gaze_x, y=gaze_y, pupil_diameter_mm=pupil_diameter, timestamp_unix_seconds=time.time())

    # Metodi che l'analizzatore chiamerà
    def receive_scene_video_frame(self):
        self.frame_count += 1
        time.sleep(1 / self.fps) # Simula il frame rate
        return SceneFrame(image=self._generate_scene_frame(), timestamp_unix_seconds=time.time())

    def receive_eyes_video_frame(self):
        return EyesFrame(image=self._generate_eyes_frame(), timestamp_unix_seconds=time.time())

    def receive_gaze_datum(self):
        return self._generate_gaze_datum()

    def close(self):
        print("Mock device connection closed.")

# --- Esempio di utilizzo per testare la GUI ---
if __name__ == '__main__':
    import tkinter as tk
    from desktop_app.GUI import RealtimeDisplayWindow

    # Monkey-patch (sostituisce temporaneamente) il metodo connect 
    # dell'analizzatore per usare sempre il nostro mock.
    from src.speed_analyzer.analysis_modules.realtime_analyzer import RealtimeNeonAnalyzer
    from desktop_app.GUI import MODELS_DIR
    
    # Salva il metodo originale
    original_connect = RealtimeNeonAnalyzer.connect
    
    # Definisci il nuovo metodo che usa il mock
    def mock_connect(analyzer_instance):
        # Simula la connessione al mock device
        mock_device = MockNeonDevice()
        analyzer_instance.device = mock_device
        print("Connected to Mock Neon Device for testing.")
        return True
        
    # Sostituisci il metodo originale con quello mockato
    RealtimeNeonAnalyzer.connect = mock_connect

    print("="*50)
    print(" AVVIO TEST CON FLUSSO SINTETICO ")
    print("="*50)
    print("Questo script avvierà la finestra di streaming in tempo reale")
    print("utilizzando dati generati localmente invece di un dispositivo reale.")
    print("Chiudi la finestra per terminare lo script.")

    root = tk.Tk()
    root.withdraw() # Nasconde la finestra principale, non ci serve
    
    # Avvia la finestra di visualizzazione che ora userà il nostro mock
    app = RealtimeDisplayWindow(root)

    # --- CORREZIONE: Simula la selezione del modello nella GUI ---
    # Imposta il modello e il task prima di avviare lo stream.
    app.yolo_model_var.set('yolov8n.pt')
    app.yolo_task_var.set('detect')

    # Simula il click sui pulsanti della GUI per avviare il test
    app.connect_to_device()

    app.start_stream()
    
    root.mainloop()