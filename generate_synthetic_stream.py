# generate_synthetic_stream.py
import cv2
import numpy as np
import time
import threading
from collections import namedtuple

# Definiamo delle tuple simili a quelle usate dall'API di Pupil Labs
GazeDatum = namedtuple('GazeDatum', ['x', 'y', 'pupil_diameter_mm', 'timestamp_unix_seconds'])
SceneFrame = namedtuple('SceneFrame', ['image', 'timestamp_unix_seconds'])
EyesFrame = namedtuple('EyesFrame', ['image', 'timestamp_unix_seconds'])

class MockNeonDevice:
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
    
    # Salva il metodo originale
    original_connect = RealtimeNeonAnalyzer.connect
    
    # Definisci il nuovo metodo che usa il mock
    def mock_connect(self):
        mock_device = MockNeonDevice()
        return original_connect(self, mock_device=mock_device)
        
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
    
    root.mainloop()