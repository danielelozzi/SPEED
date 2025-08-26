# lsl_stream_simulator.py
import time
import random
import threading
import numpy as np
import cv2
from pylsl import StreamInfo, StreamOutlet, local_clock

# --- COSTANTI DI CONFIGURAZIONE ---
# Impostazioni dello stream
GAZE_STREAM_NAME = 'SimulatedGaze'
GAZE_STREAM_TYPE = 'Gaze'
GAZE_CHANNEL_COUNT = 3  # (x, y, pupil_diameter)
GAZE_RATE = 120  # Hz

VIDEO_STREAM_NAME = 'SimulatedVideo'
VIDEO_STREAM_TYPE = 'Video'
VIDEO_RATE = 30  # FPS

EVENT_STREAM_NAME = 'SimulatedEvents'
EVENT_STREAM_TYPE = 'Markers'
EVENT_RATE = 0.2  # Hz (un evento ogni 5 secondi in media)

# Impostazioni del video
WIDTH, HEIGHT = 1280, 720

def simulate_gaze_stream(outlet: StreamOutlet):
    """
    Simula e invia dati di sguardi attraverso un outlet LSL.
    Lo sguardo seguirà un oggetto che si muove in modo sinusoidale.
    """
    print("Gaze stream started.")
    frame_count = 0
    pupil_base = 4.0
    pupil_amplitude = 0.8
    
    while True:
        # Calcola la posizione target dello sguardo (un movimento a "otto")
        angle = local_clock() * 0.5
        target_x = WIDTH / 2 + np.sin(angle) * 300
        target_y = HEIGHT / 2 + np.cos(angle * 2) * 200
        
        # Aggiungi rumore
        gaze_x = target_x + random.uniform(-10, 10)
        gaze_y = target_y + random.uniform(-10, 10)
        
        # Simula il diametro della pupilla
        pupil_diameter = pupil_base + np.sin(frame_count / 50) * pupil_amplitude
        
        # Crea il campione e invialo
        sample = [gaze_x, gaze_y, pupil_diameter]
        timestamp = local_clock()
        outlet.push_sample(sample, timestamp)
        
        frame_count += 1
        time.sleep(1 / GAZE_RATE)

def simulate_video_stream(outlet: StreamOutlet):
    """
    Genera e invia frame video simulati come array di byte.
    """
    print("Video stream started.")
    frame_count = 0
    
    while True:
        # Crea un'immagine nera
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # Disegna un cerchio rosso che si muove
        angle1 = local_clock() * 0.5
        obj1_pos = (
            int(WIDTH / 2 + np.sin(angle1) * 300),
            int(HEIGHT / 2 + np.cos(angle1 * 2) * 200)
        )
        cv2.circle(frame, obj1_pos, 40, (0, 0, 255), -1) # Rosso in BGR
        
        # Disegna un rettangolo verde che si muove orizzontalmente
        obj2_pos = (
            int(WIDTH * 0.75 + np.sin(local_clock()) * 150),
            int(HEIGHT * 0.75)
        )
        cv2.rectangle(frame, (obj2_pos[0]-30, obj2_pos[1]-30),
                      (obj2_pos[0]+30, obj2_pos[1]+30), (0, 255, 0), -1) # Verde in BGR

        # Aggiungi testo informativo
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Invia il frame
        # Nota: LSL invia array 1D, quindi appiattiamo l'immagine.
        # Il ricevitore dovrà conoscere le dimensioni (WIDTH, HEIGHT, canali) per ricostruirla.
        outlet.push_sample(frame.flatten())
        
        frame_count += 1
        time.sleep(1 / VIDEO_RATE)

def simulate_event_stream(outlet: StreamOutlet):
    """
    Simula e invia eventi (marcatori) a intervalli casuali.
    """
    print("Event stream started.")
    possible_events = ["Task_Start", "Task_End", "Target_Appeared", "User_Response", "Pause"]
    
    while True:
        # Scegli un evento casuale e invialo
        event = random.choice(possible_events)
        outlet.push_sample([event])
        print(f"--> Sent event: {event}")
        
        # Attendi un tempo casuale prima del prossimo evento
        time.sleep(random.uniform(3, 8))


if __name__ == "__main__":
    print("--- Avvio del Simulatore di Streaming LSL ---")
    
    # --- Setup Stream Gaze ---
    info_gaze = StreamInfo(GAZE_STREAM_NAME, GAZE_STREAM_TYPE, GAZE_CHANNEL_COUNT, GAZE_RATE, 'float32', 'GazeSimID1')
    info_gaze.desc().append_child_value("manufacturer", "Simulated Inc.")
    channels = info_gaze.desc().append_child("channels")
    channels.append_child("channel").append_child_value("label", "x_position_px")
    channels.append_child("channel").append_child_value("label", "y_position_px")
    channels.append_child("channel").append_child_value("label", "pupil_diameter_mm")
    outlet_gaze = StreamOutlet(info_gaze)

    # --- Setup Stream Video ---
    # Per il video, inviamo un grande array. Il numero di canali è larghezza * altezza * 3 (BGR)
    video_channel_count = WIDTH * HEIGHT * 3
    info_video = StreamInfo(VIDEO_STREAM_NAME, VIDEO_STREAM_TYPE, video_channel_count, VIDEO_RATE, 'uint8', 'VideoSimID1')
    info_video.desc().append_child_value("width", str(WIDTH))
    info_video.desc().append_child_value("height", str(HEIGHT))
    info_video.desc().append_child_value("color_format", "BGR")
    outlet_video = StreamOutlet(info_video)

    # --- Setup Stream Eventi ---
    info_event = StreamInfo(EVENT_STREAM_NAME, EVENT_STREAM_TYPE, 1, 0, 'string', 'EventSimID1') # Rate 0 per eventi irregolari
    outlet_event = StreamOutlet(info_event)

    # --- Avvio dei thread per ogni stream ---
    thread_gaze = threading.Thread(target=simulate_gaze_stream, args=(outlet_gaze,), daemon=True)
    thread_video = threading.Thread(target=simulate_video_stream, args=(outlet_video,), daemon=True)
    thread_event = threading.Thread(target=simulate_event_stream, args=(outlet_event,), daemon=True)
    
    thread_gaze.start()
    thread_video.start()
    thread_event.start()
    
    print("\nStreams attivi. Premi Ctrl+C per terminare.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n--- Simulazione terminata. ---")