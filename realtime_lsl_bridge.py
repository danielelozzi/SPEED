# realtime_lsl_bridge.py
import time
import threading
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

# Importa la classe per la gestione del dispositivo dai tuoi file
# Assicurati che il percorso sia corretto in base alla struttura del tuo progetto
from src.speed_analyzer.analysis_modules.realtime_analyzer import RealtimeNeonAnalyzer

# --- COSTANTI DI CONFIGURAZIONE LSL ---
GAZE_STREAM_NAME = 'PupilNeonGaze'
GAZE_STREAM_TYPE = 'Gaze'
GAZE_CHANNEL_COUNT = 3  # (x, y, pupil_diameter)
# Il rate è nominale, i timestamp LSL garantiranno la sincronia
GAZE_RATE = 120 

VIDEO_STREAM_NAME = 'PupilNeonVideo'
VIDEO_STREAM_TYPE = 'Video'
VIDEO_RATE = 30 # FPS nominali

EVENT_STREAM_NAME = 'ManualEvents'
EVENT_STREAM_TYPE = 'Markers'

def run_lsl_bridge(analyzer: RealtimeNeonAnalyzer):
    """
    Funzione principale che recupera i dati dall'analizzatore e li invia a LSL.
    """
    print("--- Avvio del Ponte LSL in tempo reale ---")

    # --- Setup Stream Gaze ---
    info_gaze = StreamInfo(GAZE_STREAM_NAME, GAZE_STREAM_TYPE, GAZE_CHANNEL_COUNT, GAZE_RATE, 'float32', 'PupilNeonGazeID1')
    info_gaze.desc().append_child_value("manufacturer", "Pupil Labs")
    channels_gaze = info_gaze.desc().append_child("channels")
    channels_gaze.append_child("channel").append_child_value("label", "x_position_px")
    channels_gaze.append_child("channel").append_child_value("label", "y_position_px")
    channels_gaze.append_child("channel").append_child_value("label", "pupil_diameter_mm")
    outlet_gaze = StreamOutlet(info_gaze)

    # --- Setup Stream Video ---
    # Le dimensioni del video vengono recuperate dopo la prima immagine
    outlet_video = None
    
    # --- Setup Stream Eventi ---
    info_event = StreamInfo(EVENT_STREAM_NAME, EVENT_STREAM_TYPE, 1, 0, 'string', 'ManualEventID1')
    outlet_event = StreamOutlet(info_event)

    print("Stream LSL creati. In attesa di dati dal dispositivo...")
    
    # Avvia un thread separato per l'invio manuale di eventi
    event_thread = threading.Thread(target=listen_for_manual_events, args=(outlet_event,), daemon=True)
    event_thread.start()

    while True:
        # Recupera i dati più recenti dal dispositivo
        scene_frame, _, gaze_datum = analyzer.get_latest_frames_and_gaze()

        # Invia i dati di sguardo se disponibili
        if gaze_datum:
            sample = [
                gaze_datum.x, 
                gaze_datum.y, 
                gaze_datum.pupil_diameter_mm if hasattr(gaze_datum, 'pupil_diameter_mm') else 0.0
            ]
            # Usa il timestamp Unix fornito dal dispositivo per una maggiore precisione
            outlet_gaze.push_sample(sample, gaze_datum.timestamp_unix_seconds)

        # Invia i frame video se disponibili
        if scene_frame:
            frame_image = scene_frame.image
            
            # Se è il primo frame, inizializza lo stream video con le dimensioni corrette
            if outlet_video is None:
                h, w, channels = frame_image.shape
                video_channel_count = h * w * channels
                info_video = StreamInfo(VIDEO_STREAM_NAME, VIDEO_STREAM_TYPE, video_channel_count, VIDEO_RATE, 'uint8', 'PupilNeonVideoID1')
                info_video.desc().append_child_value("width", str(w))
                info_video.desc().append_child_value("height", str(h))
                info_video.desc().append_child_value("color_format", "RGB") # La camera del Neon è RGB
                outlet_video = StreamOutlet(info_video)
                print(f"Video stream inizializzato con dimensioni {w}x{h}.")

            # Invia il frame appiattito
            outlet_video.push_sample(frame_image.flatten(), scene_frame.timestamp_unix_seconds)

        # Evita un ciclo troppo stretto se non ci sono dati
        time.sleep(1 / (GAZE_RATE * 2))


def listen_for_manual_events(outlet: StreamOutlet):
    """Ascolta l'input da terminale per inviare eventi LSL."""
    event_counter = 1
    while True:
        event_name = input("Premi Invio per inviare un evento, o scrivi un nome e premi Invio:\n")
        if not event_name:
            event_name = f"Marker_{event_counter}"
            event_counter += 1
        
        outlet.push_sample([event_name])
        print(f"--> Inviato evento: {event_name}")


if __name__ == "__main__":
    # Inizializza l'analizzatore che si connette al dispositivo
    neon_analyzer = RealtimeNeonAnalyzer()
    
    print("Ricerca del dispositivo Pupil Labs Neon in corso...")
    if not neon_analyzer.connect():
        print("Impossibile trovare un dispositivo. Assicurati che sia connesso alla stessa rete.")
    else:
        try:
            run_lsl_bridge(neon_analyzer)
        except KeyboardInterrupt:
            print("\n--- Chiusura del ponte LSL. ---")
        finally:
            neon_analyzer.close()