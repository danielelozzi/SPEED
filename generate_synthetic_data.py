# generate_synthetic_data.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
import random
from tqdm import tqdm

# --- 1. CONFIGURAZIONE DEL LOGGING ---
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)
log_file_path = log_dir / f"synthetic_data_log_{time.strftime('%Y%m%d-%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

# --- 2. COSTANTI DI CONFIGURAZIONE ---
logging.info("Configuration constants set.")
# Impostazioni Generali
OUTPUT_DIR = Path("./synthetic_data_output")
FPS = 60
DURATION_S = 45 # Durata totale della registrazione in secondi
NUM_FRAMES = DURATION_S * FPS

# Impostazioni Video
WIDTH, HEIGHT = 1280, 720
# Impostazioni Dati
NUM_EVENTS = 8
PUPIL_BASE = 4.0  # mm
PUPIL_AMPLITUDE = 0.8 # mm
NS_TO_S = 1e9

# --- 3. CREAZIONE DELLA STRUTTURA DELLE CARTELLE ---
logging.info(f"Creating directory structure inside {OUTPUT_DIR}")
raw_dir = OUTPUT_DIR / "RAW"
unenriched_dir = OUTPUT_DIR / "un-enriched"
enriched_dir = OUTPUT_DIR / "enriched"

raw_dir.mkdir(parents=True, exist_ok=True)
unenriched_dir.mkdir(parents=True, exist_ok=True)
enriched_dir.mkdir(parents=True, exist_ok=True)

# --- 4. FUNZIONI PER LA GENERAZIONE DEI VIDEO ---
def generate_videos():
    """
    Genera i file external.mp4 e internal.mp4 e restituisce le traiettorie degli oggetti.
    """
    ext_video_path = unenriched_dir / "external.mp4"
    int_video_path = raw_dir / "Neon Sensor Module v1 ps1.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_ext = cv2.VideoWriter(str(ext_video_path), fourcc, FPS, (WIDTH, HEIGHT))
    writer_int = cv2.VideoWriter(str(int_video_path), fourcc, FPS, (200, 200))
    
    surface_corners = []
    object_paths = {1: [], 2: []} # track_id 1 e 2

    logging.info("Generating external.mp4 and internal.mp4...")
    for i in tqdm(range(NUM_FRAMES), desc="Generating Videos"):
        # Frame esterno (scena)
        frame_ext = np.full((HEIGHT, WIDTH, 3), (20, 20, 20), dtype=np.uint8)
        angle = i / NUM_FRAMES * 2 * np.pi
        center_x = WIDTH / 2 + np.sin(angle) * 150
        center_y = HEIGHT / 2 + np.cos(angle * 0.7) * 100
        surface_w, surface_h = 400, 300
        tl = (int(center_x - surface_w/2), int(center_y - surface_h/2))
        br = (int(center_x + surface_w/2), int(center_y + surface_h/2))
        cv2.rectangle(frame_ext, tl, br, (150, 150, 150), -1)
        surface_corners.append({
            'tl x [px]': tl[0], 'tl y [px]': tl[1], 'tr x [px]': br[0], 'tr y [px]': tl[1],
            'br x [px]': br[0], 'br y [px]': br[1], 'bl x [px]': tl[0], 'bl y [px]': br[1],
        })

        # Disegna e traccia oggetti per YOLO
        obj1_pos = (int(WIDTH * 0.2 + np.sin(i / 30) * 50), int(HEIGHT * 0.3))
        obj2_pos = (int(WIDTH * 0.8 - np.cos(i / 45) * 60), int(HEIGHT * 0.7))
        cv2.circle(frame_ext, obj1_pos, 30, (0, 0, 255), -1) # Oggetto rosso
        cv2.circle(frame_ext, obj2_pos, 25, (0, 255, 0), -1) # Oggetto verde
        object_paths[1].append({'center_x': obj1_pos[0], 'center_y': obj1_pos[1], 'radius': 30})
        object_paths[2].append({'center_x': obj2_pos[0], 'center_y': obj2_pos[1], 'radius': 25})
        writer_ext.write(frame_ext)

        # Frame interno (occhio)
        frame_int = np.zeros((200, 200, 3), dtype=np.uint8)
        pupil_radius = int(15 + np.sin(i / 10) * 5)
        cv2.circle(frame_int, (100, 100), pupil_radius, (255, 255, 255), -1)
        writer_int.write(frame_int)

    writer_ext.release()
    writer_int.release()
    logging.info("Video generation complete.")
    return pd.DataFrame(surface_corners), object_paths

# --- 5. GENERAZIONE DI TUTTI I DATI CSV ---
def generate_csv_data(surface_corners_df):
    """Genera tutti i file CSV richiesti dall'applicazione."""
    logging.info("Generating base timestamp data...")
    start_ts = int(time.time() * 1e9)
    timestamps = (start_ts + np.arange(NUM_FRAMES) * (NS_TO_S / FPS)).astype('int64')

    world_ts_df = pd.DataFrame({'timestamp [ns]': timestamps, 'frame': range(NUM_FRAMES)})
    world_ts_df.to_csv(unenriched_dir / 'world_timestamps.csv', index=False)

    logging.info("Generating events.csv...")
    event_indices = sorted(random.sample(range(100, NUM_FRAMES - 100), NUM_EVENTS))
    events_df = pd.DataFrame({
        'timestamp [ns]': timestamps[event_indices],
        'name': [f'Event_{chr(65+i)}' for i in range(NUM_EVENTS)],
        'recording id': ['rec_001'] * NUM_EVENTS,
        'source': ['default'] * NUM_EVENTS # NUOVO: Aggiunge la fonte
    })
    events_df.to_csv(unenriched_dir / 'events.csv', index=False)

    # NUOVO: Crea anche un file modified_events.csv per simulare l'output dell'editor
    logging.info("Generating modified_events.csv to simulate editor output...")
    modified_events_df = events_df.copy()
    modified_events_df.loc[0, 'name'] = 'Edited_Event_A' # Simula una modifica
    modified_events_df = modified_events_df.drop(1) # Simula una rimozione
    new_event_ts = timestamps[random.randint(100, NUM_FRAMES - 100)]
    new_event = pd.DataFrame([{'timestamp [ns]': new_event_ts, 'name': 'Manually_Added_Event', 'recording id': 'rec_001', 'source': 'manual'}])
    modified_events_df = pd.concat([modified_events_df, new_event], ignore_index=True).sort_values('timestamp [ns]')
    modified_events_df.to_csv(OUTPUT_DIR / 'modified_events.csv', index=False)
    
    logging.info("Generating behavioral data (gaze, fixations, saccades, blinks)...")
    gaze_data, pupil_data, blinks_data, saccades_data, fixations_data = [], [], [], [], []
    fix_id, sac_id, blink_id = 0, 0, 0
    
    gaze_x, gaze_y = WIDTH / 2, HEIGHT / 2
    state = "FIXATING"
    state_frames_left = random.randint(30, 120)
    fix_start_ts = timestamps[0]

    for i in tqdm(range(NUM_FRAMES), desc="Behavioral Data"):
        ts = timestamps[i]
        
        state_frames_left -= 1
        if state_frames_left <= 0:
            if state == "FIXATING":
                fixations_data.append({'fixation id': fix_id, 'start timestamp [ns]': fix_start_ts, 'duration [ms]': (ts - fix_start_ts) / 1e6, 'fixation x [px]': gaze_x, 'fixation y [px]': gaze_y})
                fix_id += 1
                state = "SACCADING"; state_frames_left = random.randint(5, 15); saccade_start_ts = ts
                start_x, start_y = gaze_x, gaze_y
                target_x, target_y = random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)

            elif state == "SACCADING":
                duration_ms = (ts - saccade_start_ts) / 1e6
                duration_s = duration_ms / 1000
                amplitude = np.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)
                mean_vel = amplitude / duration_s if duration_s > 0 else 0
                saccades_data.append({
                    'saccade id': sac_id, 'start timestamp [ns]': saccade_start_ts, 'duration [ms]': duration_ms,
                    'amplitude [px]': amplitude, 'mean velocity [px/s]': mean_vel, 'peak velocity [px/s]': mean_vel * (1.5 + random.uniform(-0.2, 0.2))
                })
                sac_id += 1
                gaze_x, gaze_y = target_x, target_y
                
                if random.random() < 0.1:
                    state = "BLINKING"; state_frames_left = random.randint(10, 25); blink_start_ts = ts
                else:
                    state = "FIXATING"; state_frames_left = random.randint(30, 120); fix_start_ts = ts
            
            elif state == "BLINKING":
                blinks_data.append({'blink id': blink_id, 'start timestamp [ns]': blink_start_ts, 'end timestamp [ns]': ts, 'duration [ms]': (ts - blink_start_ts) / 1e6})
                blink_id += 1
                state = "FIXATING"; state_frames_left = random.randint(30, 120); fix_start_ts = ts
        
        px, py, pupil_l, pupil_r = np.nan, np.nan, np.nan, np.nan
        if state == "FIXATING":
            px = int(gaze_x + random.uniform(-1, 1) * 3); py = int(gaze_y + random.uniform(-1, 1) * 3)
        elif state == "SACCADING":
            progress = 1 - (state_frames_left / (random.randint(5,15) + 1)); px = int(start_x + (target_x - start_x) * progress); py = int(start_y + (target_y - start_y) * progress)

        if state != "BLINKING":
            pupil_l = PUPIL_BASE + np.sin(i / 100) * PUPIL_AMPLITUDE + random.uniform(-0.1, 0.1); pupil_r = PUPIL_BASE + np.sin(i / 100 + 0.1) * PUPIL_AMPLITUDE + random.uniform(-0.1, 0.1)

        gaze_data.append({'timestamp [ns]': ts, 'gaze x [px]': px, 'gaze y [px]': py})
        pupil_data.append({'timestamp [ns]': ts, 'pupil diameter left [mm]': pupil_l, 'pupil diameter right [mm]': pupil_r})

    logging.info("Saving behavioral CSV files...")
    gaze_df = pd.DataFrame(gaze_data); gaze_df.to_csv(unenriched_dir / 'gaze.csv', index=False)
    pd.DataFrame(pupil_data).to_csv(unenriched_dir / '3d_eye_states.csv', index=False)
    pd.DataFrame(blinks_data).to_csv(unenriched_dir / 'blinks.csv', index=False)
    fixations_df = pd.DataFrame(fixations_data); fixations_df.to_csv(unenriched_dir / 'fixations.csv', index=False)
    pd.DataFrame(saccades_data).to_csv(unenriched_dir / 'saccades.csv', index=False)
    
    logging.info("Generating enriched data...")
    surface_corners_df['timestamp [ns]'] = timestamps
    surface_corners_df.to_csv(enriched_dir / 'surface_positions.csv', index=False)

    gaze_enriched_data = []
    for i, row in gaze_df.iterrows():
        if pd.notna(row['gaze x [px]']):
            surf = surface_corners_df.iloc[i]; x, y = row['gaze x [px]'], row['gaze y [px]']
            on_surface = (surf['tl x [px]'] < x < surf['br x [px]']) and (surf['tl y [px]'] < y < surf['br y [px]'])
            norm_x, norm_y = (x - surf['tl x [px]']) / (surf['br x [px]'] - surf['tl x [px]']), (y - surf['tl y [px]']) / (surf['br y [px]'] - surf['tl y [px]']) if on_surface else (np.nan, np.nan)
            gaze_enriched_data.append({'timestamp [ns]': row['timestamp [ns]'], 'gaze detected on surface': on_surface, 'gaze position on surface x [normalized]': norm_x, 'gaze position on surface y [normalized]': norm_y})
    pd.DataFrame(gaze_enriched_data).to_csv(enriched_dir / 'gaze.csv', index=False)

    fix_enriched_df = fixations_df.copy(); fix_enriched_df['fixation x [normalized]'] = fix_enriched_df['fixation x [px]'] / WIDTH; fix_enriched_df['fixation y [normalized]'] = fix_enriched_df['fixation y [px]'] / HEIGHT
    fix_enriched_df['fixation detected on surface'] = True; fix_enriched_df.to_csv(enriched_dir / 'fixations.csv', index=False)
    
    logging.info("All input CSV files have been generated.")


if __name__ == "__main__":
    logging.info("--- Starting Synthetic Data Generation ---")
    try:
        surface_df, object_paths = generate_videos()
        generate_csv_data(surface_df)

        logging.info("="*60)
        logging.info("SYNTHETIC DATA GENERATION COMPLETE!")
        logging.info(f"All data saved in: {OUTPUT_DIR.resolve()}")
        logging.info("\nCOME USARE QUESTI DATI IN SPEED:")
        logging.info("1. Esegui `GUI.py`.")
        logging.info("2. Nella GUI, imposta le cartelle come segue:")
        logging.info(f"   - Participant Name: synthetic_test")
        logging.info(f"   - Output Folder:      {OUTPUT_DIR.resolve()}")
        logging.info(f"   - RAW Data Folder:      {raw_dir.resolve()}")
        logging.info(f"   - Un-enriched Folder:   {unenriched_dir.resolve()}")
        logging.info(f"   - Enriched Folder:      {enriched_dir.resolve()}")
        logging.info("3. Gli eventi di default (`events.csv`) verranno caricati automaticamente.")
        logging.info("4. Clicca su 'Edit on Video'. Vedrai gli eventi e potrai modificarli.")
        logging.info("5. Dopo aver salvato, un file `modified_events.csv` verrà creato in `synthetic_data_output`.")
        logging.info("6. Esegui 'RUN CORE ANALYSIS'. L'analisi userà automaticamente `modified_events.csv`.")
        logging.info("7. Procedi con la generazione di plot e video come al solito.")
        logging.info("="*60)

    except Exception as e:
        logging.error(f"An error occurred during data generation: {e}", exc_info=True)