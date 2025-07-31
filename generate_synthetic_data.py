# generate_synthetic_data.py
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
import random

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
    """Genera i file external.mp4 e internal.mp4"""
    
    # --- External Video (Scena) ---
    logging.info("Generating external.mp4...")
    ext_video_path = unenriched_dir / "external.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer_ext = cv2.VideoWriter(str(ext_video_path), fourcc, FPS, (WIDTH, HEIGHT))
    
    # Dati per la superficie e gli oggetti in movimento
    surface_w, surface_h = 400, 300
    surface_corners = []
    
    for i in tqdm(range(NUM_FRAMES), desc="External Video"):
        frame = np.full((HEIGHT, WIDTH, 3), (20, 20, 20), dtype=np.uint8)
        
        # Disegna la superficie mobile (rettangolo)
        angle = i / NUM_FRAMES * 2 * np.pi
        center_x = WIDTH / 2 + np.sin(angle) * 150
        center_y = HEIGHT / 2 + np.cos(angle * 0.7) * 100
        tl = (int(center_x - surface_w/2), int(center_y - surface_h/2))
        br = (int(center_x + surface_w/2), int(center_y + surface_h/2))
        cv2.rectangle(frame, tl, br, (150, 150, 150), -1)
        # Salva le coordinate degli angoli per surface_positions.csv
        surface_corners.append({
            'tl x [px]': tl[0], 'tl y [px]': tl[1],
            'tr x [px]': br[0], 'tr y [px]': tl[1],
            'br x [px]': br[0], 'br y [px]': br[1],
            'bl x [px]': tl[0], 'bl y [px]': br[1],
        })

        # Disegna oggetti mobili (cerchi) per YOLO
        obj1_x = int(WIDTH * 0.2 + np.sin(i / 30) * 50)
        obj1_y = int(HEIGHT * 0.3)
        cv2.circle(frame, (obj1_x, obj1_y), 30, (0, 0, 255), -1) # Oggetto rosso

        obj2_x = int(WIDTH * 0.8 - np.cos(i / 45) * 60)
        obj2_y = int(HEIGHT * 0.7)
        cv2.circle(frame, (obj2_x, obj2_y), 25, (0, 255, 0), -1) # Oggetto verde

        writer_ext.write(frame)
    writer_ext.release()
    logging.info("external.mp4 generation complete.")

    # --- Internal Video (Occhio) ---
    logging.info("Generating internal.mp4...")
    int_video_path = raw_dir / "Neon Sensor Module v1 ps1.mp4"
    writer_int = cv2.VideoWriter(str(int_video_path), fourcc, FPS, (200, 200))
    for _ in tqdm(range(NUM_FRAMES), desc="Internal Video"):
        frame_int = np.zeros((200, 200, 3), dtype=np.uint8)
        pupil_radius = int(15 + np.sin(_ / 10) * 5)
        cv2.circle(frame_int, (100, 100), pupil_radius, (255, 255, 255), -1)
        writer_int.write(frame_int)
    writer_int.release()
    logging.info("internal.mp4 generation complete.")
    
    return pd.DataFrame(surface_corners)

# --- 5. GENERAZIONE DI TUTTI I DATI CSV ---
def generate_csv_data(surface_corners_df):
    """Genera tutti i file CSV richiesti dall'applicazione."""
    logging.info("Generating base timestamp data...")
    start_ts = int(time.time() * 1e9)
    timestamps = start_ts + np.arange(NUM_FRAMES) * (1e9 / FPS)
    timestamps = timestamps.astype('int64')

    # World Timestamps
    world_ts_df = pd.DataFrame({'timestamp [ns]': timestamps})
    world_ts_df.to_csv(unenriched_dir / 'world_timestamps.csv', index=False)

    # Events
    logging.info("Generating events.csv...")
    event_indices = sorted(random.sample(range(NUM_FRAMES), NUM_EVENTS))
    events_df = pd.DataFrame({
        'timestamp [ns]': timestamps[event_indices],
        'name': [f'Event_{chr(65+i)}' for i in range(NUM_EVENTS)],
        'recording id': ['rec_001'] * NUM_EVENTS
    })
    events_df.to_csv(unenriched_dir / 'events.csv', index=False)

    # Generazione dati comportamentali (Gaze, Fixations, Blinks, Saccades)
    logging.info("Generating behavioral data (gaze, fixations, etc.)...")
    gaze_data, pupil_data, blinks_data, saccades_data, fixations_data = [], [], [], [], []
    fix_id, sac_id, blink_id = 0, 0, 0
    
    current_gaze_x, current_gaze_y = WIDTH / 2, HEIGHT / 2
    state = "FIXATING"
    state_frames_left = random.randint(30, 120)
    fix_start_ts = timestamps[0]

    for i in tqdm(range(NUM_FRAMES), desc="Behavioral Data"):
        ts = timestamps[i]
        
        # Transizioni di stato
        state_frames_left -= 1
        if state_frames_left <= 0:
            if state == "FIXATING":
                # Fine fissazione, inizio saccade
                fixations_data.append({
                    'fixation id': fix_id,
                    'start timestamp [ns]': fix_start_ts,
                    'duration [ms]': (ts - fix_start_ts) / 1e6,
                    'fixation x [px]': current_gaze_x,
                    'fixation y [px]': current_gaze_y
                })
                fix_id += 1
                
                state = "SACCADING"
                state_frames_left = random.randint(5, 15)
                target_gaze_x = random.randint(50, WIDTH - 50)
                target_gaze_y = random.randint(50, HEIGHT - 50)
                saccade_start_ts = ts
                saccades_data.append({'saccade id': sac_id, 'start timestamp [ns]': ts}) # Dati incompleti per ora

            elif state == "SACCADING":
                # Fine saccade, inizio fissazione o blink
                if 'duration [ms]' not in saccades_data[-1]:
                     saccades_data[-1]['duration [ms]'] = (ts - saccade_start_ts) / 1e6
                
                current_gaze_x, current_gaze_y = target_gaze_x, target_gaze_y
                if random.random() < 0.1: # 10% probabilità di un blink
                    state = "BLINKING"
                    state_frames_left = random.randint(10, 25)
                    blinks_data.append({'blink id': blink_id, 'start timestamp [ns]': ts})
                else:
                    state = "FIXATING"
                    state_frames_left = random.randint(30, 120)
                    fix_start_ts = ts
            
            elif state == "BLINKING":
                blinks_data[-1]['end timestamp [ns]'] = ts
                blinks_data[-1]['duration [ms]'] = (ts - blinks_data[-1]['start timestamp [ns]']) / 1e6
                blink_id += 1
                state = "FIXATING"
                state_frames_left = random.randint(30, 120)
                fix_start_ts = ts
        
        # Genera dati in base allo stato
        pupil_l, pupil_r = np.nan, np.nan
        gaze_x, gaze_y = np.nan, np.nan

        if state == "BLINKING":
            pass # i dati rimangono NaN
        elif state == "SACCADING":
            progress = 1 - (state_frames_left / (random.randint(5,15) + 1))
            gaze_x = int(current_gaze_x + (target_gaze_x - current_gaze_x) * progress)
            gaze_y = int(current_gaze_y + (target_gaze_y - current_gaze_y) * progress)
        else: # FIXATING
            gaze_x = int(current_gaze_x + random.uniform(-1, 1) * 3) # Micro-movimenti
            gaze_y = int(current_gaze_y + random.uniform(-1, 1) * 3)

        if state != "BLINKING":
            pupil_l = PUPIL_BASE + np.sin(i / 100) * PUPIL_AMPLITUDE + random.uniform(-0.1, 0.1)
            pupil_r = PUPIL_BASE + np.sin(i / 100 + 0.1) * PUPIL_AMPLITUDE + random.uniform(-0.1, 0.1)

        gaze_data.append({'timestamp [ns]': ts, 'gaze x [px]': gaze_x, 'gaze y [px]': gaze_y})
        pupil_data.append({'timestamp [ns]': ts, 'pupil diameter left [mm]': pupil_l, 'pupil diameter right [mm]': pupil_r})

    # Conversione in DataFrame e salvataggio
    logging.info("Saving behavioral CSV files...")
    gaze_df = pd.DataFrame(gaze_data); gaze_df.to_csv(unenriched_dir / 'gaze.csv', index=False)
    pupil_df = pd.DataFrame(pupil_data); pupil_df.to_csv(unenriched_dir / '3d_eye_states.csv', index=False)
    blinks_df = pd.DataFrame(blinks_data); blinks_df.to_csv(unenriched_dir / 'blinks.csv', index=False)
    fixations_df = pd.DataFrame(fixations_data); fixations_df.to_csv(unenriched_dir / 'fixations.csv', index=False)
    saccades_df = pd.DataFrame(saccades_data); saccades_df.to_csv(unenriched_dir / 'saccades.csv', index=False)
    
    # Generazione dati Enriched
    logging.info("Generating enriched data...")
    surface_corners_df['timestamp [ns]'] = timestamps
    surface_corners_df.to_csv(enriched_dir / 'surface_positions.csv', index=False)

    gaze_enriched_data = []
    for i, row in gaze_df.iterrows():
        if pd.notna(row['gaze x [px]']):
            surf = surface_corners_df.iloc[i]
            x, y = row['gaze x [px]'], row['gaze y [px]']
            on_surface = (surf['tl x [px]'] < x < surf['br x [px]']) and (surf['tl y [px]'] < y < surf['br y [px]'])
            
            norm_x, norm_y = np.nan, np.nan
            if on_surface:
                norm_x = (x - surf['tl x [px]']) / (surf['br x [px]'] - surf['tl x [px]'])
                norm_y = (y - surf['tl y [px]']) / (surf['br y [px]'] - surf['tl y [px]'])

            gaze_enriched_data.append({
                'timestamp [ns]': row['timestamp [ns]'],
                'gaze detected on surface': on_surface,
                'gaze position on surface x [normalized]': norm_x,
                'gaze position on surface y [normalized]': norm_y,
            })
    gaze_enriched_df = pd.DataFrame(gaze_enriched_data)
    gaze_enriched_df.to_csv(enriched_dir / 'gaze.csv', index=False)

    # Crea un fixations_enriched.csv fittizio
    fix_enriched_df = fixations_df.copy()
    fix_enriched_df['fixation x [normalized]'] = fix_enriched_df['fixation x [px]'] / WIDTH
    fix_enriched_df['fixation y [normalized]'] = fix_enriched_df['fixation y [px]'] / HEIGHT
    fix_enriched_df['fixation detected on surface'] = True
    fix_enriched_df.to_csv(enriched_dir / 'fixations.csv', index=False)
    
    logging.info("All CSV files have been generated.")


if __name__ == "__main__":
    logging.info("--- Starting Synthetic Data Generation ---")
    try:
        surface_df = generate_videos()
        generate_csv_data(surface_df)
        logging.info("="*50)
        logging.info(f"SYNTHETIC DATA GENERATION COMPLETE!")
        logging.info(f"Data saved in: {OUTPUT_DIR.resolve()}")
        logging.info("You can now use these folders in the SPEED application:")
        logging.info(f"  - RAW Folder: {raw_dir.resolve()}")
        logging.info(f"  - Un-enriched Folder: {unenriched_dir.resolve()}")
        logging.info(f"  - Enriched Folder: {enriched_dir.resolve()}")
        logging.info("="*50)

    except Exception as e:
        logging.error(f"An error occurred during data generation: {e}", exc_info=True)