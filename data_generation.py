import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path

# --- CONFIGURAZIONE ---
OUTPUT_DIR = Path("SPEED_TEST_DATA")
RAW_DIR = OUTPUT_DIR / "Raw_Data"
ENR_DIR = OUTPUT_DIR / "Enrichment_EdgeCase"

DURATION_SEC = 5
FPS = 30
TOTAL_FRAMES = DURATION_SEC * FPS
START_TS = 1000000000000  # Timestamp arbitrario in nanosecondi

def ensure_dir(directory):
    directory.mkdir(parents=True, exist_ok=True)

def generate_video(path, width=640, height=480, fps=30, frames=150):
    print(f"Generazione video dummy: {path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for _ in range(frames):
        # Frame nero con rumore casuale per simulare video
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

def main():
    print("Inizio generazione dati di test...")
    ensure_dir(RAW_DIR)
    ensure_dir(ENR_DIR)

    # 1. TIMESTAMPS
    timestamps = np.linspace(START_TS, START_TS + (DURATION_SEC * 1e9), TOTAL_FRAMES).astype(int)
    df_ts = pd.DataFrame({'timestamp [ns]': timestamps})
    df_ts.to_csv(RAW_DIR / "world_timestamps.csv", index=False)

    # 2. EVENTS
    # Creiamo eventi per segmentare i dati
    events_data = [
        {'name': 'recording.begin', 'timestamp [ns]': timestamps[0]},
        {'name': 'test_event_1', 'timestamp [ns]': timestamps[30]},  # Evento al secondo 1
        {'name': 'test_event_2', 'timestamp [ns]': timestamps[90]},  # Evento al secondo 3
        {'name': 'recording.end', 'timestamp [ns]': timestamps[-1]}
    ]
    pd.DataFrame(events_data).to_csv(RAW_DIR / "events.csv", index=False)

    # 3. GAZE RAW (Standard)
    df_gaze = pd.DataFrame({
        'timestamp [ns]': timestamps,
        'gaze x [normalized]': np.random.rand(TOTAL_FRAMES),
        'gaze y [normalized]': np.random.rand(TOTAL_FRAMES),
        'confidence': np.ones(TOTAL_FRAMES)
    })
    df_gaze.to_csv(RAW_DIR / "gaze.csv", index=False)

    # 4. FIXATIONS RAW (Standard)
    # Generiamo 5 fissazioni casuali
    fix_starts = np.sort(np.random.choice(timestamps, 5))
    df_fix = pd.DataFrame({
        'start timestamp [ns]': fix_starts,
        'duration [ms]': np.random.randint(100, 300, 5),
        'fixation x [normalized]': np.random.rand(5),
        'fixation y [normalized]': np.random.rand(5)
    })
    df_fix.to_csv(RAW_DIR / "fixations.csv", index=False)

    # 5. ALTRI FILE OBBLIGATORI (Blinks, Pupil, Saccades)
    pd.DataFrame({
        'start timestamp [ns]': [timestamps[10]],
        'duration [ms]': [150]
    }).to_csv(RAW_DIR / "blinks.csv", index=False)

    pd.DataFrame({
        'start timestamp [ns]': [timestamps[20]],
        'duration [ms]': [30]
    }).to_csv(RAW_DIR / "saccades.csv", index=False)

    # Pupil data (necessario per evitare warning nei plot)
    df_pupil = pd.DataFrame({
        'timestamp [ns]': timestamps,
        'pupil diameter left [mm]': np.random.uniform(3, 5, TOTAL_FRAMES),
        'pupil diameter right [mm]': np.random.uniform(3, 5, TOTAL_FRAMES)
    })
    df_pupil.to_csv(RAW_DIR / "3d_eye_states.csv", index=False)

    # 6. VIDEO
    generate_video(RAW_DIR / "external.mp4", frames=TOTAL_FRAMES)

    # ---------------------------------------------------------
    # 7. ENRICHMENT "EDGE CASE" (Per testare la tua fix)
    # ---------------------------------------------------------
    print("Generazione dati Enrichment per testare il bug...")
    
    # GAZE ENRICHED: Mettiamo dati corretti qui
    df_gaze_enr = df_gaze.copy()
    df_gaze_enr['gaze detected on surface'] = True
    df_gaze_enr['gaze position on surface x [normalized]'] = np.random.rand(TOTAL_FRAMES)
    df_gaze_enr['gaze position on surface y [normalized]'] = np.random.rand(TOTAL_FRAMES)
    df_gaze_enr.to_csv(ENR_DIR / "gaze.csv", index=False)

    # FIXATIONS ENRICHED (IL TEST CRITICO):
    # Simuliamo il caso che faceva crashare il software:
    # Flag 'fixation detected on surface' presente e True, MA coordinate ASSENTI.
    df_fix_enr = df_fix.copy()
    df_fix_enr['fixation detected on surface'] = True
    # NOTA: NON aggiungiamo le colonne 'fixation position on surface x [normalized]'
    # Se la tua fix funziona, il software leggerà questo file senza crashare.
    df_fix_enr.to_csv(ENR_DIR / "fixations.csv", index=False)

    print(f"\nFATTO! Cartella generata: {OUTPUT_DIR.resolve()}")
    print("Usa 'Raw_Data' come Data Folder e 'Enrichment_EdgeCase' come Enrichment Folder.")

if __name__ == "__main__":
    main()