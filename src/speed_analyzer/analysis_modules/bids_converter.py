# src/speed_analyzer/analysis_modules/bids_converter.py
import pandas as pd
import json
import shutil
import gzip
from pathlib import Path
import logging
import glob

def convert_to_bids(unenriched_dir: Path, output_bids_dir: Path, subject_id: str, session_id: str, task_name: str):
    """
    Converte i dati di eye-tracking nel formato BIDS.
    """
    logging.info(f"--- AVVIO CONVERSIONE BIDS per sub-{subject_id}, ses-{session_id}, task-{task_name} ---")

    # Struttura delle cartelle BIDS
    session_dir = output_bids_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "eyetrack"
    session_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Cartella di output BIDS creata: {session_dir}")

    base_name = f"sub-{subject_id}_ses-{session_id}_task-{task_name}"

    # 1. dataset_description.json
    dataset_desc = {
        "Name": "SPEED Eye-Tracking Dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "Authors": ["Dr. Daniele Lozzi, LabSCoC"],
    }
    with open(output_bids_dir / "dataset_description.json", 'w') as f:
        json.dump(dataset_desc, f, indent=4)

    # 2. Conversione Dati Eye-Tracking (*_eyetrack.tsv.gz)
    gaze_file = unenriched_dir / "gaze.csv"
    pupil_file = unenriched_dir / "3d_eye_states.csv"
    if gaze_file.exists() and pupil_file.exists():
        gaze_df = pd.read_csv(gaze_file)
        pupil_df = pd.read_csv(pupil_file)
        
        # Merge per aggiungere i dati pupillometrici
        merged_df = pd.merge_asof(gaze_df.sort_values('timestamp [ns]'), 
                                  pupil_df[['timestamp [ns]', 'pupil diameter left [mm]']].sort_values('timestamp [ns]'),
                                  on='timestamp [ns]', direction='nearest', tolerance=pd.Timedelta('50ms').value)

        start_time_ns = merged_df['timestamp [ns]'].min()
        merged_df['time'] = (merged_df['timestamp [ns]'] - start_time_ns) / 1e9
        
        bids_eyetrack_df = pd.DataFrame({
            'time': merged_df['time'],
            'eye1_x_coordinate': merged_df['gaze x [px]'],
            'eye1_y_coordinate': merged_df['gaze y [px]'],
            'eye1_pupil_size': merged_df['pupil diameter left [mm]']
        })

        eyetrack_tsv_path = session_dir / f"{base_name}_eyetrack.tsv"
        bids_eyetrack_df.to_csv(eyetrack_tsv_path, sep='\t', index=False, na_rep='n/a')
        
        with open(eyetrack_tsv_path, 'rb') as f_in, gzip.open(f"{eyetrack_tsv_path}.gz", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        eyetrack_tsv_path.unlink()
        logging.info("File _eyetrack.tsv.gz creato.")

        # 3. Creazione Sidecar JSON per Eye-Tracking (*_eyetrack.json)
        eyetrack_json = {
            "SamplingFrequency": 200, 
            "StartTime": 0,
            "Columns": ["time", "eye1_x_coordinate", "eye1_y_coordinate", "eye1_pupil_size"],
            "eye1_x_coordinate": {"Units": "pixels"},
            "eye1_y_coordinate": {"Units": "pixels"},
            "eye1_pupil_size": {"Units": "mm"}
        }
        with open(session_dir / f"{base_name}_eyetrack.json", 'w') as f:
            json.dump(eyetrack_json, f, indent=4)
        logging.info("File _eyetrack.json creato.")

    # 4. Conversione Eventi (*_events.tsv)
    events_file = unenriched_dir / "events.csv"
    if events_file.exists():
        events_df = pd.read_csv(events_file)
        # Assumiamo che il tempo 0 di BIDS corrisponda al primo timestamp
        start_time_ns_events = events_df['timestamp [ns]'].min()
        
        bids_events_df = pd.DataFrame({
            'onset': (events_df['timestamp [ns]'] - start_time_ns_events) / 1e9,
            'duration': 0,
            'trial_type': events_df['name']
        })
        bids_events_df.to_csv(session_dir / f"{base_name}_events.tsv", sep='\t', index=False)
        logging.info("File _events.tsv creato.")

        # 5. Creazione Sidecar JSON per Eventi (*_events.json)
        events_json = {
            "onset": {"Description": "Onset of the event in seconds relative to the start of the eyetracking recording."},
            "duration": {"Description": "Duration of the event in seconds (0 for instantaneous)."},
            "trial_type": {"Description": "Type of event."}
        }
        with open(session_dir / f"{base_name}_events.json", 'w') as f:
            json.dump(events_json, f, indent=4)
        logging.info("File _events.json creato.")
    
    # 6. Copia del video di registrazione
    video_file = next(unenriched_dir.glob('*.mp4'), None)
    if video_file:
        shutil.copy(video_file, session_dir / f"{base_name}_recording.mp4")
        logging.info("File _recording.mp4 copiato.")

    logging.info("--- CONVERSIONE BIDS COMPLETATA ---")


def load_from_bids(bids_dir: Path, subject_id: str, session_id: str, task_name: str) -> Path:
    """
    Carica un dataset BIDS e lo converte in una cartella temporanea "un-enriched" per SPEED.
    """
    logging.info(f"--- CARICAMENTO DATI BIDS per sub-{subject_id}, ses-{session_id}, task-{task_name} ---")
    session_dir = bids_dir / f"sub-{subject_id}" / f"ses-{session_id}" / "eyetrack"
    if not session_dir.is_dir():
        raise FileNotFoundError(f"La cartella BIDS specificata non è stata trovata: {session_dir}")

    # Crea una cartella temporanea per i dati convertiti
    temp_unenriched_dir = bids_dir / "derivatives" / "speed_temp_unenriched"
    temp_unenriched_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Cartella temporanea 'un-enriched' creata in: {temp_unenriched_dir}")

    base_name = f"sub-{subject_id}_ses-{session_id}_task-{task_name}"

    # 1. Carica e converti _eyetrack.tsv.gz
    eyetrack_file = session_dir / f"{base_name}_eyetrack.tsv.gz"
    if eyetrack_file.exists():
        df = pd.read_csv(eyetrack_file, sep='\t', na_values='n/a')
        
        # Ricrea i timestamp in nanosecondi (assumendo tempo 0 all'inizio)
        df['timestamp [ns]'] = (df['time'] * 1e9).astype('int64')

        # Dividi in gaze.csv e 3d_eye_states.csv
        gaze_df = df[['timestamp [ns]', 'eye1_x_coordinate', 'eye1_y_coordinate']].rename(columns={
            'eye1_x_coordinate': 'gaze x [px]',
            'eye1_y_coordinate': 'gaze y [px]'
        })
        
        pupil_df = df[['timestamp [ns]', 'eye1_pupil_size']].rename(columns={
            'eye1_pupil_size': 'pupil diameter left [mm]'
        })
        # Aggiungi una colonna fittizia per la pupilla destra se necessario
        pupil_df['pupil diameter right [mm]'] = pupil_df['pupil diameter left [mm]']

        gaze_df.to_csv(temp_unenriched_dir / "gaze.csv", index=False)
        pupil_df.to_csv(temp_unenriched_dir / "3d_eye_states.csv", index=False)
        logging.info("Convertiti gaze.csv e 3d_eye_states.csv")

    # 2. Carica e converti _events.tsv
    events_file = session_dir / f"{base_name}_events.tsv"
    if events_file.exists():
        df_events = pd.read_csv(events_file, sep='\t')
        
        # Ricrea i timestamp in nanosecondi
        start_ts = gaze_df['timestamp [ns]'].min()
        df_events['timestamp [ns]'] = (start_ts + (df_events['onset'] * 1e9)).astype('int64')
        
        events_speed_df = df_events.rename(columns={'trial_type': 'name'})
        events_speed_df['recording id'] = 'bids_import'
        
        events_speed_df[['name', 'timestamp [ns]', 'recording id']].to_csv(temp_unenriched_dir / "events.csv", index=False)
        logging.info("Convertito events.csv")
        
    # 3. Copia il video di registrazione
    video_file = session_dir / f"{base_name}_recording.mp4"
    if video_file.exists():
        shutil.copy(video_file, temp_unenriched_dir / "external.mp4")
        logging.info("Copiato external.mp4")
        
    # 4. Crea file fittizi ma necessari per SPEED
    # Crea un world_timestamps.csv basato sui dati di gaze
    if not gaze_df.empty:
        world_ts_df = pd.DataFrame({'timestamp [ns]': gaze_df['timestamp [ns]']})
        world_ts_df.to_csv(temp_unenriched_dir / 'world_timestamps.csv', index=False)

    # Crea file vuoti per gli altri dati comportamentali per evitare errori
    pd.DataFrame(columns=['fixation id', 'start timestamp [ns]', 'duration [ms]', 'fixation x [px]', 'fixation y [px]']).to_csv(temp_unenriched_dir / 'fixations.csv', index=False)
    pd.DataFrame(columns=['saccade id', 'start timestamp [ns]', 'duration [ms]']).to_csv(temp_unenriched_dir / 'saccades.csv', index=False)
    pd.DataFrame(columns=['blink id', 'start timestamp [ns]', 'duration [ms]']).to_csv(temp_unenriched_dir / 'blinks.csv', index=False)
    logging.info("Creati file comportamentali fittizi (fixations, saccades, blinks).")
    
    logging.info(f"--- CARICAMENTO BIDS COMPLETATO. Dati pronti in: {temp_unenriched_dir} ---")
    return temp_unenriched_dir