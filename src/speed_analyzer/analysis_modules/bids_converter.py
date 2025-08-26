# src/speed_analyzer/analysis_modules/bids_converter.py
import pandas as pd
import json
import shutil
import gzip
from pathlib import Path
import logging

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
            "SamplingFrequency": 200, # Assumendo 200Hz, da rendere dinamico se necessario
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
        start_time_ns = events_df['timestamp [ns]'].min()
        
        bids_events_df = pd.DataFrame({
            'onset': (events_df['timestamp [ns]'] - start_time_ns) / 1e9,
            'duration': 0,  # Durata istantanea di default
            'trial_type': events_df['name']
        })
        bids_events_df.to_csv(session_dir / f"{base_name}_events.tsv", sep='\t', index=False)
        logging.info("File _events.tsv creato.")

        # 5. Creazione Sidecar JSON per Eventi (*_events.json)
        events_json = {
            "onset": {"Description": "Onset of the event in seconds."},
            "duration": {"Description": "Duration of the event in seconds."},
            "trial_type": {"Description": "Name of the event."}
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