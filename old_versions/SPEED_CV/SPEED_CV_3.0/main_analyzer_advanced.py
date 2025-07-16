# main_analyzer_advanced.py
from pathlib import Path
import traceback
import json
import shutil

# Importa i moduli dell'applicazione
import speed_script_events as speed_events
import video_generator

def _prepare_eyetracking_files(output_dir: Path, raw_dir: Path, unenriched_dir: Path, enriched_dir: Path, un_enriched_mode: bool):
    """
    Copia e rinomina i file necessari dalle cartelle di input alla cartella di lavoro.
    """
    data_dir = output_dir / 'eyetracking_file'
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparazione dei file nella cartella di lavoro: {data_dir}")

    # Cerca l'unico file .mp4 nella cartella un-enriched
    external_videos = list(unenriched_dir.glob('*.mp4'))
    if not external_videos:
        raise FileNotFoundError(f"File video esterno (.mp4) non trovato nella cartella Un-enriched: {unenriched_dir}")
    if len(external_videos) > 1:
        raise Exception(f"Trovati {len(external_videos)} file video (.mp4) nella cartella Un-enriched. Ne è ammesso solo uno.")
    
    external_video_source_path = external_videos[0]

    # Definisce i file da copiare, la loro origine e il nome di destinazione
    file_map = {
        # File dalla cartella RAW
        'internal.mp4': (raw_dir / 'Neon Sensor Module v1 ps1.mp4', False),
        
        # File dalla cartella Un-enriched
        'external.mp4': (external_video_source_path, True),
        'events.csv': (unenriched_dir / 'events.csv', True),
        'fixations.csv': (unenriched_dir / 'fixations.csv', True),
        'gaze.csv': (unenriched_dir / 'gaze.csv', True),
        'blinks.csv': (unenriched_dir / 'blinks.csv', True),
        'saccades.csv': (unenriched_dir / 'saccades.csv', True),
        '3d_eye_states.csv': (unenriched_dir / '3d_eye_states.csv', True),
        'world_timestamps.csv': (unenriched_dir / 'world_timestamps.csv', True),
        
        # ===== MODIFICATO =====
        # File dalla cartella Enriched (opzionale, ma cercato qui)
        'surface_positions.csv': (enriched_dir / 'surface_positions.csv', False),
    }

    # Aggiunge i file Enriched (gaze/fixations) solo se non siamo in modalità "un-enriched only"
    if not un_enriched_mode:
        if not enriched_dir or not enriched_dir.exists():
            raise FileNotFoundError(f"La cartella dei dati Enriched è richiesta ma non è stata fornita o non esiste: {enriched_dir}")
        file_map.update({
            'fixations_enriched.csv': (enriched_dir / 'fixations.csv', True),
            'gaze_enriched.csv': (enriched_dir / 'gaze.csv', True),
        })

    # Ciclo di copia e rinomina
    for dest_name, (source_path, required) in file_map.items():
        if source_path.exists():
            dest_path = data_dir / dest_name
            shutil.copy(source_path, dest_path)
            print(f"Copiato: {source_path.name} -> {dest_path.name}")
        elif required:
            raise FileNotFoundError(f"File obbligatorio non trovato: {source_path}")
        else:
            print(f"Attenzione: File opzionale non trovato e non copiato: {source_path}")
    
    return data_dir


def run_core_analysis(
    subj_name: str, 
    output_dir_str: str, 
    raw_dir_str: str, 
    unenriched_dir_str: str, 
    enriched_dir_str: str, 
    un_enriched_mode: bool, 
    run_yolo: bool
):
    """
    Esegue l'analisi completa, gestendo la nuova logica basata su cartelle.
    """
    print(f"--- STARTING CORE ANALYSIS FOR SUBJECT: {subj_name} ---")
    output_dir = Path(output_dir_str)
    
    try:
        working_data_dir = _prepare_eyetracking_files(
            output_dir,
            Path(raw_dir_str),
            Path(unenriched_dir_str),
            Path(enriched_dir_str) if enriched_dir_str else Path(),
            un_enriched_mode
        )

        config = {
            "source_folders": { "raw": raw_dir_str, "unenriched": unenriched_dir_str, "enriched": enriched_dir_str },
            "unenriched_mode": un_enriched_mode, "yolo_mode": run_yolo
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print("\n>>> RUNNING STANDARD EVENT-BASED ANALYSIS (DATA ONLY)...")
        speed_events.run_analysis(
            subj_name=subj_name,
            data_dir_str=str(working_data_dir),
            output_dir_str=str(output_dir),
            un_enriched_mode=un_enriched_mode
        )
        print(">>> Standard data analysis finished.")

        if run_yolo:
            print("\n>>> RUNNING YOLO OBJECT DETECTION (DATA ONLY)...")
            print("NOTE: YOLO analysis is a placeholder in this version.")
        
        print(f"\n--- CORE ANALYSIS FOR {subj_name} COMPLETED ---")
        print(f"Processed data saved in: {output_dir / 'processed_data'}")

    except Exception as e:
        print(f"!!! Error during core analysis: {e}")
        traceback.print_exc()
        raise

def generate_selected_plots(output_dir_str: str, subj_name: str, plot_selections: dict):
    """
    Generates only the plots selected by the user, loading the necessary data.
    """
    print("\n>>> GENERATING SELECTED PLOTS...")
    output_dir = Path(output_dir_str)
    
    config_path = output_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError("config.json not found. Please run Core Analysis first.")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    try:
        speed_events.generate_plots_on_demand(
            output_dir_str=str(output_dir),
            subj_name=subj_name,
            plot_selections=plot_selections,
            un_enriched_mode=config.get("unenriched_mode", False)
        )
        print(">>> Plot generation finished.")
    except Exception as e:
        print(f"!!! Error during plot generation: {e}")
        traceback.print_exc()
        raise

def generate_custom_video(output_dir_str: str, subj_name: str, video_options: dict):
    """
    Calls the video generator with the specified configuration.
    """
    print("\n>>> GENERATING CUSTOM VIDEO...")
    output_dir = Path(output_dir_str)
    
    config_path = output_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError("config.json not found. Please run Core Analysis first.")

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    try:
        video_generator.create_custom_video(
            data_dir=output_dir / 'eyetracking_file',
            output_dir=output_dir,
            subj_name=subj_name,
            options=video_options,
            un_enriched_mode=config.get("unenriched_mode", False)
        )
        print(">>> Video generation finished.")
    except Exception as e:
        print(f"!!! Error during video generation: {e}")
        traceback.print_exc()
        raise