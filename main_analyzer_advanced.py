# main_analyzer_advanced.py
from pathlib import Path
import traceback
import json
import shutil

# Import application modules
import speed_script_events as speed_events
import video_generator
import yolo_analyzer 

def _prepare_eyetracking_files(output_dir: Path, raw_dir: Path, unenriched_dir: Path, enriched_dir: Path, un_enriched_mode: bool, custom_event_path: str = None):
    """
    Copies and renames the necessary files from the input folders to the working folder.
    MODIFIED: Uses the custom event file if provided.
    """
    data_dir = output_dir / 'eyetracking_file'
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing files in the working directory: {data_dir}")

    # Search for the single .mp4 file in the un-enriched folder
    external_videos = list(unenriched_dir.glob('*.mp4'))
    if not external_videos:
        raise FileNotFoundError(f"External video file (.mp4) not found in the Un-enriched folder: {unenriched_dir}")
    if len(external_videos) > 1:
        raise Exception(f"Found {len(external_videos)} video files (.mp4) in the Un-enriched folder. Only one is allowed.")
    
    external_video_source_path = external_videos[0]

    # Define the files to copy, their source, and destination name
    file_map = {
        # Files from the RAW folder
        'internal.mp4': (raw_dir / 'Neon Sensor Module v1 ps1.mp4', False),
        
        # Files from the Un-enriched folder
        'external.mp4': (external_video_source_path, True),
        'fixations.csv': (unenriched_dir / 'fixations.csv', True),
        'gaze.csv': (unenriched_dir / 'gaze.csv', True),
        'blinks.csv': (unenriched_dir / 'blinks.csv', True),
        'saccades.csv': (unenriched_dir / 'saccades.csv', True),
        '3d_eye_states.csv': (unenriched_dir / '3d_eye_states.csv', True),
        'world_timestamps.csv': (unenriched_dir / 'world_timestamps.csv', True),
        
        # Files from the Enriched folder (optional, but looked for here)
        'surface_positions.csv': (enriched_dir / 'surface_positions.csv', False),
    }

    # --- MODIFIED LOGIC ---
    # If a custom event path is provided (from the GUI), copy that file.
    # Otherwise, fall back to the one in the un-enriched folder.
    if custom_event_path and Path(custom_event_path).exists():
        print(f"Using user-modified event file from: {custom_event_path}")
        shutil.copy(custom_event_path, data_dir / 'events.csv')
    # If no custom path, get it from the standard location
    else:
        file_map['events.csv'] = (unenriched_dir / 'events.csv', True)
    # --- END MODIFIED LOGIC ---


    # Add Enriched files (gaze/fixations) only if not in "un-enriched only" mode
    if not un_enriched_mode:
        if not enriched_dir or not enriched_dir.exists():
            raise FileNotFoundError(f"The Enriched data folder is required but was not provided or does not exist: {enriched_dir}")
        file_map.update({
            'fixations_enriched.csv': (enriched_dir / 'fixations.csv', True),
            'gaze_enriched.csv': (enriched_dir / 'gaze.csv', True),
        })

    # Copy and rename loop
    for dest_name, (source_path, required) in file_map.items():
        if source_path.exists():
            dest_path = data_dir / dest_name
            # Don't overwrite events.csv if it was already handled
            if dest_name == 'events.csv' and (data_dir / 'events.csv').exists():
                continue
            shutil.copy(source_path, dest_path)
            print(f"Copied: {source_path.name} -> {dest_path.name}")
        elif required:
            raise FileNotFoundError(f"Required file not found: {source_path}")
        else:
            print(f"Warning: Optional file not found and not copied: {source_path}")
    
    return data_dir


def run_core_analysis(
    subj_name: str, 
    output_dir_str: str, 
    raw_dir_str: str, 
    unenriched_dir_str: str, 
    enriched_dir_str: str, 
    un_enriched_mode: bool, 
    run_yolo: bool,
    selected_events: list,
    custom_event_path: str = None # NUOVO PARAMETRO
):
    """
    Executes the complete analysis, including YOLO analysis if requested.
    """
    print(f"--- STARTING CORE ANALYSIS FOR SUBJECT: {subj_name} ---")
    output_dir = Path(output_dir_str)
    
    try:
        working_data_dir = _prepare_eyetracking_files(
            output_dir,
            Path(raw_dir_str),
            Path(unenriched_dir_str),
            Path(enriched_dir_str) if enriched_dir_str else Path(),
            un_enriched_mode,
            custom_event_path # Passa il parametro
        )

        config = {
            "source_folders": { "raw": raw_dir_str, "unenriched": unenriched_dir_str, "enriched": enriched_dir_str },
            "unenriched_mode": un_enriched_mode, "yolo_mode": run_yolo,
            "selected_events": selected_events
        }
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print("\n>>> RUNNING STANDARD EVENT-BASED ANALYSIS (DATA ONLY)...")
        speed_events.run_analysis(
            subj_name=subj_name,
            data_dir_str=str(working_data_dir),
            output_dir_str=str(output_dir),
            un_enriched_mode=un_enriched_mode,
            selected_events=selected_events
        )
        print(">>> Standard data analysis finished.")

        # --- MODIFIED: Actual execution of YOLO analysis ---
        if run_yolo:
            print("\n>>> RUNNING YOLO OBJECT DETECTION AND CORRELATION...")
            try:
                yolo_analyzer.run_yolo_analysis(
                    data_dir=working_data_dir,
                    output_dir=output_dir,
                    subj_name=subj_name
                )
                print(">>> YOLO analysis finished.")
            except KeyError as ke:
                print(f"!!! A KeyError occurred during YOLO analysis: {ke}")
                print("!!! This might be due to an outdated 'yolo_detections_cache.csv' file.")
                print("!!! Please try deleting the cache file in the output directory and run the analysis again.")
                raise
        # ------------------------------------------------------
        
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
    
    selected_events_for_trimming = config.get("selected_events", [])
        
    try:
        video_generator.create_custom_video(
            data_dir=output_dir / 'eyetracking_file',
            output_dir=output_dir,
            subj_name=subj_name,
            options=video_options,
            un_enriched_mode=config.get("unenriched_mode", False),
            selected_events=selected_events_for_trimming
        )
        print(">>> Video generation finished.")
    except Exception as e:
        print(f"!!! Error during video generation: {e}")
        traceback.print_exc()
        raise