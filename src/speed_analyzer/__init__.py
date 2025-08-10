# src/speed_analyzer/__init__.py
import pandas as pd
from pathlib import Path
import json
import shutil
import logging
from typing import Optional, Dict

# Importa i moduli di analisi dalla sottocartella
from .analysis_modules import speed_script_events
from .analysis_modules import yolo_analyzer
from .analysis_modules import video_generator

# Esporta la funzione principale per renderla accessibile con "from speed_analyzer import run_full_analysis"
__all__ = ["run_full_analysis"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _prepare_working_directory(output_dir: Path, raw_dir: Path, unenriched_dir: Path, enriched_dir: Optional[Path], events_df: pd.DataFrame):
    working_dir = output_dir / 'eyetracking_files'
    working_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Preparing working directory at: {working_dir}")
    try:
        external_video_path = next(unenriched_dir.glob('*.mp4'))
    except StopIteration:
        raise FileNotFoundError(f"No .mp4 file found in {unenriched_dir}")
    file_map = {
        'internal.mp4': raw_dir / 'Neon Sensor Module v1 ps1.mp4',
        'external.mp4': external_video_path,
        'fixations.csv': unenriched_dir / 'fixations.csv',
        'gaze.csv': unenriched_dir / 'gaze.csv',
        'blinks.csv': unenriched_dir / 'blinks.csv',
        'saccades.csv': unenriched_dir / 'saccades.csv',
        '3d_eye_states.csv': unenriched_dir / '3d_eye_states.csv',
        'world_timestamps.csv': unenriched_dir / 'world_timestamps.csv',
    }
    if enriched_dir:
        file_map.update({
            'surface_positions.csv': enriched_dir / 'surface_positions.csv',
            'gaze_enriched.csv': enriched_dir / 'gaze.csv',
            'fixations_enriched.csv': enriched_dir / 'fixations.csv',
        })
    for dest, source in file_map.items():
        if source and source.exists():
            shutil.copy(source, working_dir / dest)
        else:
            logging.warning(f"Optional file not found and not copied: {source}")
    if not events_df.empty:
        events_df.to_csv(working_dir / 'events.csv', index=False)
    return working_dir

def run_full_analysis(
    raw_data_path: str, unenriched_data_path: str, output_path: str, subject_name: str,
    enriched_data_path: Optional[str] = None, events_df: Optional[pd.DataFrame] = None,
    run_yolo: bool = False, yolo_model_path: str = 'yolov8n.pt',
    generate_plots: bool = True, plot_selections: Optional[Dict[str, bool]] = None,
    generate_video: bool = True, video_options: Optional[Dict] = None
) -> Path:
    raw_dir = Path(raw_data_path)
    unenriched_dir = Path(unenriched_data_path)
    output_dir = Path(output_path)
    enriched_dir = Path(enriched_data_path) if enriched_data_path else None
    output_dir.mkdir(parents=True, exist_ok=True)
    un_enriched_mode = enriched_dir is None

    if events_df is None:
        logging.info("No events DataFrame provided, loading 'events.csv' from un-enriched folder.")
        events_file = unenriched_dir / 'events.csv'
        events_df = pd.read_csv(events_file) if events_file.exists() else pd.DataFrame()

    working_dir = _prepare_working_directory(output_dir, raw_dir, unenriched_dir, enriched_dir, events_df)
    selected_event_names = events_df['name'].tolist() if not events_df.empty else []

    logging.info(f"--- STARTING CORE ANALYSIS FOR {subject_name} ---")
    speed_script_events.run_analysis(
        subj_name=subject_name, data_dir_str=str(working_dir), output_dir_str=str(output_dir),
        un_enriched_mode=un_enriched_mode, selected_events=selected_event_names
    )
    logging.info("--- CORE ANALYSIS COMPLETE ---")

    if run_yolo:
        logging.info("--- STARTING YOLO ANALYSIS ---")
        yolo_analyzer.run_yolo_analysis(
            data_dir=working_dir, output_dir=output_dir, subj_name=subject_name, model_path=yolo_model_path
        )
        logging.info("--- YOLO ANALYSIS COMPLETE ---")

    if generate_plots:
        logging.info("--- STARTING PLOT GENERATION ---")
        if plot_selections is None:
            plot_selections = { "path_plots": True, "heatmaps": True, "histograms": True, "pupillometry": True, "advanced_timeseries": True, "fragmentation": True }
        config = {"unenriched_mode": un_enriched_mode, "source_folders": {"unenriched": str(unenriched_dir)}}
        with open(output_dir / 'config.json', 'w') as f: json.dump(config, f)
        speed_script_events.generate_plots_on_demand(
            output_dir_str=str(output_dir), subj_name=subject_name,
            plot_selections=plot_selections, un_enriched_mode=un_enriched_mode
        )
        logging.info("--- PLOT GENERATION COMPLETE ---")

    if generate_video:
        logging.info("--- STARTING VIDEO GENERATION ---")
        if video_options is None:
            video_options = { "output_filename": f"video_output_{subject_name}.mp4", "overlay_gaze": True, "overlay_event_text": True }
        video_generator.create_custom_video(
            data_dir=working_dir, output_dir=output_dir, subj_name=subject_name,
            options=video_options, un_enriched_mode=un_enriched_mode,
            selected_events=selected_event_names
        )
        logging.info("--- VIDEO GENERATION COMPLETE ---")

    logging.info(f"Analysis complete. Results saved in: {output_dir.resolve()}")
    return output_dir