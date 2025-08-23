import pandas as pd
from pathlib import Path
import json
import shutil
import logging
import numpy as np
from typing import Optional, Dict, Any

from .analysis_modules import speed_script_events
from .analysis_modules import yolo_analyzer
from .analysis_modules import video_generator

__all__ = ["run_full_analysis"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _generate_enriched_from_static_aoi(unenriched_dir: Path, output_dir: Path, coords: Dict[str, int]):
    """Genera file enriched da un'AOI statica."""
    logging.info(f"Generating enriched data from static AOI: {coords}")
    gaze_df = pd.read_csv(unenriched_dir / 'gaze.csv')
    fixations_df = pd.read_csv(unenriched_dir / 'fixations.csv')
    world_ts = pd.read_csv(unenriched_dir / 'world_timestamps.csv')

    # 1. Crea surface_positions.csv
    x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
    surface_df = pd.DataFrame({
        'timestamp [ns]': world_ts['timestamp [ns]'],
        'surface_name': 'static_aoi',
        'tl x [px]': x1, 'tl y [px]': y1, 'tr x [px]': x2, 'tr y [px]': y1,
        'br x [px]': x2, 'br y [px]': y2, 'bl x [px]': x1, 'bl y [px]': y2,
    })
    surface_df.to_csv(output_dir / 'surface_positions.csv', index=False)

    # 2. Crea gaze_enriched.csv
    w, h = x2 - x1, y2 - y1
    gaze_df['gaze detected on surface'] = (gaze_df['gaze x [px]'] >= x1) & (gaze_df['gaze x [px]'] <= x2) & \
                                          (gaze_df['gaze y [px]'] >= y1) & (gaze_df['gaze y [px]'] <= y2)
    gaze_df['gaze position on surface x [normalized]'] = np.where(gaze_df['gaze detected on surface'], (gaze_df['gaze x [px]'] - x1) / w, np.nan)
    gaze_df['gaze position on surface y [normalized]'] = np.where(gaze_df['gaze detected on surface'], (gaze_df['gaze y [px]'] - y1) / h, np.nan)
    gaze_df[['timestamp [ns]', 'gaze detected on surface', 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]']].to_csv(output_dir / 'gaze.csv', index=False)

    # 3. Crea fixations_enriched.csv
    fixations_df['fixation detected on surface'] = (fixations_df['fixation x [px]'] >= x1) & (fixations_df['fixation x [px]'] <= x2) & \
                                                   (fixations_df['fixation y [px]'] >= y1) & (fixations_df['fixation y [px]'] <= y2)
    fixations_df['fixation x [normalized]'] = np.where(fixations_df['fixation detected on surface'], (fixations_df['fixation x [px]'] - x1) / w, np.nan)
    fixations_df['fixation y [normalized]'] = np.where(fixations_df['fixation detected on surface'], (fixations_df['fixation y [px]'] - y1) / h, np.nan)
    fixations_df.to_csv(output_dir / 'fixations.csv', index=False)
    logging.info("Enriched data generation from static AOI complete.")

def _generate_enriched_from_dynamic_aoi(unenriched_dir: Path, analysis_output_dir: Path, new_enriched_dir: Path, track_id: int):
    """Genera file enriched tracciando un oggetto con YOLO."""
    logging.info(f"Generating enriched data from dynamic AOI by tracking object ID: {track_id}")
    yolo_cache_path = analysis_output_dir / 'yolo_detections_cache.csv'
    if not yolo_cache_path.exists():
        raise FileNotFoundError("YOLO cache file not found. Run YOLO analysis before creating a dynamic AOI.")

    detections_df = pd.read_csv(yolo_cache_path)
    tracked_object_df = detections_df[detections_df['track_id'] == track_id].set_index('frame_idx')

    if tracked_object_df.empty:
        raise ValueError(f"Track ID {track_id} not found in YOLO detections.")

    world_ts = pd.read_csv(unenriched_dir / 'world_timestamps.csv')
    world_ts['frame_idx'] = world_ts.index

    # 1. Crea surface_positions.csv
    surface_df = world_ts.join(tracked_object_df, on='frame_idx').fillna(method='ffill')
    surface_df.rename(columns={'x1': 'tl x [px]', 'y1': 'tl y [px]', 'x2': 'br x [px]', 'y2': 'br y [px]'}, inplace=True)
    surface_df['tr x [px]'] = surface_df['br x [px]']
    surface_df['tr y [px]'] = surface_df['tl y [px]']
    surface_df['bl x [px]'] = surface_df['tl x [px]']
    surface_df['bl y [px]'] = surface_df['br y [px]']
    surface_df['surface_name'] = f'tracked_object_{track_id}'
    surface_df.to_csv(new_enriched_dir / 'surface_positions.csv', index=False)

    # 2. & 3. Genera gaze e fixations enriched (riutilizzando la logica statica per ogni frame)
    gaze_df = pd.read_csv(unenriched_dir / 'gaze.csv')
    fixations_df = pd.read_csv(unenriched_dir / 'fixations.csv')
    
    merged_gaze = pd.merge_asof(gaze_df.sort_values('timestamp [ns]'), surface_df.sort_values('timestamp [ns]'), on='timestamp [ns]')
    merged_fixations = pd.merge_asof(fixations_df.sort_values('start timestamp [ns]'), surface_df.rename(columns={'timestamp [ns]': 'start timestamp [ns]'}).sort_values('start timestamp [ns]'), on='start timestamp [ns]')

    for df, prefix in [(merged_gaze, 'gaze'), (merged_fixations, 'fixation')]:
        x, y = f'{prefix} x [px]', f'{prefix} y [px]'
        x1, y1, x2, y2 = 'tl x [px]', 'tl y [px]', 'br x [px]', 'br y [px]'
        w, h = df[x2] - df[x1], df[y2] - df[y1]
        
        df[f'{prefix} detected on surface'] = (df[x] >= df[x1]) & (df[x] <= df[x2]) & (df[y] >= df[y1]) & (df[y] <= df[y2])
        df[f'{prefix} x [normalized]'] = np.where(df[f'{prefix} detected on surface'], (df[x] - df[x1]) / w, np.nan)
        df[f'{prefix} y [normalized]'] = np.where(df[f'{prefix} detected on surface'], (df[y] - df[y1]) / h, np.nan)
        
        if prefix == 'gaze':
            df[['timestamp [ns]', f'{prefix} detected on surface', f'{prefix} x [normalized]', f'{prefix} y [normalized]']].to_csv(new_enriched_dir / 'gaze.csv', index=False)
        else:
            df.to_csv(new_enriched_dir / 'fixations.csv', index=False)

    logging.info("Enriched data generation from dynamic AOI complete.")

def _generate_enriched_from_manual_aoi(unenriched_dir: Path, new_enriched_dir: Path, keyframes: Dict[int, Any]):
    """Genera file enriched da keyframe manuali tramite interpolazione."""
    logging.info(f"Generating enriched data from {len(keyframes)} manual keyframes.")
    
    world_ts = pd.read_csv(unenriched_dir / 'world_timestamps.csv')
    total_frames = len(world_ts)
    
    # 1. Interpola per creare surface_positions.csv
    kf_frames = np.array(list(keyframes.keys()))
    kf_coords = np.array(list(keyframes.values())) # Shape: (n_keyframes, 4)
    
    all_frames = np.arange(total_frames)
    interp_coords = np.zeros((total_frames, 4))
    
    for i in range(4): # Interpola x1, y1, x2, y2
        interp_coords[:, i] = np.interp(all_frames, kf_frames, kf_coords[:, i])

    surface_df = pd.DataFrame(interp_coords, columns=['tl x [px]', 'tl y [px]', 'br x [px]', 'br y [px]'])
    surface_df['timestamp [ns]'] = world_ts['timestamp [ns]']
    surface_df['tr x [px]'] = surface_df['br x [px]']
    surface_df['tr y [px]'] = surface_df['tl y [px]']
    surface_df['bl x [px]'] = surface_df['tl x [px]']
    surface_df['bl y [px]'] = surface_df['br y [px]']
    surface_df['surface_name'] = 'manual_aoi'
    surface_df.to_csv(new_enriched_dir / 'surface_positions.csv', index=False)

    # 2. & 3. Genera il resto come per l'AOI dinamica
    _generate_enriched_from_dynamic_aoi(unenriched_dir, new_enriched_dir, -1) # -1 track_id fittizio, la funzione riutilizza i file appena creati
    logging.info("Enriched data generation from manual keyframes complete.")


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
    generate_video: bool = True, video_options: Optional[Dict] = None,
    aoi_coordinates: Optional[Dict[str, int]] = None,
    aoi_track_id: Optional[int] = None,
    aoi_keyframes: Optional[Dict[int, Any]] = None
) -> Path:
    raw_dir = Path(raw_data_path)
    unenriched_dir = Path(unenriched_data_path)
    output_dir = Path(output_path)
    enriched_dir = Path(enriched_data_path) if enriched_data_path else None
    output_dir.mkdir(parents=True, exist_ok=True)
    
    un_enriched_mode = True
    
    # --- NUOVA LOGICA PER GESTIRE L'AOI ---
    if enriched_data_path:
        un_enriched_mode = False
        enriched_dir = Path(enriched_data_path)
    elif aoi_coordinates or aoi_track_id or aoi_keyframes:
        un_enriched_mode = False
        enriched_dir = output_dir / 'enriched_from_AOI'
        enriched_dir.mkdir(exist_ok=True)
        
        if aoi_coordinates:
            _generate_enriched_from_static_aoi(unenriched_dir, enriched_dir, aoi_coordinates)
        elif aoi_track_id:
            if not run_yolo:
                logging.warning("Dynamic AOI requires YOLO. Forcing YOLO analysis to run.")
                run_yolo = True
            # L'analisi YOLO deve avvenire prima della generazione dei file
        elif aoi_keyframes:
            _generate_enriched_from_manual_aoi(unenriched_dir, enriched_dir, aoi_keyframes)
    # --- FINE NUOVA LOGICA ---

    if events_df is None:
        logging.info("No events DataFrame provided, loading 'events.csv' from un-enriched folder.")
        events_file = unenriched_dir / 'events.csv'
        events_df = pd.read_csv(events_file) if events_file.exists() else pd.DataFrame()

    working_dir = _prepare_working_directory(output_dir, raw_dir, unenriched_dir, enriched_dir, events_df)
    selected_event_names = events_df['name'].tolist() if not events_df.empty else []

    if run_yolo:
        logging.info("--- STARTING YOLO ANALYSIS ---")
        yolo_analyzer.run_yolo_analysis(
            data_dir=working_dir, output_dir=output_dir, subj_name=subject_name, model_path=yolo_model_path
        )
        logging.info("--- YOLO ANALYSIS COMPLETE ---")
    
    # Se l'AOI è dinamica, ora possiamo generare i file enriched
    if aoi_track_id:
        _generate_enriched_from_dynamic_aoi(unenriched_dir, output_dir, enriched_dir, aoi_track_id)
        # Ricopia i file appena creati nella cartella di lavoro
        shutil.copy(enriched_dir / 'surface_positions.csv', working_dir / 'surface_positions.csv')
        shutil.copy(enriched_dir / 'gaze.csv', working_dir / 'gaze_enriched.csv')
        shutil.copy(enriched_dir / 'fixations.csv', working_dir / 'fixations_enriched.csv')


    logging.info(f"--- STARTING CORE ANALYSIS FOR {subject_name} ---")
    speed_script_events.run_analysis(
        subj_name=subject_name, data_dir_str=str(working_dir), output_dir_str=str(output_dir),
        un_enriched_mode=un_enriched_mode, selected_events=selected_event_names
    )
    logging.info("--- CORE ANALYSIS COMPLETE ---")

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
