import pandas as pd
from pathlib import Path
import json
import shutil
import logging
import numpy as np
from typing import Optional, Dict, Any, List

from .analysis_modules import speed_script_events
from .analysis_modules import yolo_analyzer
from .analysis_modules import video_generator
from .analysis_modules.bids_converter import convert_to_bids, load_from_bids
from .analysis_modules.device_converters import convert_device_data

__all__ = ["run_full_analysis", "convert_to_bids", "load_from_bids", "convert_device_data"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NUOVA SEZIONE: Logica per la generazione di dati Enriched da AOI Multiple ---

def _generate_enriched_from_multiple_aois(
    unenriched_dir: Path,
    new_enriched_dir: Path,
    aois: List[Dict[str, Any]],
    analysis_output_dir: Path
):
    """
    Genera file enriched mappando i dati di sguardo su una lista di AOI definite.
    Crea una colonna 'aoi_name' che identifica in quale AOI si trova lo sguardo.
    """
    logging.info(f"Generating enriched data from {len(aois)} user-defined AOIs.")

    # Carica i dati base
    gaze_df = pd.read_csv(unenriched_dir / 'gaze.csv')
    fixations_df = pd.read_csv(unenriched_dir / 'fixations.csv')
    world_ts = pd.read_csv(unenriched_dir / 'world_timestamps.csv')
    world_ts['frame_idx'] = world_ts.index

    yolo_detections = pd.DataFrame()
    yolo_cache_path = analysis_output_dir / 'yolo_detections_cache.csv'
    if yolo_cache_path.exists():
        yolo_detections = pd.read_csv(yolo_cache_path)

    # Pre-calcola le posizioni di tutte le AOI per ogni frame
    aoi_positions_per_frame = {}
    for i, aoi in enumerate(aois):
        aoi_name = aoi['name']
        aoi_type = aoi['type']
        aoi_data = aoi['data']

        if aoi_type == 'static':
            coords = aoi_data
            aoi_positions_per_frame[aoi_name] = pd.DataFrame({
                'frame_idx': world_ts['frame_idx'],
                'x1': coords['x1'], 'y1': coords['y1'], 'x2': coords['x2'], 'y2': coords['y2']
            }).set_index('frame_idx')

        elif aoi_type == 'dynamic_auto':
            track_id = aoi_data
            if yolo_detections.empty:
                raise FileNotFoundError("YOLO cache not found, cannot process dynamic AOI.")

            tracked_obj = yolo_detections[yolo_detections['track_id'] == track_id].set_index('frame_idx')
            if tracked_obj.empty:
                raise ValueError(f"Track ID {track_id} for AOI '{aoi_name}' not found in detections.")

            # Unisci con world_timestamps per avere una riga per ogni frame, riempiendo i vuoti
            full_track = world_ts.join(tracked_obj, on='frame_idx').fillna(method='ffill')
            aoi_positions_per_frame[aoi_name] = full_track[['x1', 'y1', 'x2', 'y2']]

        elif aoi_type == 'dynamic_manual':
            keyframes = aoi_data
            kf_frames = np.array(list(keyframes.keys()))
            kf_coords = np.array(list(keyframes.values()))
            all_frames = np.arange(len(world_ts))
            interp_coords = np.zeros((len(world_ts), 4))
            for j in range(4):
                interp_coords[:, j] = np.interp(all_frames, kf_frames, kf_coords[:, j])

            aoi_positions_per_frame[aoi_name] = pd.DataFrame(interp_coords, columns=['x1', 'y1', 'x2', 'y2'])

    # Ora mappa ogni punto di sguardo/fissazione all'AOI corrispondente
    def map_points_to_aois(points_df, timestamp_col):
        points_df = pd.merge_asof(
            points_df.sort_values(timestamp_col),
            world_ts.sort_values('timestamp [ns]'),
            left_on=timestamp_col,
            right_on='timestamp [ns]'
        )

        def find_aoi(row):
            for aoi_name, positions in aoi_positions_per_frame.items():
                try:
                    aoi_pos = positions.loc[row['frame_idx']]
                    x, y = row[f'{points_df.prefix} x [px]'], row[f'{points_df.prefix} y [px]']
                    if pd.notna(x) and aoi_pos['x1'] <= x <= aoi_pos['x2'] and aoi_pos['y1'] <= y <= aoi_pos['y2']:
                        return aoi_name
                except (KeyError, IndexError):
                    continue
            return None

        points_df.prefix = 'gaze' if 'gaze' in points_df.columns[1] else 'fixation'
        points_df['aoi_name'] = points_df.apply(find_aoi, axis=1)
        return points_df

    enriched_gaze = map_points_to_aois(gaze_df.copy(), 'timestamp [ns]')
    enriched_fixations = map_points_to_aois(fixations_df.copy(), 'start timestamp [ns]')

    # Salva i file enriched
    enriched_gaze.to_csv(new_enriched_dir / 'gaze_enriched.csv', index=False)
    enriched_fixations.to_csv(new_enriched_dir / 'fixations_enriched.csv', index=False)
    logging.info("Enriched data generation from multiple AOIs complete.")

    return enriched_gaze # Restituisce per il calcolo SI

def _prepare_working_directory(
    output_dir: Path,
    raw_dir: Path,
    unenriched_dir: Path,
    enriched_dirs: List[Path],
    events_df: pd.DataFrame,
    concatenated_video_path: Optional[Path] = None,
    viv_events_path: Optional[Path] = None
):
    working_dir = output_dir / 'eyetracking_files'
    working_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Preparing working directory at: {working_dir}")

    # --- MODIFICA: Usa il video concatenato se fornito, altrimenti cerca quello di default ---
    if concatenated_video_path and concatenated_video_path.exists():
        external_video_path = concatenated_video_path
        logging.info(f"Using provided concatenated video: {external_video_path}")
    else:
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
    if enriched_dirs:
        # --- MODIFICA: Logica per unire file da più cartelle ---
        all_enriched_gaze = []
        all_enriched_fixations = []

        for enrich_dir in enriched_dirs:
            aoi_name_from_folder = enrich_dir.name

            gaze_path = enrich_dir / 'gaze.csv'
            if gaze_path.exists():
                gaze_df = pd.read_csv(gaze_path)
                gaze_df['aoi_name'] = aoi_name_from_folder
                all_enriched_gaze.append(gaze_df)

            fix_path = enrich_dir / 'fixations.csv'
            if fix_path.exists():
                fix_df = pd.read_csv(fix_path)
                fix_df['aoi_name'] = aoi_name_from_folder
                all_enriched_fixations.append(fix_df)

        if all_enriched_gaze:
            pd.concat(all_enriched_gaze).to_csv(working_dir / 'gaze_enriched.csv', index=False)
        if all_enriched_fixations:
            pd.concat(all_enriched_fixations).to_csv(working_dir / 'fixations_enriched.csv', index=False)

    for dest, source in file_map.items():
        if source and source.exists():
            shutil.copy(source, working_dir / dest)
        else:
            logging.warning(f"Optional file not found and not copied: {source}")

    # --- MODIFICA: Usa il file di eventi ViV se esiste, altrimenti quello standard ---
    if viv_events_path and viv_events_path.exists():
        shutil.copy(viv_events_path, working_dir / 'events.csv')
    elif not events_df.empty:
        events_df.to_csv(working_dir / 'events.csv', index=False)

    return working_dir

def run_full_analysis(
    raw_data_path: str, unenriched_data_path: str, output_path: str, subject_name: str,
    enriched_data_paths: Optional[List[str]] = None, events_df: Optional[pd.DataFrame] = None, 
    yolo_models: Optional[Dict[str, str]] = None, yolo_custom_classes: Optional[List[str]] = None,
    yolo_detections_df: Optional[pd.DataFrame] = None, generate_plots: bool = True, 
    plot_selections: Optional[Dict[str, bool]] = None,
    generate_video: bool = True, video_options: Optional[Dict[str, Any]] = None,
    concatenated_video_path: Optional[str] = None,
    defined_aois: Optional[List[Dict[str, Any]]] = None
) -> Path:
    raw_dir = Path(raw_data_path)
    unenriched_dir = Path(unenriched_data_path)
    output_dir = Path(output_path)
    enriched_dirs = [Path(p) for p in enriched_data_paths] if enriched_data_paths else []
    output_dir.mkdir(parents=True, exist_ok=True)

    un_enriched_mode = not bool(enriched_dirs) and not bool(defined_aois)
    enriched_gaze_df = pd.DataFrame()
    
    # --- NUOVO: Percorso per il file di eventi ViV ---
    viv_events_path = output_dir / "SPEED_workspace" / "events-video-in-video.csv"
    viv_events_path = viv_events_path if viv_events_path.exists() else None

    if events_df is None:
        logging.info("No events DataFrame provided, loading 'events.csv' from un-enriched folder.")
        events_file = unenriched_dir / 'events.csv'
        events_df = pd.read_csv(events_file) if events_file.exists() else pd.DataFrame()

    if (yolo_models or (defined_aois and any(aoi['type'] == 'dynamic_auto' for aoi in defined_aois))) and (yolo_detections_df is None or yolo_detections_df.empty):
        logging.info("--- STARTING YOLO ANALYSIS ---")
        yolo_analyzer.run_yolo_analysis(
            data_dir=unenriched_dir,
            output_dir=output_dir,
            subj_name=subject_name,
            yolo_detections_df=yolo_detections_df,
            yolo_models=yolo_models,
            custom_classes=yolo_custom_classes # Nuovo
        )
        logging.info("--- YOLO ANALYSIS COMPLETE ---")

    if defined_aois:
        un_enriched_mode = False
        enriched_dir_for_aois = output_dir / 'enriched_from_AOIs'
        enriched_dir_for_aois.mkdir(exist_ok=True)
        enriched_gaze_df = _generate_enriched_from_multiple_aois(unenriched_dir, enriched_dir_for_aois, defined_aois, output_dir)
        # Aggiungi questa nuova cartella alla lista di quelle da processare
        enriched_dirs.append(enriched_dir_for_aois)


    working_dir = _prepare_working_directory(
        output_dir,
        raw_dir,
        unenriched_dir,
        enriched_dirs,
        events_df,
        Path(concatenated_video_path) if concatenated_video_path else None,
        viv_events_path
    )
    selected_event_names = events_df['name'].tolist() if not events_df.empty else []

    # --- NUOVO: Calcolo Switching Index ---
    if defined_aois and len(defined_aois) >= 2 and not enriched_gaze_df.empty:
        logging.info("--- CALCULATING SWITCHING INDEX ---")
        yolo_analyzer.calculate_switching_index_from_gaze(
            enriched_gaze_df, output_dir, subject_name
        )
        logging.info("--- SWITCHING INDEX CALCULATION COMPLETE ---")


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