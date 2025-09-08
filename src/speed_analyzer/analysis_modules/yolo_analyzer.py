# yolo_analyzer.py
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import torch
from typing import Optional, List

try:
    from ultralytics import YOLO
except ImportError:
    logging.error("Ultralytics (YOLO) not installed. Cannot run object detection.")
    YOLO = None

# --- FUNZIONI PER LO SWITCHING INDEX (ORA PIÙ GENERICHE) ---

def _calculate_aoi_sequence(mapped_aoi_series: pd.Series) -> list:
    """
    Calcola la sequenza di AOI visitate (V).
    L'input è una serie di nomi/ID di AOI mappati per ogni punto di sguardo.
    """
    v_sequence = []
    last_aoi = None
    for curr_aoi in mapped_aoi_series.dropna():
        if curr_aoi != last_aoi:
            v_sequence.append(curr_aoi)
            last_aoi = curr_aoi
    return v_sequence

def _calculate_switching_index(v_sequence_len: int, total_gaze_points: int) -> float:
    """
    Calcola il Normalized Switching Index (SI).
    """
    if total_gaze_points <= 1:
        return 0.0
    
    k = v_sequence_len
    l_in = total_gaze_points
    
    si = max(0, k - 1) / (l_in - 1)
    return si

def calculate_switching_index_from_gaze(enriched_gaze_df: pd.DataFrame, output_dir: Path, subj_name: str):
    """
    Funzione principale per calcolare lo SI da un dataframe di sguardi già mappato su AOI.
    MODIFIED: Always creates the output file, even if no gaze is mapped to AOIs.
    """
    logging.info("Calculating AOI sequence (V_G) and Switching Index (SI_G)...")
    
    # Initialize default values
    v_gaze_sequence = []
    k_gaze = 0
    l_in_gaze = len(enriched_gaze_df.dropna(subset=['gaze x [px]']))
    si_gaze = 0.0

    if 'aoi_name' in enriched_gaze_df.columns and not enriched_gaze_df['aoi_name'].dropna().empty:
        # 1. Calcola la sequenza V (V_Gaze)
        v_gaze_sequence = _calculate_aoi_sequence(enriched_gaze_df['aoi_name'])
        
        # 2. Calcola lo Switching Index (SI_Gaze)
        k_gaze = len(v_gaze_sequence)
        si_gaze = _calculate_switching_index(k_gaze, l_in_gaze)
        logging.info(f"Switching Index analysis complete. SI_G = {si_gaze:.4f}")
    else:
        logging.warning("No gaze points were mapped to any manually defined AOI. Switching Index is 0.")

    # 3. Salva i risultati, SEMPRE
    si_results = {
        'participant': subj_name,
        'total_gaze_points_analyzed (L_in)': l_in_gaze,
        'aoi_sequence_length (K)': k_gaze,
        'gaze_switching_index (SI_G)': si_gaze,
        'aoi_sequence (V_G)': v_gaze_sequence
    }
    pd.DataFrame([si_results]).to_csv(output_dir / 'switching_index_results.csv', index=False)
    
    logging.info(f"Results for manually defined AOIs saved to: {output_dir / 'switching_index_results.csv'}")


# --- FUNZIONI DI ANALISI YOLO ---

def _get_yolo_device():
    """Determines the optimal device for YOLO inference (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        logging.info("CUDA GPU detected. Using 'cuda' for YOLO.")
        return 'cuda'
    elif torch.backends.mps.is_available():
        logging.info("Apple MPS detected. Using 'mps' for YOLO.")
        return 'mps'
    else:
        logging.info("No compatible GPU detected. Using 'cpu' for YOLO.")
        return 'cpu'

def _load_and_sync_data(data_dir: Path):
    """Loads and synchronizes fixation, pupil, and world timestamps."""
    try:
        world_ts = pd.read_csv(data_dir / 'world_timestamps.csv')
        world_ts['frame'] = world_ts.index
        world_ts.sort_values('timestamp [ns]', inplace=True)
        fixations = pd.read_csv(data_dir / 'fixations.csv').sort_values('start timestamp [ns]')
        pupil = pd.read_csv(data_dir / '3d_eye_states.csv').sort_values('timestamp [ns]')
        gaze = pd.read_csv(data_dir / 'gaze.csv').sort_values('timestamp [ns]')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing required file for YOLO analysis: {e}")

    merged_fix = pd.merge_asof(world_ts, fixations, left_on='timestamp [ns]', right_on='start timestamp [ns]', direction='backward', suffixes=('', '_fix'))
    merged_fix['duration_s'] = merged_fix['duration [ms]'] / 1000.0
    duration_ns = (merged_fix['duration_s'] * 1e9).round()
    merged_fix['end_ts_ns'] = merged_fix['start timestamp [ns]'] + duration_ns.astype('Int64')
    synced_data_fixations = merged_fix[merged_fix['timestamp [ns]'] <= merged_fix['end_ts_ns']].copy()
    synced_data_fixations = pd.merge_asof(synced_data_fixations, pupil[['timestamp [ns]', 'pupil diameter left [mm]']], left_on='timestamp [ns]', right_on='timestamp [ns]', direction='nearest', suffixes=('', '_pupil'))
    
    synced_data_gaze = pd.merge_asof(world_ts, gaze[['timestamp [ns]', 'gaze x [px]', 'gaze y [px]']], on='timestamp [ns]', direction='nearest')

    return synced_data_fixations, synced_data_gaze


def _is_inside(px, py, x1, y1, x2, y2):
    """Checks if a point (px, py) is inside a bounding box (x1, y1, x2, y2)."""
    return x1 <= px <= x2 and y1 <= py <= y2

def run_yolo_analysis(
    data_dir: Path, 
    output_dir: Path, 
    subj_name: str,
    yolo_model_name: str = 'yolov8n.pt',
    custom_classes: Optional[List[str]] = None
):
    """
    Runs YOLO object detection, correlates with fixations, and saves statistics.
    MODIFIED: Always calculates Switching Index based on detected objects as AOIs.
    """
    if YOLO is None:
        logging.warning("Skipping YOLO analysis because Ultralytics is not installed.")
        return

    video_path = next(data_dir.glob('*.mp4'), None)
    if not video_path:
        logging.warning(f"Skipping YOLO analysis: no .mp4 file found in {data_dir}.")
        return

    yolo_device = _get_yolo_device()
    try:
        model = YOLO(yolo_model_name)
        logging.info(f"Loaded YOLO model: {yolo_model_name}")
        if custom_classes:
            logging.info(f"Setting custom classes for zero-shot detection: {custom_classes}")
            model.set_classes(custom_classes)

    except Exception as e:
        logging.error(f"Error loading YOLO model ({yolo_model_name}): {e}. Skipping YOLO analysis.")
        return

    try:
        synced_et_data_fix, synced_et_data_gaze = _load_and_sync_data(data_dir)
    except Exception as e:
        logging.error(f"Error loading/syncing eye-tracking data for YOLO: {e}. Skipping YOLO analysis.")
        return

    yolo_cache_path = output_dir / 'yolo_detections_cache.csv'
    if yolo_cache_path.exists():
        logging.info(f"YOLO cache found. Loading detections from: {yolo_cache_path}")
        detections_df = pd.read_csv(yolo_cache_path)
    else:
        logging.info("YOLO cache not found. Starting video tracking...")
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        detections = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        effective_device = yolo_device
        pbar = tqdm(total=total_frames, desc=f"YOLO Tracking on {effective_device.upper()}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            try:
                results = model.track(frame, persist=True, verbose=False, device=effective_device)
            except NotImplementedError as e:
                if "torchvision::nms" in str(e) and effective_device != 'cpu':
                    logging.warning(f"GPU operation failed: {e}. Falling back to CPU.")
                    effective_device = 'cpu'
                    pbar.set_description(f"YOLO Tracking on {effective_device.upper()} (Fallback)")
                    results = model.track(frame, persist=True, verbose=False, device=effective_device)
                else:
                    raise e

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    detections.append({'frame_idx': frame_idx, 'track_id': track_id, 'class_id': class_id, 'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]})
            
            frame_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()
        
        detections_df = pd.DataFrame(detections)
        logging.info(f"Saving YOLO detections to cache at: {yolo_cache_path}")
        detections_df.to_csv(yolo_cache_path, index=False)

    # --- INIZIA LA LOGICA DI ANALISI ---
    # Crea sempre i file di output, anche se vuoti.
    stats_instance_df = pd.DataFrame(columns=['instance', 'n_fixations', 'normalized_fixation_count', 'avg_pupil_diameter_mm', 'total_frames_detected'])
    stats_class_df = pd.DataFrame(columns=['class', 'n_fixations', 'normalized_fixation_count', 'avg_pupil_diameter_mm', 'total_frames_detected'])
    id_map_df = pd.DataFrame(columns=['track_id', 'class_id', 'class_name', 'instance_name'])

    if not detections_df.empty:
        logging.info("Correlating detections with fixations...")
        merged_df_fix = pd.merge(detections_df, synced_et_data_fix, left_on='frame_idx', right_on='frame', how='inner')
        fixation_hits = [row for _, row in merged_df_fix.iterrows() if pd.notna(row['fixation x [px]']) and _is_inside(row['fixation x [px]'], row['fixation y [px]'], row['x1'], row['y1'], row['x2'], row['y2'])]
        
        if fixation_hits:
            hits_df = pd.DataFrame(fixation_hits)
            logging.info("Calculating statistics for fixations...")
            class_map = {int(k): v for k, v in model.names.items()}
            hits_df['class_name'] = hits_df['class_id'].map(class_map)
            
            stats_instance = hits_df.groupby('instance_name').agg(
                n_fixations=('fixation id', 'nunique'),
                avg_pupil_diameter_mm=('pupil diameter left [mm]', 'mean')
            ).reset_index()
            
            total_detections_instance = detections_df.groupby('instance_name').size().reset_index(name='total_frames_detected')
            stats_instance = pd.merge(stats_instance, total_detections_instance, on='instance_name')
            stats_instance['normalized_fixation_count'] = stats_instance['n_fixations'] / stats_instance['total_frames_detected']
            
            stats_class = hits_df.groupby('class_name').agg(
                n_fixations=('fixation id', 'nunique'),
                avg_pupil_diameter_mm=('pupil diameter left [mm]', 'mean')
            ).reset_index()

            total_detections_class = detections_df.groupby('class_name').size().reset_index(name='total_frames_detected')
            stats_class = pd.merge(stats_class, total_detections_class, on='class_name')
            stats_class['normalized_fixation_count'] = stats_class['n_fixations'] / stats_class['total_frames_detected']
            
            id_map_df = hits_df[['track_id', 'class_id', 'class_name', 'instance_name']].drop_duplicates()
            logging.info("Fixation-based statistics calculated.")
        else:
            logging.warning("No fixations were found inside any detected object bounding boxes.")
    else:
        logging.warning("No objects were detected by YOLO.")

    stats_instance_df.to_csv(output_dir / 'stats_per_instance.csv', index=False)
    stats_class_df.to_csv(output_dir / 'stats_per_class.csv', index=False)
    id_map_df.to_csv(output_dir / 'class_id_map.csv', index=False)

    # --- CALCOLO DELLO SWITCHING INDEX (SEMPRE ESEGUITO CON YOLO) ---
    logging.info("Correlating detections with gaze data for Switching Index calculation...")
    merged_df_gaze = pd.merge(detections_df, synced_et_data_gaze, left_on='frame_idx', right_on='frame', how='inner')
    
    # Mappa ogni punto di sguardo a un track_id
    mapped_gaze_to_tid = []
    if not merged_df_gaze.empty:
        merged_df_gaze['mapped_track_id'] = merged_df_gaze.apply(
            lambda row: row['track_id'] if pd.notna(row['gaze x [px]']) and _is_inside(row['gaze x [px]'], row['gaze y [px]'], row['x1'], row['y1'], row['x2'], row['y2']) else None,
            axis=1
        )
        final_gaze_mapping = merged_df_gaze.groupby('timestamp [ns]')['mapped_track_id'].first() # Prendi il primo oggetto se ci sono sovrapposizioni
    else:
        final_gaze_mapping = pd.Series([])

    # Calcola e salva SI
    v_gaze_sequence = _calculate_aoi_sequence(final_gaze_mapping)
    k_gaze = len(v_gaze_sequence)
    l_in_gaze = len(synced_et_data_gaze.dropna(subset=['gaze x [px]']))
    si_gaze = _calculate_switching_index(k_gaze, l_in_gaze)
    
    si_results = {
        'participant': subj_name,
        'total_gaze_points_analyzed (L_in)': l_in_gaze,
        'aoi_sequence_length (K)': k_gaze,
        'gaze_switching_index (SI_G)': si_gaze,
        'aoi_sequence (V_G)': v_gaze_sequence
    }
    pd.DataFrame([si_results]).to_csv(output_dir / 'switching_index_yolo_results.csv', index=False)
    logging.info(f"YOLO-based Switching Index calculated and saved. SI_G = {si_gaze:.4f}")

    logging.info("YOLO analysis part completed.")