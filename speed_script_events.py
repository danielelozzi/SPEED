import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import traceback

# --- Constants ---
SAMPLING_FREQ = 200  # Hz
NS_TO_S = 1e9

def euclidean_distance(x1, y1, x2, y2):
    """Calculates euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def load_all_data(data_dir: Path, un_enriched_mode: bool):
    """Loads all necessary CSV files from the data directory."""
    files_to_load = {
        'events': 'events.csv', 'fixations_not_enr': 'fixations.csv', 'gaze_not_enr': 'gaze.csv',
        'pupil': '3d_eye_states.csv', 'blinks': 'blinks.csv', 'saccades': 'saccades.csv'
    }
    if not un_enriched_mode:
        files_to_load.update({'gaze': 'gaze_enriched.csv', 'fixations_enr': 'fixations_enriched.csv'})

    dataframes = {}
    for name, filename in files_to_load.items():
        try:
            dataframes[name] = pd.read_csv(data_dir / filename)
        except FileNotFoundError:
            if name in ['gaze', 'fixations_enr', 'fixations_not_enr', 'gaze_not_enr']:
                print(f"Info: Optional or base file {filename} not found, proceeding without it.")
                dataframes[name] = pd.DataFrame()
            else:
                raise FileNotFoundError(f"Required data file not found: {filename}")
    return dataframes

def get_timestamp_col(df):
    """Gets the correct timestamp column from a dataframe."""
    for col in ['start timestamp [ns]', 'timestamp [ns]']:
        if col in df.columns:
            return col
    return None

def filter_data_by_segment(all_data, start_ts, end_ts, rec_id):
    """Filters all dataframes for a specific time segment [start_ts, end_ts)."""
    segment_data = {}
    for name, df in all_data.items():
        if df.empty or name == 'events':
            segment_data[name] = df
            continue
        ts_col = get_timestamp_col(df)
        if ts_col:
            mask = (df[ts_col] >= start_ts) & (df[ts_col] < end_ts)
            if 'recording id' in df.columns:
                mask &= (df['recording id'] == rec_id)
            segment_data[name] = df[mask].copy().reset_index(drop=True)
        else:
            segment_data[name] = pd.DataFrame(columns=df.columns)
    return segment_data

def process_gaze_movements(gaze_df, un_enriched_mode: bool):
    """Identifies and processes gaze movements from ENRICHED gaze data."""
    if un_enriched_mode or gaze_df.empty or 'fixation id' not in gaze_df.columns or 'gaze detected on surface' not in gaze_df.columns:
        return pd.DataFrame()
    
    gaze_df['fixation id'].fillna(-1, inplace=True)
    gaze_on_surface = gaze_df[gaze_df['gaze detected on surface'] == True].copy()
    if gaze_on_surface.empty:
        return pd.DataFrame()
    
    is_movement = gaze_on_surface['fixation id'] == -1
    gaze_on_surface.loc[is_movement, 'movement_id'] = (is_movement != is_movement.shift()).cumsum()[is_movement]
    
    movements = []
    for _, group in gaze_on_surface.dropna(subset=['movement_id']).groupby('movement_id'):
        if len(group) < 2:
            continue
        start_row, end_row = group.iloc[0], group.iloc[-1]
        x, y = group['gaze position on surface x [normalized]'], group['gaze position on surface y [normalized]']
        movements.append({
            'duration_ns': end_row['timestamp [ns]'] - start_row['timestamp [ns]'],
            'total_displacement': euclidean_distance(x.shift(), y.shift(), x, y).sum(),
            'effective_displacement': euclidean_distance(x.iloc[0], y.iloc[0], x.iloc[-1], y.iloc[-1])
        })
    return pd.DataFrame(movements)

def calculate_summary_features(data, movements_df, subj_name, event_name, un_enriched_mode: bool, video_width: int, video_height: int):
    """Calculates a dictionary of summary features, including normalization from pixels."""
    pupil, blinks, saccades = data.get('pupil', pd.DataFrame()), data.get('blinks', pd.DataFrame()), data.get('saccades', pd.DataFrame())
    gaze_enr, gaze_not_enr = data.get('gaze', pd.DataFrame()), data.get('gaze_not_enr', pd.DataFrame())
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())

    results = {
        'participant': subj_name, 'event': event_name, 'n_fixation': np.nan, 'fixation_avg_duration_ms': np.nan,
        'fixation_std_duration_ms': np.nan, 'fixation_avg_x': np.nan, 'fixation_std_x': np.nan,
        'fixation_avg_y': np.nan, 'fixation_std_y': np.nan, 'n_blink': np.nan, 'blink_avg_duration_ms': np.nan,
        'blink_std_duration_ms': np.nan, 'pupil_start_mm': np.nan, 'pupil_end_mm': np.nan, 'pupil_avg_mm': np.nan,
        'pupil_std_mm': np.nan, 'n_movements': np.nan, 'sum_time_movement_s': np.nan, 'avg_time_movement_s': np.nan,
        'std_time_movement_s': np.nan, 'total_disp_sum': np.nan, 'total_disp_avg': np.nan, 'total_disp_std': np.nan,
        'effective_disp_sum': np.nan, 'effective_disp_avg': np.nan, 'effective_disp_std': np.nan,
        'n_gaze_per_fixation_avg': np.nan
    }

    # --- Fixation Features ---
    fixations_to_analyze = fixations_not_enr if not fixations_not_enr.empty else pd.DataFrame()
    if not un_enriched_mode and not fixations_enr.empty and 'fixation detected on surface' in fixations_enr.columns:
        enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        if not enriched_on_surface.empty:
            fixations_to_analyze = enriched_on_surface
    
    if not fixations_to_analyze.empty:
        results.update({'n_fixation': fixations_to_analyze['fixation id'].nunique(), 'fixation_avg_duration_ms': fixations_to_analyze['duration [ms]'].mean(), 'fixation_std_duration_ms': fixations_to_analyze['duration [ms]'].std()})
        
        x_coords, y_coords = pd.Series(dtype='float64'), pd.Series(dtype='float64')
        if 'fixation x [normalized]' in fixations_to_analyze.columns:
            print("DEBUG: Using pre-normalized coordinates from enriched file.")
            x_coords, y_coords = fixations_to_analyze['fixation x [normalized]'], fixations_to_analyze['fixation y [normalized]']
        elif 'fixation x [px]' in fixations_to_analyze.columns:
            if video_width and video_height and video_width > 0 and video_height > 0:
                print(f"DEBUG: Normalizing pixel coordinates using video dimensions {video_width}x{video_height}.")
                x_coords = fixations_to_analyze['fixation x [px]'] / video_width
                y_coords = fixations_to_analyze['fixation y [px]'] / video_height
            else:
                print("WARNING: Fixation coordinates are in pixels, but video dimensions are unavailable. Cannot normalize.")
        
        if not x_coords.empty:
            results.update({
                'fixation_avg_x': x_coords.mean(), 'fixation_std_x': x_coords.std(),
                'fixation_avg_y': y_coords.mean(), 'fixation_std_y': y_coords.std()
            })

    # --- Other Features ---
    gaze_for_fix_count = pd.DataFrame()
    if not un_enriched_mode and not gaze_enr.empty and 'fixation id' in gaze_enr.columns:
        gaze_for_fix_count = gaze_enr
    elif not gaze_not_enr.empty and 'fixation id' in gaze_not_enr.columns:
        gaze_for_fix_count = gaze_not_enr
    if not gaze_for_fix_count.empty:
        results['n_gaze_per_fixation_avg'] = gaze_for_fix_count.groupby('fixation id').size().mean()

    if not blinks.empty:
        results.update({'n_blink': len(blinks), 'blink_avg_duration_ms': blinks['duration [ms]'].mean(), 'blink_std_duration_ms': blinks['duration [ms]'].std()})

    if not pupil.empty and 'pupil diameter left [mm]' in pupil.columns and not pupil['pupil diameter left [mm]'].dropna().empty:
        pupil_diam = pupil['pupil diameter left [mm]'].dropna()
        results.update({'pupil_start_mm': pupil_diam.iloc[0], 'pupil_end_mm': pupil_diam.iloc[-1], 'pupil_avg_mm': pupil_diam.mean(), 'pupil_std_mm': pupil_diam.std()})

    if not un_enriched_mode and not movements_df.empty:
        results.update({
            'n_movements': len(movements_df), 'sum_time_movement_s': movements_df['duration_ns'].sum() / NS_TO_S,
            'avg_time_movement_s': movements_df['duration_ns'].mean() / NS_TO_S, 'std_time_movement_s': movements_df['duration_ns'].std() / NS_TO_S,
            'total_disp_sum': movements_df['total_displacement'].sum(), 'total_disp_avg': movements_df['total_displacement'].mean(),
            'total_disp_std': movements_df['total_displacement'].std(), 'effective_disp_sum': movements_df['effective_displacement'].sum(),
            'effective_disp_avg': movements_df['effective_displacement'].mean(), 'effective_disp_std': movements_df['effective_displacement'].std()
        })
    elif not saccades.empty:
        print("DEBUG: Calculating movement features from saccades.csv")
        sacc_duration_s = saccades['duration [ms]'] / 1000
        amplitude = saccades['amplitude [deg]'] if 'amplitude [deg]' in saccades.columns else pd.Series(dtype='float64')
        results.update({
            'n_movements': len(saccades), 'sum_time_movement_s': sacc_duration_s.sum(), 'avg_time_movement_s': sacc_duration_s.mean(),
            'std_time_movement_s': sacc_duration_s.std(), 'total_disp_sum': amplitude.sum(), 'total_disp_avg': amplitude.mean(),
            'total_disp_std': amplitude.std(), 'effective_disp_sum': amplitude.sum(), 'effective_disp_avg': amplitude.mean(),
            'effective_disp_std': amplitude.std()
        })
            
    return results

def process_segment(event_row, start_ts, end_ts, all_data, subj_name, output_dir, un_enriched_mode, video_width, video_height):
    """Main processing pipeline for a single event segment."""
    event_name = event_row.get('name', f"segment_{event_row.name}")
    print(f"--- Processing segment for event: '{event_name}' ---")
    rec_id = event_row['recording id']
    
    segment_data = filter_data_by_segment(all_data, start_ts, end_ts, rec_id)
    if all(df.empty for name, df in segment_data.items() if name != 'events'):
        print(f"  -> Skipping segment '{event_name}' due to no data in the interval.")
        return None
    
    movements_df = process_gaze_movements(segment_data.get('gaze', pd.DataFrame()), un_enriched_mode)
    results = calculate_summary_features(segment_data, movements_df, subj_name, event_name, un_enriched_mode, video_width, video_height)
    # generate_plots(segment_data, movements_df, subj_name, event_name, output_dir, un_enriched_mode)
    return results

def get_video_dimensions(video_path: Path):
    """Gets the width and height of a video file."""
    if not video_path.exists():
        print(f"WARNING: Video file not found at {video_path}. Cannot get dimensions for normalization.")
        return None, None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"WARNING: Could not open video file {video_path}.")
        return None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def run_analysis(subj_name='subj_01', data_dir_str='./eyetracking_file', output_dir_str='./results', un_enriched_mode=False, generate_video=True):
    """Main function to run the complete analysis pipeline using event-based segmentation."""
    pd.options.mode.chained_assignment = None
    data_dir, output_dir = Path(data_dir_str), Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video dimensions for potential normalization
    video_width, video_height = get_video_dimensions(data_dir / 'external.mp4')
    
    try:
        all_data = load_all_data(data_dir, un_enriched_mode)
    except FileNotFoundError as e:
        print(f"Analysis stopped. {e}")
        return
        
    events_df = all_data.get('events')
    if events_df is None or events_df.empty:
        print("Error: events.csv not loaded or is empty. Cannot proceed.")
        return
        
    all_results = []
    if len(events_df) > 1:
        print(f"\nFound {len(events_df)} events, processing {len(events_df) - 1} segments.")
        for i in range(len(events_df) - 1):
            event_row, start_ts, end_ts = events_df.iloc[i], events_df.iloc[i]['timestamp [ns]'], events_df.iloc[i+1]['timestamp [ns]']
            try:
                event_results = process_segment(event_row, start_ts, end_ts, all_data, subj_name, output_dir, un_enriched_mode, video_width, video_height)
                if event_results:
                    all_results.append(event_results)
            except Exception as e:
                print(f"Could not process segment for event '{event_row.get('name', i)}'. Error: {e}")
                traceback.print_exc()
    else:
        print("Warning: Less than two events found. Cannot process segments.")
        
    if all_results:
        results_df = pd.DataFrame(all_results)
        column_order = [
            'participant', 'event', 'n_fixation', 'fixation_avg_duration_ms', 'fixation_std_duration_ms',
            'fixation_avg_x', 'fixation_std_x', 'fixation_avg_y', 'fixation_std_y', 'n_gaze_per_fixation_avg',
            'n_blink', 'blink_avg_duration_ms', 'blink_std_duration_ms', 'pupil_start_mm', 'pupil_end_mm',
            'pupil_avg_mm', 'pupil_std_mm', 'n_movements', 'sum_time_movement_s', 'avg_time_movement_s',
            'std_time_movement_s', 'total_disp_sum', 'total_disp_avg', 'total_disp_std', 'effective_disp_sum',
            'effective_disp_avg', 'effective_disp_std'
        ]
        final_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[final_columns]
        results_filename = output_dir / f'summary_results_{subj_name}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"\nAggregated results saved to {results_filename}")
    else:
        print("\nNo analysis results were generated.")
        
    if generate_video:
        # Placeholder for video creation logic if needed
        # create_analysis_video(data_dir, output_dir)
        pass