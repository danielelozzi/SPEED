# speed_script_events.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from pathlib import Path
import traceback
import pickle
from scipy.signal import welch, spectrogram
from scipy.stats import gaussian_kde
import gc
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
#------------------------------------------------

# --- Constants ---
SAMPLING_FREQ = 200  # Hz
NS_TO_S = 1e9
PLOT_DPI = 25

# ==============================================================================
# HELPER AND DATA LOADING FUNCTIONS
# ==============================================================================

def euclidean_distance(x1, y1, x2, y2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def load_all_data(data_dir: Path, un_enriched_mode: bool):
    """Loads all necessary CSV files from the data directory."""
    files_to_load = {
        'events': 'events.csv', 'fixations_not_enr': 'fixations.csv', 'gaze_not_enr': 'gaze.csv',
        'pupil': '3d_eye_states.csv', 'blinks': 'blinks.csv', 'saccades': 'saccades.csv'
    }
    if not un_enriched_mode:
        files_to_load.update({'gaze_enr': 'gaze_enriched.csv', 'fixations_enr': 'fixations_enriched.csv'})

    dataframes = {}
    for name, filename in files_to_load.items():
        try:
            dataframes[name] = pd.read_csv(data_dir / filename)
        except FileNotFoundError:
            if name in ['gaze_enr', 'fixations_enr']:
                 dataframes[name] = pd.DataFrame()
            else:
                raise FileNotFoundError(f"Required data file not found: {filename} in {data_dir}")
    return dataframes

def get_timestamp_col(df):
    """Gets the correct timestamp column from a dataframe."""
    for col in ['start timestamp [ns]', 'timestamp [ns]']:
        if col in df.columns:
            return col
    return None

def filter_data_by_segment(all_data, start_ts, end_ts, rec_id, is_last=False):
    """Filters all dataframes for a specific time segment."""
    segment_data = {}
    for name, df in all_data.items():
        if df.empty or name == 'events':
            segment_data[name] = df
            continue
        ts_col = get_timestamp_col(df)
        if ts_col:
            if is_last:
                mask = (df[ts_col] >= start_ts) & (df[ts_col] <= end_ts)
            else:
                mask = (df[ts_col] >= start_ts) & (df[ts_col] < end_ts)

            if 'recording id' in df.columns and rec_id is not None:
                mask &= (df['recording id'] == rec_id)
            segment_data[name] = df[mask].copy().reset_index(drop=True)
        else:
            segment_data[name] = pd.DataFrame(columns=df.columns)
    return segment_data

def get_video_dimensions(video_path: Path):
    """Gets width and height from a video file."""
    if not video_path.exists():
        logging.warning(f"Video file not found at {video_path}. Cannot get dimensions.")
        return None, None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Could not open video file {video_path}.")
        return None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

# ==============================================================================
# CORE DATA ANALYSIS FUNCTIONS
# ==============================================================================

def calculate_summary_features(data, subj_name, event_name, un_enriched_mode: bool, video_width: int, video_height: int):
    """Calculates a dictionary of summary features for a segment."""
    pupil, blinks, saccades = data.get('pupil', pd.DataFrame()), data.get('blinks', pd.DataFrame()), data.get('saccades', pd.DataFrame())
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())
    gaze_not_enr = data.get('gaze_not_enr', pd.DataFrame())

    results = {'participant': subj_name, 'event': event_name}

    fixations_to_analyze = fixations_not_enr
    is_enriched_fixation = False
    if not un_enriched_mode and not fixations_enr.empty and 'fixation detected on surface' in fixations_enr.columns:
        fixations_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        if not fixations_on_surface.empty:
            fixations_to_analyze = fixations_on_surface
            is_enriched_fixation = True

    if not fixations_to_analyze.empty and 'duration [ms]' in fixations_to_analyze.columns:
        results.update({
            'n_fixation': fixations_to_analyze['fixation id'].nunique(),
            'fixation_avg_duration_ms': fixations_to_analyze['duration [ms]'].mean(),
            'fixation_std_duration_ms': fixations_to_analyze['duration [ms]'].std()
        })
        x_coords, y_coords = pd.Series(dtype='float64'), pd.Series(dtype='float64')
        if is_enriched_fixation and 'fixation x [normalized]' in fixations_to_analyze.columns:
            x_coords, y_coords = fixations_to_analyze['fixation x [normalized]'], fixations_to_analyze['fixation y [normalized]']
        elif 'fixation x [px]' in fixations_to_analyze.columns and video_width and video_height:
            x_coords = fixations_to_analyze['fixation x [px]'] / video_width
            y_coords = fixations_to_analyze['fixation y [px]'] / video_height
        if not x_coords.empty:
            results.update({'fixation_avg_x': x_coords.mean(), 'fixation_std_x': x_coords.std(), 'fixation_avg_y': y_coords.mean(), 'fixation_std_y': y_coords.std()})

    if not blinks.empty and 'duration [ms]' in blinks.columns:
        results.update({'n_blink': len(blinks), 'blink_avg_duration_ms': blinks['duration [ms]'].mean()})
    if not saccades.empty and 'duration [ms]' in saccades.columns:
        results.update({'n_saccade': len(saccades), 'saccade_avg_duration_ms': saccades['duration [ms]'].mean()})
    if not pupil.empty and 'pupil diameter left [mm]' in pupil.columns:
        pupil_diam = pupil['pupil diameter left [mm]'].dropna()
        if not pupil_diam.empty:
            results.update({'pupil_avg_mm': pupil_diam.mean(), 'pupil_std_mm': pupil_diam.std()})
    if not gaze_not_enr.empty and 'gaze_speed_px_per_s' in gaze_not_enr.columns:
        results['fragmentation_avg_px_per_s'] = gaze_not_enr['gaze_speed_px_per_s'].mean()
        results['fragmentation_std_px_per_s'] = gaze_not_enr['gaze_speed_px_per_s'].std()
    return results

def run_analysis(subj_name: str, data_dir_str: str, output_dir_str: str, un_enriched_mode: bool, selected_events: list):
    """
    Performs data analysis, calculates stats, and saves processed data for later use.
    """
    pd.options.mode.chained_assignment = None
    data_dir, output_dir = Path(data_dir_str), Path(output_dir_str)
    processed_data_dir = output_dir / 'processed_data'
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    video_width, video_height = get_video_dimensions(data_dir / 'external.mp4')
    
    all_data = load_all_data(data_dir, un_enriched_mode)
    events_df = all_data.get('events')
    if events_df is None or events_df.empty:
        raise ValueError("events.csv not found or is empty. Cannot proceed with analysis.")
    
    if 'name' in events_df.columns:
        events_df['name'] = events_df['name'].astype(str).str.replace(r'[\\/]', '_', regex=True)
        
    if selected_events:
        initial_count = len(events_df)
        events_df = events_df[events_df['name'].isin(selected_events)].copy()
        logging.info(f"Filtered events based on selection. Kept {len(events_df)} out of {initial_count} events.")
        
    gaze_not_enr = all_data.get('gaze_not_enr')
    if gaze_not_enr is not None and not gaze_not_enr.empty:
        gaze_not_enr.sort_values('timestamp [ns]', inplace=True)
        gaze_not_enr['gaze_speed_px_per_s'] = euclidean_distance(gaze_not_enr['gaze x [px]'].shift(), gaze_not_enr['gaze y [px]'].shift(), gaze_not_enr['gaze x [px]'], gaze_not_enr['gaze y [px]']) / (gaze_not_enr['timestamp [ns]'].diff() / NS_TO_S)
        all_data['gaze_not_enr'] = gaze_not_enr
        
    all_results = []
    events_df.sort_values('timestamp [ns]', inplace=True)
    logging.info(f"Found {len(events_df)} selected events. Processing segments between them.")
    
    for i in range(len(events_df)):
        event_row = events_df.iloc[i]
        start_ts = event_row['timestamp [ns]']
        next_events_in_df = events_df[events_df['timestamp [ns]'] > start_ts]
        
        if not next_events_in_df.empty:
            end_ts = next_events_in_df.iloc[0]['timestamp [ns]']
        else:
            max_ts = start_ts
            for df_name in ['pupil', 'gaze_not_enr', 'gaze_enr']:
                df = all_data.get(df_name)
                if df is not None and not df.empty:
                    ts_col = get_timestamp_col(df)
                    if ts_col: max_ts = max(max_ts, df[ts_col].max())
            end_ts = max_ts
            
        event_name = event_row.get('name', f"segment_{i}")
        rec_id = event_row.get('recording id')
        logging.info(f"--- Analyzing segment for event: '{event_name}' ({i+1}/{len(events_df)}) ---")
        
        is_last_event = (i == len(events_df) - 1)
        segment_data = filter_data_by_segment(all_data, start_ts, end_ts, rec_id, is_last=is_last_event)

        if all(df.empty for name, df in segment_data.items() if name != 'events'):
            logging.warning(f"  -> Skipping segment '{event_name}' due to no data in the time range.")
            continue
            
        event_results = calculate_summary_features(segment_data, subj_name, event_name, un_enriched_mode, video_width, video_height)
        all_results.append(event_results)
        safe_event_name = "".join(c for c in event_name if c.isalnum() or c in ('_', '-')).rstrip()
        segment_output_path = processed_data_dir / f"segment_{i}_{safe_event_name}.pkl"
        with open(segment_output_path, 'wb') as f:
            pickle.dump(segment_data, f)
        logging.info(f"  -> Saved processed data to {segment_output_path}")
        
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / f'summary_results_{subj_name}.csv', index=False)
        logging.info("\nAggregated summary results saved.")
    else:
        logging.warning("\nNo results were generated from the analysis.")

# ==============================================================================
# ON-DEMAND PLOT GENERATION FUNCTIONS
# ==============================================================================

def _plot_histogram(data_series, title, xlabel, output_path):
    try:
        if data_series.dropna().empty: return
        plt.figure(figsize=(10, 6), dpi=PLOT_DPI)
        plt.hist(data_series.dropna(), bins=25, color='royalblue', edgecolor='black', alpha=0.7)
        plt.title(title, fontsize=15); plt.xlabel(xlabel, fontsize=12); plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
        plt.savefig(output_path)
    except Exception as e:
        logging.error(f"Failed to generate plot '{title}': {e}", exc_info=True)
    finally:
        plt.close('all'); gc.collect()

def _plot_path(df, x_col, y_col, title, output_path, is_normalized, color, w=None, h=None):
    try:
        if df.empty or x_col not in df.columns or y_col not in df.columns or df[x_col].isnull().all(): return
        plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
        plt.plot(df[x_col], df[y_col], marker='o', linestyle='-', color=color, markersize=4, alpha=0.6)
        plt.title(title, fontsize=15)
        if is_normalized:
            plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.xlim(0, 1); plt.ylim(1, 0)
        else:
            plt.xlabel('Pixel X'); plt.ylabel('Pixel Y'); plt.gca().invert_yaxis()
            if w and h: plt.xlim(0, w); plt.ylim(h, 0)
        plt.grid(True); plt.tight_layout(); plt.savefig(output_path)
    except Exception as e:
        logging.error(f"Failed to generate plot '{title}': {e}", exc_info=True)
    finally:
        plt.close('all'); gc.collect()

def _plot_heatmap(df, x_col, y_col, title, output_path, is_normalized, w=None, h=None):
    try:
        if df.empty or x_col not in df.columns or y_col not in df.columns: return
        x = df[x_col].dropna(); y = df[y_col].dropna()
        if len(x) < 3: return
        k = gaussian_kde(np.vstack([x, y]))
        x_range = (x.min(), x.max()) if not is_normalized else (0,1)
        y_range = (y.min(), y.max()) if not is_normalized else (0,1)
        xi, yi = np.mgrid[x_range[0]:x_range[1]:100j, y_range[0]:y_range[1]:100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.figure(figsize=(10, 8), dpi=PLOT_DPI)
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds')
        plt.title(title, fontsize=15)
        if is_normalized:
            plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.xlim(0, 1); plt.ylim(1, 0)
        else:
            plt.xlabel('Pixel X'); plt.ylabel('Pixel Y'); plt.gca().invert_yaxis()
            if w and h: plt.xlim(0, w); plt.ylim(h, 0)
        plt.colorbar(label="Density"); plt.grid(True); plt.tight_layout(); plt.savefig(output_path)
    except Exception as e:
        logging.error(f"Failed during heatmap generation for '{title}': {e}", exc_info=True)
    finally:
        plt.close('all'); gc.collect()

def _plot_spectral_analysis(pupil_series, title_prefix, output_dir):
    ts = pupil_series.dropna()
    if len(ts) <= SAMPLING_FREQ: return
    try:
        freqs, Pxx = welch(ts.to_numpy(), fs=SAMPLING_FREQ, nperseg=min(len(ts), 256))
        plt.figure(figsize=(10, 5), dpi=PLOT_DPI); plt.semilogy(freqs, Pxx)
        plt.title(f'Periodogram - {title_prefix}'); plt.xlabel('Frequency [Hz]'); plt.ylabel('PSD')
        plt.savefig(output_dir / f'periodogram_{title_prefix}.pdf')
    except Exception as e: logging.warning(f"Failed Periodogram for '{title_prefix}': {e}")
    finally: plt.close('all')
    try:
        f, t, Sxx = spectrogram(ts.to_numpy(), fs=SAMPLING_FREQ, nperseg=min(len(ts), 256))
        plt.figure(figsize=(10, 5), dpi=PLOT_DPI)
        if Sxx.size > 0:
            plt.pcolormesh(t, f, 10 * np.log10(np.maximum(Sxx, 1e-10)), shading='gouraud')
            plt.colorbar(label='Power [dB]')
        plt.title(f'Spectrogram - {title_prefix}'); plt.ylabel('Frequency [Hz]'); plt.xlabel('Time [s]')
        plt.savefig(output_dir / f'spectrogram_{title_prefix}.pdf')
    except Exception as e: logging.warning(f"Failed Spectrogram for '{title_prefix}': {e}")
    finally:
        plt.close('all'); gc.collect()

def _plot_generic_timeseries(x_data, y_data_dict, title, xlabel, ylabel, output_path, span_df=None):
    """
    MODIFIED: Draws green spans for 'on surface' and red spans for 'off surface'.
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=PLOT_DPI)
        
        # --- LOGICA PER SFONDO VERDE/ROSSO ---
        if span_df is not None and 'gaze detected on surface' in span_df.columns and 'time_sec' in span_df.columns:
            time_diffs = span_df['time_sec'].diff().fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            on_surface_spans = span_df[span_df['gaze detected on surface'] == True]
            for index, row in on_surface_spans.iterrows():
                start_time = row['time_sec']
                duration = time_diffs.loc[index]
                ax.axvspan(start_time, start_time + duration, facecolor='green', alpha=0.2, edgecolor='none')

            off_surface_spans = span_df[span_df['gaze detected on surface'] == False]
            for index, row in off_surface_spans.iterrows():
                start_time = row['time_sec']
                duration = time_diffs.loc[index]
                ax.axvspan(start_time, start_time + duration, facecolor='red', alpha=0.15, edgecolor='none')
        # --- FINE LOGICA ---

        for label, y_data in y_data_dict.items():
            ax.plot(x_data, y_data, label=label, alpha=0.8)

        ax.set_title(title, fontsize=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        plt.savefig(output_path)
    except Exception as e:
        logging.error(f"Failed to generate generic time series plot '{title}': {e}", exc_info=True)
    finally:
        plt.close('all')
        gc.collect()

def _plot_binary_timeseries(df, start_col, duration_col, total_duration, title, ylabel, output_path):
    try:
        fig, ax = plt.subplots(figsize=(12, 3), dpi=PLOT_DPI)
        for _, row in df.iterrows():
            ax.add_patch(mpatches.Rectangle((row[start_col], 0), row[duration_col], 1, facecolor='red', alpha=0.5))
        ax.set_xlim(0, total_duration); ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=15); ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
        ax.get_yaxis().set_ticks([]); plt.tight_layout(); plt.savefig(output_path)
    except Exception as e:
        logging.error(f"Failed to generate binary plot '{title}': {e}", exc_info=True)
    finally:
        plt.close('all'); gc.collect()

def _generate_plots_for_segment_process(pkl_file: Path, plots_dir: Path, plot_selections: dict, un_enriched_mode: bool, video_width: int, video_height: int):
    """
    This function is executed in a separate process to generate all plots for a single segment.
    MODIFIED: Uses a unique segment identifier for plot filenames to prevent overwriting.
    """
    try:
        plt.switch_backend('Agg')
        
        # --- MODIFICA PER NOMI FILE UNICI ---
        unique_filename_id = pkl_file.stem 
        event_name_for_title = "_".join(unique_filename_id.split('_')[2:])
        logging.info(f"--- [Process] Generating plots for event segment: '{event_name_for_title}' ({unique_filename_id}) ---")
        # --- FINE MODIFICA ---

        with open(pkl_file, 'rb') as f:
            segment_data = pickle.load(f)

        fixations_enr = segment_data.get('fixations_enr', pd.DataFrame())
        fixations_not_enr = segment_data.get('fixations_not_enr', pd.DataFrame())
        gaze_enr = segment_data.get('gaze_enr', pd.DataFrame())
        gaze_not_enr = segment_data.get('gaze_not_enr', pd.DataFrame())
        blinks = segment_data.get('blinks', pd.DataFrame())
        saccades = segment_data.get('saccades', pd.DataFrame())
        pupil = segment_data.get('pupil', pd.DataFrame())

        t_min, total_duration = 0, 1
        base_df_for_time = pupil if not pupil.empty else (gaze_not_enr if not gaze_not_enr.empty else pd.DataFrame())
        if not base_df_for_time.empty:
            ts_col = get_timestamp_col(base_df_for_time)
            if ts_col:
                t_min = base_df_for_time[ts_col].min()
                for df_name, df in segment_data.items():
                    if not df.empty and df_name != 'events':
                        ts_col_inner = get_timestamp_col(df)
                        if ts_col_inner:
                            df['time_sec'] = (df[ts_col_inner] - t_min) / NS_TO_S
                if 'time_sec' in base_df_for_time.columns:
                    total_duration = base_df_for_time['time_sec'].max()

        span_df = None
        if not un_enriched_mode and not pupil.empty and not gaze_enr.empty and 'gaze detected on surface' in gaze_enr.columns:
            if 'time_sec' not in gaze_enr.columns and 'timestamp [ns]' in gaze_enr.columns:
                 gaze_enr['time_sec'] = (gaze_enr['timestamp [ns]'] - t_min) / NS_TO_S
            if 'time_sec' in pupil.columns and 'time_sec' in gaze_enr.columns:
                span_df = pd.merge_asof(pupil.sort_values('time_sec'), gaze_enr[['time_sec', 'gaze detected on surface']].sort_values('time_sec'), on='time_sec', direction='nearest')

        # --- MODIFICA PER NOMI FILE UNICI ---
        if plot_selections.get("histograms"):
            if 'duration [ms]' in fixations_not_enr.columns: _plot_histogram(fixations_not_enr['duration [ms]'], f"Fixation Duration (Un-enriched) - {event_name_for_title}", "Duration [ms]", plots_dir / f"hist_fix_unenriched_{unique_filename_id}.pdf")
            if not un_enriched_mode and 'duration [ms]' in fixations_enr.columns: _plot_histogram(fixations_enr['duration [ms]'], f"Fixation Duration (Enriched) - {event_name_for_title}", "Duration [ms]", plots_dir / f"hist_fix_enriched_{unique_filename_id}.pdf")
            if 'duration [ms]' in blinks.columns: _plot_histogram(blinks['duration [ms]'], f"Blink Duration - {event_name_for_title}", "Duration [ms]", plots_dir / f"hist_blinks_{unique_filename_id}.pdf")
            if 'duration [ms]' in saccades.columns: _plot_histogram(saccades['duration [ms]'], f"Saccade Duration - {event_name_for_title}", "Duration [ms]", plots_dir / f"hist_saccades_{unique_filename_id}.pdf")

        if plot_selections.get("path_plots"):
            _plot_path(fixations_not_enr, 'fixation x [px]', 'fixation y [px]', f"Fixation Path (Un-enriched) - {event_name_for_title}", plots_dir / f"path_fix_unenriched_{unique_filename_id}.pdf", False, 'purple', video_width, video_height)
            _plot_path(gaze_not_enr, 'gaze x [px]', 'gaze y [px]', f"Gaze Path (Un-enriched) - {event_name_for_title}", plots_dir / f"path_gaze_unenriched_{unique_filename_id}.pdf", False, 'blue', video_width, video_height)
            if not un_enriched_mode and not fixations_enr.empty: _plot_path(fixations_enr, 'fixation x [normalized]', 'fixation y [normalized]', f"Fixation Path (Enriched) - {event_name_for_title}", plots_dir / f"path_fix_enriched_{unique_filename_id}.pdf", True, 'green')
            if not un_enriched_mode and not gaze_enr.empty: _plot_path(gaze_enr, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', f"Gaze Path (Enriched) - {event_name_for_title}", plots_dir / f"path_gaze_enriched_{unique_filename_id}.pdf", True, 'red')

        if plot_selections.get("heatmaps"):
            _plot_heatmap(fixations_not_enr, 'fixation x [px]', 'fixation y [px]', f"Fixation Heatmap (Un-enriched) - {event_name_for_title}", plots_dir / f"heatmap_fix_unenriched_{unique_filename_id}.pdf", False, video_width, video_height)
            _plot_heatmap(gaze_not_enr, 'gaze x [px]', 'gaze y [px]', f"Gaze Heatmap (Un-enriched) - {event_name_for_title}", plots_dir / f"heatmap_gaze_unenriched_{unique_filename_id}.pdf", False, video_width, video_height)
            if not un_enriched_mode and not fixations_enr.empty: _plot_heatmap(fixations_enr, 'fixation x [normalized]', 'fixation y [normalized]', f"Fixation Heatmap (Enriched) - {event_name_for_title}", plots_dir / f"heatmap_fix_enriched_{unique_filename_id}.pdf", True)
            if not un_enriched_mode and not gaze_enr.empty: _plot_heatmap(gaze_enr, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', f"Gaze Heatmap (Enriched) - {event_name_for_title}", plots_dir / f"heatmap_gaze_enriched_{unique_filename_id}.pdf", True)

        if plot_selections.get("pupillometry") and not pupil.empty and 'time_sec' in pupil.columns:
            y_data = {}
            if 'pupil diameter left [mm]' in pupil.columns: y_data['Left Pupil'] = pupil['pupil diameter left [mm]']
            if 'pupil diameter right [mm]' in pupil.columns: y_data['Right Pupil'] = pupil['pupil diameter right [mm]']
            if y_data: _plot_generic_timeseries(pupil['time_sec'], y_data, f"Pupil Diameter - {event_name_for_title}", "Time (s)", "Diameter [mm]", plots_dir / f"pupillometry_{unique_filename_id}.pdf", span_df)
            if 'pupil diameter left [mm]' in pupil.columns: _plot_spectral_analysis(pupil['pupil diameter left [mm]'], f"total_{unique_filename_id}", plots_dir)
            if span_df is not None and not span_df.empty and 'gaze detected on surface' in span_df.columns:
                if 'pupil diameter left [mm]' in span_df.columns: _plot_spectral_analysis(span_df[span_df['gaze detected on surface'] == True]['pupil diameter left [mm]'], f"onsurface_{unique_filename_id}", plots_dir)

        if plot_selections.get("fragmentation") and not gaze_not_enr.empty and 'gaze_speed_px_per_s' in gaze_not_enr.columns and 'time_sec' in gaze_not_enr.columns:
            _plot_generic_timeseries(gaze_not_enr['time_sec'], {'Fragmentation': gaze_not_enr['gaze_speed_px_per_s']}, f"Gaze Fragmentation (Speed) - {event_name_for_title}", "Time (s)", "Speed (pixels/sec)", plots_dir / f"fragmentation_{unique_filename_id}.pdf")

        if plot_selections.get("advanced_timeseries"):
            if not pupil.empty and 'pupil diameter left [mm]' in pupil.columns and 'pupil diameter right [mm]' in pupil.columns and 'time_sec' in pupil.columns:
                pupil['pupil_mean_mm'] = pupil[['pupil diameter left [mm]', 'pupil diameter right [mm]']].mean(axis=1)
                _plot_generic_timeseries(pupil['time_sec'], {'Mean Pupil Diameter': pupil['pupil_mean_mm']}, f"Mean Pupil Diameter - {event_name_for_title}", "Time (s)", "Diameter [mm]", plots_dir / f"pupil_diameter_mean_{unique_filename_id}.pdf", span_df)
            if not saccades.empty and 'time_sec' in saccades.columns:
                vel_y_data, vel_unit = {}, None
                if 'mean velocity [px/s]' in saccades.columns: vel_y_data['Mean Velocity'] = saccades['mean velocity [px/s]']; vel_y_data['Peak Velocity'] = saccades['peak velocity [px/s]']; vel_unit = "px/s"
                if vel_y_data: _plot_generic_timeseries(saccades['time_sec'], vel_y_data, f"Saccade Velocity - {event_name_for_title}", "Time (s)", f"Velocity [{vel_unit}]", plots_dir / f"saccade_velocities_{unique_filename_id}.pdf")
                amp_y_data, amp_unit = {}, None
                if 'amplitude [px]' in saccades.columns: amp_y_data['Amplitude'] = saccades['amplitude [px]']; amp_unit = "px"
                if amp_y_data: _plot_generic_timeseries(saccades['time_sec'], amp_y_data, f"Saccade Amplitude - {event_name_for_title}", "Time (s)", f"Amplitude [{amp_unit}]", plots_dir / f"saccade_amplitude_{unique_filename_id}.pdf")
            if not blinks.empty and 'start timestamp [ns]' in blinks.columns and 'duration [ms]' in blinks.columns and 'time_sec' in blinks.columns:
                blinks['duration_sec'] = blinks['duration [ms]'] / 1000
                _plot_binary_timeseries(blinks, 'time_sec', 'duration_sec', total_duration, f"Blink Events - {event_name_for_title}", "Blink", plots_dir / f"blink_time_series_{unique_filename_id}.pdf")
        
        return f"Successfully generated plots for {event_name_for_title}"
    except Exception as e:
        error_message = f"ERROR: Failed to generate plots for event segment '{pkl_file.stem}'. Error: {e}\n{traceback.format_exc()}"
        logging.error(error_message)
        return error_message

def generate_plots_on_demand(output_dir_str: str, subj_name: str, plot_selections: dict, un_enriched_mode: bool):
    """
    Generates plots using a ProcessPoolExecutor for memory efficiency and robustness.
    """
    output_dir = Path(output_dir_str)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir = output_dir / 'processed_data'

    if not processed_data_dir.exists():
        raise FileNotFoundError("Processed data directory not found. Please run the Core Analysis first.")

    try:
        with open(output_dir / 'config.json', 'r') as f:
            config = json.load(f)
        un_enriched_mode = config.get("unenriched_mode", False) # Sovrascrive per sicurezza
        data_dir = Path(config['source_folders']['unenriched'])
        video_width, video_height = get_video_dimensions(next(data_dir.glob('*.mp4')))
    except (FileNotFoundError, StopIteration, KeyError):
        logging.warning("Could not determine original video dimensions. Plots might be affected.")
        video_width, video_height = None, None

    pkl_files = sorted(processed_data_dir.glob("*.pkl"))

    if not pkl_files:
        logging.warning("No processed data segment (.pkl) files found. Cannot generate plots.")
        return

    worker_function = partial(
        _generate_plots_for_segment_process,
        plots_dir=plots_dir,
        plot_selections=plot_selections,
        un_enriched_mode=un_enriched_mode,
        video_width=video_width,
        video_height=video_height
    )

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(worker_function, pkl_file): pkl_file for pkl_file in pkl_files}
        for future in as_completed(futures):
            result_message = future.result()
            if "ERROR" in result_message:
                logging.error(f"A process failed: {result_message}")
            else:
                logging.info(result_message)
    
    logging.info("--- Plot generation finished. ---")