import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import math
import cv2
from scipy.signal import welch, spectrogram
from scipy.stats import gaussian_kde
from pathlib import Path

# --- Constants ---
SAMPLING_FREQ = 200  # Hz
NS_TO_S = 1e9


def euclidean_distance(x1, y1, x2, y2):
    """Calculates euclidean distance between two points, works on scalars or series."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def load_all_data(data_dir: Path, un_enriched_mode: bool):
    """Loads all necessary CSV files from the data directory, adjusting for un-enriched mode."""
    files_to_load = {
        'events': 'events.csv',
        'fixations_not_enr': 'fixations.csv', # Un-enriched fixations (now fixations.csv)
        'gaze_not_enr': 'gaze.csv', # Un-enriched gaze (now gaze.csv)
        'pupil': '3d_eye_states.csv',
        'blinks': 'blinks.csv',
        'saccades': 'saccades.csv'
    }
    
    # Conditionally add enriched files if not in un-enriched mode
    if not un_enriched_mode:
        files_to_load['gaze'] = 'gaze_enriched.csv' # Enriched gaze
        files_to_load['fixations_enr'] = 'fixations_enriched.csv' # Enriched fixations

    dataframes = {}
    missing_files_in_mode = []
    for name, filename in files_to_load.items():
        try:
            dataframes[name] = pd.read_csv(data_dir / filename)
        except FileNotFoundError:
            # Handle optional files that might be missing
            if name in ['gaze', 'fixations_enr']:
                print(f"Info: Optional enriched file {filename} not found, proceeding without it.")
                dataframes[name] = pd.DataFrame() # Create empty dataframe to avoid KeyError
            elif name in ['fixations_not_enr']:
                print(f"Info: File {filename} not found, fixation analysis will be skipped unless enriched version is present.")
                dataframes[name] = pd.DataFrame()
            else: # These are always required
                missing_files_in_mode.append(filename)
                
    if missing_files_in_mode:
        print(f"Error: Required data files not found: {', '.join(missing_files_in_mode)}")
        raise FileNotFoundError(f"Required data files not found: {', '.join(missing_files_in_mode)}")
            
    return dataframes


def filter_data_by_event(all_data, event_timestamp, rec_id, un_enriched_mode: bool):
    """Filters all dataframes for a specific event based on timestamp and recording ID, adapting for un-enriched mode."""
    event_data = {}
    
    # Filter enriched gaze data ('gaze_enriched.csv')
    if 'gaze' in all_data and not all_data['gaze'].empty and not un_enriched_mode:
        event_data['gaze'] = all_data['gaze'][
            (all_data['gaze']['timestamp [ns]'] > event_timestamp) &
            (all_data['gaze']['recording id'] == rec_id)
        ].copy().reset_index(drop=True)
    else:
        event_data['gaze'] = pd.DataFrame()

    # Filter un-enriched gaze data ('gaze.csv')
    if 'gaze_not_enr' in all_data and not all_data['gaze_not_enr'].empty:
        event_data['gaze_not_enr'] = all_data['gaze_not_enr'][
            (all_data['gaze_not_enr']['timestamp [ns]'] > event_timestamp)
        ].copy().reset_index(drop=True)
    else:
        event_data['gaze_not_enr'] = pd.DataFrame()

    # Pupil data
    if 'pupil' in all_data and not all_data['pupil'].empty:
        event_data['pupil'] = all_data['pupil'][
            (all_data['pupil']['timestamp [ns]'] > event_timestamp)
        ].copy().reset_index(drop=True)
    else:
        event_data['pupil'] = pd.DataFrame()

    # Filter enriched fixations data ('fixations_enriched.csv')
    if 'fixations_enr' in all_data and not all_data['fixations_enr'].empty and not un_enriched_mode:
        event_data['fixations_enr'] = all_data['fixations_enr'][
            (all_data['fixations_enr']['start timestamp [ns]'] > event_timestamp) &
            (all_data['fixations_enr']['recording id'] == rec_id)
        ].copy().reset_index(drop=True)
    else:
        event_data['fixations_enr'] = pd.DataFrame()

    # Filter un-enriched fixations data ('fixations.csv')
    if 'fixations_not_enr' in all_data and not all_data['fixations_not_enr'].empty:
        event_data['fixations_not_enr'] = all_data['fixations_not_enr'][
            (all_data['fixations_not_enr']['start timestamp [ns]'] > event_timestamp)
        ].copy().reset_index(drop=True)
    else:
        event_data['fixations_not_enr'] = pd.DataFrame()

    # Blinks
    if 'blinks' in all_data and not all_data['blinks'].empty:
        event_data['blinks'] = all_data['blinks'][
            (all_data['blinks']['start timestamp [ns]'] > event_timestamp)
        ].copy().reset_index(drop=True)
    else:
        event_data['blinks'] = pd.DataFrame()

    # Saccades
    if 'saccades' in all_data and not all_data['saccades'].empty:
        event_data['saccades'] = all_data['saccades'][
            (all_data['saccades']['start timestamp [ns]'] > event_timestamp)
        ].copy().reset_index(drop=True)
    else:
        event_data['saccades'] = pd.DataFrame()

    return event_data


def process_gaze_movements(gaze_df, un_enriched_mode: bool):
    """
    Identifies and processes gaze movements (saccades) from gaze data.
    Adapts for un-enriched mode by not calculating movements if 'fixation id' is not available.
    """
    # This function specifically uses enriched gaze data, which contains 'fixation id'
    if un_enriched_mode or gaze_df.empty or 'fixation id' not in gaze_df.columns or 'gaze detected on surface' not in gaze_df.columns:
        return pd.DataFrame()

    gaze_df['fixation id'].fillna(-1, inplace=True)
    gaze_on_surface = gaze_df[gaze_df['gaze detected on surface'] == True].copy()
    
    if gaze_on_surface.empty:
        return pd.DataFrame()

    # Identify movements (periods where fixation id is -1)
    is_movement = gaze_on_surface['fixation id'] == -1
    movement_groups = (is_movement != is_movement.shift()).cumsum()
    gaze_on_surface.loc[is_movement, 'movement_id'] = movement_groups[is_movement]

    movements = []
    movement_data = gaze_on_surface.dropna(subset=['movement_id'])
    
    for _, group in movement_data.groupby('movement_id'):
        if len(group) < 2:
            continue

        start_row = group.iloc[0]
        end_row = group.iloc[-1]

        start_time = start_row['timestamp [ns]']
        end_time = end_row['timestamp [ns]']
        
        if 'gaze position on surface x [normalized]' in group.columns and 'gaze position on surface y [normalized]' in group.columns:
            start_pos = (start_row['gaze position on surface x [normalized]'], start_row['gaze position on surface y [normalized]'])
            end_pos = (end_row['gaze position on surface x [normalized]'], end_row['gaze position on surface y [normalized]'])
            x = group['gaze position on surface x [normalized]']
            y = group['gaze position on surface y [normalized]']
            total_displacement = euclidean_distance(x.shift(), y.shift(), x, y).sum()
            effective_displacement = euclidean_distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
        else:
            start_pos = (np.nan, np.nan)
            end_pos = (np.nan, np.nan)
            total_displacement = np.nan
            effective_displacement = np.nan
            print("Warning: Missing 'gaze position on surface' columns for movement calculation.")

        movements.append({
            'movement_id': start_row['movement_id'],
            'start_time': start_time,
            'end_time': end_time,
            'duration_ns': end_time - start_time,
            'surface': start_row['gaze detected on surface'],
            'total_displacement': total_displacement,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'effective_displacement': effective_displacement
        })

    return pd.DataFrame(movements)


def calculate_summary_features(data, movements_df, subj_name, event_name, un_enriched_mode: bool):
    """Calculates a dictionary of summary features from the processed data."""
    fixations_enr = data.get('fixations_enr', pd.DataFrame()) # from fixations_enriched.csv
    fixations_not_enr = data.get('fixations_not_enr', pd.DataFrame()) # from fixations.csv
    blinks = data.get('blinks', pd.DataFrame())
    pupil = data.get('pupil', pd.DataFrame())
    gaze_enr = data.get('gaze', pd.DataFrame()) # Enriched gaze data (from gaze_enriched.csv)
    gaze_not_enr = data.get('gaze_not_enr', pd.DataFrame()) # Un-enriched gaze data (from gaze.csv)

    results = {'participant': subj_name, 'event': event_name}
    
    # --- Fixation features with precedence logic ---
    fixations_to_analyze = pd.DataFrame()
    # 1. Prefer enriched fixations on surface
    if not un_enriched_mode and not fixations_enr.empty and 'fixation detected on surface' in fixations_enr.columns:
        fixations_to_analyze = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        print(f"Using {len(fixations_to_analyze)} fixations from enriched file (on surface).")
    # 2. Fallback to un-enriched fixations
    elif not fixations_not_enr.empty:
        fixations_to_analyze = fixations_not_enr.copy()
        print(f"Using {len(fixations_to_analyze)} fixations from un-enriched file.")

    if not fixations_to_analyze.empty:
        results.update({
            'n_fixation': fixations_to_analyze['fixation id'].nunique(),
            'fixation_avg_duration_ms': fixations_to_analyze['duration [ms]'].mean(),
            'fixation_std_duration_ms': fixations_to_analyze['duration [ms]'].std(),
            'fixation_avg_x': fixations_to_analyze['fixation x [normalized]'].mean(),
            'fixation_std_x': fixations_to_analyze['fixation x [normalized]'].std(),
            'fixation_avg_y': fixations_to_analyze['fixation y [normalized]'].mean(),
            'fixation_std_y': fixations_to_analyze['fixation y [normalized]'].std(),
        })
    else:
        for col in ['n_fixation', 'fixation_avg_duration_ms', 'fixation_std_duration_ms', 'fixation_avg_x', 'fixation_std_x', 'fixation_avg_y', 'fixation_std_y']:
            results[col] = np.nan

    # Blink features
    if not blinks.empty:
        results.update({
            'n_blink': len(blinks),
            'blink_avg_duration_ms': blinks['duration [ms]'].mean(),
            'blink_std_duration_ms': blinks['duration [ms]'].std(),
        })

    # Pupillometry features
    if not pupil.empty and 'pupil diameter left [mm]' in pupil.columns:
        pupil_diam = pupil['pupil diameter left [mm]']
        results.update({
            'pupil_start_mm': pupil_diam.iloc[0],
            'pupil_end_mm': pupil_diam.iloc[-1],
            'pupil_avg_mm': pupil_diam.mean(),
            'pupil_std_mm': pupil_diam.std(),
        })

    # Movement features (from enriched gaze data)
    if not un_enriched_mode and not movements_df.empty:
        results.update({
            'n_movements': len(movements_df),
            'sum_time_movement_s': movements_df['duration_ns'].sum() / NS_TO_S,
            'avg_time_movement_s': movements_df['duration_ns'].mean() / NS_TO_S,
            'std_time_movement_s': movements_df['duration_ns'].std() / NS_TO_S,
            'total_disp_sum': movements_df['total_displacement'].sum(),
            'total_disp_avg': movements_df['total_displacement'].mean(),
            'total_disp_std': movements_df['total_displacement'].std(),
            'effective_disp_sum': movements_df['effective_displacement'].sum(),
            'effective_disp_avg': movements_df['effective_displacement'].mean(),
            'effective_disp_std': movements_df['effective_displacement'].std(),
        })

    # Gaze per fixation (from enriched gaze data)
    if not un_enriched_mode and not gaze_enr.empty and not fixations_to_analyze.empty and 'fixation id' in gaze_enr.columns:
        gaze_per_fix = gaze_enr.groupby('fixation id').size().mean()
        results['n_gaze_per_fixation_avg'] = gaze_per_fix

    return results


def _plot_histogram(data_series, title, xlabel, output_path):
    """Helper function to create a standardized, aesthetically pleasing histogram."""
    if data_series.dropna().empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(data_series.dropna(), bins=25, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_plots(data, movements_df, subj_name, event_name, output_dir: Path, un_enriched_mode: bool):
    """Generates and saves all plots for the event, adapting for un-enriched mode."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine which fixation data to use for plots ---
    fixations_enr = data.get('fixations_enr', pd.DataFrame())
    fixations_not_enr = data.get('fixations_not_enr', pd.DataFrame())
    fixations_for_plots = pd.DataFrame()

    if not un_enriched_mode and not fixations_enr.empty and 'fixation detected on surface' in fixations_enr.columns:
        fixations_for_plots = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        print("Plotting fixations from enriched file (on surface).")
    elif not fixations_not_enr.empty:
        fixations_for_plots = fixations_not_enr.copy()
        print("Plotting fixations from un-enriched file.")

    # Plot Periodogram and Spectrogram (Pupil data)
    if 'pupil' in data and not data['pupil'].empty and 'pupil diameter left [mm]' in data['pupil'].columns:
        ts = data['pupil']['pupil diameter left [mm]'].to_numpy()
        if len(ts) > SAMPLING_FREQ:
            freqs, Pxx = welch(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 100))
            plt.figure(figsize=(10, 5))
            plt.semilogy(freqs, Pxx)
            plt.title(f'Periodogram - {subj_name} - {event_name}')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power Spectral Density [V^2/Hz]')
            plt.grid(True)
            plt.savefig(output_dir / f'periodogram_{subj_name}_{event_name}.pdf')
            plt.close()

            f, t, Sxx = spectrogram(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 256), noverlap=min(len(ts)//2, 50))
            plt.figure(figsize=(10, 5))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.title(f'Spectrogram - {subj_name} - {event_name}')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [s]')
            plt.colorbar(label='Power [dB]')
            plt.savefig(output_dir / f'spectrogram_{subj_name}_{event_name}.pdf')
            plt.close()
        else:
            print(f"Skipping Periodogram/Spectrogram for {event_name} due to insufficient pupil data points.")

    # Histograms
    gaze_elevation_data = pd.Series(dtype=float)
    gaze_elevation_label = ""
    # Plot from un-enriched gaze data ('gaze.csv') if available
    if 'gaze_not_enr' in data and not data['gaze_not_enr'].empty and 'elevation [deg]' in data['gaze_not_enr'].columns:
        gaze_elevation_data = data['gaze_not_enr']['elevation [deg]']
        gaze_elevation_label = "Gaze Elevation (Un-enriched)"
        
    # Fallback to enriched gaze data ('gaze_enriched.csv')
    elif not un_enriched_mode and 'gaze' in data and not data['gaze'].empty and 'gaze direction elevation [deg]' in data['gaze'].columns:
        gaze_elevation_data = data['gaze']['gaze direction elevation [deg]']
        gaze_elevation_label = "Gaze Elevation (Enriched)"

    if not gaze_elevation_data.empty:
        _plot_histogram(gaze_elevation_data,
                        f"{gaze_elevation_label} Histogram - {subj_name} - {event_name}",
                        "Elevation [deg]",
                        output_dir / f'hist_gaze_elevation_{subj_name}_{event_name}.pdf')

    if 'pupil' in data and not data['pupil'].empty and 'pupil diameter left [mm]' in data['pupil'].columns:
        _plot_histogram(data['pupil']['pupil diameter left [mm]'],
                        f"Pupil Diameter Histogram - {subj_name} - {event_name}",
                        "Diameter [mm]",
                        output_dir / f'hist_pupillometry_{subj_name}_{event_name}.pdf')

    if not fixations_for_plots.empty and 'duration [ms]' in fixations_for_plots.columns:
        _plot_histogram(fixations_for_plots['duration [ms]'],
                        f"Fixation Duration Histogram - {subj_name} - {event_name}",
                        "Duration [ms]",
                        output_dir / f'hist_fixations_{subj_name}_{event_name}.pdf')

    if 'blinks' in data and not data['blinks'].empty and 'duration [ms]' in data['blinks'].columns:
        _plot_histogram(data['blinks']['duration [ms]'],
                        f"Blink Duration Histogram - {subj_name} - {event_name}",
                        "Duration [ms]",
                        output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')

    if 'saccades' in data and not data['saccades'].empty and 'duration [ms]' in data['saccades'].columns:
        _plot_histogram(data['saccades']['duration [ms]'],
                        f"Saccade Duration Histogram - {subj_name} - {event_name}",
                        "Duration [ms]",
                        output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')

    # Path graphs
    # Plot path from un-enriched gaze data ('gaze.csv') if available
    if 'gaze_not_enr' in data and not data['gaze_not_enr'].empty and 'gaze x [px]' in data['gaze_not_enr'].columns and 'gaze y [px]' in data['gaze_not_enr'].columns:
        plt.plot(data['gaze_not_enr']['gaze x [px]'], data['gaze_not_enr']['gaze y [px]'], marker='o', linestyle='-', color='green')
        plt.title(f"Gaze Path (Un-enriched) - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'path_gaze_unenriched_{subj_name}_{event_name}.pdf')
        plt.close()
    # Fallback to enriched gaze data ('gaze_enriched.csv')
    elif not un_enriched_mode and 'gaze' in data and not data['gaze'].empty and 'gaze position on surface x [normalized]' in data['gaze'].columns and 'gaze position on surface y [normalized]' in data['gaze'].columns:
         plt.plot(data['gaze']['gaze position on surface x [normalized]'], data['gaze']['gaze position on surface y [normalized]'], marker='o', linestyle='-', color='green')
         plt.title(f"Gaze Path (Enriched) - {subj_name} - {event_name}")
         plt.savefig(output_dir / f'path_gaze_enriched_{subj_name}_{event_name}.pdf')
         plt.close()

    if not fixations_for_plots.empty and 'fixation x [normalized]' in fixations_for_plots.columns and 'fixation y [normalized]' in fixations_for_plots.columns:
        plt.plot(fixations_for_plots['fixation x [normalized]'], fixations_for_plots['fixation y [normalized]'], marker='o', linestyle='-', color='green')
        plt.title(f"Fixation Path - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'path_fixation_{subj_name}_{event_name}.pdf')
        plt.close()

    # Movement paths (from enriched data)
    if not un_enriched_mode and not movements_df.empty:
        plt.figure(figsize=(8, 6))
        for _, row in movements_df.iterrows():
            start_x, start_y = row['start_pos']
            end_x, end_y = row['end_pos']
            if not np.isnan(start_x) and not np.isnan(start_y) and not np.isnan(end_x) and not np.isnan(end_y):
                plt.plot([start_x, end_x], [start_y, end_y], linestyle='-', c='b', alpha=0.5)
                plt.scatter([start_x, end_x], [start_y, end_y], marker='o', c='b', s=10)
        plt.title(f"Total Movement Path - {subj_name} - {event_name}")
        plt.xlabel("X Normalized")
        plt.ylabel("Y Normalized")
        plt.grid(True)
        plt.savefig(output_dir / f'total_mov_{subj_name}_{event_name}.pdf')
        plt.close()

    # Heatmaps (from fixation data)
    scale = 1000
    if not fixations_for_plots.empty and 'fixation x [normalized]' in fixations_for_plots.columns and 'fixation y [normalized]' in fixations_for_plots.columns:
        fix_x = fixations_for_plots['fixation x [normalized]'].dropna() * scale
        fix_y = fixations_for_plots['fixation y [normalized]'].dropna() * scale
        if len(fix_x) > 1:
            kde = gaussian_kde([fix_x, fix_y])
            x_grid, y_grid = np.mgrid[0:scale:complex(scale), 0:scale:complex(scale)]
            z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))
            plt.figure(figsize=(8, 4))
            plt.contourf(x_grid, y_grid, z.reshape(x_grid.shape), cmap='Reds', alpha=0.7)
            plt.colorbar()
            plt.scatter(fix_x, fix_y, alpha=0.4, s=5)
            plt.xlim(0, scale)
            plt.ylim(0, scale)
            plt.title(f"Fixation Heatmap - {subj_name} - {event_name}")
            plt.savefig(output_dir / f'cloud_fix_{subj_name}_{event_name}.pdf')
            plt.close()
        else:
            print(f"Skipping Heatmap for {event_name} due to insufficient fixation data points.")


def process_event(event_row, all_data, subj_name, output_dir, un_enriched_mode: bool):
    """Main processing pipeline for a single event."""
    event_name = event_row.get('name', event_row.name)
    print(f"Processing event: {event_name} for participant: {subj_name}")

    timestamp = event_row['timestamp [ns]']
    rec_id = event_row['recording id']

    event_data = filter_data_by_event(all_data, timestamp, rec_id, un_enriched_mode)
    movements_df = process_gaze_movements(event_data.get('gaze', pd.DataFrame()), un_enriched_mode)
    results = calculate_summary_features(event_data, movements_df, subj_name, event_name, un_enriched_mode)
    generate_plots(event_data, movements_df, subj_name, event_name, output_dir, un_enriched_mode)
    
    return results


def downsample_video(input_file, output_file, input_fps, output_fps):
    """Downsamples a video file to a lower FPS."""
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(str(output_file), fourcc, output_fps, (width, height))

    if out.isOpened():
        frame_interval = int(input_fps / output_fps)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret: break
            if i % frame_interval == 0:
                out.write(frame)
        print(f"Video downsampled to {output_file}")
    else:
        print(f"Error: Could not open output video file for downsampling: {output_file}")
    cap.release()
    out.release()


def create_analysis_video(data_dir: Path, output_dir: Path):
    """Creates a video combining eye tracking, external view, and pupil diameter."""
    print("Creating video...")
    
    internal_video_path = data_dir / 'internal.mp4'
    external_video_path = data_dir / 'external.mp4'
    pupillometry_data_path = data_dir / '3d_eye_states.csv'

    if not all([p.exists() for p in [internal_video_path, external_video_path, pupillometry_data_path]]):
        print("Skipping video creation: One or more required files (internal.mp4, external.mp4, 3d_eye_states.csv) not found.")
        return

    try:
        downsampled_video_path = output_dir / 'downsampled_internal_video.mp4'
        if os.path.getsize(internal_video_path) > 0:
            downsample_video(internal_video_path, downsampled_video_path, 200, 40)
        else:
            print(f"Internal video {internal_video_path} is empty, skipping downsampling.")
            return

        pupillometry_data = pd.read_csv(pupillometry_data_path)
        if 'pupil diameter left [mm]' not in pupillometry_data.columns:
            print("Skipping video creation: 'pupil diameter left [mm]' column not found in pupillometry data.")
            return
        time_series = pupillometry_data['pupil diameter left [mm]'].values.flatten()

        cap1 = cv2.VideoCapture(str(downsampled_video_path))
        cap2 = cv2.VideoCapture(str(external_video_path))

        if not cap1.isOpened() or not cap2.isOpened():
            print("Error opening video files for animation after downsampling.")
            return

        fig, (video_axes1, video_axes2, time_series_axes) = plt.subplots(3, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)
        output_video_path = output_dir / 'output_analysis_video.mp4'
        fps = cap1.get(cv2.CAP_PROP_FPS)
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        if not out.isOpened():
            print(f"Error: Could not create output video file: {output_video_path}")
            cap1.release(); cap2.release(); plt.close(fig)
            return

        video_axes1.set_title("Internal View (Eye)"); video_axes1.axis('off')
        video_axes2.set_title("External View"); video_axes2.axis('off')
        time_series_axes.set_title("Pupil Diameter Time Series")
        time_series_axes.set_xlabel('Frame (n)'); time_series_axes.set_ylabel('Diameter (mm)')
        
        num_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)), len(time_series))

        for i in range(num_frames):
            ret1, frame1 = cap1.read(); ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                print(f"Warning: Could not read frame {i}. Exiting video creation loop.")
                break
            
            video_axes1.clear(); video_axes2.clear(); time_series_axes.clear()
            video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)); video_axes1.axis('off')
            video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)); video_axes2.axis('off')

            window_size = 1000
            idx_start = max(0, i - window_size // 2)
            idx_end = min(len(time_series), i + window_size // 2)
            idx = np.arange(idx_start, idx_end)
            
            if idx.size > 0:
                time_series_axes.plot(idx, time_series[idx], 'b-')
                time_series_axes.plot(i, time_series[i], 'ro', markersize=10) # Current point
                time_series_axes.set_xlim(idx_start, idx_end)
                time_series_axes.set_ylim(np.nanmin(time_series) * 0.95, np.nanmax(time_series) * 1.05)
                time_series_axes.set_xlabel('Frame (n)'); time_series_axes.set_ylabel('Diameter (mm)')

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if (i+1) % 100 == 0: print(f"  ...processed frame {i+1}/{num_frames}")

        cap1.release(); cap2.release(); out.release(); plt.close(fig)
        if downsampled_video_path.exists():
            os.remove(downsampled_video_path)
            print(f"Removed temporary downsampled video: {downsampled_video_path}")
        print(f"Analysis video saved to {output_video_path}")

    except Exception as e:
        print(f"An unexpected error occurred during video creation: {e}")
        import traceback
        traceback.print_exc()


def run_analysis(subj_name='subj_01', data_dir_str='./eyetracking_file', output_dir_str='./results', un_enriched_mode=False):
    """Main function to run the complete analysis pipeline."""
    pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', 250)
    data_dir = Path(data_dir_str); output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_data = load_all_data(data_dir, un_enriched_mode)
    except FileNotFoundError:
        print("Analysis stopped due to missing required files.")
        return

    events_df = all_data['events']; all_results = []
    for _, event_row in events_df.iterrows():
        try:
            event_results = process_event(event_row, all_data, subj_name, output_dir, un_enriched_mode)
            all_results.append(event_results)
        except Exception as e:
            event_name = event_row.get('name', event_row.name)
            print(f"Could not process event '{event_name}'. Error: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_filename = output_dir / f'summary_results_{subj_name}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"\nAggregated results saved to {results_filename}")
    else:
        print("\nNo analysis results were generated.")

    create_analysis_video(data_dir, output_dir)


if __name__ == '__main__':
    # This block is for standalone execution and testing.
    # The GUI provides these values dynamically.
    SUBJECT_ID = 'subj_01'
    DATA_DIRECTORY = './eyetracking_file'
    RESULTS_DIRECTORY = f'./analysis_results_{SUBJECT_ID}'
    UN_ENRICHED = False 

    run_analysis(
        subj_name=SUBJECT_ID,
        data_dir_str=DATA_DIRECTORY,
        output_dir_str=RESULTS_DIRECTORY,
        un_enriched_mode=UN_ENRICHED
    )