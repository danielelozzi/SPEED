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
        'gaze_not_enr': 'gaze_not_enr.csv',
        'pupil': '3d_eye_states.csv',
        'blinks': 'blinks.csv',
        'saccades': 'saccades.csv'
    }
    
    # Conditionally add enriched files if not in un-enriched mode
    if not un_enriched_mode:
        files_to_load['gaze'] = 'gaze.csv'
        files_to_load['fixations'] = 'fixations.csv'

    dataframes = {}
    missing_files_in_mode = []
    for name, filename in files_to_load.items():
        try:
            dataframes[name] = pd.read_csv(data_dir / filename)
        except FileNotFoundError:
            # If a file is genuinely missing (and not optional for the current mode),
            # or if it's explicitly required, raise an error.
            # In un-enriched mode, 'gaze' and 'fixations' might not be present,
            # but that's handled by not attempting to load them.
            if not un_enriched_mode and (name == 'gaze' or name == 'fixations'):
                # This should ideally be caught by the GUI, but as a safeguard:
                print(f"Warning: File {filename} not found, proceeding without it (enriched mode).")
                dataframes[name] = pd.DataFrame() # Create empty dataframe to avoid KeyError
            elif name in ['events', 'gaze_not_enr', 'pupil', 'blinks', 'saccades']:
                # These are always required (or their absence indicates a problem)
                missing_files_in_mode.append(filename)
                
    if missing_files_in_mode:
        print(f"Error: Required data files not found: {', '.join(missing_files_in_mode)}")
        raise FileNotFoundError(f"Required data files not found: {', '.join(missing_files_in_mode)}")
            
    return dataframes


def filter_data_by_event(all_data, event_timestamp, rec_id, un_enriched_mode: bool):
    """Filters all dataframes for a specific event based on timestamp and recording ID, adapting for un-enriched mode."""
    event_data = {}
    
    # Gaze (enriched) - only if available
    if 'gaze' in all_data and not all_data['gaze'].empty and not un_enriched_mode:
        event_data['gaze'] = all_data['gaze'][
            (all_data['gaze']['timestamp [ns]'] > event_timestamp) &
            (all_data['gaze']['recording id'] == rec_id)
        ].copy().reset_index(drop=True)
    else:
        event_data['gaze'] = pd.DataFrame() # Ensure it's an empty DataFrame if not used

    # Gaze (not enriched) - always try to load if exists
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

    # Fixations - only if available
    if 'fixations' in all_data and not all_data['fixations'].empty and not un_enriched_mode:
        event_data['fixations'] = all_data['fixations'][
            (all_data['fixations']['start timestamp [ns]'] > event_timestamp) &
            (all_data['fixations']['recording id'] == rec_id)
        ].copy().reset_index(drop=True)
    else:
        event_data['fixations'] = pd.DataFrame()

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
    if un_enriched_mode or gaze_df.empty or 'fixation id' not in gaze_df.columns or 'gaze detected on surface' not in gaze_df.columns:
        # Cannot calculate movements without enriched data (fixation id and surface detection)
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
        
        # Check if gaze position on surface columns exist before accessing
        if 'gaze position on surface x [normalized]' in group.columns and 'gaze position on surface y [normalized]' in group.columns:
            start_pos = (start_row['gaze position on surface x [normalized]'], start_row['gaze position on surface y [normalized]'])
            end_pos = (end_row['gaze position on surface x [normalized]'], end_row['gaze position on surface y [normalized]'])

            # Calculate total path length
            x = group['gaze position on surface x [normalized]']
            y = group['gaze position on surface y [normalized]']
            total_displacement = euclidean_distance(x.shift(), y.shift(), x, y).sum()

            # Calculate effective displacement (start to end)
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
    fixations = data.get('fixations', pd.DataFrame()) # Use .get() for safe access
    blinks = data.get('blinks', pd.DataFrame())
    pupil = data.get('pupil', pd.DataFrame())
    gaze = data.get('gaze', pd.DataFrame()) # Enriched gaze
    gaze_not_enr = data.get('gaze_not_enr', pd.DataFrame()) # Un-enriched gaze

    results = {'participant': subj_name, 'event': event_name}

    # Fixation features (only if not in un_enriched_mode)
    if not un_enriched_mode and not fixations.empty:
        results.update({
            'n_fixation': fixations['fixation id'].nunique(),
            'fixation_avg_duration_ms': fixations['duration [ms]'].mean(),
            'fixation_std_duration_ms': fixations['duration [ms]'].std(),
            'fixation_avg_x': fixations['fixation x [normalized]'].mean(),
            'fixation_std_x': fixations['fixation x [normalized]'].std(),
            'fixation_avg_y': fixations['fixation y [normalized]'].mean(),
            'fixation_std_y': fixations['fixation y [normalized]'].std(),
        })

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

    # Movement features (only if not in un_enriched_mode)
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

    # Gaze per fixation (only if not in un_enriched_mode)
    if not un_enriched_mode and not gaze.empty and not fixations.empty and 'fixation id' in gaze.columns:
        gaze_per_fix = gaze.groupby('fixation id').size().mean()
        results['n_gaze_per_fixation_avg'] = gaze_per_fix

    return results


def generate_plots(data, movements_df, subj_name, event_name, output_dir: Path, un_enriched_mode: bool):
    """Generates and saves all plots for the event, adapting for un-enriched mode."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot Periodogram and Spectrogram (Pupil data, always attempted if available)
    if 'pupil' in data and not data['pupil'].empty and 'pupil diameter left [mm]' in data['pupil'].columns:
        ts = data['pupil']['pupil diameter left [mm]'].to_numpy()
        if len(ts) > SAMPLING_FREQ: # Ensure enough data points for spectral analysis
            freqs, Pxx = welch(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 100))
            plt.figure(figsize=(10, 5))
            plt.semilogy(freqs, Pxx)
            plt.title(f'Periodogramma - {subj_name} - {event_name}')
            plt.xlabel('Frequenza [Hz]')
            plt.ylabel('Densità spettrale di potenza [V^2/Hz]')
            plt.grid(True)
            plt.savefig(output_dir / f'periodogram_{subj_name}_{event_name}.pdf')
            plt.close()

            f, t, Sxx = spectrogram(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 256), noverlap=min(len(ts)//2, 50))
            plt.figure(figsize=(10, 5))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.title(f'Spettrogramma - {subj_name} - {event_name}')
            plt.ylabel('Frequenza [Hz]')
            plt.xlabel('Tempo [s]')
            plt.colorbar(label='Potenza [dB]')
            plt.savefig(output_dir / f'spectrogram_{subj_name}_{event_name}.pdf')
            plt.close()
        else:
            print(f"Skipping Periodogram/Spectrogram for {event_name} due to insufficient pupil data points.")


    # Histograms
    if 'gaze_not_enr' in data and not data['gaze_not_enr'].empty and 'elevation [deg]' in data['gaze_not_enr'].columns:
        plt.hist(data['gaze_not_enr']['elevation [deg]'].dropna())
        plt.title(f"Istogramma Elevazione Sguardo (Non Arricchito) - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_gaze_not_enr_elevation_{subj_name}_{event_name}.pdf')
        plt.close()
    elif not un_enriched_mode and 'gaze' in data and not data['gaze'].empty and 'gaze direction elevation [deg]' in data['gaze'].columns:
        plt.hist(data['gaze']['gaze direction elevation [deg]'].dropna())
        plt.title(f"Istogramma Elevazione Sguardo (Arricchito) - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_gaze_enriched_elevation_{subj_name}_{event_name}.pdf')
        plt.close()


    if 'pupil' in data and not data['pupil'].empty and 'pupil diameter left [mm]' in data['pupil'].columns:
        plt.hist(data['pupil']['pupil diameter left [mm]'].dropna())
        plt.title(f"Istogramma Diametro Pupilla - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_pupillometry_{subj_name}_{event_name}.pdf')
        plt.close()

    if not un_enriched_mode and 'fixations' in data and not data['fixations'].empty and 'duration [ms]' in data['fixations'].columns:
        plt.hist(data['fixations']['duration [ms]'].dropna())
        plt.title(f"Istogramma Durata Fissazioni - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_fixations_{subj_name}_{event_name}.pdf')
        plt.close()

    if 'blinks' in data and not data['blinks'].empty and 'duration [ms]' in data['blinks'].columns:
        plt.hist(data['blinks']['duration [ms]'].dropna())
        plt.title(f"Istogramma Durata Blink - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')
        plt.close()

    if 'saccades' in data and not data['saccades'].empty and 'duration [ms]' in data['saccades'].columns:
        plt.hist(data['saccades']['duration [ms]'].dropna())
        plt.title(f"Istogramma Durata Saccadi - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')
        plt.close()

    # Path graphs
    if 'gaze_not_enr' in data and not data['gaze_not_enr'].empty and 'gaze x [px]' in data['gaze_not_enr'].columns and 'gaze y [px]' in data['gaze_not_enr'].columns:
        plt.plot(data['gaze_not_enr']['gaze x [px]'], data['gaze_not_enr']['gaze y [px]'], marker='o', linestyle='-', color='green')
        plt.title(f"Percorso Sguardo (Non Arricchito) - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'path_gaze_not_enr_{subj_name}_{event_name}.pdf')
        plt.close()
    elif not un_enriched_mode and 'gaze' in data and not data['gaze'].empty and 'gaze position on surface x [normalized]' in data['gaze'].columns and 'gaze position on surface y [normalized]' in data['gaze'].columns:
         plt.plot(data['gaze']['gaze position on surface x [normalized]'], data['gaze']['gaze position on surface y [normalized]'], marker='o', linestyle='-', color='green')
         plt.title(f"Percorso Sguardo (Arricchito) - {subj_name} - {event_name}")
         plt.savefig(output_dir / f'path_gaze_enriched_{subj_name}_{event_name}.pdf')
         plt.close()


    if not un_enriched_mode and 'fixations' in data and not data['fixations'].empty and 'fixation x [normalized]' in data['fixations'].columns and 'fixation y [normalized]' in data['fixations'].columns:
        plt.plot(data['fixations']['fixation x [normalized]'], data['fixations']['fixation y [normalized]'], marker='o', linestyle='-', color='green')
        plt.title(f"Percorso Fissazioni - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'path_fixation_{subj_name}_{event_name}.pdf')
        plt.close()

    # Movement paths (only if not in un_enriched_mode)
    if not un_enriched_mode and not movements_df.empty:
        plt.figure(figsize=(8, 6))
        for _, row in movements_df.iterrows():
            start_x, start_y = row['start_pos']
            end_x, end_y = row['end_pos']
            if not np.isnan(start_x) and not np.isnan(start_y) and not np.isnan(end_x) and not np.isnan(end_y):
                plt.plot([start_x, end_x], [start_y, end_y], linestyle='-', c='b', alpha=0.5)
                plt.scatter([start_x, end_x], [start_y, end_y], marker='o', c='b', s=10)
        plt.title(f"Percorso Movimenti Totali - {subj_name} - {event_name}")
        plt.xlabel("X Normalized")
        plt.ylabel("Y Normalized")
        plt.grid(True)
        plt.savefig(output_dir / f'total_mov_{subj_name}_{event_name}.pdf')
        plt.close()

    # Heatmaps (only if not in un_enriched_mode)
    scale = 1000
    if not un_enriched_mode and 'fixations' in data and not data['fixations'].empty and 'fixation x [normalized]' in data['fixations'].columns and 'fixation y [normalized]' in data['fixations'].columns:
        fix_x = data['fixations']['fixation x [normalized]'].dropna() * scale
        fix_y = data['fixations']['fixation y [normalized]'].dropna() * scale
        if len(fix_x) > 1: # Need at least two points for KDE
            kde = gaussian_kde([fix_x, fix_y])
            x_grid, y_grid = np.mgrid[0:scale:complex(scale), 0:scale:complex(scale)]
            z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))
            plt.figure(figsize=(8, 4))
            plt.contourf(x_grid, y_grid, z.reshape(x_grid.shape), cmap='Reds', alpha=0.7)
            plt.colorbar()
            plt.scatter(fix_x, fix_y, alpha=0.4, s=5)
            plt.xlim(0, scale)
            plt.ylim(0, scale)
            plt.title(f"Mappa di Calore Fissazioni - {subj_name} - {event_name}")
            plt.savefig(output_dir / f'cloud_fix_{subj_name}_{event_name}.pdf')
            plt.close()
        else:
            print(f"Skipping Heatmap for {event_name} due to insufficient fixation data points.")


def process_event(event_row, all_data, subj_name, output_dir, un_enriched_mode: bool):
    """Main processing pipeline for a single event."""
    event_name = event_row.get('name', event_row.name)
    print(f"Elaborazione evento: {event_name} per il partecipante: {subj_name}")

    timestamp = event_row['timestamp [ns]']
    rec_id = event_row['recording id']

    # 1. Filter data for the current event
    event_data = filter_data_by_event(all_data, timestamp, rec_id, un_enriched_mode)

    # 2. Process gaze data to find movements (only if not un_enriched_mode)
    movements_df = pd.DataFrame()
    if not un_enriched_mode:
        movements_df = process_gaze_movements(event_data.get('gaze', pd.DataFrame()), un_enriched_mode)

    # 3. Calculate summary features
    results = calculate_summary_features(event_data, movements_df, subj_name, event_name, un_enriched_mode)

    # 4. Generate all plots
    generate_plots(event_data, movements_df, subj_name, event_name, output_dir, un_enriched_mode)
    
    return results


def downsample_video(input_file, output_file, input_fps, output_fps):
    """Downsamples a video file to a lower FPS."""
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il file video {input_file}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Use MP4V codec which is generally widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(str(output_file), fourcc, output_fps, (width, height))

    if out.isOpened():
        frame_interval = int(input_fps / output_fps)
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_interval == 0:
                out.write(frame)
        print(f"Video sottocampionato in {output_file}")
    else:
        print(f"Errore: Impossibile aprire il file video di output per il downsampling: {output_file}")


    cap.release()
    out.release()


def create_analysis_video(data_dir: Path, output_dir: Path):
    """Creates a video combining eye tracking, external view, and pupil diameter."""
    print("Creazione video in corso...")
    
    internal_video_path = data_dir / 'internal.mp4'
    external_video_path = data_dir / 'external.mp4'
    pupillometry_data_path = data_dir / '3d_eye_states.csv'

    if not internal_video_path.exists():
        print(f"Skipping video creation: Internal video not found at {internal_video_path}")
        return
    if not external_video_path.exists():
        print(f"Skipping video creation: External video not found at {external_video_path}")
        return
    if not pupillometry_data_path.exists():
        print(f"Skipping video creation: Pupillometry data not found at {pupillometry_data_path}")
        return

    try:
        # Downsample internal video
        downsampled_video_path = output_dir / 'downsampled_internal_video.mp4'
        # Check if file exists and has content before attempting to downsample
        if os.path.getsize(internal_video_path) > 0:
            downsample_video(internal_video_path, downsampled_video_path, 200, 40)
        else:
            print(f"Internal video {internal_video_path} is empty, skipping downsampling.")
            return

        # Load data for video
        pupillometry_data = pd.read_csv(pupillometry_data_path)
        if 'pupil diameter left [mm]' not in pupillometry_data.columns:
            print("Skipping video creation: 'pupil diameter left [mm]' column not found in pupillometry data.")
            return
        time_series = pupillometry_data['pupil diameter left [mm]'].values.flatten()

        cap1 = cv2.VideoCapture(str(downsampled_video_path))
        cap2 = cv2.VideoCapture(str(external_video_path))

        if not cap1.isOpened() or not cap2.isOpened():
            print("Errore nell'apertura dei file video per l'animazione dopo il downsampling.")
            return

        # Setup plot
        fig, (video_axes1, video_axes2, time_series_axes) = plt.subplots(3, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)

        # Init video writer
        output_video_path = output_dir / 'output_analysis_video.mp4'
        fps = cap1.get(cv2.CAP_PROP_FPS) # Use downsampled internal video's FPS
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for output video
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        if not out.isOpened():
            print(f"Errore: Impossibile creare il file video di output: {output_video_path}")
            cap1.release()
            cap2.release()
            plt.close(fig)
            return

        # Init plots
        video_axes1.set_title("Vista Interna (Occhio)")
        video_axes1.axis('off')
        video_axes2.set_title("Vista Esterna")
        video_axes2.axis('off')
        time_series_axes.set_title("Serie Temporale Diametro Pupilla")
        time_series_axes.set_xlabel('Frame (n)')
        time_series_axes.set_ylabel('Diametro (mm)')
        
        # Determine the number of frames to process based on the shortest component
        num_frames_video1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames_video2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames_timeseries = len(time_series)

        num_frames = min(num_frames_video1, num_frames_video2, num_frames_timeseries)

        # Manual loop to create video frames
        for i in range(num_frames):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            # Check if frames were read successfully
            if not ret1 or not ret2:
                print(f"Warning: Could not read frame {i}. Exiting video creation loop.")
                break
            
            # Clear axes for new frame
            video_axes1.clear()
            video_axes2.clear()
            time_series_axes.clear()

            # Redraw video frames
            video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
            video_axes1.axis('off')
            video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
            video_axes2.axis('off')

            # Redraw time series plot
            window_size = 1000 # Keep window size fixed
            idx_start = max(0, i - window_size // 2)
            idx_end = min(len(time_series), i + window_size // 2)
            idx = np.arange(idx_start, idx_end)
            
            # Ensure idx is not empty before plotting
            if idx.size > 0:
                time_series_axes.plot(idx, time_series[idx], 'b-')
                time_series_axes.plot(i, time_series[i], 'ro', markersize=10) # Punto corrente
                time_series_axes.set_xlim(idx_start, idx_end)
                time_series_axes.set_ylim(np.nanmin(time_series) * 0.95, np.nanmax(time_series) * 1.05)
                time_series_axes.set_xlabel('Frame (n)')
                time_series_axes.set_ylabel('Diametro (mm)')
            else:
                print(f"Warning: Empty index for time series plot at frame {i}.")


            # Draw figure and write to video
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img)
            
            if (i+1) % 100 == 0:
                print(f"  ...elaborato frame {i+1}/{num_frames}")

        # Release resources
        cap1.release()
        cap2.release()
        out.release()
        plt.close(fig)
        
        # Clean up downsampled internal video file
        if downsampled_video_path.exists():
            os.remove(downsampled_video_path)
            print(f"Removed temporary downsampled video: {downsampled_video_path}")

        print(f"Video di analisi salvato in {output_video_path}")

    except FileNotFoundError as e:
        print(f"Errore durante la creazione del video, file non trovato: {e}")
    except Exception as e:
        print(f"Si è verificato un errore imprevisto durante la creazione del video: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging


def run_analysis(subj_name='subj_01', data_dir_str='./eyetracking_file', output_dir_str='./results', un_enriched_mode=False):
    """
    Main function to run the complete analysis pipeline.
    It processes each event, saves summary results and plots, and generates a final analysis video.
    Includes a flag for un-enriched data analysis.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 250)

    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_data = load_all_data(data_dir, un_enriched_mode)
    except FileNotFoundError:
        print("Analysis stopped due to missing required files.")
        return # Stop execution if essential files are missing

    events_df = all_data['events']
    all_results = []

    # --- Process each event ---
    for _, event_row in events_df.iterrows():
        try:
            event_results = process_event(event_row, all_data, subj_name, output_dir, un_enriched_mode)
            all_results.append(event_results)
        except Exception as e:
            event_name = event_row.get('name', event_row.name)
            print(f"Impossibile elaborare l'evento '{event_name}'. Errore: {e}")
            import traceback
            traceback.print_exc()

    # --- Save aggregated results ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_filename = output_dir / f'summary_results_{subj_name}.csv'
        results_df.to_csv(results_filename, index=False)
        print(f"\nRisultati aggregati salvati in {results_filename}")
    else:
        print("\nNessun risultato di analisi generato.")

    # --- Generate analysis video (runs on full data, not per event) ---
    create_analysis_video(data_dir, output_dir)


if __name__ == '__main__':
    # Esempio di come eseguire lo script
    # È possibile modificare questi parametri secondo necessità
    SUBJECT_ID = 'subj_01'
    DATA_DIRECTORY = './eyetracking_file'
    RESULTS_DIRECTORY = f'./analysis_results_{SUBJECT_ID}'

    # Imposta a True per eseguire l'analisi in modalità non arricchita
    UN_ENRICHED = False 

    run_analysis(
        subj_name=SUBJECT_ID,
        data_dir_str=DATA_DIRECTORY,
        output_dir_str=RESULTS_DIRECTORY,
        un_enriched_mode=UN_ENRICHED
    )