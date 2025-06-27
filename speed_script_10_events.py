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


def load_all_data(data_dir: Path):
    """Loads all necessary CSV files from the data directory."""
    files_to_load = {
        'events': 'events.csv',
        'gaze': 'gaze.csv',
        'gaze_not_enr': 'gaze_not_enr.csv',
        'pupil': '3d_eye_states.csv',
        'fixations': 'fixations.csv',
        'blinks': 'blinks.csv',
        'saccades': 'saccades.csv'
    }
    dataframes = {}
    try:
        for name, filename in files_to_load.items():
            dataframes[name] = pd.read_csv(data_dir / filename)
    except FileNotFoundError as e:
        print(f"Errore: Un file dati richiesto non è stato trovato: {e}")
        raise
    return dataframes


def filter_data_by_event(all_data, event_timestamp, rec_id):
    """Filters all dataframes for a specific event based on timestamp and recording ID."""
    event_data = {}
    event_data['gaze'] = all_data['gaze'][
        (all_data['gaze']['timestamp [ns]'] > event_timestamp) &
        (all_data['gaze']['recording id'] == rec_id)
    ].copy().reset_index(drop=True)

    event_data['pupil'] = all_data['pupil'][
        (all_data['pupil']['timestamp [ns]'] > event_timestamp)
    ].copy().reset_index(drop=True)

    event_data['fixations'] = all_data['fixations'][
        (all_data['fixations']['start timestamp [ns]'] > event_timestamp) &
        (all_data['fixations']['recording id'] == rec_id)
    ].copy().reset_index(drop=True)

    event_data['blinks'] = all_data['blinks'][
        (all_data['blinks']['start timestamp [ns]'] > event_timestamp)
    ].copy().reset_index(drop=True)

    event_data['saccades'] = all_data['saccades'][
        (all_data['saccades']['start timestamp [ns]'] > event_timestamp)
    ].copy().reset_index(drop=True)
    
    # This data seems to be used for plotting only
    event_data['gaze_not_enr'] = all_data['gaze_not_enr'][
        (all_data['gaze_not_enr']['timestamp [ns]'] > event_timestamp)
    ].copy().reset_index(drop=True)

    return event_data


def process_gaze_movements(gaze_df):
    """Identifies and processes gaze movements (saccades) from gaze data."""
    if gaze_df.empty:
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
        
        start_pos = (start_row['gaze position on surface x [normalized]'], start_row['gaze position on surface y [normalized]'])
        end_pos = (end_row['gaze position on surface x [normalized]'], end_row['gaze position on surface y [normalized]'])

        # Calculate total path length
        x = group['gaze position on surface x [normalized]']
        y = group['gaze position on surface y [normalized]']
        total_displacement = euclidean_distance(x.shift(), y.shift(), x, y).sum()

        # Calculate effective displacement (start to end)
        effective_displacement = euclidean_distance(start_pos[0], start_pos[1], end_pos[0], end_pos[1])

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


def calculate_summary_features(data, movements_df, subj_name, event_name):
    """Calculates a dictionary of summary features from the processed data."""
    fixations = data['fixations']
    blinks = data['blinks']
    pupil = data['pupil']
    gaze = data['gaze']

    results = {'participant': subj_name, 'event': event_name}

    # Fixation features
    if not fixations.empty:
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
    if not pupil.empty:
        pupil_diam = pupil['pupil diameter left [mm]']
        results.update({
            'pupil_start_mm': pupil_diam.iloc[0],
            'pupil_end_mm': pupil_diam.iloc[-1],
            'pupil_avg_mm': pupil_diam.mean(),
            'pupil_std_mm': pupil_diam.std(),
        })

    # Movement features
    if not movements_df.empty:
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

    # Gaze per fixation/movement
    if not gaze.empty and not fixations.empty:
        gaze_per_fix = gaze.groupby('fixation id').size().mean()
        results['n_gaze_per_fixation_avg'] = gaze_per_fix

    return results


def generate_plots(data, movements_df, subj_name, event_name, output_dir: Path):
    """Generates and saves all plots for the event."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot Periodogram and Spectrogram
    if not data['pupil'].empty:
        ts = data['pupil']['pupil diameter left [mm]'].to_numpy()
        freqs, Pxx = welch(ts, fs=SAMPLING_FREQ, nperseg=100)
        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, Pxx)
        plt.title(f'Periodogramma - {subj_name} - {event_name}')
        plt.xlabel('Frequenza [Hz]')
        plt.ylabel('Densità spettrale di potenza [V^2/Hz]')
        plt.grid(True)
        plt.savefig(output_dir / f'periodogram_{subj_name}_{event_name}.pdf')
        plt.close()

        f, t, Sxx = spectrogram(ts, fs=SAMPLING_FREQ, nperseg=256, noverlap=50)
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title(f'Spettrogramma - {subj_name} - {event_name}')
        plt.ylabel('Frequenza [Hz]')
        plt.xlabel('Tempo [s]')
        plt.colorbar(label='Potenza [dB]')
        plt.savefig(output_dir / f'spectrogram_{subj_name}_{event_name}.pdf')
        plt.close()

    # Histograms
    if not data['gaze_not_enr'].empty:
        plt.hist(data['gaze_not_enr']['elevation [deg]'].dropna())
        plt.title(f"Istogramma Elevazione Sguardo - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_gaze_{subj_name}_{event_name}.pdf')
        plt.close()

    if not data['pupil'].empty:
        plt.hist(data['pupil']['pupil diameter left [mm]'].dropna())
        plt.title(f"Istogramma Diametro Pupilla - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_pupillometry_{subj_name}_{event_name}.pdf')
        plt.close()

    if not data['fixations'].empty:
        plt.hist(data['fixations']['duration [ms]'].dropna())
        plt.title(f"Istogramma Durata Fissazioni - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_fixations_{subj_name}_{event_name}.pdf')
        plt.close()

    if not data['blinks'].empty:
        plt.hist(data['blinks']['duration [ms]'].dropna())
        plt.title(f"Istogramma Durata Blink - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')
        plt.close()

    if not data['saccades'].empty:
        plt.hist(data['saccades']['duration [ms]'].dropna())
        plt.title(f"Istogramma Durata Saccadi - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')
        plt.close()

    # Path graphs
    if not data['gaze_not_enr'].empty:
        plt.plot(data['gaze_not_enr']['gaze x [px]'], data['gaze_not_enr']['gaze y [px]'], marker='o', linestyle='-', color='green')
        plt.title(f"Percorso Sguardo - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'path_gaze_{subj_name}_{event_name}.pdf')
        plt.close()

    if not data['fixations'].empty:
        plt.plot(data['fixations']['fixation x [normalized]'], data['fixations']['fixation y [normalized]'], marker='o', linestyle='-', color='green')
        plt.title(f"Percorso Fissazioni - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'path_fixation_{subj_name}_{event_name}.pdf')
        plt.close()

    # Movement paths
    if not movements_df.empty:
        for _, row in movements_df.iterrows():
            start_x, start_y = row['start_pos']
            end_x, end_y = row['end_pos']
            plt.plot([start_x, end_x], [start_y, end_y], linestyle='-', c='b', alpha=0.5)
            plt.scatter([start_x, end_x], [start_y, end_y], marker='o', c='b', s=10)
        plt.title(f"Percorso Movimenti Totali - {subj_name} - {event_name}")
        plt.savefig(output_dir / f'total_mov_{subj_name}_{event_name}.pdf')
        plt.close()

    # Heatmaps
    scale = 1000
    if not data['fixations'].empty:
        fix_x = data['fixations']['fixation x [normalized]'].dropna() * scale
        fix_y = data['fixations']['fixation y [normalized]'].dropna() * scale
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
            plt.title(f"Mappa di Calore Fissazioni - {subj_name} - {event_name}")
            plt.savefig(output_dir / f'cloud_fix_{subj_name}_{event_name}.pdf')
            plt.close()


def process_event(event_row, all_data, subj_name, output_dir):
    """Main processing pipeline for a single event."""
    event_name = event_row.get('name', event_row.name)
    print(f"Elaborazione evento: {event_name} per il partecipante: {subj_name}")

    timestamp = event_row['timestamp [ns]']
    rec_id = event_row['recording id']

    # 1. Filter data for the current event
    event_data = filter_data_by_event(all_data, timestamp, rec_id)

    # 2. Process gaze data to find movements
    movements_df = process_gaze_movements(event_data['gaze'])

    # 3. Calculate summary features
    results = calculate_summary_features(event_data, movements_df, subj_name, event_name)

    # 4. Save results to a temporary file (or append to a list)
    # This part is handled in the main loop to save a single file at the end.

    # 5. Generate all plots
    generate_plots(event_data, movements_df, subj_name, event_name, output_dir)
    
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_file), fourcc, output_fps, (width, height))

    frame_interval = int(input_fps / output_fps)

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            out.write(frame)

    cap.release()
    out.release()
    print(f"Video sottocampionato in {output_file}")


def create_analysis_video(data_dir: Path, output_dir: Path):
    """Creates a video combining eye tracking, external view, and pupil diameter."""
    print("Creazione video in corso...")
    try:
        # Downsample internal video
        internal_video_path = data_dir / 'internal.mp4'
        downsampled_video_path = output_dir / 'downsampled_internal_video.mp4'
        downsample_video(internal_video_path, downsampled_video_path, 200, 40)

        # Load data for video
        pupillometry_data = pd.read_csv(data_dir / '3d_eye_states.csv')
        time_series = pupillometry_data['pupil diameter left [mm]'].values.flatten()

        cap1 = cv2.VideoCapture(str(downsampled_video_path))
        cap2 = cv2.VideoCapture(str(data_dir / 'external.mp4'))

        if not cap1.isOpened() or not cap2.isOpened():
            print("Errore nell'apertura dei file video per l'animazione.")
            return

        # Setup plot
        fig, (video_axes1, video_axes2, time_series_axes) = plt.subplots(3, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)

        # Init video writer
        output_video_path = output_dir / 'output_analysis_video.mp4'
        fps = cap1.get(cv2.CAP_PROP_FPS)
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        # Init plots
        video_axes1.set_title("Vista Interna (Occhio)")
        video_axes1.axis('off')
        video_axes2.set_title("Vista Esterna")
        video_axes2.axis('off')
        time_series_axes.set_title("Serie Temporale Diametro Pupilla")
        time_series_axes.set_xlabel('Frame (n)')
        time_series_axes.set_ylabel('Diametro (mm)')
        
        num_frames = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT), len(time_series)))

        # Manual loop to create video frames
        for i in range(num_frames):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
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
            window_size = 1000
            idx_start = max(0, i - window_size // 2)
            idx_end = min(len(time_series), i + window_size // 2)
            idx = np.arange(idx_start, idx_end)
            
            time_series_axes.plot(idx, time_series[idx], 'b-')
            time_series_axes.plot(i, time_series[i], 'ro', markersize=10) # Punto corrente
            time_series_axes.set_xlim(idx_start, idx_end)
            time_series_axes.set_ylim(np.nanmin(time_series) * 0.95, np.nanmax(time_series) * 1.05)
            time_series_axes.set_xlabel('Frame (n)')
            time_series_axes.set_ylabel('Diametro (mm)')

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
        print(f"Video di analisi salvato in {output_video_path}")

    except FileNotFoundError as e:
        print(f"Errore durante la creazione del video, file non trovato: {e}")
    except Exception as e:
        print(f"Si è verificato un errore imprevisto durante la creazione del video: {e}")


def run_analysis(subj_name='subj_01', data_dir_str='./eyetracking_file', output_dir_str='./results'):
    """
    Main function to run the complete analysis pipeline.
    It processes each event, saves summary results and plots, and generates a final analysis video.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 250)

    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_data = load_all_data(data_dir)
    except FileNotFoundError:
        return # Stop execution if essential files are missing

    events_df = all_data['events']
    all_results = []

    # --- Process each event ---
    for _, event_row in events_df.iterrows():
        try:
            event_results = process_event(event_row, all_data, subj_name, output_dir)
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

    # --- Generate analysis video (runs on full data, not per event) ---
    create_analysis_video(data_dir, output_dir)


if __name__ == '__main__':
    # Esempio di come eseguire lo script
    # È possibile modificare questi parametri secondo necessità
    SUBJECT_ID = 'subj_01'
    DATA_DIRECTORY = './eyetracking_file'
    RESULTS_DIRECTORY = f'./analysis_results_{SUBJECT_ID}'

    run_analysis(
        subj_name=SUBJECT_ID,
        data_dir_str=DATA_DIRECTORY,
        output_dir_str=RESULTS_DIRECTORY
    )