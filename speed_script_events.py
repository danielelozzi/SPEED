import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import traceback
import os
from scipy.signal import welch, spectrogram
from scipy.stats import gaussian_kde

# --- Constants ---
SAMPLING_FREQ = 200  # Hz
NS_TO_S = 1e9

def euclidean_distance(x1, y1, x2, y2):
    """Calcola la distanza euclidea tra due punti."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def load_all_data(data_dir: Path, un_enriched_mode: bool):
    """Carica tutti i file CSV necessari dalla directory dei dati."""
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
                print(f"Info: File opzionale o di base {filename} non trovato, si procede senza.")
                dataframes[name] = pd.DataFrame()
            else:
                raise FileNotFoundError(f"File dati richiesto non trovato: {filename}")
    return dataframes

def get_timestamp_col(df):
    """Ottiene la colonna di timestamp corretta da un dataframe."""
    for col in ['start timestamp [ns]', 'timestamp [ns]']:
        if col in df.columns:
            return col
    return None

def filter_data_by_segment(all_data, start_ts, end_ts, rec_id):
    """Filtra tutti i dataframe per uno specifico segmento temporale [start_ts, end_ts)."""
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
    """Identifica ed elabora i movimenti dello sguardo dai dati gaze ARRICCHITI."""
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
    """Calcola un dizionario di feature di riepilogo, inclusa la normalizzazione dai pixel."""
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

    # --- Feature delle Fissazioni ---
    fixations_to_analyze = fixations_not_enr if not fixations_not_enr.empty else pd.DataFrame()
    if not un_enriched_mode and not fixations_enr.empty and 'fixation detected on surface' in fixations_enr.columns:
        enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        if not enriched_on_surface.empty:
            fixations_to_analyze = enriched_on_surface
    
    if not fixations_to_analyze.empty:
        results.update({'n_fixation': fixations_to_analyze['fixation id'].nunique(), 'fixation_avg_duration_ms': fixations_to_analyze['duration [ms]'].mean(), 'fixation_std_duration_ms': fixations_to_analyze['duration [ms]'].std()})
        
        x_coords, y_coords = pd.Series(dtype='float64'), pd.Series(dtype='float64')
        if 'fixation x [normalized]' in fixations_to_analyze.columns:
            print("DEBUG: Uso le coordinate pre-normalizzate dal file arricchito.")
            x_coords, y_coords = fixations_to_analyze['fixation x [normalized]'], fixations_to_analyze['fixation y [normalized]']
        elif 'fixation x [px]' in fixations_to_analyze.columns:
            if video_width and video_height and video_width > 0 and video_height > 0:
                print(f"DEBUG: Normalizzo le coordinate in pixel usando le dimensioni del video {video_width}x{video_height}.")
                x_coords = fixations_to_analyze['fixation x [px]'] / video_width
                y_coords = fixations_to_analyze['fixation y [px]'] / video_height
            else:
                print("ATTENZIONE: Le coordinate delle fissazioni sono in pixel, ma le dimensioni del video non sono disponibili. Impossibile normalizzare.")
        
        if not x_coords.empty:
            results.update({
                'fixation_avg_x': x_coords.mean(), 'fixation_std_x': x_coords.std(),
                'fixation_avg_y': y_coords.mean(), 'fixation_std_y': y_coords.std()
            })

    # --- Altre Feature ---
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
        print("DEBUG: Calcolo le feature dei movimenti da saccades.csv")
        sacc_duration_s = saccades['duration [ms]'] / 1000
        amplitude = saccades['amplitude [deg]'] if 'amplitude [deg]' in saccades.columns else pd.Series(dtype='float64')
        results.update({
            'n_movements': len(saccades), 'sum_time_movement_s': sacc_duration_s.sum(), 'avg_time_movement_s': sacc_duration_s.mean(),
            'std_time_movement_s': sacc_duration_s.std(), 'total_disp_sum': amplitude.sum(), 'total_disp_avg': amplitude.mean(),
            'total_disp_std': amplitude.std(), 'effective_disp_sum': amplitude.sum(), 'effective_disp_avg': amplitude.mean(),
            'effective_disp_std': amplitude.std()
        })
            
    return results

def _plot_histogram(data_series, title, xlabel, output_path):
    """Funzione ausiliaria per creare un istogramma standardizzato."""
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
    """Genera e salva tutti i grafici per l'evento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())
    gaze_enr = data.get('gaze', pd.DataFrame()) # Get enriched gaze data
    pupil_df = data.get('pupil', pd.DataFrame()) # Get pupil data
    blinks_df = data.get('blinks', pd.DataFrame()) # Get blinks data
    saccades_df = data.get('saccades', pd.DataFrame()) # Get saccades data
    
    fixations_for_plots = pd.DataFrame()

    if not un_enriched_mode and not fixations_enr.empty and 'fixation detected on surface' in fixations_enr.columns:
        fixations_for_plots = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
    elif not fixations_not_enr.empty:
        fixations_for_plots = fixations_not_enr.copy()

    # --- Pupillometry Plot with Gaze On Surface Indicator ---
    if not pupil_df.empty and 'pupil diameter left [mm]' in pupil_df.columns:
        # Merge pupil data with enriched gaze data to get 'gaze detected on surface' status
        # This assumes that 'timestamp [ns]' is the common column and sufficiently aligned.
        
        # Ensure timestamps are unique in gaze_enr for merging
        gaze_enr_unique_ts = gaze_enr.drop_duplicates(subset=['timestamp [ns]']).copy()
        
        pupil_with_gaze_status = pd.merge(
            pupil_df,
            gaze_enr_unique_ts[['timestamp [ns]', 'gaze detected on surface']],
            on='timestamp [ns]',
            how='left'
        )
        # Fill NaN in 'gaze detected on surface' if no direct match, assume False if not detected
        pupil_with_gaze_status['gaze detected on surface'].fillna(False, inplace=True)

        if not pupil_with_gaze_status.empty:
            plt.figure(figsize=(12, 6))
            
            # Plot left pupil diameter
            if 'pupil diameter left [mm]' in pupil_with_gaze_status.columns:
                plt.plot(pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S, pupil_with_gaze_status['pupil diameter left [mm]'], label='Pupil Diameter Left [mm]', color='blue', alpha=0.7)
            
            # Plot right pupil diameter (if available)
            if 'pupil diameter right [mm]' in pupil_with_gaze_status.columns:
                plt.plot(pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S, pupil_with_gaze_status['pupil diameter right [mm]'], label='Pupil Diameter Right [mm]', color='purple', alpha=0.7)

            # Color background based on 'gaze detected on surface'
            current_status = None
            start_time = None

            timestamps_seconds = pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S
            epsilon = (timestamps_seconds.iloc[1] - timestamps_seconds.iloc[0]) / 2 if len(timestamps_seconds) > 1 else 0.01

            for i, row in pupil_with_gaze_status.iterrows():
                ts = row['timestamp [ns]'] / NS_TO_S
                gaze_on_surface = row['gaze detected on surface']

                if current_status is None:
                    current_status = gaze_on_surface
                    start_time = ts
                elif gaze_on_surface != current_status:
                    color = 'lightgreen' if current_status else 'lightcoral'
                    plt.axvspan(start_time - epsilon, ts - epsilon, facecolor=color, alpha=0.5)
                    current_status = gaze_on_surface
                    start_time = ts
            
            if start_time is not None:
                color = 'lightgreen' if current_status else 'lightcoral'
                plt.axvspan(start_time - epsilon, timestamps_seconds.iloc[-1] + epsilon, facecolor=color, alpha=0.5)

            plt.title(f"Pupil Diameter with Gaze On Surface - {subj_name} - {event_name}", fontsize=15)
            plt.xlabel('Time [s]', fontsize=12)
            plt.ylabel('Pupil Diameter [mm]', fontsize=12)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_dir / f'pupil_diameter_gaze_surface_{subj_name}_{event_name}.pdf')
            plt.close()
        else:
            print(f"ATTENZIONE: Nessun dato pupillare arricchito disponibile per l'evento '{event_name}' per il grafico con indicatore di superficie.")

    # Periodogramma e Spettrogramma
    if 'pupil' in data and not data['pupil'].empty and 'pupil diameter left [mm]' in data['pupil'].columns:
        ts = data['pupil']['pupil diameter left [mm]'].dropna().to_numpy()
        if len(ts) > SAMPLING_FREQ:
            freqs, Pxx = welch(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 100))
            plt.figure(figsize=(10, 5)); plt.semilogy(freqs, Pxx); plt.title(f'Periodogram - {subj_name} - {event_name}'); plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density [V^2/Hz]'); plt.grid(True); plt.savefig(output_dir / f'periodogram_{subj_name}_{event_name}.pdf'); plt.close()
            
            f, t, Sxx = spectrogram(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 256), noverlap=min(len(ts)//2, 50))
            plt.figure(figsize=(10, 5)); plt.pcolormesh(t, f, 10 * np.log10(np.maximum(Sxx, 1e-10)), shading='gouraud'); plt.title(f'Spectrogram - {subj_name} - {event_name}'); plt.ylabel('Frequency [Hz]'); plt.xlabel('Time [s]'); plt.colorbar(label='Power [dB]'); plt.savefig(output_dir / f'spectrogram_{subj_name}_{event_name}.pdf'); plt.close()

    # Istogrammi
    if not fixations_for_plots.empty and 'duration [ms]' in fixations_for_plots.columns:
        _plot_histogram(fixations_for_plots['duration [ms]'], f"Fixation Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_fixations_{subj_name}_{event_name}.pdf')
    if 'blinks' in data and not data['blinks'].empty and 'duration [ms]' in data['blinks'].columns:
        _plot_histogram(data['blinks']['duration [ms]'], f"Blink Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')
    if 'saccades' in data and not data['saccades'].empty and 'duration [ms]' in data['saccades'].columns:
        _plot_histogram(data['saccades']['duration [ms]'], f"Saccade Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')

    # Grafici di percorso
    if not fixations_for_plots.empty:
        x_col_plot = 'fixation x [normalized]' if 'fixation x [normalized]' in fixations_for_plots.columns else 'norm_pos_x'
        if x_col_plot in fixations_for_plots.columns: # Controlla se esiste una colonna di posizione
            y_col_plot = 'fixation y [normalized]' if 'fixation y [normalized]' in fixations_for_plots.columns else 'norm_pos_y'
            plt.plot(fixations_for_plots[x_col_plot], fixations_for_plots[y_col_plot], marker='o', linestyle='-', color='green')
            plt.title(f"Fixation Path - {subj_name} - {event_name}"); plt.savefig(output_dir / f'path_fixation_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Plot per Figure 5 (Saccade Velocities) ---
    if not saccades_df.empty and 'mean velocity [px/s]' in saccades_df.columns and 'peak velocity [px/s]' in saccades_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(saccades_df.index, saccades_df['mean velocity [px/s]'], label='mean velocity', color='blue')
        plt.plot(saccades_df.index, saccades_df['peak velocity [px/s]'], label='peak velocity', color='orange')
        plt.title(f'Mean and Peak Saccade Velocity - {subj_name} - {event_name}', fontsize=15)
        plt.xlabel('Frames (n)', fontsize=12)
        plt.ylabel('px/s', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f'saccade_velocities_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"ATTENZIONE: Dati di velocità delle saccadi insufficienti per l'evento '{event_name}'.")

    # --- Plot per Figure 6 (Saccade Amplitude) ---
    if not saccades_df.empty and 'amplitude [px]' in saccades_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(saccades_df.index, saccades_df['amplitude [px]'], label='amplitude', color='teal')
        plt.title(f'Saccade Amplitude - {subj_name} - {event_name}', fontsize=15)
        plt.xlabel('Frames (n)', fontsize=12)
        plt.ylabel('px', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f'saccade_amplitude_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"ATTENZIONE: Dati di ampiezza delle saccadi insufficienti per l'evento '{event_name}'.")

    # --- Plot per Figure 7 (Blink Time Series) ---
    if not blinks_df.empty and 'start timestamp [ns]' in blinks_df.columns and 'duration [ms]' in blinks_df.columns:
        # Create a time series for blinks (binary: 0 or 1)
        # Find the overall start and end timestamps for the segment to define the plot range
        all_timestamps = []
        if 'timestamp [ns]' in pupil_df.columns:
            all_timestamps.extend(pupil_df['timestamp [ns]'].tolist())
        if 'start timestamp [ns]' in blinks_df.columns:
            all_timestamps.extend(blinks_df['start timestamp [ns]'].tolist())
        if 'start timestamp [ns]' in saccades_df.columns:
            all_timestamps.extend(saccades_df['start timestamp [ns]'].tolist())
        
        if not all_timestamps:
            print(f"ATTENZIONE: Nessun timestamp disponibile per creare la serie temporale dei blink per l'evento '{event_name}'.")
            return

        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)

        # Create a series with 0s for the entire duration of the segment
        # We need a granular timestamp array to represent each 'frame' or a small interval
        # Let's use the sampling frequency to estimate the number of points needed
        # Assuming SAMPLING_FREQ is the rate for gaze/pupil data, we can use it for time resolution
        duration_s = (max_ts - min_ts) / NS_TO_S
        num_points = int(duration_s * SAMPLING_FREQ)
        
        if num_points <= 0:
            print(f"ATTENZIONE: Durata del segmento troppo breve per generare la serie temporale dei blink per l'evento '{event_name}'.")
            return

        # Create a time axis for the plot (frames or a scaled time)
        time_axis_s = np.linspace(0, duration_s, num_points)
        blink_time_series = np.zeros(num_points)

        for _, row in blinks_df.iterrows():
            blink_start_ns = row['start timestamp [ns]']
            blink_end_ns = blink_start_ns + (row['duration [ms]'] * 1_000_000) # Convert ms to ns

            # Map blink timestamps to the time_axis_s indices
            start_idx = int(((blink_start_ns - min_ts) / NS_TO_S) * SAMPLING_FREQ)
            end_idx = int(((blink_end_ns - min_ts) / NS_TO_S) * SAMPLING_FREQ)

            start_idx = max(0, start_idx)
            end_idx = min(num_points, end_idx) # Ensure index doesn't exceed array bounds

            if start_idx < end_idx:
                blink_time_series[start_idx:end_idx] = 1

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis_s, blink_time_series, drawstyle='steps-post', color='blue')
        plt.title(f'Blink Time Series - {subj_name} - {event_name}', fontsize=15)
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Blink (0 = No, 1 = Yes)', fontsize=12)
        plt.yticks([0, 1]) # Ensure y-axis only shows 0 and 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(-0.1, 1.1) # Add a small buffer for better visualization
        plt.tight_layout()
        plt.savefig(output_dir / f'blink_time_series_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"ATTENZIONE: Dati di blink insufficienti per l'evento '{event_name}'.")


def process_segment(event_row, start_ts, end_ts, all_data, subj_name, output_dir, un_enriched_mode, video_width, video_height):
    """Pipeline di elaborazione principale per un singolo segmento di evento."""
    event_name = event_row.get('name', f"segment_{event_row.name}")
    print(f"--- Elaborazione segmento per l'evento: '{event_name}' ---")
    rec_id = event_row['recording id']
    
    segment_data = filter_data_by_segment(all_data, start_ts, end_ts, rec_id)
    if all(df.empty for name, df in segment_data.items() if name != 'events'):
        print(f"  -> Salto il segmento '{event_name}' perché non ci sono dati nell'intervallo.")
        return None
    
    movements_df = process_gaze_movements(segment_data.get('gaze', pd.DataFrame()), un_enriched_mode)
    results = calculate_summary_features(segment_data, movements_df, subj_name, event_name, un_enriched_mode, video_width, video_height)
    generate_plots(segment_data, movements_df, subj_name, event_name, output_dir, un_enriched_mode)
    return results

def get_video_dimensions(video_path: Path):
    """Ottiene larghezza e altezza di un file video."""
    if not video_path.exists():
        print(f"ATTENZIONE: File video non trovato in {video_path}. Impossibile ottenere le dimensioni per la normalizzazione.")
        return None, None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ATTENZIONE: Impossibile aprire il file video {video_path}.")
        return None, None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def downsample_video(input_file, output_file, input_fps, output_fps):
    """Downsample un file video a un FPS inferiore."""
    cap = cv2.VideoCapture(str(input_file))
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il file video {input_file}")
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
        print(f"Video downsampled a {output_file}")
    else:
        print(f"Errore: Impossibile aprire il file video di output per il downsampling: {output_file}")
    cap.release()
    out.release()

def create_analysis_video(data_dir: Path, output_dir: Path):
    """Crea un video che combina eye tracking, vista esterna e diametro pupillare."""
    print("\nCreazione del video di analisi in corso...")
    
    internal_video_path = data_dir / 'internal.mp4'
    external_video_path = data_dir / 'external.mp4'
    pupillometry_data_path = data_dir / '3d_eye_states.csv'

    if not all([p.exists() for p in [internal_video_path, external_video_path, pupillometry_data_path]]):
        print("Salto la creazione del video: Uno o più file richiesti (internal.mp4, external.mp4, 3d_eye_states.csv) non trovati.")
        return

    try:
        downsampled_video_path = output_dir / 'downsampled_internal_video.mp4'
        if os.path.getsize(internal_video_path) > 0:
            downsample_video(internal_video_path, downsampled_video_path, 200, 40)
        else:
            print(f"Il video interno {internal_video_path} è vuoto, salto il downsampling.")
            return

        pupillometry_data = pd.read_csv(pupillometry_data_path)
        if 'pupil diameter left [mm]' not in pupillometry_data.columns:
            print("Salto la creazione del video: colonna 'pupil diameter left [mm]' non trovata.")
            return
        time_series = pupillometry_data['pupil diameter left [mm]'].values.flatten()

        cap1 = cv2.VideoCapture(str(downsampled_video_path))
        cap2 = cv2.VideoCapture(str(external_video_path))

        if not cap1.isOpened() or not cap2.isOpened():
            print("Errore nell'apertura dei file video per l'animazione.")
            return

        fig, (video_axes1, video_axes2, time_series_axes) = plt.subplots(3, 1, figsize=(10, 8))
        fig.tight_layout(pad=3.0)
        output_video_path = output_dir / 'output_analysis_video.mp4'
        fps = cap1.get(cv2.CAP_PROP_FPS)
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        if not out.isOpened():
            print(f"Errore: Impossibile creare il file video di output: {output_video_path}")
            cap1.release(); cap2.release(); plt.close(fig)
            return

        video_axes1.set_title("Internal View (Eye)"); video_axes1.axis('off')
        video_axes2.set_title("External View"); video_axes2.axis('off')
        time_series_axes.set_title("Pupil Diameter Time Series")
        
        num_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)), len(time_series))

        for i in range(num_frames):
            ret1, frame1 = cap1.read(); ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                print(f"Attenzione: Impossibile leggere il frame {i}. Interruzione della creazione del video.")
                break
            
            video_axes1.clear(); video_axes2.clear(); time_series_axes.clear()
            video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)); video_axes1.axis('off')
            video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)); video_axes2.axis('off')

            window_size = 1000
            idx_start, idx_end = max(0, i - window_size // 2), min(len(time_series), i + window_size // 2)
            idx = np.arange(idx_start, idx_end)
            
            if idx.size > 0:
                time_series_axes.plot(idx, time_series[idx], 'b-')
                time_series_axes.plot(i, time_series[i], 'ro', markersize=10)
                time_series_axes.set_xlim(idx_start, idx_end)
                time_series_axes.set_ylim(np.nanmin(time_series) * 0.95, np.nanmax(time_series) * 1.05)
                time_series_axes.set_xlabel('Frame (n)'); time_series_axes.set_ylabel('Diameter (mm)')

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (3,))
            out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if (i+1) % 100 == 0: print(f"  ...elaborato il frame {i+1}/{num_frames}")

        cap1.release(); cap2.release(); out.release(); plt.close(fig)
        if downsampled_video_path.exists():
            os.remove(downsampled_video_path)
            print(f"Rimosso video temporaneo downsampled: {downsampled_video_path}")
        print(f"Video di analisi salvato in {output_video_path}")

    except Exception as e:
        print(f"Si è verificato un errore inaspettato durante la creazione del video: {e}")
        traceback.print_exc()

def run_analysis(subj_name='subj_01', data_dir_str='./eyetracking_file', output_dir_str='./results', un_enriched_mode=False, generate_video=True):
    """Funzione principale per eseguire l'intera pipeline di analisi basata su segmenti di eventi."""
    pd.options.mode.chained_assignment = None
    data_dir, output_dir = Path(data_dir_str), Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_width, video_height = get_video_dimensions(data_dir / 'external.mp4')
    
    try:
        all_data = load_all_data(data_dir, un_enriched_mode)
    except FileNotFoundError as e:
        print(f"Analisi interrotta. {e}")
        return
        
    events_df = all_data.get('events')
    if events_df is None or events_df.empty:
        print("Errore: events.csv non caricato o vuoto. Impossibile procedere.")
        return
        
    all_results = []
    if len(events_df) > 1:
        print(f"\nTrovati {len(events_df)} eventi, elaborazione di {len(events_df) - 1} segmenti.")
        for i in range(len(events_df) - 1):
            event_row, start_ts, end_ts = events_df.iloc[i], events_df.iloc[i]['timestamp [ns]'], events_df.iloc[i+1]['timestamp [ns]']
            try:
                event_results = process_segment(event_row, start_ts, end_ts, all_data, subj_name, output_dir, un_enriched_mode, video_width, video_height)
                if event_results:
                    all_results.append(event_results)
            except Exception as e:
                print(f"Impossibile elaborare il segmento per l'evento '{event_row.get('name', i)}'. Errore: {e}")
                traceback.print_exc()
    else:
        print("Attenzione: Trovati meno di due eventi. Impossibile elaborare segmenti.")
        
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
        print(f"\nRisultati aggregati salvati in {results_filename}")
    else:
        print("\nNessun risultato di analisi è stato generato.")
        
    if generate_video:
        create_analysis_video(data_dir, output_dir)