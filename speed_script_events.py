import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import traceback
import os
from scipy.signal import welch, spectrogram
from scipy.stats import gaussian_kde

# --- Costants ---
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
    
    gaze_df['fixation id'] = gaze_df['fixation id'].fillna(-1)
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

def get_timestamp_col(df):
    """Ottiene la colonna di timestamp corretta da un dataframe."""
    for col in ['start timestamp [ns]', 'timestamp [ns]']:
        if col in df.columns:
            return col
    return None

def generate_plots(data, movements_df, subj_name, event_name, output_dir: Path, un_enriched_mode: bool, video_width: int, video_height: int):
    """Genera e salva tutti i grafici per l'evento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())
    gaze_enr = data.get('gaze', pd.DataFrame()) # Get enriched gaze data

    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())
    gaze_enr = data.get('gaze', pd.DataFrame()) # Get enriched gaze data
    gaze_not_enr = data.get('gaze_not_enr', pd.DataFrame()) # Aggiungi questa riga per inizializzare gaze_not_enr
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
        pupil_with_gaze_status['gaze detected on surface'] = pupil_with_gaze_status['gaze detected on surface'].fillna(False)

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
    if 'blinks' in data and not data['blinks'].empty and 'duration [ms]' in data['blinks'].columns:
        _plot_histogram(data['blinks']['duration [ms]'], f"Blink Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')
    if 'saccades' in data and not data['saccades'].empty and 'duration [ms]' in data['saccades'].columns:
        _plot_histogram(data['saccades']['duration [ms]'], f"Saccade Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')

    # Grafici di percorso (Fixation Path)
    # Fixation Path - Enriched
    if not un_enriched_mode and not fixations_enr.empty and 'fixation x [normalized]' in fixations_enr.columns:
        fixations_enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        if not fixations_enriched_on_surface.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(fixations_enriched_on_surface['fixation x [normalized]'], fixations_enriched_on_surface['fixation y [normalized]'], marker='o', linestyle='-', color='green')
            plt.title(f"Fixation Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True)
            plt.savefig(output_dir / f'path_fixation_enriched_{subj_name}_{event_name}.pdf')
            plt.close()
        else:
            print(f"ATTENZIONE: Nessuna fissazione arricchita rilevata sulla superficie per l'evento '{event_name}'. Nessun grafico generato.")

    # Fixation Path - Un-enriched
    if not fixations_not_enr.empty and 'fixation x [px]' in fixations_not_enr.columns:
        plt.figure(figsize=(10, 6))
        
        x_coords_px = fixations_not_enr['fixation x [px]']
        y_coords_px = fixations_not_enr['fixation y [px]']

        # Normalizzazione se le dimensioni del video sono disponibili
        if video_width and video_height and video_width > 0 and video_height > 0:
            x_coords_norm = x_coords_px / video_width
            y_coords_norm = y_coords_px / video_height
            xlabel_text, ylabel_text = 'Normalized X', 'Normalized Y'
        else:
            x_coords_norm, y_coords_norm = x_coords_px, y_coords_px
            xlabel_text, ylabel_text = 'Pixel X', 'Pixel Y'
            print("ATTENZIONE: Le coordinate delle fissazioni non arricchite sono in pixel e le dimensioni del video non sono disponibili per la normalizzazione.")

        plt.plot(x_coords_norm, y_coords_norm, marker='o', linestyle='-', color='purple')
        plt.title(f"Fixation Path (Un-enriched) - {subj_name} - {event_name}"); plt.xlabel(xlabel_text); plt.ylabel(ylabel_text); plt.grid(True)
        plt.savefig(output_dir / f'path_fixation_not_enriched_{subj_name}_{event_name}.pdf')
        plt.close()
    else:
        print(f"ATTENZIONE: Dati di fissazione non arricchiti insufficienti per l'evento '{event_name}'. Nessun grafico generato.")
    
    # Grafici di percorso (Gaze Path)
    gaze_for_plots = pd.DataFrame()
    if not un_enriched_mode and not gaze_enr.empty and 'gaze position on surface x [normalized]' in gaze_enr.columns:
        gaze_for_plots = gaze_enr[gaze_enr['gaze detected on surface'] == True].copy()
        if not gaze_for_plots.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(gaze_for_plots['gaze position on surface x [normalized]'], gaze_for_plots['gaze position on surface y [normalized]'], marker='.', linestyle='-', color='red', alpha=0.5)
            plt.title(f"Gaze Path - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True)
            plt.savefig(output_dir / f'path_gaze_enriched_{subj_name}_{event_name}.pdf')
            plt.close()

    # Gaze Path - Un-enriched
    plt.figure(figsize=(10, 6))
    plt.plot(gaze_not_enr['gaze x [px]'], gaze_not_enr['gaze y [px]'], marker='.', linestyle='-', color='red', alpha=0.5)
    plt.title(f"Gaze Path - {subj_name} - {event_name}"); plt.xlabel('X'); plt.ylabel('Y'); plt.grid(True)
    plt.savefig(output_dir / f'path_gaze_not_enriched_{subj_name}_{event_name}.pdf')
    plt.close()
    
# --- Grafici di percorso (Gaze Path) ---
    if not un_enriched_mode and not gaze_enr.empty and 'gaze position on surface x [normalized]' in gaze_enr.columns:
        gaze_on_surface = gaze_enr[gaze_enr['gaze detected on surface'] == True]
        if not gaze_on_surface.empty:
            plt.figure(figsize=(10, 6)); plt.plot(gaze_on_surface['gaze position on surface x [normalized]'], gaze_on_surface['gaze position on surface y [normalized]'], marker='.', linestyle='-', color='red', alpha=0.5); plt.title(f"Gaze Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True); plt.savefig(output_dir / f'path_gaze_enriched_{subj_name}_{event_name}.pdf'); plt.close()
    
    # MODIFICA QUI: Aggiunto fallback a coordinate [norm]
    if not gaze_not_enr.empty:
        x_col, y_col, coords_type = None, None, None
        if 'gaze position x [px]' in gaze_not_enr.columns and 'gaze position y [px]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze position x [px]', 'gaze position y [px]', 'pixel'
        elif 'gaze x [norm]' in gaze_not_enr.columns and 'gaze y [norm]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze x [norm]', 'gaze y [norm]', 'normalized'
        
        if x_col:
            plt.figure(figsize=(10, 6))
            x, y = gaze_not_enr[x_col], gaze_not_enr[y_col]
            
            if coords_type == 'pixel' and video_width and video_height:
                x, y = x / video_width, y / video_height
                xlabel, ylabel = 'Normalized X', 'Normalized Y'
            elif coords_type == 'normalized':
                 xlabel, ylabel = 'Normalized X', 'Normalized Y'
            else:
                xlabel, ylabel = 'Pixel X', 'Pixel Y'
                
            plt.plot(x, y, marker='.', linestyle='-', color='blue', alpha=0.5)
            plt.title(f"Gaze Path (Un-enriched) - {subj_name} - {event_name}")
            plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True)
            plt.savefig(output_dir / f'path_gaze_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()


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


    # --- Heatmaps di densità (come Figura 2) ---
    # Questa sezione genera heatmap che mostrano la densità dei punti di fissazione e di sguardo,
    # sia per dati arricchiti (normalizzati) che non arricchiti (pixel o normalizzati da pixel).

    # Heatmap delle Fissazioni (Arricchite, su superficie)
    if not un_enriched_mode and not fixations_enr.empty and 'fixation x [normalized]' in fixations_enr.columns:
        fixations_enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True].copy()
        if not fixations_enriched_on_surface.empty:
            x = fixations_enriched_on_surface['fixation x [normalized]'].dropna()
            y = fixations_enriched_on_surface['fixation y [normalized]'].dropna()
            
            if len(x) > 2:
                try:
                    k = gaussian_kde(np.vstack([x, y]))
                    xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                    plt.figure(figsize=(10, 8))
                    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds', alpha=0.8)
                    plt.plot(x, y, 'k.', markersize=4, alpha=0.4)
                    plt.title(f"Fixation Density Heatmap (Enriched) - {subj_name} - {event_name}", fontsize=15)
                    plt.xlabel('Normalized X'); plt.ylabel('Normalized Y')
                    plt.xlim(0, 1); plt.ylim(0, 1)
                    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
                    plt.savefig(output_dir / f'heatmap_fixation_enriched_{subj_name}_{event_name}.pdf'); plt.close()
                except np.linalg.LinAlgError:
                    print(f"ATTENZIONE: Impossibile generare heatmap per fissazioni arricchite (matrice singolare).")

    # Heatmap delle Fissazioni (Non Arricchite)
    if not fixations_not_enr.empty and 'fixation x [px]' in fixations_not_enr.columns:
        x_px = fixations_not_enr['fixation x [px]'].dropna()
        y_px = fixations_not_enr['fixation y [px]'].dropna()

        if len(x_px) > 2:
            if video_width and video_height and video_width > 0 and video_height > 0:
                x_coords, y_coords = x_px / video_width, y_px / video_height
                xlabel_text, ylabel_text = 'Normalized X', 'Normalized Y'
                xlim, ylim = (0, 1), (0, 1)
            else:
                x_coords, y_coords = x_px, y_px
                xlabel_text, ylabel_text = 'Pixel X', 'Pixel Y'
                xlim, ylim = (0, x_coords.max()), (0, y_coords.max())

            try:
                k = gaussian_kde(np.vstack([x_coords, y_coords]))
                xi, yi = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
                zi = k(np.vstack([xi.flatten(), yi.flatten()]))

                plt.figure(figsize=(10, 8))
                plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds', alpha=0.8)
                plt.plot(x_coords, y_coords, 'k.', markersize=4, alpha=0.4)
                plt.title(f"Fixation Density Heatmap (Un-enriched) - {subj_name} - {event_name}", fontsize=15)
                plt.xlabel(xlabel_text); plt.ylabel(ylabel_text)
                plt.xlim(xlim); plt.ylim(ylim)
                plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
                plt.savefig(output_dir / f'heatmap_fixation_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()
            except np.linalg.LinAlgError:
                print(f"ATTENZIONE: Impossibile generare heatmap per fissazioni non arricchite (matrice singolare).")

    # Heatmap dello Sguardo (Arricchito, su superficie)
    if not un_enriched_mode and not gaze_enr.empty and 'gaze position on surface x [normalized]' in gaze_enr.columns:
        gaze_on_surface = gaze_enr[gaze_enr['gaze detected on surface'] == True].copy()
        if not gaze_on_surface.empty:
            x = gaze_on_surface['gaze position on surface x [normalized]'].dropna()
            y = gaze_on_surface['gaze position on surface y [normalized]'].dropna()

            if len(x) > 2:
                try:
                    k = gaussian_kde(np.vstack([x, y]))
                    xi, yi = np.mgrid[0:1:100j, 0:1:100j]
                    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
                    
                    plt.figure(figsize=(10, 8))
                    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds', alpha=0.8)
                    plt.plot(x, y, 'k.', markersize=2, alpha=0.2)
                    plt.title(f"Gaze Density Heatmap (Enriched) - {subj_name} - {event_name}", fontsize=15)
                    plt.xlabel('Normalized X'); plt.ylabel('Normalized Y')
                    plt.xlim(0, 1); plt.ylim(0, 1)
                    plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
                    plt.savefig(output_dir / f'heatmap_gaze_enriched_{subj_name}_{event_name}.pdf'); plt.close()
                except np.linalg.LinAlgError:
                    print(f"ATTENZIONE: Impossibile generare heatmap per sguardi arricchiti (matrice singolare).")

    # Heatmap dello Sguardo (Non Arricchito)
    
    x_px = gaze_not_enr['gaze x [px]'].dropna()
    y_px = gaze_not_enr['gaze y [px]'].dropna()
    
    if len(x_px) > 2:
        if video_width and video_height and video_width > 0 and video_height > 0:
            x_coords, y_coords = x_px / video_width, y_px / video_height
            xlabel_text, ylabel_text = 'Normalized X', 'Normalized Y'
            xlim, ylim = (0, 1), (0, 1)
        else:
            x_coords, y_coords = x_px, y_px
            xlabel_text, ylabel_text = 'Pixel X', 'Pixel Y'
            xlim, ylim = (0, x_coords.max()), (0, y_coords.max())

        try:
            k = gaussian_kde(np.vstack([x_coords, y_coords]))
            xi, yi = np.mgrid[xlim[0]:xlim[1]:100j, ylim[0]:ylim[1]:100j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            plt.figure(figsize=(10, 8))
            plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds', alpha=0.8)
            plt.plot(x_coords, y_coords, 'k.', markersize=2, alpha=0.2)
            plt.title(f"Gaze Density Heatmap (Un-enriched) - {subj_name} - {event_name}", fontsize=15)
            plt.xlabel(xlabel_text); plt.ylabel(ylabel_text)
            plt.xlim(xlim); plt.ylim(ylim)
            plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
            plt.savefig(output_dir / f'heatmap_gaze_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()
        except np.linalg.LinAlgError:
            print(f"ATTENZIONE: Impossibile generare heatmap per sguardi non arricchiti (matrice singolare).")

def generate_plots_2(data, movements_df, subj_name, event_name, output_dir: Path, un_enriched_mode: bool, video_width: int, video_height: int):
    """Genera e salva tutti i grafici per l'evento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fixations_enr, fixations_not_enr = data.get('fixations_enr', pd.DataFrame()), data.get('fixations_not_enr', pd.DataFrame())
    gaze_enr, gaze_not_enr = data.get('gaze', pd.DataFrame()), data.get('gaze_not_enr', pd.DataFrame())
    pupil_df, blinks_df, saccades_df = data.get('pupil', pd.DataFrame()), data.get('blinks', pd.DataFrame()), data.get('saccades', pd.DataFrame())

    # --- Plot Pupillometria (Originale: Destro e Sinistro separati) ---
    if not pupil_df.empty and 'pupil diameter left [mm]' in pupil_df.columns and not un_enriched_mode:
        gaze_enr_unique_ts = gaze_enr.drop_duplicates(subset=['timestamp [ns]']).copy()
        pupil_with_gaze_status = pd.merge(pupil_df, gaze_enr_unique_ts[['timestamp [ns]', 'gaze detected on surface']], on='timestamp [ns]', how='left')
        pupil_with_gaze_status['gaze detected on surface'] = pupil_with_gaze_status['gaze detected on surface'].fillna(False)

        if not pupil_with_gaze_status.empty:
            plt.figure(figsize=(12, 6))
            if 'pupil diameter left [mm]' in pupil_with_gaze_status.columns:
                plt.plot(pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S, pupil_with_gaze_status['pupil diameter left [mm]'], label='Pupil Diameter Left [mm]', color='blue', alpha=0.7)
            if 'pupil diameter right [mm]' in pupil_with_gaze_status.columns:
                plt.plot(pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S, pupil_with_gaze_status['pupil diameter right [mm]'], label='Pupil Diameter Right [mm]', color='purple', alpha=0.7)

            timestamps_seconds = pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S
            epsilon = (timestamps_seconds.iloc[1] - timestamps_seconds.iloc[0]) / 2 if len(timestamps_seconds) > 1 else 0.01
            # Background coloring
            start_time = timestamps_seconds.iloc[0]
            current_status = pupil_with_gaze_status.iloc[0]['gaze detected on surface']
            for i in range(1, len(pupil_with_gaze_status)):
                if pupil_with_gaze_status.iloc[i]['gaze detected on surface'] != current_status:
                    end_time = timestamps_seconds.iloc[i]
                    color = 'lightgreen' if current_status else 'lightcoral'
                    plt.axvspan(start_time - epsilon, end_time - epsilon, facecolor=color, alpha=0.3, lw=0)
                    start_time = end_time
                    current_status = pupil_with_gaze_status.iloc[i]['gaze detected on surface']
            # Color last segment
            color = 'lightgreen' if current_status else 'lightcoral'
            plt.axvspan(start_time - epsilon, timestamps_seconds.iloc[-1] + epsilon, facecolor=color, alpha=0.3, lw=0)


            plt.title(f"Pupil Diameter with Gaze On Surface - {subj_name} - {event_name}", fontsize=15)
            plt.xlabel('Time [s]'); plt.ylabel('Pupil Diameter [mm]'); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
            plt.savefig(output_dir / f'pupil_diameter_gaze_surface_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Plot Pupillometria (Media di Destro e Sinistro) ---
    if not pupil_df.empty and 'pupil diameter left [mm]' in pupil_df.columns and 'pupil diameter right [mm]' in pupil_df.columns and not un_enriched_mode:
        pupil_df_mean = pupil_df.copy()
        pupil_df_mean['pupil_diameter_mean'] = pupil_df_mean[['pupil diameter left [mm]', 'pupil diameter right [mm]']].mean(axis=1)

        gaze_enr_unique_ts = gaze_enr.drop_duplicates(subset=['timestamp [ns]']).copy()
        pupil_with_gaze_status = pd.merge(pupil_df_mean, gaze_enr_unique_ts[['timestamp [ns]', 'gaze detected on surface']], on='timestamp [ns]', how='left')
        pupil_with_gaze_status['gaze detected on surface'] = pupil_with_gaze_status['gaze detected on surface'].fillna(False)
        pupil_with_gaze_status.dropna(subset=['pupil_diameter_mean'], inplace=True)
        pupil_with_gaze_status.reset_index(drop=True, inplace=True)


        if not pupil_with_gaze_status.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S, pupil_with_gaze_status['pupil_diameter_mean'], label='Mean Pupil Diameter [mm]', color='black', alpha=0.8)

            timestamps_seconds = pupil_with_gaze_status['timestamp [ns]'] / NS_TO_S
            epsilon = (timestamps_seconds.diff().mean() / 2) if len(timestamps_seconds) > 1 else 0.01
            
            # Background coloring
            start_time = timestamps_seconds.iloc[0]
            current_status = pupil_with_gaze_status.iloc[0]['gaze detected on surface']
            for i in range(1, len(pupil_with_gaze_status)):
                if pupil_with_gaze_status.iloc[i]['gaze detected on surface'] != current_status:
                    end_time = timestamps_seconds.iloc[i]
                    color = 'lightgreen' if current_status else 'lightcoral'
                    plt.axvspan(start_time - epsilon, end_time - epsilon, facecolor=color, alpha=0.3, lw=0)
                    start_time = end_time
                    current_status = pupil_with_gaze_status.iloc[i]['gaze detected on surface']
            # Color last segment
            color = 'lightgreen' if current_status else 'lightcoral'
            plt.axvspan(start_time - epsilon, timestamps_seconds.iloc[-1] + epsilon, facecolor=color, alpha=0.3, lw=0)

            plt.title(f"Mean Pupil Diameter with Gaze On Surface - {subj_name} - {event_name}", fontsize=15)
            plt.xlabel('Time [s]'); plt.ylabel('Mean Pupil Diameter [mm]'); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
            plt.savefig(output_dir / f'pupil_diameter_mean_gaze_surface_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Periodogramma e Spettrogramma ---
    if 'pupil' in data and not data['pupil'].empty and 'pupil diameter left [mm]' in data['pupil'].columns:
        ts = data['pupil']['pupil diameter left [mm]'].dropna().to_numpy()
        if len(ts) > SAMPLING_FREQ:
            try:
                freqs, Pxx = welch(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 100))
                plt.figure(figsize=(10, 5)); plt.semilogy(freqs, Pxx); plt.title(f'Periodogram - {subj_name} - {event_name}'); plt.xlabel('Frequency [Hz]'); plt.ylabel('Power Spectral Density [V^2/Hz]'); plt.grid(True); plt.savefig(output_dir / f'periodogram_{subj_name}_{event_name}.pdf'); plt.close()
                
                f, t, Sxx = spectrogram(ts, fs=SAMPLING_FREQ, nperseg=min(len(ts), 256), noverlap=min(len(ts)//2, 50))
                plt.figure(figsize=(10, 5)); plt.pcolormesh(t, f, 10 * np.log10(np.maximum(Sxx, 1e-10)), shading='gouraud'); plt.title(f'Spectrogram - {subj_name} - {event_name}'); plt.ylabel('Frequency [Hz]'); plt.xlabel('Time [s]'); plt.colorbar(label='Power [dB]'); plt.savefig(output_dir / f'spectrogram_{subj_name}_{event_name}.pdf'); plt.close()
            except Exception as e:
                print(f"ATTENZIONE: Impossibile generare periodogramma/spettrogramma per '{event_name}'. Errore: {e}")

    # --- Istogrammi separati per Fissazioni (Enriched e Not Enriched) ---
    if not un_enriched_mode and not fixations_enr.empty and 'duration [ms]' in fixations_enr.columns:
        fixations_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        if not fixations_on_surface.empty:
            _plot_histogram(fixations_on_surface['duration [ms]'], f"Fixation Duration (Enriched on Surface) - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_fixations_enriched_{subj_name}_{event_name}.pdf')
    if not fixations_not_enr.empty and 'duration [ms]' in fixations_not_enr.columns:
        _plot_histogram(fixations_not_enr['duration [ms]'], f"Fixation Duration (Un-enriched) - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_fixations_not_enriched_{subj_name}_{event_name}.pdf')

    if 'blinks' in data and not data['blinks'].empty:
        _plot_histogram(data['blinks']['duration [ms]'], f"Blink Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_blinks_{subj_name}_{event_name}.pdf')
    if 'saccades' in data and not data['saccades'].empty:
        _plot_histogram(data['saccades']['duration [ms]'], f"Saccade Duration Histogram - {subj_name} - {event_name}", "Duration [ms]", output_dir / f'hist_saccades_{subj_name}_{event_name}.pdf')

    # --- Grafici di percorso (Fixation Path) ---
    if not un_enriched_mode and not fixations_enr.empty and 'fixation x [normalized]' in fixations_enr.columns:
        enriched_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        if not enriched_on_surface.empty:
            plt.figure(figsize=(10, 6)); plt.plot(enriched_on_surface['fixation x [normalized]'], enriched_on_surface['fixation y [normalized]'], marker='o', linestyle='-', color='green'); plt.title(f"Fixation Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True); plt.savefig(output_dir / f'path_fixation_enriched_{subj_name}_{event_name}.pdf'); plt.close()
    if not fixations_not_enr.empty and 'fixation x [px]' in fixations_not_enr.columns:
        plt.figure(figsize=(10, 6))
        x, y = (fixations_not_enr['fixation x [px]'] / video_width, fixations_not_enr['fixation y [px]'] / video_height) if video_width and video_height else (fixations_not_enr['fixation x [px]'], fixations_not_enr['fixation y [px]'])
        xlabel, ylabel = ('Normalized X', 'Normalized Y') if video_width and video_height else ('Pixel X', 'Pixel Y')
        plt.plot(x, y, marker='o', linestyle='-', color='purple'); plt.title(f"Fixation Path (Un-enriched) - {subj_name} - {event_name}"); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True); plt.savefig(output_dir / f'path_fixation_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Grafici di percorso (Gaze Path) ---
    if not un_enriched_mode and not gaze_enr.empty and 'gaze position on surface x [normalized]' in gaze_enr.columns:
        gaze_on_surface = gaze_enr[gaze_enr['gaze detected on surface'] == True]
        if not gaze_on_surface.empty:
            plt.figure(figsize=(10, 6)); plt.plot(gaze_on_surface['gaze position on surface x [normalized]'], gaze_on_surface['gaze position on surface y [normalized]'], marker='.', linestyle='-', color='red', alpha=0.5); plt.title(f"Gaze Path (Enriched) - {subj_name} - {event_name}"); plt.xlabel('Normalized X'); plt.ylabel('Normalized Y'); plt.grid(True); plt.savefig(output_dir / f'path_gaze_enriched_{subj_name}_{event_name}.pdf'); plt.close()
    
    if not gaze_not_enr.empty and 'gaze position x [px]' in gaze_not_enr.columns:
        plt.figure(figsize=(10, 6))
        x, y = (gaze_not_enr['gaze position x [px]'] / video_width, gaze_not_enr['gaze position y [px]'] / video_height) if video_width and video_height else (gaze_not_enr['gaze position x [px]'], gaze_not_enr['gaze position y [px]'])
        xlabel, ylabel = ('Normalized X', 'Normalized Y') if video_width and video_height else ('Pixel X', 'Pixel Y')
        plt.plot(x, y, marker='.', linestyle='-', color='blue', alpha=0.5); plt.title(f"Gaze Path (Un-enriched) - {subj_name} - {event_name}"); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True); plt.savefig(output_dir / f'path_gaze_not_enriched_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Plot Saccadi e Blink ---
    if not saccades_df.empty and 'mean velocity [px/s]' in saccades_df.columns:
        plt.figure(figsize=(10, 6)); plt.plot(saccades_df.index, saccades_df['mean velocity [px/s]'], label='mean velocity'); plt.plot(saccades_df.index, saccades_df['peak velocity [px/s]'], label='peak velocity'); plt.title(f'Mean and Peak Saccade Velocity - {subj_name} - {event_name}'); plt.xlabel('Frames (n)'); plt.ylabel('px/s'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(output_dir / f'saccade_velocities_{subj_name}_{event_name}.pdf'); plt.close()
    if not saccades_df.empty and 'amplitude [px]' in saccades_df.columns:
        plt.figure(figsize=(10, 6)); plt.plot(saccades_df.index, saccades_df['amplitude [px]'], label='amplitude', color='teal'); plt.title(f'Saccade Amplitude - {subj_name} - {event_name}'); plt.xlabel('Frames (n)'); plt.ylabel('px'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(output_dir / f'saccade_amplitude_{subj_name}_{event_name}.pdf'); plt.close()
    if not blinks_df.empty:
        all_timestamps = pd.concat([df[get_timestamp_col(df)] for df in [pupil_df, blinks_df, saccades_df] if not df.empty and get_timestamp_col(df) is not None]).dropna()
        if not all_timestamps.empty:
            min_ts, max_ts = all_timestamps.min(), all_timestamps.max()
            duration_s = (max_ts - min_ts) / NS_TO_S
            num_points = int(duration_s * SAMPLING_FREQ)
            if num_points > 0:
                time_axis_s = np.linspace(0, duration_s, num_points)
                blink_time_series = np.zeros(num_points)
                for _, row in blinks_df.iterrows():
                    start_idx = max(0, int(((row['start timestamp [ns]'] - min_ts) / NS_TO_S) * SAMPLING_FREQ))
                    end_idx = min(num_points, int(((row['start timestamp [ns]'] + row['duration [ms]'] * 1e6 - min_ts) / NS_TO_S) * SAMPLING_FREQ))
                    if start_idx < end_idx: blink_time_series[start_idx:end_idx] = 1
                plt.figure(figsize=(12, 4)); plt.plot(time_axis_s, blink_time_series, drawstyle='steps-post'); plt.title(f'Blink Time Series - {subj_name} - {event_name}'); plt.xlabel('Time [s]'); plt.ylabel('Blink (0=No, 1=Yes)'); plt.yticks([0, 1]); plt.grid(axis='y'); plt.ylim(-0.1, 1.1); plt.tight_layout(); plt.savefig(output_dir / f'blink_time_series_{subj_name}_{event_name}.pdf'); plt.close()

    # --- Heatmap con gestione errori ---
    def plot_heatmap(x, y, title, output_path, xlim=None, ylim=None, xlabel='X', ylabel='Y'):
        x = x.dropna(); y = y.dropna()
        valid_points = pd.concat([x, y], axis=1).dropna()
        if len(valid_points) < 3:
            print(f"ATTENZIONE: Dati insufficienti (meno di 3 punti validi) per generare heatmap: {title}")
            return
        try:
            k = gaussian_kde(valid_points.values.T)
            xi, yi = np.mgrid[valid_points.iloc[:,0].min():valid_points.iloc[:,0].max():100j, valid_points.iloc[:,1].min():valid_points.iloc[:,1].max():100j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            plt.figure(figsize=(10, 8)); plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='Reds'); plt.plot(valid_points.iloc[:,0], valid_points.iloc[:,1], 'k.', markersize=2, alpha=0.2); plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
            if xlim: plt.xlim(xlim)
            if ylim: plt.ylim(ylim)
            plt.grid(True, linestyle='--'); plt.tight_layout(); plt.savefig(output_path); plt.close()
        except np.linalg.LinAlgError:
            print(f"ATTENZIONE: Impossibile generare heatmap per '{title}' (matrice singolare).")
        except Exception as e:
            print(f"ATTENzione: Errore imprevisto durante la generazione di heatmap per '{title}': {e}")

    # Heatmap Fissazioni Arricchite
    if not un_enriched_mode and not fixations_enr.empty:
        fixations_on_surface = fixations_enr[fixations_enr['fixation detected on surface'] == True]
        plot_heatmap(fixations_on_surface['fixation x [normalized]'], fixations_on_surface['fixation y [normalized]'], f"Fixation Density Heatmap (Enriched) - {subj_name} - {event_name}", output_dir / f'heatmap_fixation_enriched_{subj_name}_{event_name}.pdf', xlim=(0, 1), ylim=(0, 1), xlabel='Normalized X', ylabel='Normalized Y')
    # Heatmap Fissazioni Non Arricchite
    if not fixations_not_enr.empty and 'fixation x [px]' in fixations_not_enr.columns:
        x_coords, y_coords = (fixations_not_enr['fixation x [px]'] / video_width, fixations_not_enr['fixation y [px]'] / video_height) if video_width and video_height else (fixations_not_enr['fixation x [px]'], fixations_not_enr['fixation y [px]'])
        xlim, ylim = ((0, 1), (0, 1)) if video_width and video_height else (None, None)
        xlabel, ylabel = ('Normalized X', 'Normalized Y') if video_width and video_height else ('Pixel X', 'Pixel Y')
        plot_heatmap(x_coords, y_coords, f"Fixation Density Heatmap (Un-enriched) - {subj_name} - {event_name}", output_dir / f'heatmap_fixation_not_enriched_{subj_name}_{event_name}.pdf', xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)

    # Heatmap Sguardo Arricchito
    if not un_enriched_mode and not gaze_enr.empty:
        gaze_on_surface = gaze_enr[gaze_enr['gaze detected on surface'] == True]
        plot_heatmap(gaze_on_surface['gaze position on surface x [normalized]'], gaze_on_surface['gaze position on surface y [normalized]'], f"Gaze Density Heatmap (Enriched) - {subj_name} - {event_name}", output_dir / f'heatmap_gaze_enriched_{subj_name}_{event_name}.pdf', xlim=(0, 1), ylim=(0, 1), xlabel='Normalized X', ylabel='Normalized Y')

    # Heatmap Sguardo Non Arricchito
    # MODIFICA QUI: Aggiunto fallback a coordinate [norm]
    if not gaze_not_enr.empty:
        x_col, y_col, coords_type = None, None, None
        if 'gaze position x [px]' in gaze_not_enr.columns and 'gaze position y [px]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze position x [px]', 'gaze position y [px]', 'pixel'
        elif 'gaze x [norm]' in gaze_not_enr.columns and 'gaze y [norm]' in gaze_not_enr.columns:
            x_col, y_col, coords_type = 'gaze x [norm]', 'gaze y [norm]', 'normalized'
            
        if x_col:
            x_coords, y_coords = gaze_not_enr[x_col], gaze_not_enr[y_col]
            
            if coords_type == 'pixel' and video_width and video_height:
                x_coords, y_coords = x_coords / video_width, y_coords / video_height
                xlim, ylim, xlabel, ylabel = (0, 1), (0, 1), 'Normalized X', 'Normalized Y'
            elif coords_type == 'normalized':
                xlim, ylim, xlabel, ylabel = (0, 1), (0, 1), 'Normalized X', 'Normalized Y'
            else:
                xlim, ylim, xlabel, ylabel = None, None, 'Pixel X', 'Pixel Y'
                
            plot_heatmap(x_coords, y_coords, f"Gaze Density Heatmap (Un-enriched) - {subj_name} - {event_name}", output_dir / f'heatmap_gaze_not_enriched_{subj_name}_{event_name}.pdf', xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
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
    generate_plots(segment_data, movements_df, subj_name, event_name, output_dir, un_enriched_mode, video_width, video_height)
    generate_plots_2(segment_data, movements_df, subj_name, event_name, output_dir, un_enriched_mode, video_width, video_height)

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
    """
    Crea un video di analisi che combina eye tracking, vista esterna e diametro pupillare.
    VERSIONE MODIFICATA per sincronizzazione temporale e grafici migliorati.
    """
    print("\nCreazione del video di analisi in corso...")

    # --- Percorsi dei file ---
    internal_video_path = data_dir / 'internal.mp4'
    external_video_path = data_dir / 'external.mp4'
    pupil_data_path = data_dir / '3d_eye_states.csv'
    blinks_data_path = data_dir / 'blinks.csv'

    if not all([p.exists() for p in [internal_video_path, external_video_path, pupil_data_path, blinks_data_path]]):
        print("Salto la creazione del video: file richiesti non trovati (internal.mp4, external.mp4, 3d_eye_states.csv, blinks.csv).")
        return

    try:
        # --- Caricamento e preparazione dati ---
        pupil_data = pd.read_csv(pupil_data_path)
        blinks_data = pd.read_csv(blinks_data_path)

        # Trova il timestamp iniziale per normalizzare l'asse temporale
        t0 = pupil_data['timestamp [ns]'].min()
        
        # Converti i timestamp in secondi dall'inizio
        pupil_data['time_sec'] = (pupil_data['timestamp [ns]'] - t0) / 1e9
        blinks_data['start_sec'] = (blinks_data['start timestamp [ns]'] - t0) / 1e9
        blinks_data['end_sec'] = blinks_data['start_sec'] + (blinks_data['duration [ms]'] / 1000)
        
        # *** NUOVO: Calcola la media dei diametri delle pupille ***
        pupil_data['pupil_diameter_mean'] = pupil_data[['pupil diameter left [mm]', 'pupil diameter right [mm]']].mean(axis=1)


        # --- Setup Video ---
        cap1 = cv2.VideoCapture(str(internal_video_path))
        cap2 = cv2.VideoCapture(str(external_video_path))
        fps = cap1.get(cv2.CAP_PROP_FPS)

        if not cap1.isOpened() or not cap2.isOpened():
            print("Errore nell'apertura dei file video per l'animazione.")
            return

        # --- Setup Grafico ---
        fig, (video_axes1, video_axes2, ts_axes) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 3, 2]})
        fig.tight_layout(pad=4.0)
        
        # Plotta l'intera serie temporale una sola volta
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil diameter left [mm]'], color='dodgerblue', alpha=0.8, label='Pupilla Sinistra')
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil diameter right [mm]'], color='orchid', alpha=0.8, label='Pupilla Destra')
        # *** NUOVO: Aggiungi la linea della media al grafico ***
        ts_axes.plot(pupil_data['time_sec'], pupil_data['pupil_diameter_mean'], color='black', linestyle='--', lw=1.5, label='Media Pupille')


        # Aggiungi le strisce rosse per i blink
        for _, blink in blinks_data.iterrows():
            ts_axes.axvspan(blink['start_sec'], blink['end_sec'], color='red', alpha=0.4, lw=0)

        ts_axes.set_title("Pupil Diameter Over Time", fontsize=16)
        ts_axes.set_xlabel("Time (s)")
        ts_axes.set_ylabel("Diameter (mm)")
        ts_axes.legend()
        ts_axes.grid(True, linestyle='--', alpha=0.6)
        
        # Imposta i limiti dell'asse per evitare che cambino
        max_time = pupil_data['time_sec'].max()
        ts_axes.set_xlim(0, max_time)
        min_pupil = min(pupil_data['pupil diameter left [mm]'].min(), pupil_data['pupil diameter right [mm]'].min())
        max_pupil = max(pupil_data['pupil diameter left [mm]'].max(), pupil_data['pupil diameter right [mm]'].max())
        ts_axes.set_ylim(min_pupil * 0.95, max_pupil * 1.05)

        # Aggiungi la linea verticale che indicherà il tempo corrente
        time_indicator = ts_axes.axvline(x=0, color='red', linestyle='-', lw=2)

        # --- Setup Video Writer ---
        output_video_path = output_dir / 'output_analysis_video.mp4'
        w, h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        if not out.isOpened():
            print(f"Errore: Impossibile creare il file video di output: {output_video_path}")
            cap1.release(); cap2.release(); plt.close(fig)
            return

        # --- Loop di Creazione Video ---
        frame_count = 0
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            # Sincronizza tramite il timestamp del video
            current_video_time_sec = cap1.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Aggiorna le viste video
            video_axes1.clear(); video_axes2.clear()
            video_axes1.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)); video_axes1.axis('off'); video_axes1.set_title("Internal View (Eye)")
            video_axes2.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)); video_axes2.axis('off'); video_axes2.set_title("External View")
            
            # Aggiorna solo la posizione della linea verticale
            time_indicator.set_xdata([current_video_time_sec])

            # Cattura il frame del grafico e scrivilo nel video
            fig.canvas.draw()
            img_buf = fig.canvas.buffer_rgba()
            img = np.array(img_buf)
            out.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  ...elaborato frame {frame_count} ({current_video_time_sec:.2f}s)")

        # --- Pulizia ---
        cap1.release(); cap2.release(); out.release(); plt.close(fig)
        print(f"Video di analisi salvato in {output_video_path}")

    except Exception as e:
        print(f"Si è verificato un errore inaspettato durante la creazione del video: {e}")
        import traceback
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