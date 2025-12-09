import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2
import shutil
from pathlib import Path
import os
from scipy.stats import gaussian_kde
from PIL import Image, ImageTk
from tqdm import tqdm
import threading
import time
import queue  # Aggiunto per Thread Safety

# --- CONFIGURATION ---
PLOT_DPI = 150
GAZE_COLOR_DEFAULT = (255, 0, 0)  # Blue (BGR) per Raw
GAZE_COLOR_IN = (0, 255, 0)       # Green (BGR) per Surface In
GAZE_COLOR_OUT = (0, 0, 255)      # Red (BGR) per Surface Out

PUPIL_LEFT_COLOR = 'blue'
PUPIL_RIGHT_COLOR = 'orange'
PUPIL_LEFT_COLOR_BGR = (255, 255, 0)  # Cyan for video plot
PUPIL_RIGHT_COLOR_BGR = (0, 165, 255) # Orange for video plot

# Heatmap Colormaps
CMAP_RAW = 'winter' # Toni freddi per i dati RAW
CMAP_ENRICHED = 'jet' # Classico o toni caldi per i dati Enrichment/Surface

# ==============================================================================
# 1. FILE MANAGEMENT AND DATA PREPARATION
# ==============================================================================

def prepare_working_directory(data_dir, enrichment_dir, output_dir):
    """
    Creates the 'files' folder.
    ALWAYS creates *_raw.csv files from Data.
    IF enrichment is present, creates *_enriched.csv files.
    Also manages a 'gaze.csv' / 'fixations.csv' default link for the video generator (preferring enriched).
    
    Returns: files_dir (Path), warnings (list of strings)
    """
    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    
    # Map of required files and their default source
    files_map = [
        ('events.csv', data_dir, False),
        ('blinks.csv', data_dir, True),
        ('3d_eye_states.csv', data_dir, True),
        ('world_timestamps.csv', data_dir, False),
        ('saccades.csv', data_dir, True)
    ]
    
    # Video handling: Look for an mp4 in Data
    video_files = list(data_dir.glob("*.mp4"))
    if video_files:
        shutil.copy(video_files[0], files_dir / "external.mp4")
    else:
        raise FileNotFoundError("No .mp4 file found in the Data folder.")

    # Copy base files from Data
    for filename, source, optional in files_map:
        src_path = source / filename
        if src_path.exists():
            shutil.copy(src_path, files_dir / filename)
        elif not optional:
            raise FileNotFoundError(f"Required file missing: {filename} in {source}")

    # --- HANDLE GAZE AND FIXATIONS (RAW vs ENRICHED) ---
    
    # 1. ALWAYS copy RAW files explicitly
    if (data_dir / 'gaze.csv').exists():
        shutil.copy(data_dir / 'gaze.csv', files_dir / 'gaze_raw.csv')
        # Default gaze.csv initially points to Raw
        shutil.copy(data_dir / 'gaze.csv', files_dir / 'gaze.csv')
    else:
        # Create dummy if missing (though unlikely for valid recording)
        pd.DataFrame().to_csv(files_dir / 'gaze_raw.csv')
        pd.DataFrame().to_csv(files_dir / 'gaze.csv')
        warnings.append("Gaze RAW mancante. Creato file vuoto.")

    if (data_dir / 'fixations.csv').exists():
        shutil.copy(data_dir / 'fixations.csv', files_dir / 'fixations_raw.csv')
        shutil.copy(data_dir / 'fixations.csv', files_dir / 'fixations.csv')
    else:
        pd.DataFrame().to_csv(files_dir / 'fixations_raw.csv')
        pd.DataFrame().to_csv(files_dir / 'fixations.csv')
        warnings.append("Fixations RAW mancante. Creato file vuoto.")

    # 2. Handle Enrichment if provided
    if enrichment_dir:
        # Gaze Enriched
        if (enrichment_dir / 'gaze.csv').exists():
            shutil.copy(enrichment_dir / 'gaze.csv', files_dir / 'gaze_enriched.csv')
            # Overwrite the default gaze.csv for the video to use the Enriched version (best quality)
            shutil.copy(enrichment_dir / 'gaze.csv', files_dir / 'gaze.csv')
            print("Enrichment gaze found and prepared.")
        
        # Fixations Enriched
        if (enrichment_dir / 'fixations.csv').exists():
            shutil.copy(enrichment_dir / 'fixations.csv', files_dir / 'fixations_enriched.csv')
            shutil.copy(enrichment_dir / 'fixations.csv', files_dir / 'fixations.csv')
            print("Enrichment fixations found and prepared.")
            
        # Surface Positions (only in enrichment usually)
        if (enrichment_dir / 'surface_positions.csv').exists():
            shutil.copy(enrichment_dir / 'surface_positions.csv', files_dir / 'surface_positions.csv')

    return files_dir, warnings

# ==============================================================================
# 2. FEATURE CALCULATION AND PLOTTING
# ==============================================================================

def calculate_metrics(df_seg, video_res=(1280, 720)):
    """
    Calculates metrics for a segment.
    """
    metrics = {}
    
    fix = df_seg.get('fixations', pd.DataFrame())
    gaze = df_seg.get('gaze', pd.DataFrame())
    pupil = df_seg.get('pupil', pd.DataFrame())
    blinks = df_seg.get('blinks', pd.DataFrame())
    saccades = df_seg.get('saccades', pd.DataFrame())

    # --- 1. Base Fixation Metrics ---
    if not fix.empty and 'duration [ms]' in fix.columns:
        metrics['n_fixations'] = len(fix)
        metrics['mean_duration_fixations'] = fix['duration [ms]'].mean()
        metrics['std_duration_fixations'] = fix['duration [ms]'].std()
        
        if 'fixation x [px]' in fix.columns:
            fx = fix['fixation x [px]'] / video_res[0]
            fy = fix['fixation y [px]'] / video_res[1]
        else: 
            fx = fix.get('fixation x [normalized]', pd.Series(dtype=float))
            fy = fix.get('fixation y [normalized]', pd.Series(dtype=float))
            
        metrics['mean_x_pos_fixations'] = fx.mean()
        metrics['mean_y_pos_fixations'] = fy.mean()
        metrics['std_x_pos_fixations'] = fx.std()
        metrics['std_y_pos_fixations'] = fy.std()

        # Enriched/Normalized Metrics (only if columns exist)
        if 'fixation detected on surface' in fix.columns:
            surface_fix = fix[fix['fixation detected on surface'] == True]
            if not surface_fix.empty:
                fx_norm = surface_fix['fixation position on surface x [normalized]']
                fy_norm = surface_fix['fixation position on surface y [normalized]']
                
                metrics['mean_x_pos_fixations[norm]'] = fx_norm.mean()
                metrics['mean_y_pos_fixations[norm]'] = fy_norm.mean()
                metrics['std_x_pos_fixations[norm]'] = fx_norm.std()
                metrics['std_y_pos_fixations[norm]'] = fy_norm.std()

    else:
        for k in ['n_fixations', 'mean_duration_fixations', 'std_duration_fixations', 
                  'mean_x_pos_fixations', 'mean_y_pos_fixations', 'std_x_pos_fixations', 'std_y_pos_fixations']:
            metrics[k] = np.nan

    # --- Gaze Metrics ---
    if not gaze.empty:
        if 'gaze x [px]' in gaze.columns:
            gx = gaze['gaze x [px]'] / video_res[0]
            gy = gaze['gaze y [px]'] / video_res[1]
        else: 
            gx = gaze.get('gaze x [normalized]', pd.Series(dtype=float))
            gy = gaze.get('gaze y [normalized]', pd.Series(dtype=float))

        metrics['mean_x_gaze'] = gx.mean()
        metrics['mean_y_gaze'] = gy.mean()
        metrics['std_x_gaze'] = gx.std()
        metrics['std_y_gaze'] = gy.std()

        if 'gaze detected on surface' in gaze.columns:
            surface_gaze = gaze[gaze['gaze detected on surface'] == True]
            if not surface_gaze.empty:
                gx_norm = surface_gaze['gaze position on surface x [normalized]']
                gy_norm = surface_gaze['gaze position on surface y [normalized]']

                metrics['mean_x_gaze[norm]'] = gx_norm.mean()
                metrics['mean_y_gaze[norm]'] = gy_norm.mean()
                metrics['std_x_gaze[norm]'] = gx_norm.std()
                metrics['std_y_gaze[norm]'] = gy_norm.std()
    else:
        for k in ['mean_x_gaze', 'mean_y_gaze', 'std_x_gaze', 'std_y_gaze']:
            metrics[k] = np.nan
            
    # --- 3. Other Metrics ---
    metrics['n_blink'] = len(blinks)

    if not pupil.empty:
        if 'pupil diameter right [mm]' in pupil.columns:
            p_dx = pupil['pupil diameter right [mm]']
            metrics['mean_mm_dx_pupil'] = p_dx.mean()
            metrics['std_mm_dx_pupil'] = p_dx.std()
        else:
            metrics['mean_mm_dx_pupil'] = np.nan
            metrics['std_mm_dx_pupil'] = np.nan
            
        if 'pupil diameter left [mm]' in pupil.columns:
            p_sx = pupil['pupil diameter left [mm]']
            metrics['mean_mm_sx_pupil'] = p_sx.mean()
            metrics['std_mm_sx_pupil'] = p_sx.std()
        else:
            metrics['mean_mm_sx_pupil'] = np.nan
            metrics['std_mm_sx_pupil'] = np.nan
            
        if 'pupil diameter right [mm]' in pupil.columns and 'pupil diameter left [mm]' in pupil.columns:
            p_avg = (pupil['pupil diameter right [mm]'] + pupil['pupil diameter left [mm]']) / 2
            metrics['mean_mm_(dx+sx/2)_pupil'] = p_avg.mean()
            metrics['std_mm_(dx+sx/2)_pupil'] = p_avg.std()
        else:
            metrics['mean_mm_(dx+sx/2)_pupil'] = np.nan
            metrics['std_mm_(dx+sx/2)_pupil'] = np.nan
    else:
        for k in ['mean_mm_dx_pupil', 'std_mm_dx_pupil', 'mean_mm_sx_pupil', 'std_mm_sx_pupil', 
                  'mean_mm_(dx+sx/2)_pupil', 'std_mm_(dx+sx/2)_pupil']:
            metrics[k] = np.nan

    metrics['n_saccades'] = len(saccades)
    if not saccades.empty and 'duration [ms]' in saccades.columns:
        metrics['mean_saccades'] = saccades['duration [ms]'].mean()
        metrics['std_saccades'] = saccades['duration [ms]'].std()
    else:
        metrics['mean_saccades'] = np.nan
        metrics['std_saccades'] = np.nan
        if 'n_saccades' not in metrics:
            metrics['n_saccades'] = 0
        
    return metrics

def generate_heatmap_pdf(df, x_col, y_col, title, filename, output_dir, video_res=(1280, 720), cmap='jet'):
    """Generates a KDE heatmap and saves it to PDF with selectable colormap."""
    if df.empty or x_col not in df.columns or len(df) < 5:
        return

    x = df[x_col].dropna()
    y = df[y_col].dropna()
    
    is_surface_plot = 'surface' in filename.lower()

    if is_surface_plot:
        x_limit, y_limit = 1.0, 1.0
        x_label, y_label = 'X [normalized]', 'Y [normalized]'
    else:
        if x.max() <= 1.0: # Normalized -> Pixel
            x = x * video_res[0]
            y = y * video_res[1]
        x_limit, y_limit = video_res[0], video_res[1]
        x_label, y_label = 'X [px]', 'Y [px]'

    plt.figure(figsize=(10, 6))
    try:
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[0:x_limit:100j, 0:y_limit:100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi_reshaped = zi.reshape(xi.shape)

        numpy_dir = output_dir.parent / "Numpy_Matrices"
        numpy_dir.mkdir(exist_ok=True)
        np.save(numpy_dir / f"{filename}_xi.npy", xi)
        np.save(numpy_dir / f"{filename}_yi.npy", yi)
        np.save(numpy_dir / f"{filename}_zi.npy", zi_reshaped)

        # Use the passed cmap
        plt.pcolormesh(xi, yi, zi_reshaped, shading='auto', cmap=cmap)
        plt.colorbar(label='Density')
        plt.xlim(0, x_limit)
        plt.ylim(y_limit, 0)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        out_path = output_dir / f"{filename}.pdf"
        plt.savefig(out_path, format='pdf', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating heatmap {title}: {e}")
        plt.close()

def generate_pupil_timeseries_pdf(pupil_df, gaze_df, blinks_df, start_ts, title, filename, output_dir):
    """Generates the pupillometry plot (Left and Right) in PDF."""
    if pupil_df.empty: return
    
    plt.figure(figsize=(12, 6))
    
    # Normalize timestamps
    pupil_df['time_norm'] = pupil_df['timestamp [ns]'] - start_ts

    # --- Draw On-Surface Bands ---
    surface_col = 'gaze detected on surface' if 'gaze detected on surface' in gaze_df.columns else 'on_surface'

    if not gaze_df.empty and surface_col in gaze_df.columns:
        gaze_df['time_norm'] = gaze_df['timestamp [ns]'] - start_ts
        gaze_df['block'] = (gaze_df[surface_col] != gaze_df[surface_col].shift()).cumsum()
        
        for _, group in gaze_df.groupby('block'):
            start_band = group['time_norm'].iloc[0]
            end_band = group['time_norm'].iloc[-1]
            is_on_surface = group[surface_col].iloc[0]
            
            color = 'green' if is_on_surface else 'red'
            plt.axvspan(start_band, end_band, color=color, alpha=0.15, lw=0)

    # --- Draw Blink Bands ---
    if not blinks_df.empty:
        for _, blink_row in blinks_df.iterrows():
            blink_start = blink_row['start timestamp [ns]']
            blink_duration_ms = blink_row['duration [ms]']
            blink_end = blink_start + (blink_duration_ms * 1_000_000)
            norm_blink_start = blink_start - start_ts
            norm_blink_end = blink_end - start_ts
            plt.axvspan(norm_blink_start, norm_blink_end, color='blue', alpha=0.2, lw=0)

    has_data = False
    if 'pupil diameter left [mm]' in pupil_df.columns:
        plt.plot(pupil_df['time_norm'], pupil_df['pupil diameter left [mm]'], 
                 label='Left Pupil', color=PUPIL_LEFT_COLOR, alpha=0.7)
        has_data = True
        
    if 'pupil diameter right [mm]' in pupil_df.columns:
        plt.plot(pupil_df['time_norm'], pupil_df['pupil diameter right [mm]'], 
                 label='Right Pupil', color=PUPIL_RIGHT_COLOR, alpha=0.7)
        has_data = True
        
    if has_data:
        plt.title(f"Pupillometry Timeseries - {title}")
        plt.xlabel("Time from event start (ns)")
        plt.ylabel("Diameter [mm]")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        out_path = output_dir / f"{filename}.pdf"

        try:
            numpy_dir = output_dir.parent / "Numpy_Matrices"
            numpy_dir.mkdir(exist_ok=True)
            
            time_col = pupil_df['time_norm'].values
            left_col = pupil_df.get('pupil diameter left [mm]', pd.Series(np.nan, index=pupil_df.index)).values
            right_col = pupil_df.get('pupil diameter right [mm]', pd.Series(np.nan, index=pupil_df.index)).values
            
            timeseries_data = np.vstack([time_col, left_col, right_col]).T
            np.save(numpy_dir / f"{filename}_timeseries.npy", timeseries_data)
        except Exception as e:
            print(f"Could not save timeseries numpy array for {filename}: {e}")

        plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

def generate_fragmentation_plot(gaze_df, x_col, y_col, start_ts, title, filename, output_dir, events_df=None, show_surface_bands=False):
    """Generates a fragmentation plot."""
    if gaze_df.empty or x_col not in gaze_df.columns or y_col not in gaze_df.columns:
        return

    df = gaze_df.copy()
    df['time_norm'] = df['timestamp [ns]'] - start_ts
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    df['distance'] = np.sqrt(dx**2 + dy**2)

    plt.figure(figsize=(12, 6))
    plt.plot(df['time_norm'], df['distance'], label='Gaze Fragmentation', color='purple', alpha=0.8)

    if show_surface_bands and 'gaze detected on surface' in df.columns:
        df['block'] = (df['gaze detected on surface'] != df['gaze detected on surface'].shift()).cumsum()
        for _, group in df.groupby('block'):
            start_band = group['time_norm'].iloc[0]
            end_band = group['time_norm'].iloc[-1]
            color = 'green' if group['gaze detected on surface'].iloc[0] else 'red'
            plt.axvspan(start_band, end_band, color=color, alpha=0.15, lw=0)

    if events_df is not None:
        for _, event_row in events_df.iterrows():
            event_time = event_row['timestamp [ns]'] - start_ts
            plt.axvline(x=event_time, color='k', linestyle='--', linewidth=1)

    plt.title(title)
    plt.xlabel("Time from start (ns)")
    plt.ylabel("Euclidean Distance")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    out_path_pdf = output_dir / f"{filename}.pdf"
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    numpy_dir = output_dir.parent / "Numpy_Matrices"
    numpy_dir.mkdir(exist_ok=True)
    fragmentation_data = df[['time_norm', 'distance']].dropna().values
    np.save(numpy_dir / f"{filename}_fragmentation.npy", fragmentation_data)

# ==============================================================================
# 3. VIDEO GENERATION
# ==============================================================================

def generate_full_video(data_dir, output_file, events_df, enrichment_dir=None):
    """Generates the video with all overlays."""
    print("Starting video generation...")
    
    try:
        w_ts = pd.read_csv(data_dir / 'world_timestamps.csv')
        gaze = pd.read_csv(data_dir / 'gaze.csv') # Uses the 'best' version prepared
        blinks = pd.read_csv(data_dir / 'blinks.csv')
        pupil = pd.read_csv(data_dir / '3d_eye_states.csv')
        
        merged = pd.merge_asof(w_ts, gaze, on='timestamp [ns]', direction='nearest')
        merged = pd.merge_asof(merged, blinks.add_suffix('_blink'), left_on='timestamp [ns]', right_on='start timestamp [ns]_blink', direction='nearest')
        merged = pd.merge_asof(merged, pupil, on='timestamp [ns]', direction='nearest')

        surface_file = data_dir / "surface_positions.csv"
        if surface_file.exists():
            surf_df = pd.read_csv(surface_file)
            merged = pd.merge_asof(merged, surf_df.add_suffix('_surf'), left_on='timestamp [ns]', right_on='timestamp [ns]_surf', direction='nearest')

    except Exception as e:
        print(f"Error loading data for video: {e}")
        return

    cap = cv2.VideoCapture(str(data_dir / "external.mp4"))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    gaze_history = []
    pupil_history_l = []
    pupil_history_r = []
    fragmentation_history = []
    prev_gaze_point = None
    
    has_surface_info = 'gaze detected on surface' in merged.columns

    for i in tqdm(range(total_frames), desc="Rendering Video"):
        ret, frame = cap.read()
        if not ret: break
        
        try:
            row = merged.iloc[i]
            ts = row['timestamp [ns]']
            
            # 1. Event Overlay
            curr_event = events_df[events_df['timestamp [ns]'] <= ts]
            if not curr_event.empty:
                evt_name = curr_event.iloc[-1]['name']
                if evt_name not in ['recording.begin', 'recording.end']:
                    text = f"Event: {evt_name}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.rectangle(frame, (20, 50 - text_height - baseline), (20 + text_width, 50 + baseline), (0, 0, 0), -1)
                    cv2.putText(frame, text, (20, 50), font, font_scale, (255, 255, 255), thickness)
            
            # 1.5 Blink & On-Surface Overlay
            if pd.notna(row.get('start timestamp [ns]_blink')) and ts >= row['start timestamp [ns]_blink'] and ts <= (row['start timestamp [ns]_blink'] + row['duration [ms]_blink'] * 1_000_000):
                blink_text = "BLINK"
                blink_font = cv2.FONT_HERSHEY_SIMPLEX
                blink_font_scale = 1.2
                blink_thickness = 3
                (text_w, text_h), blink_baseline = cv2.getTextSize(blink_text, blink_font, blink_font_scale, blink_thickness)
                text_x = width - 20 - text_w
                text_y = 50
                cv2.rectangle(frame, (text_x, text_y - text_h - blink_baseline), (text_x + text_w, text_y + blink_baseline), (0, 0, 0), -1)
                cv2.putText(frame, blink_text, (text_x, text_y), blink_font, blink_font_scale, (0, 255, 255), blink_thickness)
            
            if has_surface_info and row.get('gaze detected on surface') == True:
                 cv2.putText(frame, "ON_SURFACE", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 1.8 Draw Dynamic Surface Polygon
            if 'tl x [px]_surf' in row and pd.notna(row['tl x [px]_surf']):
                tl = (int(row['tl x [px]_surf']), int(row['tl y [px]_surf']))
                tr = (int(row['tr x [px]_surf']), int(row['tr y [px]_surf']))
                bl = (int(row['bl x [px]_surf']), int(row['bl y [px]_surf']))
                br = (int(row['br x [px]_surf']), int(row['br y [px]_surf']))
                
                surface_poly_pts = np.array([tl, tr, br, bl], dtype=np.int32)
                cv2.polylines(frame, [surface_poly_pts], isClosed=True, color=(255, 255, 0), thickness=2)

            # 2. Gaze Point & Path
            gx, gy = None, None
            if 'gaze x [px]' in row and pd.notna(row['gaze x [px]']):
                gx, gy = int(row['gaze x [px]']), int(row['gaze y [px]'])
            elif 'gaze x [normalized]' in row and pd.notna(row['gaze x [normalized]']):
                gx = int(row['gaze x [normalized]'] * width)
                gy = int(row['gaze y [normalized]'] * height)
            
            if gx is not None and gy is not None:
                # Color Logic
                if has_surface_info:
                    if row.get('gaze detected on surface') == True:
                        current_color = GAZE_COLOR_IN # Green
                    else:
                        current_color = GAZE_COLOR_OUT # Red
                else:
                    current_color = GAZE_COLOR_DEFAULT # Blue

                cv2.circle(frame, (gx, gy), 15, current_color, 2)
                
                gaze_history.append((gx, gy))
                if len(gaze_history) > 20: gaze_history.pop(0)

                if prev_gaze_point:
                    distance = np.sqrt((gx - prev_gaze_point[0])**2 + (gy - prev_gaze_point[1])**2)
                    fragmentation_history.append(distance)
                else:
                    fragmentation_history.append(0)
                if len(fragmentation_history) > 100: fragmentation_history.pop(0)
                
                prev_gaze_point = (gx, gy)
                
                for j in range(1, len(gaze_history)):
                    cv2.line(frame, gaze_history[j-1], gaze_history[j], current_color, 2)
            
            # 3. Pupil Plot
            if pd.notna(row.get('pupil diameter left [mm]')):
                val_l = row['pupil diameter left [mm]']
                pupil_history_l.append(val_l)
                if len(pupil_history_l) > 100: pupil_history_l.pop(0)

            if pd.notna(row.get('pupil diameter right [mm]')):
                val_r = row['pupil diameter right [mm]']
                pupil_history_r.append(val_r)
                if len(pupil_history_r) > 100: pupil_history_r.pop(0)
            
            plot_w, plot_h = 300, 100
            plot_x, plot_y = width - plot_w - 20, height - plot_h - 20
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "Pupil Diameter", (plot_x + 5, plot_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            p_min, p_max = 2.0, 8.0 
            if len(pupil_history_l) > 1:
                pts_l = []
                for idx, val in enumerate(pupil_history_l):
                    px = int(plot_x + (idx / 100) * plot_w)
                    py = int(plot_y + plot_h - ((val - p_min) / (p_max - p_min)) * plot_h)
                    pts_l.append((px, py))
                cv2.polylines(frame, [np.array(pts_l)], False, PUPIL_LEFT_COLOR_BGR, 2)

            if len(pupil_history_r) > 1:
                pts_r = []
                for idx, val in enumerate(pupil_history_r):
                    px = int(plot_x + (idx / 100) * plot_w)
                    py = int(plot_y + plot_h - ((val - p_min) / (p_max - p_min)) * plot_h)
                    pts_r.append((px, py))
                cv2.polylines(frame, [np.array(pts_r)], False, PUPIL_RIGHT_COLOR_BGR, 2)

            # 4. Fragmentation Plot
            frag_plot_y = plot_y - plot_h - 20
            overlay_frag = frame.copy()
            cv2.rectangle(overlay_frag, (plot_x, frag_plot_y), (plot_x + plot_w, frag_plot_y + plot_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay_frag, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "Gaze Fragmentation", (plot_x + 5, frag_plot_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if len(fragmentation_history) > 1:
                frag_min, frag_max = 0, 150
                pts_frag = []
                for idx, val in enumerate(fragmentation_history):
                    px = int(plot_x + (idx / 100) * plot_w)
                    clamped_val = min(val, frag_max)
                    py = int(frag_plot_y + plot_h - ((clamped_val - frag_min) / (frag_max - frag_min)) * plot_h)
                    pts_frag.append((px, py))
                cv2.polylines(frame, [np.array(pts_frag)], False, (128, 0, 128), 2)

        except (IndexError, KeyError):
            pass
            
        writer.write(frame)
        
    cap.release()
    writer.release()
    print(f"Video saved to: {output_file}")

# ==============================================================================
# 4. EVENT EDITOR (LITE)
# ==============================================================================

class AdvancedEventEditor(tk.Toplevel):
    def __init__(self, parent, video_path, events_df, world_ts):
        super().__init__(parent)
        self.title("Speed Lite - Advanced Event Editor")
        self.geometry("1400x800")
        
        self.video_path = video_path
        self.events_df = events_df.copy().reset_index(drop=True)
        self.world_ts = world_ts
        
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        self.current_frame = 0
        self.saved_df = None

        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        video_frame = tk.Frame(main_pane)
        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.scale = tk.Scale(video_frame, from_=0, to=self.total_frames, orient=tk.HORIZONTAL, command=self.on_slide)
        self.scale.pack(fill=tk.X, padx=10, pady=5)
        main_pane.add(video_frame, width=800)

        controls_frame = tk.Frame(main_pane)
        table_container = tk.Frame(controls_frame)
        table_container.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        cols = ("#", "Event Name", "Timestamp (ns)")
        self.tree = ttk.Treeview(table_container, columns=cols, show='headings', selectmode='extended')
        for col in cols: self.tree.heading(col, text=col)
        self.tree.column("#", width=50, anchor=tk.CENTER)
        self.tree.column("Event Name", width=200)
        self.tree.column("Timestamp (ns)", width=150)
        
        vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<<TreeviewSelect>>', self._on_row_select)

        btn_grid = tk.Frame(controls_frame)
        btn_grid.pack(fill=tk.X, padx=10)

        tk.Button(btn_grid, text="Add Event at Current Frame", command=self._add_event, bg="#c8e6c9").grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(btn_grid, text="Delete Selected", command=self._delete_events, bg="#ffcdd2").grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        tk.Button(btn_grid, text="Rename Selected", command=self._rename_event).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(btn_grid, text="Move to Current Frame", command=self._move_event).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        tk.Button(btn_grid, text="Merge Selected", command=self._merge_events, bg="#e1bee7").grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        
        tk.Button(controls_frame, text="Save and Exit", command=self.save_exit, font=("Arial", 12, "bold"), bg="#b2dfdb").pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        main_pane.add(controls_frame, width=600)
        self._populate_table()
        self.update_image()

    def on_slide(self, val):
        self.current_frame = int(val)
        self.update_image()

    def update_image(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            scale = min(800/w, 700/h)
            frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
            self.canvas.image = img

    def _populate_table(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        self.events_df = self.events_df.sort_values('timestamp [ns]').reset_index(drop=True)
        for index, row in self.events_df.iterrows():
            self.tree.insert("", "end", iid=str(index), values=(index, row['name'], row['timestamp [ns]']))

    def _get_selected_indices(self):
        return [int(item) for item in self.tree.selection()]

    def _on_row_select(self, event):
        indices = self._get_selected_indices()
        if not indices: return
        selected_idx = indices[0]
        ts = self.events_df.loc[selected_idx, 'timestamp [ns]']
        frame_idx = (self.world_ts['timestamp [ns]'] - ts).abs().idxmin()
        self.current_frame = frame_idx
        self.scale.set(self.current_frame)
        self.update_image()

    def _add_event(self):
        name = simpledialog.askstring("New Event", "Enter event name:")
        if name:
            ts = self.world_ts.iloc[self.current_frame]['timestamp [ns]']
            new_row = {'name': name, 'timestamp [ns]': ts}
            self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
            self._populate_table()

    def _delete_events(self):
        indices = self._get_selected_indices()
        if not indices:
            messagebox.showwarning("Warning", "No events selected to delete.")
            return
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete {len(indices)} event(s)?"):
            self.events_df = self.events_df.drop(indices).reset_index(drop=True)
            self._populate_table()

    def _rename_event(self):
        indices = self._get_selected_indices()
        if len(indices) != 1:
            messagebox.showwarning("Warning", "Please select exactly one event to rename.")
            return
        idx = indices[0]
        old_name = self.events_df.loc[idx, 'name']
        new_name = simpledialog.askstring("Rename Event", "Enter new event name:", initialvalue=old_name)
        if new_name and new_name != old_name:
            self.events_df.loc[idx, 'name'] = new_name
            self._populate_table()

    def _move_event(self):
        indices = self._get_selected_indices()
        if len(indices) != 1:
            messagebox.showwarning("Warning", "Please select exactly one event to move.")
            return
        idx = indices[0]
        new_ts = self.world_ts.iloc[self.current_frame]['timestamp [ns]']
        if messagebox.askyesno("Confirm Move", f"Move event '{self.events_df.loc[idx, 'name']}' to the current timestamp?"):
            self.events_df.loc[idx, 'timestamp [ns]'] = new_ts
            self._populate_table()

    def _merge_events(self):
        indices = self._get_selected_indices()
        if len(indices) < 2:
            messagebox.showwarning("Warning", "Please select at least two events to merge.")
            return
        new_name = simpledialog.askstring("Merge Events", "Enter name for the new merged event:")
        if not new_name: return
        first_event_idx = min(indices)
        new_ts = self.events_df.loc[first_event_idx, 'timestamp [ns]']
        new_row = {'name': new_name, 'timestamp [ns]': new_ts}
        self.events_df = self.events_df.drop(indices)
        self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
        self._populate_table()

    def save_exit(self):
        self.saved_df = self.events_df
        self.destroy()

# ==============================================================================
# 5. MAIN APP
# ==============================================================================

class SpeedLiteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED Light")
        self.root.geometry("600x600")
        
        self.data_dir = tk.StringVar()
        self.enrich_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        self.working_dir = None
        self.gui_queue = queue.Queue() 
        
        pad = {'padx': 10, 'pady': 5}
        
        grp_in = tk.LabelFrame(root, text="Input Folders")
        grp_in.pack(fill=tk.X, **pad)
        
        self._build_file_select(grp_in, "Data Folder (Mandatory):", self.data_dir)
        self._build_file_select(grp_in, "Enrichment Folder (Optional):", self.enrich_dir)
        self._build_file_select(grp_in, "Output Folder (Default: Data/Output):", self.output_dir)
        
        tk.Button(root, text="1. Load and Prepare Data", command=self.load_data, bg="#fff9c4").pack(fill=tk.X, **pad)
        self.btn_edit = tk.Button(root, text="2. Edit Events (Optional)", command=self.edit_events, state=tk.DISABLED)
        self.btn_edit.pack(fill=tk.X, **pad)
        self.btn_run = tk.Button(root, text="3. Estrai Features, Plot & Video", command=self.run_process, bg="#b2dfdb", state=tk.DISABLED)
        self.btn_run.pack(fill=tk.X, **pad)
        
        self.log_box = tk.Text(root, height=12)
        self.log_box.pack(fill=tk.BOTH, **pad)

        credits_text = (
            "Developed by: Dr. Daniele Lozzi (github.com/danielelozzi)\n"
            "Laboratorio di Scienze Cognitive e del Comportamento (SCoC) - Università degli Studi dell’Aquila\n"
            "https://labscoc.wordpress.com/"
        )
        credits_label = tk.Label(root, text=credits_text, justify=tk.CENTER, font=("Arial", 8), fg="grey")
        credits_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 10))
        
        self.root.after(100, self._check_queue)
        
    def _build_file_select(self, parent, label, var):
        f = tk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=label).pack(side=tk.LEFT)
        tk.Entry(f, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(f, text="Browse", command=lambda: self.browse(var)).pack(side=tk.RIGHT)
        
    def browse(self, var):
        d = filedialog.askdirectory()
        if d: var.set(d)
        
    def log(self, msg):
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        print(msg)

    def _check_queue(self):
        try:
            while True:
                msg_type, content = self.gui_queue.get_nowait()
                if msg_type == 'log':
                    self.log(content)
                elif msg_type == 'info':
                    messagebox.showinfo("Info", content)
                elif msg_type == 'error':
                    messagebox.showerror("Error", content)
                elif msg_type == 'enable_buttons':
                    self.btn_run.config(state=tk.NORMAL)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._check_queue)

    def load_data(self):
        data = Path(self.data_dir.get())
        if not data.exists():
            messagebox.showerror("Error", "The Data folder is mandatory.")
            return
            
        enrich = Path(self.enrich_dir.get()) if self.enrich_dir.get() else None
        
        if not self.output_dir.get():
            self.output_dir.set(str(data / "Output"))
        
        try:
            self.log("Preparing working directory...")
            self.working_dir, warnings = prepare_working_directory(data, enrich, Path(self.output_dir.get()))
            self.log(f"Unified files in: {self.working_dir}")
            
            if warnings:
                messagebox.showwarning("Dati Mancanti", "\n".join(warnings))

            self.btn_edit.config(state=tk.NORMAL)
            self.btn_run.config(state=tk.NORMAL)
            messagebox.showinfo("OK", "Data ready.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"Error: {e}")

    def edit_events(self):
        video_path = self.working_dir / "external.mp4"
        events_path = self.working_dir / "events.csv"
        ts_path = self.working_dir / "world_timestamps.csv"
        
        df_ev = pd.read_csv(events_path)
        df_ts = pd.read_csv(ts_path)
        
        editor = AdvancedEventEditor(self.root, video_path, df_ev, df_ts)
        self.root.wait_window(editor)
        
        if editor.saved_df is not None:
            if 'selected' in editor.saved_df.columns:
                editor.saved_df = editor.saved_df.drop(columns=['selected'])
            editor.saved_df.to_csv(events_path, index=False)
            self.log("Updated events saved.")

    def run_process(self):
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            self.gui_queue.put(('log', "Starting Analysis..."))
            
            ev = pd.read_csv(self.working_dir / "events.csv").sort_values('timestamp [ns]')
            ev = ev[~ev['name'].isin(['recording.begin', 'recording.end', 'begin', 'end'])]
            if ev.empty:
                self.gui_queue.put(('log', "Warning: No valid events found after filtering."))
                self.gui_queue.put(('enable_buttons', None))
                return

            # LOAD RAW
            gaze_raw = pd.read_csv(self.working_dir / "gaze_raw.csv")
            fix_raw = pd.read_csv(self.working_dir / "fixations_raw.csv")
            
            # LOAD ENRICHED (IF AVAILABLE)
            gaze_enr = None
            fix_enr = None
            if (self.working_dir / "gaze_enriched.csv").exists():
                gaze_enr = pd.read_csv(self.working_dir / "gaze_enriched.csv")
            if (self.working_dir / "fixations_enriched.csv").exists():
                fix_enr = pd.read_csv(self.working_dir / "fixations_enriched.csv")

            pupil = pd.read_csv(self.working_dir / "3d_eye_states.csv")
            sacc = pd.read_csv(self.working_dir / "saccades.csv")
            blinks = pd.read_csv(self.working_dir / "blinks.csv")
            
            cap = cv2.VideoCapture(str(self.working_dir / "external.mp4"))
            res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            
            all_metrics = []
            pdf_dir = Path(self.output_dir.get()) / "Plots_PDF"
            pdf_dir.mkdir(exist_ok=True)
            
            for i in range(len(ev)):
                start_ts = ev.iloc[i]['timestamp [ns]']
                end_ts = ev.iloc[i+1]['timestamp [ns]'] if i < len(ev)-1 else gaze_raw['timestamp [ns]'].max()
                evt_name = ev.iloc[i]['name']
                
                self.gui_queue.put(('log', f"Processing event: {evt_name}"))
                
                # --- PROCESS RAW DATA (MANDATORY) ---
                seg_raw = {
                    'fixations': fix_raw[(fix_raw['start timestamp [ns]'] >= start_ts) & (fix_raw['start timestamp [ns]'] < end_ts)],
                    'pupil': pupil[(pupil['timestamp [ns]'] >= start_ts) & (pupil['timestamp [ns]'] < end_ts)],
                    'blinks': blinks[(blinks['start timestamp [ns]'] >= start_ts) & (blinks['start timestamp [ns]'] < end_ts)],
                    'saccades': sacc[(sacc['start timestamp [ns]'] >= start_ts) & (sacc['start timestamp [ns]'] < end_ts)],
                    'gaze': gaze_raw[(gaze_raw['timestamp [ns]'] >= start_ts) & (gaze_raw['timestamp [ns]'] < end_ts)]
                }
                
                m_raw = calculate_metrics(seg_raw, res)
                # Rename keys to indicate RAW source
                m_raw = {k + '_RAW': v for k, v in m_raw.items()}
                
                # Plot Raw (Color 1 - Winter)
                gaze_seg = seg_raw['gaze']
                if not gaze_seg.empty:
                    gx_raw = 'gaze x [px]' if 'gaze x [px]' in gaze_seg.columns else 'gaze x [normalized]'
                    gy_raw = 'gaze y [px]' if 'gaze y [px]' in gaze_seg.columns else 'gaze y [normalized]'
                    generate_heatmap_pdf(gaze_seg, gx_raw, gy_raw, f"Gaze Heatmap (RAW) - {evt_name}", f"heatmap_gaze_RAW_{evt_name}", pdf_dir, res, cmap=CMAP_RAW)

                fix_seg = seg_raw['fixations']
                if not fix_seg.empty:
                    fx_raw = 'fixation x [px]' if 'fixation x [px]' in fix_seg.columns else 'fixation x [normalized]'
                    fy_raw = 'fixation y [px]' if 'fixation y [px]' in fix_seg.columns else 'fixation y [normalized]'
                    generate_heatmap_pdf(fix_seg, fx_raw, fy_raw, f"Fixation Heatmap (RAW) - {evt_name}", f"heatmap_fixation_RAW_{evt_name}", pdf_dir, res, cmap=CMAP_RAW)

                # --- PROCESS ENRICHED DATA (OPTIONAL) ---
                m_enr = {}
                if gaze_enr is not None and fix_enr is not None:
                    seg_enr = {
                        'fixations': fix_enr[(fix_enr['start timestamp [ns]'] >= start_ts) & (fix_enr['start timestamp [ns]'] < end_ts)],
                        'pupil': pupil[(pupil['timestamp [ns]'] >= start_ts) & (pupil['timestamp [ns]'] < end_ts)],
                        'blinks': blinks[(blinks['start timestamp [ns]'] >= start_ts) & (blinks['start timestamp [ns]'] < end_ts)],
                        'saccades': sacc[(sacc['start timestamp [ns]'] >= start_ts) & (sacc['start timestamp [ns]'] < end_ts)],
                        'gaze': gaze_enr[(gaze_enr['timestamp [ns]'] >= start_ts) & (gaze_enr['timestamp [ns]'] < end_ts)]
                    }
                    m_temp = calculate_metrics(seg_enr, res)
                    m_enr = {k + '_ENRICHED': v for k, v in m_temp.items()}

                    # Plot Enriched (Color 2 - Jet/Warm)
                    gaze_seg_enr = seg_enr['gaze']
                    if not gaze_seg_enr.empty:
                        # Standard Pixel Enriched
                        gx_enr = 'gaze x [px]' if 'gaze x [px]' in gaze_seg_enr.columns else 'gaze x [normalized]'
                        gy_enr = 'gaze y [px]' if 'gaze y [px]' in gaze_seg_enr.columns else 'gaze y [normalized]'
                        generate_heatmap_pdf(gaze_seg_enr, gx_enr, gy_enr, f"Gaze Heatmap (ENRICHED) - {evt_name}", f"heatmap_gaze_ENRICHED_{evt_name}", pdf_dir, res, cmap=CMAP_ENRICHED)

                        # Surface Plot (if available)
                        if 'gaze detected on surface' in gaze_seg_enr.columns:
                            surface_gaze = gaze_seg_enr[gaze_seg_enr['gaze detected on surface'] == True]
                            if not surface_gaze.empty:
                                generate_heatmap_pdf(surface_gaze, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]',
                                                     f"Gaze Heatmap (Surface) - {evt_name}", f"heatmap_gaze_{evt_name}_surface", pdf_dir, res, cmap=CMAP_ENRICHED)
                    
                    fix_seg_enr = seg_enr['fixations']
                    if not fix_seg_enr.empty:
                        # Surface Fixation Plot
                        if 'fixation detected on surface' in fix_seg_enr.columns:
                            surface_fix = fix_seg_enr[fix_seg_enr['fixation detected on surface'] == True]
                            if not surface_fix.empty:
                                generate_heatmap_pdf(surface_fix, 'fixation position on surface x [normalized]', 'fixation position on surface y [normalized]',
                                                     f"Fixation Heatmap (Surface) - {evt_name}", f"heatmap_fixation_{evt_name}_surface", pdf_dir, res, cmap=CMAP_ENRICHED)

                # Merge and append
                full_metrics = {**m_raw, **m_enr}
                full_metrics['event_name'] = evt_name
                all_metrics.append(full_metrics)

                # Common Plots (Pupil & Fragmentation - Raw based primarily)
                pupil_seg = seg_raw['pupil']
                if not pupil_seg.empty:
                    # Pass enriched gaze for bands if available, else raw
                    gaze_for_bands = seg_enr['gaze'] if gaze_enr is not None else seg_raw['gaze']
                    generate_pupil_timeseries_pdf(pupil_seg, gaze_for_bands, seg_raw['blinks'], start_ts, evt_name, f"pupil_ts_{evt_name}", pdf_dir)

                if not gaze_seg.empty:
                    gx_raw = 'gaze x [px]' if 'gaze x [px]' in gaze_seg.columns else 'gaze x [normalized]'
                    gy_raw = 'gaze y [px]' if 'gaze y [px]' in gaze_seg.columns else 'gaze y [normalized]'
                    generate_fragmentation_plot(gaze_seg, gx_raw, gy_raw, start_ts,
                                                f"Gaze Fragmentation (RAW) - {evt_name}", f"frag_gaze_RAW_{evt_name}", pdf_dir)
                    
                    # If enriched surface exists, do surface frag too
                    if gaze_enr is not None:
                        gaze_seg_enr = seg_enr['gaze']
                        if 'gaze detected on surface' in gaze_seg_enr.columns:
                            surface_gaze = gaze_seg_enr[gaze_seg_enr['gaze detected on surface'] == True]
                            if not surface_gaze.empty:
                                generate_fragmentation_plot(surface_gaze, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', start_ts,
                                                            f"Gaze Fragmentation (Surface) - {evt_name}", f"frag_gaze_{evt_name}_surface", pdf_dir)

            out_file = Path(self.output_dir.get()) / "Speed_Lite_Results.xlsx"
            pd.DataFrame(all_metrics).to_excel(out_file, index=False)
            self.gui_queue.put(('log', f"Metrics saved to {out_file}"))
            
            # Video Generation
            self.gui_queue.put(('log', "Generating Final Video..."))
            vid_out = Path(self.output_dir.get()) / "final_video_overlay.mp4"
            enrich_path = Path(self.enrich_dir.get()) if self.enrich_dir.get() else None
            generate_full_video(self.working_dir, vid_out, ev, enrichment_dir=enrich_path)

            # Full Duration Plots
            self.gui_queue.put(('log', "Generating full duration plots..."))
            full_duration_dir = Path(self.output_dir.get()) / "Full_Duration_Plots"
            full_duration_dir.mkdir(exist_ok=True)
            video_start_ts = gaze_raw['timestamp [ns]'].min()
            
            gaze_for_full = gaze_enr if gaze_enr is not None else gaze_raw

            generate_pupil_timeseries_pdf(pupil, gaze_for_full, blinks, video_start_ts,
                                          "Full Pupillometry Timeseries", "full_pupillometry", full_duration_dir)

            gx_raw_full = 'gaze x [px]' if 'gaze x [px]' in gaze_raw.columns else 'gaze x [normalized]'
            gy_raw_full = 'gaze y [px]' if 'gaze y [px]' in gaze_raw.columns else 'gaze y [normalized]'
            generate_fragmentation_plot(gaze_raw, gx_raw_full, gy_raw_full, video_start_ts,
                                        "Full Gaze Fragmentation (Raw)", "full_frag_raw", full_duration_dir,
                                        events_df=ev, show_surface_bands=True)

            if gaze_enr is not None and 'gaze detected on surface' in gaze_enr.columns:
                surface_gaze_full = gaze_enr[gaze_enr['gaze detected on surface'] == True]
                if not surface_gaze_full.empty:
                    generate_fragmentation_plot(surface_gaze_full, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', video_start_ts,
                                                "Full Gaze Fragmentation (Surface)", "full_frag_surface", full_duration_dir,
                                                events_df=ev)
            
            self.gui_queue.put(('log', "Analysis Complete."))
            self.gui_queue.put(('info', "All operations completed successfully!"))
            
        except Exception as e:
            self.gui_queue.put(('log', f"Error: {e}"))
            print(e)
            self.gui_queue.put(('error', str(e)))
        finally:
            self.gui_queue.put(('enable_buttons', None))

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedLiteApp(root)
    root.mainloop()