import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
import queue

# --- CONFIGURATION ---
PLOT_DPI = 150
GAZE_COLOR_RAW = (255, 0, 0)      # Blue (BGR) per Raw
GAZE_COLOR_IN = (0, 255, 0)       # Green (BGR) per Surface In
GAZE_COLOR_OUT = (0, 0, 255)      # Red (BGR) per Surface Out
SURFACE_POLY_COLOR = (0, 255, 255) # Yellow (BGR)

PUPIL_LEFT_COLOR = 'blue'
PUPIL_RIGHT_COLOR = 'orange'
PUPIL_LEFT_COLOR_BGR = (255, 255, 0)  # Cyan for video plot
PUPIL_RIGHT_COLOR_BGR = (0, 165, 255) # Orange for video plot

# Heatmap Colormaps
CMAP_RAW = 'winter' 
CMAP_ENRICHED = 'jet' 

# ==============================================================================
# 1. FILE MANAGEMENT AND DATA PREPARATION
# ==============================================================================

def prepare_working_directory(data_dir, enrichment_dirs, output_dir):
    """
    Creates the 'files' folder.
    ALWAYS creates *_raw.csv files from Data.
    For each enrichment dir in the list, creates *_enr_{i}.csv files.
    
    Args:
        enrichment_dirs: list of Path objects
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

    # --- HANDLE RAW FILES ---
    if (data_dir / 'gaze.csv').exists():
        shutil.copy(data_dir / 'gaze.csv', files_dir / 'gaze_raw.csv')
    else:
        pd.DataFrame().to_csv(files_dir / 'gaze_raw.csv')
        warnings.append("Gaze RAW mancante. Creato file vuoto.")

    if (data_dir / 'fixations.csv').exists():
        shutil.copy(data_dir / 'fixations.csv', files_dir / 'fixations_raw.csv')
    else:
        pd.DataFrame().to_csv(files_dir / 'fixations_raw.csv')
        warnings.append("Fixations RAW mancante. Creato file vuoto.")

    # --- HANDLE MULTIPLE ENRICHMENTS ---
    for i, enr_dir in enumerate(enrichment_dirs):
        if not enr_dir.exists():
            warnings.append(f"Enrichment {i} path non trovato: {enr_dir}")
            continue

        # Gaze
        if (enr_dir / 'gaze.csv').exists():
            shutil.copy(enr_dir / 'gaze.csv', files_dir / f'gaze_enr_{i}.csv')
        else:
            warnings.append(f"Gaze mancante in Enrichment {i}")

        # Fixations
        if (enr_dir / 'fixations.csv').exists():
            shutil.copy(enr_dir / 'fixations.csv', files_dir / f'fixations_enr_{i}.csv')
            
        # Surface Positions
        if (enr_dir / 'surface_positions.csv').exists():
            shutil.copy(enr_dir / 'surface_positions.csv', files_dir / f'surface_positions_enr_{i}.csv')

    return files_dir, warnings

# ==============================================================================
# 2. FEATURE CALCULATION AND PLOTTING
# ==============================================================================

def calculate_metrics(df_seg, video_res=(1280, 720)):
    """Calculates metrics for a segment."""
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

        if 'fixation detected on surface' in fix.columns:
            surface_fix = fix[fix['fixation detected on surface'] == True]
            col_x = 'fixation position on surface x [normalized]'
            col_y = 'fixation position on surface y [normalized]']
            if not surface_fix.empty and col_x in fix.columns and col_y in fix.columns:
                fx_norm = surface_fix[col_x]
                fy_norm = surface_fix[col_y]
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
        # Simplified pupil logic for brevity
        if 'pupil diameter right [mm]' in pupil.columns:
            metrics['mean_mm_dx_pupil'] = pupil['pupil diameter right [mm]'].mean()
            metrics['std_mm_dx_pupil'] = pupil['pupil diameter right [mm]'].std()
        else:
            metrics['mean_mm_dx_pupil'] = np.nan
            metrics['std_mm_dx_pupil'] = np.nan
            
        if 'pupil diameter left [mm]' in pupil.columns:
            metrics['mean_mm_sx_pupil'] = pupil['pupil diameter left [mm]'].mean()
            metrics['std_mm_sx_pupil'] = pupil['pupil diameter left [mm]'].std()
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
        
    return metrics

def generate_heatmap_pdf(df, x_col, y_col, title, filename, output_dir, video_res=(1280, 720), cmap='jet'):
    """Generates a KDE heatmap and saves it to PDF."""
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
    """Generates the pupillometry plot."""
    if pupil_df.empty: return
    
    plt.figure(figsize=(12, 6))
    pupil_df['time_norm'] = pupil_df['timestamp [ns]'] - start_ts

    # --- Draw On-Surface Bands ---
    surface_col = 'gaze detected on surface' if 'gaze detected on surface' in gaze_df.columns else 'on_surface'
    if not gaze_df.empty and surface_col in gaze_df.columns:
        gaze_df['time_norm'] = gaze_df['timestamp [ns]'] - start_ts
        gaze_df['block'] = (gaze_df[surface_col] != gaze_df[surface_col].shift()).cumsum()
        for _, group in gaze_df.groupby('block'):
            start_band = group['time_norm'].iloc[0]
            end_band = group['time_norm'].iloc[-1]
            color = 'green' if group[surface_col].iloc[0] else 'red'
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
        plt.plot(pupil_df['time_norm'], pupil_df['pupil diameter left [mm]'], label='Left Pupil', color=PUPIL_LEFT_COLOR, alpha=0.7)
        has_data = True
    if 'pupil diameter right [mm]' in pupil_df.columns:
        plt.plot(pupil_df['time_norm'], pupil_df['pupil diameter right [mm]'], label='Right Pupil', color=PUPIL_RIGHT_COLOR, alpha=0.7)
        has_data = True
        
    if has_data:
        plt.title(f"{title}")
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
            np.save(numpy_dir / f"{filename}_timeseries.npy", np.vstack([time_col, left_col, right_col]).T)
        except Exception: pass
        plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

def generate_fragmentation_plot(gaze_df, x_col, y_col, start_ts, title, filename, output_dir, events_df=None, show_surface_bands=False):
    """Generates fragmentation plot."""
    if gaze_df.empty or x_col not in gaze_df.columns or y_col not in gaze_df.columns: return

    df = gaze_df.copy()
    df['time_norm'] = df['timestamp [ns]'] - start_ts
    df['distance'] = np.sqrt(df[x_col].diff()**2 + df[y_col].diff()**2)

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
    np.save(numpy_dir / f"{filename}_fragmentation.npy", df[['time_norm', 'distance']].dropna().values)

# ==============================================================================
# 3. VIDEO GENERATION (MULTI-LAYER)
# ==============================================================================

def generate_full_video(data_dir, output_file, events_df, active_enrichment_indices=[]):
    """Generates the video with Raw + Multiple Enrichment overlays."""
    print("Starting video generation...")
    
    try:
        # 1. Base Data (Time, Blinks, Pupil)
        w_ts = pd.read_csv(data_dir / 'world_timestamps.csv')
        blinks = pd.read_csv(data_dir / 'blinks.csv')
        pupil = pd.read_csv(data_dir / '3d_eye_states.csv')
        
        merged = pd.merge_asof(w_ts, blinks.add_suffix('_blink'), left_on='timestamp [ns]', right_on='start timestamp [ns]_blink', direction='nearest')
        merged = pd.merge_asof(merged, pupil, on='timestamp [ns]', direction='nearest')

        # 2. RAW Gaze (Baseline - Always loaded)
        if (data_dir / 'gaze_raw.csv').exists():
            gaze_raw = pd.read_csv(data_dir / 'gaze_raw.csv')
            # Rinomina colonne raw per chiarezza
            cols_map = {c: f"{c}_RAW" for c in gaze_raw.columns if c != 'timestamp [ns]'}
            gaze_raw = gaze_raw.rename(columns=cols_map)
            merged = pd.merge_asof(merged, gaze_raw, on='timestamp [ns]', direction='nearest')

        # 3. Active Enrichments
        for idx in active_enrichment_indices:
            enr_file = data_dir / f'gaze_enr_{idx}.csv'
            if enr_file.exists():
                gaze_enr = pd.read_csv(enr_file)
                # Suffix for this enrichment index
                cols_map = {c: f"{c}_enr{idx}" for c in gaze_enr.columns if c != 'timestamp [ns]'}
                gaze_enr = gaze_enr.rename(columns=cols_map)
                merged = pd.merge_asof(merged, gaze_enr, on='timestamp [ns]', direction='nearest')
                
                # Load surface polygon info if available
                surf_file = data_dir / f'surface_positions_enr_{idx}.csv'
                if surf_file.exists():
                    surf_df = pd.read_csv(surf_file)
                    cols_map_surf = {c: f"{c}_surf_enr{idx}" for c in surf_df.columns if c != 'timestamp [ns]'}
                    surf_df = surf_df.rename(columns=cols_map_surf)
                    merged = pd.merge_asof(merged, surf_df, on='timestamp [ns]', direction='nearest')

    except Exception as e:
        print(f"Error loading data for video: {e}")
        return

    cap = cv2.VideoCapture(str(data_dir / "external.mp4"))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # History buffers
    gaze_histories = {} # Key: 'RAW' or 'ENR_i', Value: list of points
    pupil_history_l = []
    pupil_history_r = []
    fragmentation_history = [] # Only for RAW mainly or first enriched
    prev_gaze_point = None

    for i in tqdm(range(total_frames), desc="Rendering Video"):
        ret, frame = cap.read()
        if not ret: break
        
        try:
            row = merged.iloc[i]
            ts = row['timestamp [ns]']
            
            # --- 1. Event & Blink Overlays (Standard) ---
            curr_event = events_df[events_df['timestamp [ns]'] <= ts]
            if not curr_event.empty:
                evt_name = curr_event.iloc[-1]['name']
                if evt_name not in ['recording.begin', 'recording.end']:
                    text = f"Event: {evt_name}"
                    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
                    cv2.rectangle(frame, (20, 50 - th - bl), (20 + tw, 50 + bl), (0, 0, 0), -1)
                    cv2.putText(frame, text, (20, 50), font, scale, (255, 255, 255), thick)
            
            if pd.notna(row.get('start timestamp [ns]_blink')) and ts >= row['start timestamp [ns]_blink'] and ts <= (row['start timestamp [ns]_blink'] + row['duration [ms]_blink'] * 1_000_000):
                btxt = "BLINK"
                font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                (tw, th), bl = cv2.getTextSize(btxt, font, scale, thick)
                cv2.rectangle(frame, (width - 20 - tw, 50 - th - bl), (width - 20, 50 + bl), (0, 0, 0), -1)
                cv2.putText(frame, btxt, (width - 20 - tw, 50), font, scale, (0, 255, 255), thick)

            # --- 2. RAW Gaze Overlay ---
            gx_raw, gy_raw = None, None
            if 'gaze x [px]_RAW' in row and pd.notna(row['gaze x [px]_RAW']):
                gx_raw, gy_raw = int(row['gaze x [px]_RAW']), int(row['gaze y [px]_RAW'])
            elif 'gaze x [normalized]_RAW' in row and pd.notna(row['gaze x [normalized]_RAW']):
                gx_raw, gy_raw = int(row['gaze x [normalized]_RAW'] * width), int(row['gaze y [normalized]_RAW'] * height)
            
            if gx_raw is not None:
                # Update history
                if 'RAW' not in gaze_histories: gaze_histories['RAW'] = []
                gaze_histories['RAW'].append((gx_raw, gy_raw))
                if len(gaze_histories['RAW']) > 20: gaze_histories['RAW'].pop(0)
                
                # Draw RAW (Blue)
                cv2.circle(frame, (gx_raw, gy_raw), 10, GAZE_COLOR_RAW, 2)
                for j in range(1, len(gaze_histories['RAW'])):
                    cv2.line(frame, gaze_histories['RAW'][j-1], gaze_histories['RAW'][j], GAZE_COLOR_RAW, 2)
                
                # Frag calc on RAW
                if prev_gaze_point:
                    dist = np.sqrt((gx_raw - prev_gaze_point[0])**2 + (gy_raw - prev_gaze_point[1])**2)
                    fragmentation_history.append(dist)
                else:
                    fragmentation_history.append(0)
                if len(fragmentation_history) > 100: fragmentation_history.pop(0)
                prev_gaze_point = (gx_raw, gy_raw)

            # --- 3. Enrichment Overlays ---
            for idx in active_enrichment_indices:
                # 3a. Surface Polygon (if present)
                if f'tl x [px]_surf_enr{idx}' in row and pd.notna(row[f'tl x [px]_surf_enr{idx}']):
                    tl = (int(row[f'tl x [px]_surf_enr{idx}']), int(row[f'tl y [px]_surf_enr{idx}']))
                    tr = (int(row[f'tr x [px]_surf_enr{idx}']), int(row[f'tr y [px]_surf_enr{idx}']))
                    bl = (int(row[f'bl x [px]_surf_enr{idx}']), int(row[f'bl y [px]_surf_enr{idx}']))
                    br = (int(row[f'br x [px]_surf_enr{idx}']), int(row[f'br y [px]_surf_enr{idx}']))
                    poly_pts = np.array([tl, tr, br, bl], dtype=np.int32)
                    cv2.polylines(frame, [poly_pts], isClosed=True, color=SURFACE_POLY_COLOR, thickness=2)

                # 3b. Gaze Point
                gx_enr, gy_enr = None, None
                col_x_px = f'gaze x [px]_enr{idx}'
                col_x_norm = f'gaze x [normalized]_enr{idx}'
                
                if col_x_px in row and pd.notna(row[col_x_px]):
                    gx_enr, gy_enr = int(row[col_x_px]), int(row[f'gaze y [px]_enr{idx}'])
                elif col_x_norm in row and pd.notna(row[col_x_norm]):
                    gx_enr, gy_enr = int(row[col_x_norm] * width), int(row[f'gaze y [normalized]_enr{idx}'] * height)
                
                if gx_enr is not None:
                    # Logic Green/Red based on surface detection
                    is_on_surf = row.get(f'gaze detected on surface_enr{idx}') == True
                    color = GAZE_COLOR_IN if is_on_surf else GAZE_COLOR_OUT
                    
                    # Update History
                    key = f'ENR_{idx}'
                    if key not in gaze_histories: gaze_histories[key] = []
                    gaze_histories[key].append((gx_enr, gy_enr))
                    if len(gaze_histories[key]) > 20: gaze_histories[key].pop(0)

                    # Draw (Slightly larger or different style to distinguish from raw?)
                    cv2.circle(frame, (gx_enr, gy_enr), 12, color, 3)
                    # Label the cursor? Optional, maybe cluttering.
                    
                    for j in range(1, len(gaze_histories[key])):
                        cv2.line(frame, gaze_histories[key][j-1], gaze_histories[key][j], color, 2)
                    
                    if is_on_surf:
                         cv2.putText(frame, f"ON_SURF_{idx}", (20 + (idx*150), height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 4. Plots (Pupil & Frag)
            # ... (Drawing code identical to previous version, just ensuring data exists)
            # Pupil Plot
            if pd.notna(row.get('pupil diameter left [mm]')): pupil_history_l.append(row['pupil diameter left [mm]'])
            if len(pupil_history_l) > 100: pupil_history_l.pop(0)
            if pd.notna(row.get('pupil diameter right [mm]')): pupil_history_r.append(row['pupil diameter right [mm]'])
            if len(pupil_history_r) > 100: pupil_history_r.pop(0)
            
            plot_w, plot_h = 300, 100
            plot_x, plot_y = width - plot_w - 20, height - plot_h - 20
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "Pupil Diameter", (plot_x + 5, plot_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            p_min, p_max = 2.0, 8.0 
            if len(pupil_history_l) > 1:
                pts_l = [(int(plot_x + (i/100)*plot_w), int(plot_y + plot_h - ((v-p_min)/(p_max-p_min))*plot_h)) for i,v in enumerate(pupil_history_l)]
                cv2.polylines(frame, [np.array(pts_l)], False, PUPIL_LEFT_COLOR_BGR, 2)
            if len(pupil_history_r) > 1:
                pts_r = [(int(plot_x + (i/100)*plot_w), int(plot_y + plot_h - ((v-p_min)/(p_max-p_min))*plot_h)) for i,v in enumerate(pupil_history_r)]
                cv2.polylines(frame, [np.array(pts_r)], False, PUPIL_RIGHT_COLOR_BGR, 2)

            # Frag Plot (Using RAW data history)
            frag_y = plot_y - plot_h - 20
            overlay_f = frame.copy()
            cv2.rectangle(overlay_f, (plot_x, frag_y), (plot_x + plot_w, frag_y + plot_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay_f, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "Gaze Fragmentation (RAW)", (plot_x + 5, frag_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if len(fragmentation_history) > 1:
                f_min, f_max = 0, 150
                pts_f = [(int(plot_x + (i/100)*plot_w), int(frag_y + plot_h - ((min(v, f_max)-f_min)/(f_max-f_min))*plot_h)) for i,v in enumerate(fragmentation_history)]
                cv2.polylines(frame, [np.array(pts_f)], False, (128, 0, 128), 2)

        except (IndexError, KeyError): pass
        writer.write(frame)
        
    cap.release()
    writer.release()
    print(f"Video saved to: {output_file}")

# ==============================================================================
# 4. EVENT EDITOR (LITE) - Unchanged
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
    def _get_selected_indices(self): return [int(item) for item in self.tree.selection()]
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
            self.events_df = pd.concat([self.events_df, pd.DataFrame([{'name': name, 'timestamp [ns]': ts}])], ignore_index=True)
            self._populate_table()
    def _delete_events(self):
        indices = self._get_selected_indices()
        if indices and messagebox.askyesno("Confirm Delete", f"Delete {len(indices)} event(s)?"):
            self.events_df = self.events_df.drop(indices).reset_index(drop=True)
            self._populate_table()
    def _rename_event(self):
        indices = self._get_selected_indices()
        if len(indices) == 1:
            old_name = self.events_df.loc[indices[0], 'name']
            new_name = simpledialog.askstring("Rename", "New name:", initialvalue=old_name)
            if new_name:
                self.events_df.loc[indices[0], 'name'] = new_name
                self._populate_table()
    def _move_event(self):
        indices = self._get_selected_indices()
        if len(indices) == 1:
            if messagebox.askyesno("Confirm", "Move event?"):
                self.events_df.loc[indices[0], 'timestamp [ns]'] = self.world_ts.iloc[self.current_frame]['timestamp [ns]']
                self._populate_table()
    def _merge_events(self):
        indices = self._get_selected_indices()
        if len(indices) >= 2:
            new_name = simpledialog.askstring("Merge", "Merged event name:")
            if new_name:
                ts = self.events_df.loc[min(indices), 'timestamp [ns]']
                self.events_df = self.events_df.drop(indices)
                self.events_df = pd.concat([self.events_df, pd.DataFrame([{'name': new_name, 'timestamp [ns]': ts}])], ignore_index=True)
                self._populate_table()
    def save_exit(self):
        self.saved_df = self.events_df
        self.destroy()

# ==============================================================================
# 5. MAIN APP (Dynamic Enrichment UI)
# ==============================================================================

class SpeedLiteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED Light")
        self.root.geometry("700x750")
        
        self.data_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        
        # List of dicts: {'path': StringVar, 'video': BooleanVar, 'frame': widget}
        self.enrichment_rows = []
        
        self.working_dir = None
        self.gui_queue = queue.Queue() 
        
        pad = {'padx': 10, 'pady': 5}
        
        # --- INPUTS ---
        grp_in = tk.LabelFrame(root, text="Input Folders")
        grp_in.pack(fill=tk.X, **pad)
        
        self._build_file_select(grp_in, "Data Folder (Mandatory):", self.data_dir)
        
        # Dynamic Enrichment Area
        self.enrich_frame = tk.LabelFrame(grp_in, text="Enrichment Folders (Optional - Progressive ID)")
        self.enrich_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(self.enrich_frame, text="+ Aggiungi Enrichment", command=self.add_enrichment_row, bg="#e0f7fa").pack(anchor=tk.W, padx=5, pady=5)
        
        self._build_file_select(grp_in, "Output Folder (Default: Data/Output):", self.output_dir)
        
        # --- ACTIONS ---
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
        
    def add_enrichment_row(self):
        idx = len(self.enrichment_rows)
        row_f = tk.Frame(self.enrich_frame)
        row_f.pack(fill=tk.X, pady=2)
        
        path_var = tk.StringVar()
        video_var = tk.BooleanVar(value=True) # Default checked for video
        
        tk.Label(row_f, text=f"Enrichment {idx}:").pack(side=tk.LEFT)
        tk.Entry(row_f, textvariable=path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        tk.Button(row_f, text="Browse", command=lambda: self.browse(path_var)).pack(side=tk.LEFT)
        tk.Checkbutton(row_f, text="Video?", variable=video_var).pack(side=tk.LEFT, padx=10)
        
        self.enrichment_rows.append({'path': path_var, 'video': video_var, 'frame': row_f})

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
                if msg_type == 'log': self.log(content)
                elif msg_type == 'info': messagebox.showinfo("Info", content)
                elif msg_type == 'error': messagebox.showerror("Error", content)
                elif msg_type == 'enable_buttons': self.btn_run.config(state=tk.NORMAL)
        except queue.Empty: pass
        finally: self.root.after(100, self._check_queue)

    def load_data(self):
        data = Path(self.data_dir.get())
        if not data.exists():
            messagebox.showerror("Error", "The Data folder is mandatory.")
            return
        
        enrich_paths = []
        for row in self.enrichment_rows:
            p = row['path'].get()
            if p: enrich_paths.append(Path(p))
        
        if not self.output_dir.get():
            self.output_dir.set(str(data / "Output"))
        
        try:
            self.log("Preparing working directory...")
            self.working_dir, warnings = prepare_working_directory(data, enrich_paths, Path(self.output_dir.get()))
            self.log(f"Unified files in: {self.working_dir}")
            
            if warnings: messagebox.showwarning("Dati Mancanti", "\n".join(warnings))

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
            if 'selected' in editor.saved_df.columns: editor.saved_df = editor.saved_df.drop(columns=['selected'])
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

            # LOAD BASE FILES
            gaze_raw = pd.read_csv(self.working_dir / "gaze_raw.csv")
            fix_raw = pd.read_csv(self.working_dir / "fixations_raw.csv")
            pupil = pd.read_csv(self.working_dir / "3d_eye_states.csv")
            sacc = pd.read_csv(self.working_dir / "saccades.csv")
            blinks = pd.read_csv(self.working_dir / "blinks.csv")
            
            cap = cv2.VideoCapture(str(self.working_dir / "external.mp4"))
            res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            cap.release()
            
            all_metrics = []
            pdf_dir = Path(self.output_dir.get()) / "Plots_PDF"
            pdf_dir.mkdir(exist_ok=True)
            
            # Identify active enrichments available
            avail_enrichments = []
            for i in range(len(self.enrichment_rows)):
                if (self.working_dir / f"gaze_enr_{i}.csv").exists():
                    avail_enrichments.append(i)

            for i in range(len(ev)):
                start_ts = ev.iloc[i]['timestamp [ns]']
                end_ts = ev.iloc[i+1]['timestamp [ns]'] if i < len(ev)-1 else gaze_raw['timestamp [ns]'].max()
                evt_name = ev.iloc[i]['name']
                self.gui_queue.put(('log', f"Processing event: {evt_name}"))
                
                # --- PROCESS RAW ---
                seg_raw = {
                    'fixations': fix_raw[(fix_raw['start timestamp [ns]'] >= start_ts) & (fix_raw['start timestamp [ns]'] < end_ts)],
                    'pupil': pupil[(pupil['timestamp [ns]'] >= start_ts) & (pupil['timestamp [ns]'] < end_ts)],
                    'blinks': blinks[(blinks['start timestamp [ns]'] >= start_ts) & (blinks['start timestamp [ns]'] < end_ts)],
                    'saccades': sacc[(sacc['start timestamp [ns]'] >= start_ts) & (sacc['start timestamp [ns]'] < end_ts)],
                    'gaze': gaze_raw[(gaze_raw['timestamp [ns]'] >= start_ts) & (gaze_raw['timestamp [ns]'] < end_ts)]
                }
                m_raw = calculate_metrics(seg_raw, res)
                m_final = {k + '_RAW': v for k, v in m_raw.items()}
                
                # PLOTS RAW
                if not seg_raw['gaze'].empty:
                    gx = 'gaze x [px]' if 'gaze x [px]' in seg_raw['gaze'].columns else 'gaze x [normalized]'
                    gy = 'gaze y [px]' if 'gaze y [px]' in seg_raw['gaze'].columns else 'gaze y [normalized]'
                    generate_heatmap_pdf(seg_raw['gaze'], gx, gy, f"Gaze Heatmap (RAW) - {evt_name}", f"heatmap_gaze_RAW_{evt_name}", pdf_dir, res, cmap=CMAP_RAW)
                    generate_fragmentation_plot(seg_raw['gaze'], gx, gy, start_ts, f"Gaze Frag (RAW) - {evt_name}", f"frag_gaze_RAW_{evt_name}", pdf_dir)

                if not seg_raw['fixations'].empty:
                    fx = 'fixation x [px]' if 'fixation x [px]' in seg_raw['fixations'].columns else 'fixation x [normalized]'
                    fy = 'fixation y [px]' if 'fixation y [px]' in seg_raw['fixations'].columns else 'fixation y [normalized]'
                    generate_heatmap_pdf(seg_raw['fixations'], fx, fy, f"Fixation Heatmap (RAW) - {evt_name}", f"heatmap_fixation_RAW_{evt_name}", pdf_dir, res, cmap=CMAP_RAW)

                # --- PROCESS ENRICHMENTS ---
                for idx in avail_enrichments:
                    gaze_enr = pd.read_csv(self.working_dir / f"gaze_enr_{idx}.csv")
                    fix_enr = pd.read_csv(self.working_dir / f"fixations_enr_{idx}.csv") if (self.working_dir / f"fixations_enr_{idx}.csv").exists() else pd.DataFrame()
                    
                    seg_enr = {
                        'gaze': gaze_enr[(gaze_enr['timestamp [ns]'] >= start_ts) & (gaze_enr['timestamp [ns]'] < end_ts)],
                        'fixations': fix_enr[(fix_enr['start timestamp [ns]'] >= start_ts) & (fix_enr['start timestamp [ns]'] < end_ts)] if not fix_enr.empty else pd.DataFrame(),
                        'pupil': seg_raw['pupil'], 'blinks': seg_raw['blinks'], 'saccades': seg_raw['saccades']
                    }
                    m_enr = calculate_metrics(seg_enr, res)
                    m_final.update({k + f'_ENR_{idx}': v for k, v in m_enr.items()})

                    # PLOTS ENR
                    if not seg_enr['gaze'].empty:
                        gx = 'gaze x [px]' if 'gaze x [px]' in seg_enr['gaze'].columns else 'gaze x [normalized]'
                        gy = 'gaze y [px]' if 'gaze y [px]' in seg_enr['gaze'].columns else 'gaze y [normalized]'
                        generate_heatmap_pdf(seg_enr['gaze'], gx, gy, f"Gaze Heatmap (ENR {idx}) - {evt_name}", f"heatmap_gaze_ENR_{idx}_{evt_name}", pdf_dir, res, cmap=CMAP_ENRICHED)
                        
                        if 'gaze detected on surface' in seg_enr['gaze'].columns:
                            surf_gaze = seg_enr['gaze'][seg_enr['gaze']['gaze detected on surface'] == True]
                            if not surf_gaze.empty:
                                generate_heatmap_pdf(surf_gaze, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', f"Gaze Heatmap (Surface {idx}) - {evt_name}", f"heatmap_gaze_ENR_{idx}_SURF_{evt_name}", pdf_dir, res, cmap=CMAP_ENRICHED)
                                generate_fragmentation_plot(surf_gaze, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]', start_ts, f"Frag Surface {idx} - {evt_name}", f"frag_surf_{idx}_{evt_name}", pdf_dir)

                m_final['event_name'] = evt_name
                all_metrics.append(m_final)
                
                # Pupil Plot (Common)
                generate_pupil_timeseries_pdf(seg_raw['pupil'], seg_raw['gaze'], seg_raw['blinks'], start_ts, evt_name, f"pupil_ts_{evt_name}", pdf_dir)

            out_file = Path(self.output_dir.get()) / "Speed_Lite_Results.xlsx"
            pd.DataFrame(all_metrics).to_excel(out_file, index=False)
            self.gui_queue.put(('log', f"Metrics saved to {out_file}"))
            
            # VIDEO GENERATION
            self.gui_queue.put(('log', "Generating Final Video..."))
            vid_out = Path(self.output_dir.get()) / "final_video_overlay.mp4"
            
            # Determine active video indices from checkboxes
            active_video_indices = []
            for i, row in enumerate(self.enrichment_rows):
                if row['video'].get() and i in avail_enrichments:
                    active_video_indices.append(i)
            
            generate_full_video(self.working_dir, vid_out, ev, active_video_indices)
            
            # FULL DURATION PLOTS
            self.gui_queue.put(('log', "Generating full duration plots..."))
            fd_dir = Path(self.output_dir.get()) / "Full_Duration_Plots"
            fd_dir.mkdir(exist_ok=True)
            v_start = gaze_raw['timestamp [ns]'].min()
            generate_pupil_timeseries_pdf(pupil, gaze_raw, blinks, v_start, "Full Pupil", "full_pupil", fd_dir)
            gx = 'gaze x [px]' if 'gaze x [px]' in gaze_raw.columns else 'gaze x [normalized]'
            gy = 'gaze y [px]' if 'gaze y [px]' in gaze_raw.columns else 'gaze y [normalized]'
            generate_fragmentation_plot(gaze_raw, gx, gy, v_start, "Full Frag Raw", "full_frag_raw", fd_dir, events_df=ev, show_surface_bands=True)

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