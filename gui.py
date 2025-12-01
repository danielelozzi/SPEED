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


# --- CONFIGURATION ---
PLOT_DPI = 150
GAZE_COLOR = (0, 0, 255)  # Red BGR for video
PUPIL_LEFT_COLOR = 'blue'
PUPIL_RIGHT_COLOR = 'orange'
PUPIL_LEFT_COLOR_BGR = (255, 255, 0)  # Cyan for video plot
PUPIL_RIGHT_COLOR_BGR = (0, 165, 255) # Orange for video plot

# ==============================================================================
# 1. FILE MANAGEMENT AND DATA PREPARATION
# ==============================================================================

def prepare_working_directory(data_dir, enrichment_dir, output_dir):
    """
    Creates the 'files' folder with precedence logic:
    - Enrichment wins for gaze.csv and fixations.csv
    - Data wins for the rest
    """
    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    
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

    # Handle Enrichment precedence for Gaze and Fixations
    for filename in ['gaze.csv', 'fixations.csv', 'surface_positions.csv']:
        # If enrichment is provided AND the file exists there, take it from there
        if enrichment_dir and (enrichment_dir / filename).exists():
            shutil.copy(enrichment_dir / filename, files_dir / filename)
            print(f"Used {filename} from Enrichment.")
        # Otherwise, take it from Data
        elif (data_dir / filename).exists():
            shutil.copy(data_dir / filename, files_dir / filename)
            print(f"Used {filename} from Data.")
        # Only create dummy files for gaze and fixations, not for surface_positions
        elif filename in ['gaze.csv', 'fixations.csv']:
            # Create an empty dummy if it's missing
            pd.DataFrame().to_csv(files_dir / filename)
            print(f"Warning: {filename} not found, created empty file.")

    return files_dir

# ==============================================================================
# 2. FEATURE CALCULATION AND PLOTTING
# ==============================================================================

def calculate_metrics(df_seg, video_res=(1280, 720)):
    """
    Calculates metrics. Always calculates from raw data.
    If enrichment data is present in fixations, it adds normalized surface metrics.
    """
    metrics = {}
    
    fix = df_seg.get('fixations', pd.DataFrame())
    pupil = df_seg.get('pupil', pd.DataFrame())
    blinks = df_seg.get('blinks', pd.DataFrame())
    saccades = df_seg.get('saccades', pd.DataFrame())

    # --- 1. Base Fixation Metrics (Always from raw data) ---
    if not fix.empty and 'duration [ms]' in fix.columns:
        metrics['n_fixations'] = len(fix)
        metrics['mean_duration_fixations'] = fix['duration [ms]'].mean()
        metrics['std_duration_fixations'] = fix['duration [ms]'].std()
        
        # Use pixel data for base metrics, then normalize for output consistency
        if 'fixation x [px]' in fix.columns:
            fx = fix['fixation x [px]'] / video_res[0]
            fy = fix['fixation y [px]'] / video_res[1]
        else: # Fallback to normalized if pixel is not available
            fx = fix.get('fixation x [normalized]', pd.Series(dtype=float))
            fy = fix.get('fixation y [normalized]', pd.Series(dtype=float))
            
        metrics['mean_x_pos_fixations'] = fx.mean()
        metrics['mean_y_pos_fixations'] = fy.mean()
        metrics['std_x_pos_fixations'] = fx.std()
        metrics['std_y_pos_fixations'] = fy.std()

        # --- 2. Enriched/Normalized Metrics (Conditional) ---
        # If surface data is available, calculate additional [norm] metrics from it.
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
            
    # --- 3. Other Metrics (Blinks, Pupil, Saccades) ---
    metrics['n_blink'] = len(blinks)
    
    # 9-14: Pupil metrics
    if not pupil.empty:
        # Right
        if 'pupil diameter right [mm]' in pupil.columns:
            p_dx = pupil['pupil diameter right [mm]']
            metrics['mean_mm_dx_pupil'] = p_dx.mean()
            metrics['std_mm_dx_pupil'] = p_dx.std()
        else:
            metrics['mean_mm_dx_pupil'] = np.nan
            metrics['std_mm_dx_pupil'] = np.nan
            
        # Left
        if 'pupil diameter left [mm]' in pupil.columns:
            p_sx = pupil['pupil diameter left [mm]']
            metrics['mean_mm_sx_pupil'] = p_sx.mean()
            metrics['std_mm_sx_pupil'] = p_sx.std()
        else:
            metrics['mean_mm_sx_pupil'] = np.nan
            metrics['std_mm_sx_pupil'] = np.nan
            
        # Average (R+L)/2
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

    # 15-16: Saccades (Duration ms)
    if not saccades.empty and 'duration [ms]' in saccades.columns:
        metrics['mean_saccades'] = saccades['duration [ms]'].mean()
        metrics['std_saccades'] = saccades['duration [ms]'].std()
    else:
        metrics['mean_saccades'] = np.nan
        metrics['std_saccades'] = np.nan
        
    return metrics

def generate_heatmap_pdf(df, x_col, y_col, title, filename, output_dir, video_res=(1280, 720)):
    """Generates a KDE heatmap and saves it to PDF."""
    if df.empty or x_col not in df.columns or len(df) < 5:
        return

    x = df[x_col].dropna()
    y = df[y_col].dropna()
    
    is_surface_plot = 'surface' in filename.lower()

    if is_surface_plot:
        # For surface plots, keep data normalized (0-1) and set axes accordingly
        x_limit, y_limit = 1.0, 1.0
        x_label, y_label = 'X [normalized]', 'Y [normalized]'
    else:
        # For other plots, convert to pixels if necessary
        if x.max() <= 1.0: # Normalized -> Pixel
            x = x * video_res[0]
            y = y * video_res[1]
        x_limit, y_limit = video_res[0], video_res[1]
        x_label, y_label = 'X [px]', 'Y [px]'

    plt.figure(figsize=(10, 6))
    try:
        k = gaussian_kde(np.vstack([x, y]))
        # Adjust grid to the correct space (pixel or normalized)
        xi, yi = np.mgrid[0:x_limit:100j, 0:y_limit:100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        zi_reshaped = zi.reshape(xi.shape)

        # Save the numpy matrices
        numpy_dir = output_dir.parent / "Numpy_Matrices"
        numpy_dir.mkdir(exist_ok=True)
        np.save(numpy_dir / f"{filename}_xi.npy", xi)
        np.save(numpy_dir / f"{filename}_yi.npy", yi)
        np.save(numpy_dir / f"{filename}_zi.npy", zi_reshaped)

        
        plt.pcolormesh(xi, yi, zi_reshaped, shading='auto', cmap='jet')
        plt.colorbar(label='Density')
        plt.xlim(0, x_limit)
        plt.ylim(y_limit, 0) # Invert Y-axis
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
    
    # Normalize timestamps to start from 0 for this event
    pupil_df['time_norm'] = pupil_df['timestamp [ns]'] - start_ts

    # --- Draw On-Surface Bands ---
    # Use the correct column name for on_surface
    surface_col = 'gaze detected on surface' if 'gaze detected on surface' in gaze_df.columns else 'on_surface'

    if not gaze_df.empty and surface_col in gaze_df.columns:
        gaze_df['time_norm'] = gaze_df['timestamp [ns]'] - start_ts
        # Find blocks of consecutive True/False values
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
            blink_end = blink_start + (blink_duration_ms * 1_000_000) # convert ms to ns

            # Normalize to the event's timeline
            norm_blink_start = blink_start - start_ts
            norm_blink_end = blink_end - start_ts
            plt.axvspan(norm_blink_start, norm_blink_end, color='blue', alpha=0.2, lw=0)

    has_data = False
    # Left Pupil
    if 'pupil diameter left [mm]' in pupil_df.columns:
        plt.plot(pupil_df['time_norm'], pupil_df['pupil diameter left [mm]'], 
                 label='Left Pupil', color=PUPIL_LEFT_COLOR, alpha=0.7)
        has_data = True
        
    # Right Pupil
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

        # --- Save the timeseries data to a .npy file ---
        try:
            numpy_dir = output_dir.parent / "Numpy_Matrices"
            numpy_dir.mkdir(exist_ok=True)
            
            # Prepare data for saving: [time, left_pupil, right_pupil]
            time_col = pupil_df['time_norm'].values
            left_col = pupil_df.get('pupil diameter left [mm]', pd.Series(np.nan, index=pupil_df.index)).values
            right_col = pupil_df.get('pupil diameter right [mm]', pd.Series(np.nan, index=pupil_df.index)).values
            
            timeseries_data = np.vstack([time_col, left_col, right_col]).T
            np.save(numpy_dir / f"{filename}_timeseries.npy", timeseries_data)
        except Exception as e:
            print(f"Could not save timeseries numpy array for {filename}: {e}")

        plt.savefig(out_path, format='pdf', bbox_inches='tight')
    plt.close()

# ==============================================================================
# 3. VIDEO GENERATION (Simplified Engine)
# ==============================================================================

def generate_full_video(data_dir, output_file, events_df, enrichment_dir=None):
    """Generates the video with all overlays."""
    print("Starting video generation...")
    
    try:
        w_ts = pd.read_csv(data_dir / 'world_timestamps.csv')
        gaze = pd.read_csv(data_dir / 'gaze.csv')
        blinks = pd.read_csv(data_dir / 'blinks.csv')
        pupil = pd.read_csv(data_dir / '3d_eye_states.csv')
        
        # Merge data frame-by-frame
        merged = pd.merge_asof(w_ts, gaze, on='timestamp [ns]', direction='nearest')
        merged = pd.merge_asof(merged, blinks.add_suffix('_blink'), left_on='timestamp [ns]', right_on='start timestamp [ns]_blink', direction='nearest')
        merged = pd.merge_asof(merged, pupil, on='timestamp [ns]', direction='nearest')

        # Load and merge surface positions if the file was copied
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
                # Filter system events for a cleaner display
                if evt_name not in ['recording.begin', 'recording.end']:
                    text = f"Event: {evt_name}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    
                    # Draw a black background rectangle
                    cv2.rectangle(frame, (20, 50 - text_height - baseline), (20 + text_width, 50 + baseline), (0, 0, 0), -1)
                    # Put the text on top of the rectangle
                    cv2.putText(frame, text, (20, 50), font, font_scale, (255, 255, 255), thickness)
            
            # 1.5 Blink & On-Surface Overlay
            # Check if the current frame's timestamp is within a blink's duration
            if pd.notna(row.get('start timestamp [ns]_blink')) and ts >= row['start timestamp [ns]_blink'] and ts <= (row['start timestamp [ns]_blink'] + row['duration [ms]_blink'] * 1_000_000):
                blink_text = "BLINK"
                blink_font = cv2.FONT_HERSHEY_SIMPLEX
                blink_font_scale = 1.2
                blink_thickness = 3
                (text_w, text_h), blink_baseline = cv2.getTextSize(blink_text, blink_font, blink_font_scale, blink_thickness)
                
                text_x = width - 20 - text_w # Position from right edge
                text_y = 50
                
                cv2.rectangle(frame, (text_x, text_y - text_h - blink_baseline), (text_x + text_w, text_y + blink_baseline), (0, 0, 0), -1)
                cv2.putText(frame, blink_text, (text_x, text_y), blink_font, blink_font_scale, (0, 255, 255), blink_thickness)
            
            # Check for on_surface from enrichment data using the correct column name
            if row.get('gaze detected on surface') == True:
                 cv2.putText(frame, "ON_SURFACE", (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 1.8 Draw Dynamic Surface Polygon
            # Check if the surface position columns (with _surf suffix) exist in the merged data for this frame
            if 'tl x [px]_surf' in row and pd.notna(row['tl x [px]_surf']):
                tl = (int(row['tl x [px]_surf']), int(row['tl y [px]_surf']))
                tr = (int(row['tr x [px]_surf']), int(row['tr y [px]_surf']))
                bl = (int(row['bl x [px]_surf']), int(row['bl y [px]_surf']))
                br = (int(row['br x [px]_surf']), int(row['br y [px]_surf']))
                
                # Define the polygon vertices in order
                surface_poly_pts = np.array([tl, tr, br, bl], dtype=np.int32)
                
                # Draw the polygon
                cv2.polylines(frame, [surface_poly_pts], isClosed=True, color=(255, 255, 0), thickness=2)

            # 2. Gaze Point & Path
            # For video overlay, ALWAYS use non-normalized pixel coordinates if available.
            gx_col = 'gaze x [px]'
            gy_col = 'gaze y [px]'
            
            if pd.notna(row.get(gx_col)):
                gx, gy = row[gx_col], row[gy_col]
                
                gx, gy = int(gx), int(gy)
                cv2.circle(frame, (gx, gy), 15, GAZE_COLOR, 2)
                
                gaze_history.append((gx, gy))
                if len(gaze_history) > 20: gaze_history.pop(0)
                
                for j in range(1, len(gaze_history)):
                    cv2.line(frame, gaze_history[j-1], gaze_history[j], GAZE_COLOR, 2)
            
            # 3. Pupil Plot (bottom right)
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
            
            # Plot background
            overlay = frame.copy()
            cv2.rectangle(overlay, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "Pupil Diameter", (plot_x + 5, plot_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw lines
            p_min, p_max = 2.0, 8.0 # Assumed range for pupil diameter in mm
            
            # Left Pupil Line
            if len(pupil_history_l) > 1:
                pts_l = []
                for idx, val in enumerate(pupil_history_l):
                    px = int(plot_x + (idx / 100) * plot_w)
                    py = int(plot_y + plot_h - ((val - p_min) / (p_max - p_min)) * plot_h)
                    pts_l.append((px, py))
                cv2.polylines(frame, [np.array(pts_l)], False, PUPIL_LEFT_COLOR_BGR, 2)

            # Right Pupil Line
            if len(pupil_history_r) > 1:
                pts_r = []
                for idx, val in enumerate(pupil_history_r):
                    px = int(plot_x + (idx / 100) * plot_w)
                    py = int(plot_y + plot_h - ((val - p_min) / (p_max - p_min)) * plot_h)
                    pts_r.append((px, py))
                cv2.polylines(frame, [np.array(pts_r)], False, PUPIL_RIGHT_COLOR_BGR, 2)

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

        # Main layout
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True)

        # Left Pane: Video Player
        video_frame = tk.Frame(main_pane)
        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.scale = tk.Scale(video_frame, from_=0, to=self.total_frames, orient=tk.HORIZONTAL, command=self.on_slide)
        self.scale.pack(fill=tk.X, padx=10, pady=5)
        main_pane.add(video_frame, width=800)

        # Right Pane: Table and Controls
        controls_frame = tk.Frame(main_pane)
        
        # Table
        table_container = tk.Frame(controls_frame)
        table_container.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        cols = ("#", "Event Name", "Timestamp (ns)")
        self.tree = ttk.Treeview(table_container, columns=cols, show='headings', selectmode='extended')
        for col in cols:
            self.tree.heading(col, text=col)
        self.tree.column("#", width=50, anchor=tk.CENTER)
        self.tree.column("Event Name", width=200)
        self.tree.column("Timestamp (ns)", width=150)
        
        vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<<TreeviewSelect>>', self._on_row_select)

        # Buttons
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
        # Clear existing table
        for i in self.tree.get_children():
            self.tree.delete(i)
        # Repopulate from dataframe
        self.events_df = self.events_df.sort_values('timestamp [ns]').reset_index(drop=True)
        for index, row in self.events_df.iterrows():
            self.tree.insert("", "end", iid=str(index), values=(index, row['name'], row['timestamp [ns]']))

    def _get_selected_indices(self):
        return [int(item) for item in self.tree.selection()]

    def _on_row_select(self, event):
        indices = self._get_selected_indices()
        if not indices: return
        
        # Jump video to the first selected event's timestamp
        selected_idx = indices[0]
        ts = self.events_df.loc[selected_idx, 'timestamp [ns]']
        
        # Find the closest frame index for the timestamp
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
        if not new_name:
            return

        # The new event takes the timestamp of the earliest selected event
        first_event_idx = min(indices)
        new_ts = self.events_df.loc[first_event_idx, 'timestamp [ns]']
        
        # Create the new merged event
        new_row = {'name': new_name, 'timestamp [ns]': new_ts}
        
        # Remove old events and add the new one
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
        
        pad = {'padx': 10, 'pady': 5}
        
        # Inputs
        grp_in = tk.LabelFrame(root, text="Input Folders")
        grp_in.pack(fill=tk.X, **pad)
        
        self._build_file_select(grp_in, "Data Folder (Mandatory):", self.data_dir)
        self._build_file_select(grp_in, "Enrichment Folder (Optional):", self.enrich_dir)
        self._build_file_select(grp_in, "Output Folder (Default: Data/Output):", self.output_dir)
        
        # Actions
        tk.Button(root, text="1. Load and Prepare Data", command=self.load_data, bg="#fff9c4").pack(fill=tk.X, **pad)
        self.btn_edit = tk.Button(root, text="2. Edit Events (Optional)", command=self.edit_events, state=tk.DISABLED)
        self.btn_edit.pack(fill=tk.X, **pad)
        self.btn_run = tk.Button(root, text="3. Estrai Features, Plot & Video", command=self.run_process, bg="#b2dfdb", state=tk.DISABLED)
        self.btn_run.pack(fill=tk.X, **pad) # Changed from "Estrai..."
        
        self.log_box = tk.Text(root, height=12)
        self.log_box.pack(fill=tk.BOTH, **pad)

        # Credits
        credits_text = (
            "Developed by: Dr. Daniele Lozzi (github.com/danielelozzi)\n"
            "Laboratorio di Scienze Cognitive e del Comportamento (SCoC) - Università degli Studi dell’Aquila\n"
            "https://labscoc.wordpress.com/"
        )
        credits_label = tk.Label(root, text=credits_text, justify=tk.CENTER, font=("Arial", 8), fg="grey")
        credits_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 10))
        
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
            self.working_dir = prepare_working_directory(data, enrich, Path(self.output_dir.get()))
            self.log(f"Unified files in: {self.working_dir}")
            
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
            # Ensure 'selected' column is not saved if it exists from old format
            if 'selected' in editor.saved_df.columns:
                editor.saved_df = editor.saved_df.drop(columns=['selected'])
            editor.saved_df.to_csv(events_path, index=False)
            self.log("Updated events saved.")

    def run_process(self):
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        try:
            self.btn_run.config(state=tk.DISABLED)
            self.log("Starting Analysis...")
            
            ev = pd.read_csv(self.working_dir / "events.csv").sort_values('timestamp [ns]')
            # --- FILTER SYSTEM EVENTS ---
            ev = ev[~ev['name'].isin(['recording.begin', 'recording.end', 'begin', 'end'])]
            if ev.empty:
                self.log("Warning: No valid events found after filtering.")
                return

            gaze = pd.read_csv(self.working_dir / "gaze.csv")
            fix = pd.read_csv(self.working_dir / "fixations.csv")
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
                end_ts = ev.iloc[i+1]['timestamp [ns]'] if i < len(ev)-1 else gaze['timestamp [ns]'].max()
                evt_name = ev.iloc[i]['name']
                
                self.log(f"Processing event: {evt_name}")
                
                # Filter segments
                seg = {
                    'fixations': fix[(fix['start timestamp [ns]'] >= start_ts) & (fix['start timestamp [ns]'] < end_ts)],
                    'pupil': pupil[(pupil['timestamp [ns]'] >= start_ts) & (pupil['timestamp [ns]'] < end_ts)],
                    'blinks': blinks[(blinks['start timestamp [ns]'] >= start_ts) & (blinks['start timestamp [ns]'] < end_ts)],
                    'saccades': sacc[(sacc['start timestamp [ns]'] >= start_ts) & (sacc['start timestamp [ns]'] < end_ts)],
                    'gaze': gaze[(gaze['timestamp [ns]'] >= start_ts) & (gaze['timestamp [ns]'] < end_ts)]
                }
                
                # Calculate Metrics
                m = calculate_metrics(seg, res)
                m['event_name'] = evt_name
                all_metrics.append(m)
                
                # --- Plot Generation ---
                # Always generate plots from raw data (using pixel coordinates)
                gaze_seg = seg['gaze']
                if not gaze_seg.empty:
                    gx_raw = 'gaze x [px]' if 'gaze x [px]' in gaze_seg.columns else 'gaze x [normalized]'
                    gy_raw = 'gaze y [px]' if 'gaze y [px]' in gaze_seg.columns else 'gaze y [normalized]'
                    generate_heatmap_pdf(gaze_seg, gx_raw, gy_raw, f"Gaze Heatmap - {evt_name}", f"heatmap_gaze_{evt_name}", pdf_dir, res)

                fix_seg = seg['fixations']
                if not fix_seg.empty:
                    fx_raw = 'fixation x [px]' if 'fixation x [px]' in fix_seg.columns else 'fixation x [normalized]'
                    fy_raw = 'fixation y [px]' if 'fixation y [px]' in fix_seg.columns else 'fixation y [normalized]'
                    generate_heatmap_pdf(fix_seg, fx_raw, fy_raw, f"Fixation Heatmap - {evt_name}", f"heatmap_fixation_{evt_name}", pdf_dir, res)

                # If enrichment data is available, generate additional plots for surface data
                if 'gaze detected on surface' in gaze_seg.columns:
                    surface_gaze = gaze_seg[gaze_seg['gaze detected on surface'] == True]
                    if not surface_gaze.empty:
                        generate_heatmap_pdf(surface_gaze, 'gaze position on surface x [normalized]', 'gaze position on surface y [normalized]',
                                             f"Gaze Heatmap (Surface) - {evt_name}", f"heatmap_gaze_{evt_name}_surface", pdf_dir, res)
                
                if 'fixation detected on surface' in fix_seg.columns:
                    surface_fix = fix_seg[fix_seg['fixation detected on surface'] == True]
                    if not surface_fix.empty:
                        generate_heatmap_pdf(surface_fix, 'fixation position on surface x [normalized]', 'fixation position on surface y [normalized]',
                                             f"Fixation Heatmap (Surface) - {evt_name}", f"heatmap_fixation_{evt_name}_surface", pdf_dir, res)

                # Pupil Timeseries Plot (unchanged)
                pupil_seg = seg['pupil']
                if not pupil_seg.empty:
                    generate_pupil_timeseries_pdf(pupil_seg, seg['gaze'], seg['blinks'], start_ts, evt_name, f"pupil_ts_{evt_name}", pdf_dir)

            # Save Excel
            out_file = Path(self.output_dir.get()) / "Speed_Lite_Results.xlsx"
            pd.DataFrame(all_metrics).to_excel(out_file, index=False)
            self.log(f"Metrics saved to {out_file}")
            
            # Video Generation
            self.log("Generating Final Video...")
            vid_out = Path(self.output_dir.get()) / "final_video_overlay.mp4"
            enrich_path = Path(self.enrich_dir.get()) if self.enrich_dir.get() else None
            generate_full_video(self.working_dir, vid_out, ev, enrichment_dir=enrich_path)
            
            self.log("Analysis Complete.")
            messagebox.showinfo("Finished", "All operations completed successfully!")
            
        except Exception as e:
            self.log(f"Error: {e}")
            print(e)
        finally:
            self.btn_run.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedLiteApp(root)
    root.mainloop()

# ==============================================================================
# CREDITS
# ==============================================================================
#
# Dr. Daniele Lozzi
# github.com/danielelozzi
#
# Laboratorio di Scienze Cognitive e del Comportamento (SCoC)
# Dipartimento di Scienze Cliniche Applicate e Biotecnologiche (DISCAB)
# Università degli Studi dell’Aquila
# https://labscoc.wordpress.com/
#