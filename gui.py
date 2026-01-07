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
import random

# Tenta di importare Ultralytics per la IA
try:
    from ultralytics import YOLO, SAM
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Ultralytics non trovato. Le funzioni AI saranno disabilitate.")

# --- CONFIGURATION ---
PLOT_DPI = 150
GAZE_COLOR_RAW = (255, 0, 0)      # Blue (BGR)
GAZE_COLOR_IN = (0, 255, 0)       # Green (BGR)
GAZE_COLOR_OUT = (0, 0, 255)      # Red (BGR)
SURFACE_POLY_COLOR = (0, 255, 255) # Yellow (BGR)

PUPIL_LEFT_COLOR = 'blue'
PUPIL_RIGHT_COLOR = 'orange'
PUPIL_LEFT_COLOR_BGR = (255, 255, 0) 
PUPIL_RIGHT_COLOR_BGR = (0, 165, 255)

# Heatmap Colormaps
CMAP_RAW = 'winter' 
CMAP_ENRICHED = 'jet' 

# ==============================================================================
# 1. HELPERS & AI ENGINE
# ==============================================================================

def create_tracker():
    """Crea un tracker CSRT in modo robusto su diverse versioni di OpenCV."""
    try:
        # OpenCV standard/contrib moderno
        return cv2.TrackerCSRT_create()
    except AttributeError:
        try:
            # OpenCV legacy API
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            print("ERRORE CRITICO: TrackerCSRT non trovato. Installa 'opencv-contrib-python'.")
            return None

def import_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except: return False

def safe_rename_merge(base_df, new_df, suffix, on_col='timestamp [ns]'):
    """Rinomina le colonne aggiungendo il suffisso, MA preserva la colonna di merge."""
    if new_df.empty: return base_df
    
    # Rinomina tutte le colonne tranne quella di join
    rename_map = {c: f"{c}_{suffix}" for c in new_df.columns if c != on_col}
    new_df_renamed = new_df.rename(columns=rename_map)
    
    # Merge asof
    try:
        merged = pd.merge_asof(base_df, new_df_renamed, on=on_col, direction='nearest')
        return merged
    except Exception as e:
        print(f"Merge error for {suffix}: {e}")
        return base_df

def run_ai_extraction(video_path, timestamps_path, roi_config, output_base_dir, queue_log=None):
    """
    Esegue l'estrazione AI Ibrida:
    1. YOLO Tracking per classi standard (con Re-ID).
    2. OpenCV TrackerCSRT per oggetti custom definiti con SAM.
    """
    if not AI_AVAILABLE:
        if queue_log: queue_log.put(('error', "Libreria Ultralytics non installata."))
        return []

    try:
        world_ts = pd.read_csv(timestamps_path)
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except Exception as e:
        if queue_log: queue_log.put(('error', f"Errore apertura video/ts: {e}"))
        return []

    active_rois = [r for r in roi_config if r.get('active', True)]
    if not active_rois:
        if queue_log: queue_log.put(('log', "Nessuna ROI attiva."))
        return []

    yolo_rois = [r for r in active_rois if r.get('type') == 'yolo']
    custom_rois = [r for r in active_rois if r.get('type') == 'custom']

    # --- SETUP MODELLI ---
    loaded_models = {}
    sam_model = None 
    
    need_sam = any(r.get('use_sam', False) for r in active_rois)
    if need_sam:
        if queue_log: queue_log.put(('log', "Caricamento SAM..."))
        try:
            sam_model = SAM("sam2.1_b.pt") 
        except: 
            if queue_log: queue_log.put(('log', "SAM non trovato, funzioni limitate."))

    # --- SETUP CUSTOM TRACKERS ---
    custom_trackers = []
    for cr in custom_rois:
        custom_trackers.append({
            'roi': cr,
            'tracker': None,
            'status': 'waiting'
        })

    tracking_buffer = {} 
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    map_targets = {}
    for r in yolo_rois:
        key = (r['model_path'], r['target_class'])
        if key not in map_targets: map_targets[key] = []
        map_targets[key].append(r)

    # --- CICLO VIDEO ---
    for i in tqdm(range(len(world_ts)), desc="AI Analysis"):
        ret, frame = cap.read()
        if not ret: break
        ts = world_ts.iloc[i]['timestamp [ns]']

        # A. CUSTOM TRACKERS
        for ct in custom_trackers:
            roi = ct['roi']
            init_f = roi['init_frame']
            
            if i == init_f:
                tracker = create_tracker()
                if tracker:
                    try:
                        tracker.init(frame, tuple(roi['init_box']))
                        ct['tracker'] = tracker
                        ct['status'] = 'tracking'
                    except Exception as e:
                        print(f"Tracker init failed at frame {i}: {e}")
                        ct['status'] = 'lost'
                else:
                    ct['status'] = 'error'
            
            if ct['status'] == 'tracking':
                success, box = ct['tracker'].update(frame)
                if success:
                    final_mask = None
                    if sam_model: 
                        try:
                            x, y, w, h = [int(v) for v in box]
                            b_xyxy = [x, y, x+w, y+h]
                            sam_res = sam_model(frame, bboxes=[b_xyxy], verbose=False)
                            if sam_res[0].masks:
                                final_mask = np.array(sam_res[0].masks.xy[0], dtype=np.int32)
                        except: pass
                    
                    if final_mask is None:
                        x, y, w, h = [int(v) for v in box]
                        final_mask = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)

                    u_key = roi['name'] 
                    if u_key not in tracking_buffer:
                        tracking_buffer[u_key] = {'surface': [], 'polygons': {}, 'frames_visible': 0, 'roi_ref': roi}
                    
                    rect = cv2.minAreaRect(final_mask)
                    box_pts = np.int_(cv2.boxPoints(rect))
                    s = box_pts.sum(axis=1); diff = np.diff(box_pts, axis=1)
                    tl = box_pts[np.argmin(s)]; br = box_pts[np.argmax(s)]
                    tr = box_pts[np.argmin(diff)]; bl = box_pts[np.argmax(diff)]

                    tracking_buffer[u_key]['surface'].append({
                        'timestamp [ns]': ts,
                        'tl x [px]': tl[0], 'tl y [px]': tl[1],
                        'tr x [px]': tr[0], 'tr y [px]': tr[1],
                        'bl x [px]': bl[0], 'bl y [px]': bl[1],
                        'br x [px]': br[0], 'br y [px]': br[1]
                    })
                    tracking_buffer[u_key]['polygons'][i] = final_mask
                    tracking_buffer[u_key]['frames_visible'] += 1

        # B. YOLO TRACKERS
        unique_models = set([r['model_path'] for r in yolo_rois])
        for m_path in unique_models:
            if m_path not in loaded_models: loaded_models[m_path] = YOLO(m_path)
            model = loaded_models[m_path]
            
            try:
                results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, device='cuda' if import_torch_cuda() else 'cpu')
            except: continue
            
            if not results or not results[0].boxes or results[0].boxes.id is None: continue

            for j, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0])
                track_id = int(box.id[0])
                cls_name = model.names[cls_id]
                
                roi_candidates = map_targets.get((m_path, cls_name), [])
                if not roi_candidates and (m_path, 'All Detected Classes') in map_targets:
                    roi_candidates = map_targets[(m_path, 'All Detected Classes')]

                for roi in roi_candidates:
                    unique_obj_key = f"{roi['name']}_{track_id}"
                    if unique_obj_key not in tracking_buffer:
                        tracking_buffer[unique_obj_key] = {'surface': [], 'polygons': {}, 'frames_visible': 0, 'roi_ref': roi}
                    
                    final_mask = None
                    if roi.get('use_sam', False) and sam_model:
                        try:
                            b_xyxy = box.xyxy[0].cpu().numpy()
                            sam_res = sam_model(frame, bboxes=[b_xyxy], verbose=False)
                            if sam_res[0].masks: final_mask = np.array(sam_res[0].masks.xy[0], dtype=np.int32)
                        except: pass
                    
                    if final_mask is None and results[0].masks and len(results[0].masks) > j:
                         try: final_mask = np.array(results[0].masks.xy[j], dtype=np.int32)
                         except: pass

                    if final_mask is None:
                        b = box.xyxy[0].cpu().numpy()
                        final_mask = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=np.int32)

                    if len(final_mask) > 0:
                        rect = cv2.minAreaRect(final_mask)
                        box_pts = np.int_(cv2.boxPoints(rect))
                        s = box_pts.sum(axis=1); diff = np.diff(box_pts, axis=1)
                        tl = box_pts[np.argmin(s)]; br = box_pts[np.argmax(s)]
                        tr = box_pts[np.argmin(diff)]; bl = box_pts[np.argmax(diff)]

                        tracking_buffer[unique_obj_key]['surface'].append({
                            'timestamp [ns]': ts,
                            'tl x [px]': tl[0], 'tl y [px]': tl[1],
                            'tr x [px]': tr[0], 'tr y [px]': tr[1],
                            'bl x [px]': bl[0], 'bl y [px]': bl[1],
                            'br x [px]': br[0], 'br y [px]': br[1]
                        })
                        tracking_buffer[unique_obj_key]['polygons'][i] = final_mask
                        tracking_buffer[unique_obj_key]['frames_visible'] += 1

    cap.release()

    if queue_log: queue_log.put(('log', f"Salvataggio dati per {len(tracking_buffer)} oggetti tracciati..."))

    processed_rois = []
    for obj_key, data in tracking_buffer.items():
        if data['frames_visible'] < 5: continue 

        roi_out_dir = output_base_dir / f"ROI_{obj_key}"
        roi_out_dir.mkdir(parents=True, exist_ok=True)
        
        if data['surface']:
            pd.DataFrame(data['surface']).to_csv(roi_out_dir / 'surface_positions.csv', index=False)
        
        processed_rois.append({
            'name': obj_key, 
            'base_name': data['roi_ref']['name'], 
            'path': roi_out_dir,
            'polygons': data['polygons'],
            'frames_visible': data['frames_visible'],
            'seconds_visible': data['frames_visible'] / fps if fps > 0 else 0
        })

    return processed_rois

# ==============================================================================
# 2. FILE MANAGEMENT
# ==============================================================================

def prepare_working_directory(data_dir, enrichment_dirs, output_dir):
    files_dir = output_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    
    files_map = [('events.csv', False), ('blinks.csv', True), ('3d_eye_states.csv', True), 
                 ('world_timestamps.csv', False), ('saccades.csv', True)]
    
    video_files = list(data_dir.glob("*.mp4"))
    if video_files: shutil.copy(video_files[0], files_dir / "external.mp4")
    else: raise FileNotFoundError("No .mp4 file in Data.")

    for fname, opt in files_map:
        if (data_dir / fname).exists(): shutil.copy(data_dir / fname, files_dir / fname)
        elif not opt: raise FileNotFoundError(f"Missing {fname}")

    for f in ['gaze.csv', 'fixations.csv']:
        if (data_dir / f).exists():
            shutil.copy(data_dir / f, files_dir / f"{Path(f).stem}_raw.csv")
        else:
            pd.DataFrame().to_csv(files_dir / f"{Path(f).stem}_raw.csv")
            warnings.append(f"{f} missing (Raw).")

    for i, enr_dir in enumerate(enrichment_dirs):
        if not enr_dir.exists(): continue
        for f in ['gaze.csv', 'fixations.csv', 'surface_positions.csv']:
            if (enr_dir / f).exists():
                shutil.copy(enr_dir / f, files_dir / f"{Path(f).stem}_enr_{i}.csv")

    return files_dir, warnings

# ==============================================================================
# 3. METRICS CALCULATOR
# ==============================================================================

def calculate_metrics(df_seg, video_res=(1280, 720)):
    m = {}
    fix = df_seg.get('fixations', pd.DataFrame())
    gaze = df_seg.get('gaze', pd.DataFrame())
    pupil = df_seg.get('pupil', pd.DataFrame())
    blinks = df_seg.get('blinks', pd.DataFrame())
    
    if not fix.empty:
        m['n_fixations'] = len(fix)
        m['mean_duration_fixations'] = fix['duration [ms]'].mean()
        m['std_duration_fixations'] = fix['duration [ms]'].std()
    else:
        m['n_fixations'] = 0; m['mean_duration_fixations'] = np.nan; m['std_duration_fixations'] = np.nan

    m['n_blink'] = len(blinks)
    
    if not pupil.empty:
        if 'pupil diameter left [mm]' in pupil.columns:
            m['mean_pupil_L_global'] = pupil['pupil diameter left [mm]'].mean()
        if 'pupil diameter right [mm]' in pupil.columns:
            m['mean_pupil_R_global'] = pupil['pupil diameter right [mm]'].mean()

    # ROI Metrics
    col_fix_surf = 'fixation detected on surface'
    if not fix.empty and col_fix_surf in fix.columns:
        fix_on = fix[fix[col_fix_surf] == True]
        m['ROI_n_fixations'] = len(fix_on)
        m['ROI_mean_fix_duration'] = fix_on['duration [ms]'].mean() if not fix_on.empty else 0
        m['ROI_std_fix_duration'] = fix_on['duration [ms]'].std() if not fix_on.empty else 0
        
        if 'fixation position on surface x [normalized]' in fix.columns and 'fixation position on surface y [normalized]' in fix.columns:
            m['ROI_mean_fix_x'] = fix_on['fixation position on surface x [normalized]'].mean() if not fix_on.empty else np.nan
            m['ROI_mean_fix_y'] = fix_on['fixation position on surface y [normalized]'].mean() if not fix_on.empty else np.nan
    else:
        m['ROI_n_fixations'] = 0; m['ROI_mean_fix_duration'] = np.nan; m['ROI_std_fix_duration'] = np.nan

    col_gaze_surf = 'gaze detected on surface'
    if not gaze.empty and not pupil.empty and col_gaze_surf in gaze.columns:
        gaze_on = gaze[gaze[col_gaze_surf] == True]
        if not gaze_on.empty:
            p_on = pd.merge_asof(gaze_on.sort_values('timestamp [ns]'), 
                                 pupil[['timestamp [ns]', 'pupil diameter left [mm]', 'pupil diameter right [mm]']].sort_values('timestamp [ns]'),
                                 on='timestamp [ns]', direction='nearest', tolerance=50000000)
            
            if 'pupil diameter left [mm]' in p_on.columns:
                m['ROI_mean_pupil_L'] = p_on['pupil diameter left [mm]'].mean()
                m['ROI_std_pupil_L'] = p_on['pupil diameter left [mm]'].std()
            else:
                m['ROI_mean_pupil_L'] = np.nan; m['ROI_std_pupil_L'] = np.nan

            if 'pupil diameter right [mm]' in p_on.columns:
                m['ROI_mean_pupil_R'] = p_on['pupil diameter right [mm]'].mean()
                m['ROI_std_pupil_R'] = p_on['pupil diameter right [mm]'].std()
            else:
                m['ROI_mean_pupil_R'] = np.nan; m['ROI_std_pupil_R'] = np.nan

            if 'pupil diameter left [mm]' in p_on.columns and 'pupil diameter right [mm]' in p_on.columns:
                avg_p = (p_on['pupil diameter left [mm]'] + p_on['pupil diameter right [mm]']) / 2
                m['ROI_mean_pupil_AVG'] = avg_p.mean()
                m['ROI_std_pupil_AVG'] = avg_p.std()
            else:
                m['ROI_mean_pupil_AVG'] = np.nan; m['ROI_std_pupil_AVG'] = np.nan
        else:
            for k in ['ROI_mean_pupil_L', 'ROI_std_pupil_L', 'ROI_mean_pupil_R', 'ROI_std_pupil_R', 'ROI_mean_pupil_AVG', 'ROI_std_pupil_AVG']:
                m[k] = np.nan
    else:
        for k in ['ROI_mean_pupil_L', 'ROI_std_pupil_L', 'ROI_mean_pupil_R', 'ROI_std_pupil_R', 'ROI_mean_pupil_AVG', 'ROI_std_pupil_AVG']:
             m[k] = np.nan

    return m

# ==============================================================================
# 4. PLOTTING & VIDEO
# ==============================================================================
def generate_heatmap_pdf(df, x_col, y_col, title, filename, output_dir, video_res=(1280, 720), cmap='jet'):
    if df.empty or x_col not in df.columns or len(df) < 5: return
    x = df[x_col].dropna(); y = df[y_col].dropna()
    is_surface = 'surface' in filename.lower()
    x_lim, y_lim = (1.0, 1.0) if is_surface else video_res
    x_lbl, y_lbl = ('X [norm]', 'Y [norm]') if is_surface else ('X [px]', 'Y [px]')
    if not is_surface and x.max() <= 1.0: x *= video_res[0]; y *= video_res[1]

    plt.figure(figsize=(10, 6))
    try:
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[0:x_lim:100j, 0:y_lim:100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        np.save(output_dir.parent/"Numpy_Matrices"/f"{filename}_zi.npy", zi)
        plt.pcolormesh(xi, yi, zi, shading='auto', cmap=cmap)
        plt.colorbar(label='Density')
        plt.xlim(0, x_lim); plt.ylim(y_lim, 0)
        plt.title(title); plt.xlabel(x_lbl); plt.ylabel(y_lbl)
        plt.savefig(output_dir / f"{filename}.pdf", format='pdf', bbox_inches='tight')
    except: pass
    plt.close()

def generate_pupil_timeseries_pdf(pupil_df, gaze_df, blinks_df, start_ts, title, filename, output_dir):
    if pupil_df.empty: return
    plt.figure(figsize=(12, 6))
    pupil_df['time_norm'] = pupil_df['timestamp [ns]'] - start_ts
    
    col_surf = 'gaze detected on surface' if 'gaze detected on surface' in gaze_df.columns else 'on_surface'
    if not gaze_df.empty and col_surf in gaze_df.columns:
        gaze_df['t'] = gaze_df['timestamp [ns]'] - start_ts
        gaze_df['blk'] = (gaze_df[col_surf] != gaze_df[col_surf].shift()).cumsum()
        for _, g in gaze_df.groupby('blk'):
            plt.axvspan(g['t'].iloc[0], g['t'].iloc[-1], color='green' if g[col_surf].iloc[0] else 'red', alpha=0.15, lw=0)

    if 'pupil diameter left [mm]' in pupil_df.columns:
        plt.plot(pupil_df['time_norm'], pupil_df['pupil diameter left [mm]'], color=PUPIL_LEFT_COLOR, alpha=0.7, label='Left')
    if 'pupil diameter right [mm]' in pupil_df.columns:
        plt.plot(pupil_df['time_norm'], pupil_df['pupil diameter right [mm]'], color=PUPIL_RIGHT_COLOR, alpha=0.7, label='Right')
    
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.5)
    plt.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight')
    plt.close()

def generate_full_video_layered(data_dir, output_file, events_df, active_layers):
    print("Generating Video...")
    try:
        w_ts = pd.read_csv(data_dir / 'world_timestamps.csv')
        merged = w_ts
        # Safe Merges using rename instead of blind add_suffix
        for f in ['blinks.csv', '3d_eye_states.csv']:
            if (data_dir/f).exists():
                merged = safe_rename_merge(merged, pd.read_csv(data_dir/f), 'base')
        
        if (data_dir/'gaze_raw.csv').exists():
            merged = safe_rename_merge(merged, pd.read_csv(data_dir/'gaze_raw.csv'), 'RAW')

        for layer in active_layers:
            suf = layer['file_suffix']
            g_file = data_dir / f"gaze_{suf}.csv"
            s_file = data_dir / f"surface_positions_{suf}.csv"
            if g_file.exists():
                merged = safe_rename_merge(merged, pd.read_csv(g_file), suf)
            if s_file.exists():
                merged = safe_rename_merge(merged, pd.read_csv(s_file), suf)
                
    except Exception as e: 
        print(f"Video Data Error: {e}")
        return

    cap = cv2.VideoCapture(str(data_dir / "external.mp4"))
    W, H = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), cap.get(5), (W, H))
    
    histories = {} 
    colors = {}
    def get_col(name):
        if name not in colors: colors[name] = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        return colors[name]

    for i in tqdm(range(len(merged)), desc="Video Rendering"):
        ret, frame = cap.read()
        if not ret: break
        row = merged.iloc[i]
        
        # RAW
        if 'gaze x [px]_RAW' in row and pd.notna(row['gaze x [px]_RAW']):
            gx, gy = int(row['gaze x [px]_RAW']), int(row['gaze y [px]_RAW'])
            cv2.circle(frame, (gx, gy), 8, GAZE_COLOR_RAW, 2)

        # Layers
        for layer in active_layers:
            suf = layer['file_suffix']
            tl_x = f'tl x [px]_{suf}'
            if tl_x in row and pd.notna(row[tl_x]):
                pts = np.array([[row[f'tl x [px]_{suf}'], row[f'tl y [px]_{suf}']], [row[f'tr x [px]_{suf}'], row[f'tr y [px]_{suf}']], [row[f'br x [px]_{suf}'], row[f'br y [px]_{suf}']], [row[f'bl x [px]_{suf}'], row[f'bl y [px]_{suf}']]], np.int32)
                poly_col = get_col(layer['name'])
                cv2.polylines(frame, [pts], True, poly_col, 2)
                cv2.putText(frame, layer['name'], (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, poly_col, 2)

            gx_col = f'gaze x [px]_{suf}'
            if gx_col in row and pd.notna(row[gx_col]):
                gx, gy = int(row[gx_col]), int(row[f'gaze y [px]_{suf}'])
                is_in = row.get(f'gaze detected on surface_{suf}') == True
                col = GAZE_COLOR_IN if is_in else GAZE_COLOR_OUT
                cv2.circle(frame, (gx, gy), 10, col, 3)
                
                if suf not in histories: histories[suf] = []
                histories[suf].append((gx, gy))
                if len(histories[suf]) > 15: histories[suf].pop(0)
                for j in range(1, len(histories[suf])):
                    cv2.line(frame, histories[suf][j-1], histories[suf][j], col, 2)

        out.write(frame)
    cap.release(); out.release()

# ==============================================================================
# 5. PREVIEW & INTERACTIVE WINDOWS
# ==============================================================================

class CustomObjectDefiner(tk.Toplevel):
    def __init__(self, parent, video_path, app_ref):
        super().__init__(parent)
        self.title("Definisci Oggetto Custom (SAM) - Navigazione Completa")
        self.geometry("1100x850")
        self.app = app_ref
        self.video_path = video_path
        
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        self.frame_img = None
        self.is_playing = False
        
        try:
            self.sam = SAM("sam2.1_b.pt")
        except:
            messagebox.showerror("Error", "Modello SAM non trovato.")
            self.destroy()
            return

        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        
        ctl_frame = tk.Frame(self, bd=2, relief=tk.RAISED)
        ctl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.slider = tk.Scale(ctl_frame, from_=0, to=self.total_frames-1, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, padx=10, pady=2)
        
        btn_box = tk.Frame(ctl_frame)
        btn_box.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Button(btn_box, text="<< -30s", command=lambda: self.seek(-int(30*self.fps))).pack(side=tk.LEFT)
        tk.Button(btn_box, text="< -1 Frame", command=lambda: self.seek(-1)).pack(side=tk.LEFT)
        self.btn_play = tk.Button(btn_box, text="▶ Play", command=self.toggle_play, width=10)
        self.btn_play.pack(side=tk.LEFT, padx=20)
        tk.Button(btn_box, text="+1 Frame >", command=lambda: self.seek(1)).pack(side=tk.LEFT)
        tk.Button(btn_box, text="+30s >>", command=lambda: self.seek(int(30*self.fps))).pack(side=tk.LEFT)
        
        self.lbl_info = tk.Label(btn_box, text="Frame: 0", font=("Arial", 10, "bold"))
        self.lbl_info.pack(side=tk.RIGHT, padx=10)
        
        self.show_frame()

    def on_slider_move(self, val):
        self.current_frame_idx = int(val)
        if not self.is_playing:
            self.show_frame()

    def seek(self, delta):
        self.current_frame_idx = max(0, min(self.total_frames-1, self.current_frame_idx + delta))
        self.slider.set(self.current_frame_idx)
        self.show_frame()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.config(text="⏸ Pause" if self.is_playing else "▶ Play")
        if self.is_playing:
            self.play_loop()

    def play_loop(self):
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.slider.set(self.current_frame_idx)
            self.show_frame()
            self.after(int(1000/self.fps), self.play_loop)
        else:
            self.is_playing = False
            self.btn_play.config(text="▶ Play")

    def show_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame_img = frame
            h, w = frame.shape[:2]
            scale = min(1000/w, 650/h)
            frame_s = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            self.scale = scale
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0,0, image=img, anchor=tk.NW)
            self.canvas.image = img
            sec = self.current_frame_idx / self.fps
            self.lbl_info.config(text=f"Frame: {self.current_frame_idx} ({int(sec//60)}:{int(sec%60):02d})")

    def on_click(self, event):
        if self.frame_img is None: return
        if self.is_playing: self.toggle_play() 
        
        px = int(event.x / self.scale)
        py = int(event.y / self.scale)
        
        results = self.sam(self.frame_img, points=[[px, py]], labels=[1], verbose=False)
        if results[0].masks:
            mask = results[0].masks.xy[0].astype(np.int32)
            vis = self.frame_img.copy()
            cv2.polylines(vis, [mask], True, (0, 255, 0), 3)
            frame_s = cv2.resize(vis, (0,0), fx=self.scale, fy=self.scale)
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0,0, image=img, anchor=tk.NW)
            self.canvas.image = img
            
            existing_names = sorted(list(set([r['name'] for r in self.app.ai_rois if r.get('type') == 'custom'])))
            prompt_msg = "Nome Oggetto:"
            if existing_names: prompt_msg += f"\n(Esistenti: {', '.join(existing_names)})"
            
            name = simpledialog.askstring("Nuovo Oggetto", prompt_msg)
            if name:
                x, y, w, h = cv2.boundingRect(mask)
                new_roi = {
                    'name': name,
                    'type': 'custom',
                    'model_path': 'SAM',
                    'target_class': 'Custom',
                    'active': True,
                    'init_frame': self.current_frame_idx,
                    'init_box': (x, y, w, h)
                }
                self.app.ai_rois.append(new_roi)
                self.app.refresh_roi_list()
                
                msg = f"Oggetto '{name}' aggiunto al frame {self.current_frame_idx}."
                if name in existing_names: msg += "\nNOTA: Nome esistente! I dati verranno uniti."
                messagebox.showinfo("Tracking Avviato", msg)
                self.show_frame() 

class AIInteractiveWindow(tk.Toplevel):
    def __init__(self, parent, video_path, model_path, target_class, app_ref):
        super().__init__(parent)
        self.title("Interattiva AI: Clicca sugli oggetti per attivarli (Verde = Attivo)")
        self.geometry("1000x750")
        
        self.app = app_ref 
        self.video_path = video_path
        self.target_class = target_class
        self.is_running = True
        
        self.lbl_img = tk.Label(self)
        self.lbl_img.pack(fill=tk.BOTH, expand=True)
        self.lbl_img.bind("<Button-1>", self.on_click) 
        
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.destroy()
            return
            
        self.cap = cv2.VideoCapture(str(video_path))
        self.current_frame_detections = [] 
        self.click_queue = queue.Queue()
        
        threading.Thread(target=self.loop, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self.stop)

    def stop(self):
        self.is_running = False
        self.destroy()

    def on_click(self, event):
        self.click_queue.put((event.x, event.y, self.lbl_img.winfo_width(), self.lbl_img.winfo_height()))

    def loop(self):
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: 
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            h, w = frame.shape[:2]
            target_w = 960 
            scale = target_w / w
            frame_s = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            
            try:
                results = self.model.track(frame_s, persist=True, verbose=False, tracker="botsort.yaml")
            except: continue
            
            self.current_frame_detections = []
            
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    name = self.model.names[cls_id]
                    
                    if self.target_class == 'All Detected Classes' or name == self.target_class:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        is_active = False
                        for r in self.app.ai_rois:
                            if r.get('type') == 'yolo' and r['target_class'] == name and r['active']:
                                is_active = True
                                break
                        
                        color = (0, 255, 0) if is_active else (0, 0, 255)
                        cv2.rectangle(frame_s, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} (ACTIVE)" if is_active else name
                        cv2.putText(frame_s, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        self.current_frame_detections.append({
                            'rect': (x1, y1, x2, y2),
                            'name': name,
                            'active': is_active
                        })

            try:
                cx_gui, cy_gui, w_gui, h_gui = self.click_queue.get_nowait()
                scale_x = frame_s.shape[1] / w_gui
                scale_y = frame_s.shape[0] / h_gui
                cx_frame = cx_gui * scale_x
                cy_frame = cy_gui * scale_y
                
                for det in self.current_frame_detections:
                    x1, y1, x2, y2 = det['rect']
                    if x1 <= cx_frame <= x2 and y1 <= cy_frame <= y2:
                        target = det['name']
                        found = False
                        for i, r in enumerate(self.app.ai_rois):
                            if r.get('type') == 'yolo' and r['target_class'] == target:
                                self.app.ai_rois[i]['active'] = not self.app.ai_rois[i]['active']
                                found = True
                                break
                        
                        if not found:
                            new_roi = {
                                'name': target,
                                'type': 'yolo',
                                'model_path': self.app.cb_model.get(),
                                'target_class': target,
                                'use_sam': self.app.var_sam.get(),
                                'active': True
                            }
                            self.app.ai_rois.append(new_roi)
                        
                        self.app.refresh_roi_list()
                        break
            except queue.Empty: pass

            img = Image.fromarray(cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            if self.is_running: self.lbl_img.configure(image=imgtk); self.lbl_img.image = imgtk
            time.sleep(0.03)

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
# 6. MAIN APP
# ==============================================================================

class SpeedLiteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED Light + AI Tracker")
        self.root.geometry("900x900")
        
        self.data_dir = tk.StringVar(); self.output_dir = tk.StringVar()
        self.enrichment_rows = []
        self.ai_rois = [] 
        
        self.working_dir = None; self.gui_queue = queue.Queue() 
        pad = {'padx': 10, 'pady': 5}
        
        grp_in = tk.LabelFrame(root, text="Configurazione")
        grp_in.pack(fill=tk.X, **pad)
        self._build_file_select(grp_in, "Data Folder:", self.data_dir)
        self._build_file_select(grp_in, "Output Folder:", self.output_dir)

        # AI ROI CONFIG
        grp_ai = tk.LabelFrame(root, text="AI ROI Tracker (YOLO Re-ID + SAM Custom)")
        grp_ai.pack(fill=tk.X, **pad)
        
        frm_ai_ctl = tk.Frame(grp_ai)
        frm_ai_ctl.pack(fill=tk.X, **pad)
        
        tk.Label(frm_ai_ctl, text="Name:").pack(side=tk.LEFT)
        self.txt_roi_name = tk.Entry(frm_ai_ctl, width=10); self.txt_roi_name.pack(side=tk.LEFT, padx=2)
        
        tk.Label(frm_ai_ctl, text="Model:").pack(side=tk.LEFT)
        self.cb_model = ttk.Combobox(frm_ai_ctl, values=["yolov8n-seg.pt", "yolov8x-seg.pt"], width=15)
        self.cb_model.set("yolov8x-seg.pt"); self.cb_model.pack(side=tk.LEFT, padx=2)
        
        tk.Label(frm_ai_ctl, text="Class:").pack(side=tk.LEFT)
        self.cb_class = ttk.Combobox(frm_ai_ctl, values=["All Detected Classes", "laptop", "cup", "person", "cell phone", "keyboard"], width=15)
        self.cb_class.set("All Detected Classes"); self.cb_class.pack(side=tk.LEFT, padx=2)
        
        self.var_sam = tk.BooleanVar(value=False)
        tk.Checkbutton(frm_ai_ctl, text="Use SAM Refine?", variable=self.var_sam).pack(side=tk.LEFT, padx=5)
        
        # PULSANTI AI
        tk.Button(frm_ai_ctl, text="🎨 Definisci Oggetti Custom (SAM)", command=self.open_custom_definer, bg="#e1bee7").pack(side=tk.LEFT, padx=5)
        tk.Button(frm_ai_ctl, text="✨ IA Interattiva", command=self.open_interactive_ai, bg="#ffecb3").pack(side=tk.LEFT, padx=5)
        tk.Button(frm_ai_ctl, text="+ Add YOLO ROI", command=self.add_ai_roi).pack(side=tk.LEFT, padx=5)
        tk.Button(frm_ai_ctl, text="🎬 Genera Video AI", command=self.run_video_only, bg="#b3e5fc").pack(side=tk.LEFT, padx=5)

        self.lst_rois = tk.Listbox(grp_ai, height=5)
        self.lst_rois.pack(fill=tk.X, padx=10, pady=2)
        self.lst_rois.bind("<Double-Button-1>", self.toggle_roi_active)
        
        btn_roi_act = tk.Frame(grp_ai)
        btn_roi_act.pack(fill=tk.X, padx=10, pady=2)
        tk.Button(btn_roi_act, text="Remove Selected", command=self.remove_ai_roi, bg="#ffcdd2").pack(side=tk.RIGHT)
        tk.Button(btn_roi_act, text="Toggle Active (Double Click)", command=self.toggle_roi_active).pack(side=tk.RIGHT, padx=5)

        grp_man = tk.LabelFrame(root, text="Manual Enrichment Folders")
        grp_man.pack(fill=tk.X, **pad)
        tk.Button(grp_man, text="+ Add Manual Folder", command=self.add_enrichment_row).pack(anchor=tk.W, padx=5)
        self.frm_man_rows = tk.Frame(grp_man); self.frm_man_rows.pack(fill=tk.X)

        btn_fr = tk.Frame(root); btn_fr.pack(fill=tk.X, **pad)
        tk.Button(btn_fr, text="1. Load Data", command=self.load_data, bg="#fff9c4").pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.btn_edit = tk.Button(btn_fr, text="2. Edit Events", command=self.edit_events, state=tk.DISABLED)
        self.btn_edit.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.btn_run = tk.Button(btn_fr, text="3. RUN FULL ANALYSIS (Metrics & Excel)", command=self.run_process, bg="#b2dfdb", state=tk.DISABLED)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.log_box = tk.Text(root, height=12); self.log_box.pack(fill=tk.BOTH, **pad)
        self.root.after(100, self._check_queue)

    def _build_file_select(self, p, l, v):
        f = tk.Frame(p); f.pack(fill=tk.X, pady=2)
        tk.Label(f, text=l).pack(side=tk.LEFT)
        tk.Entry(f, textvariable=v).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(f, text="Browse", command=lambda: self.browse(v)).pack(side=tk.RIGHT)
    
    def browse(self, v): 
        d = filedialog.askdirectory()
        if d: v.set(d)
        
    def _check_queue(self):
        try:
            while True:
                t, c = self.gui_queue.get_nowait()
                if t=='log': self.log_box.insert(tk.END, c+"\n"); self.log_box.see(tk.END)
                elif t=='info': messagebox.showinfo("Info", c)
                elif t=='error': messagebox.showerror("Error", c)
                elif t=='enable': self.btn_run.config(state=tk.NORMAL); self.btn_edit.config(state=tk.NORMAL)
        except: pass
        self.root.after(100, self._check_queue)

    def add_enrichment_row(self):
        idx = len(self.enrichment_rows)
        row = tk.Frame(self.frm_man_rows); row.pack(fill=tk.X, pady=2)
        v = tk.StringVar(); chk = tk.BooleanVar(value=True)
        tk.Label(row, text=f"Manual {idx}:").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=v).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(row, text="Browse", command=lambda: self.browse(v)).pack(side=tk.LEFT)
        tk.Checkbutton(row, text="Video?", variable=chk).pack(side=tk.LEFT)
        self.enrichment_rows.append({'path': v, 'video': chk})

    def open_interactive_ai(self):
        d = self.data_dir.get()
        if not d: return
        vid = list(Path(d).glob("*.mp4"))
        if not vid: messagebox.showerror("Err", "No video in Data"); return
        AIInteractiveWindow(self.root, vid[0], self.cb_model.get(), self.cb_class.get(), self)

    def open_custom_definer(self):
        d = self.data_dir.get()
        if not d: return
        vid = list(Path(d).glob("*.mp4"))
        if not vid: messagebox.showerror("Err", "No video in Data"); return
        CustomObjectDefiner(self.root, vid[0], self)

    def add_ai_roi(self):
        name = self.txt_roi_name.get()
        if not name: name = f"{self.cb_class.get()}_{len(self.ai_rois)}"
        roi = {'name': name, 'type': 'yolo', 'model_path': self.cb_model.get(), 'target_class': self.cb_class.get(), 'use_sam': self.var_sam.get(), 'active': True}
        self.ai_rois.append(roi)
        self.refresh_roi_list()

    def refresh_roi_list(self):
        self.lst_rois.delete(0, tk.END)
        for r in self.ai_rois:
            status = "[x]" if r['active'] else "[ ]"
            tag = "(YOLO)" if r.get('type') == 'yolo' else "(CUSTOM)"
            sam_tag = "+SAM" if r.get('use_sam') else ""
            self.lst_rois.insert(tk.END, f"{status} {r['name']} {tag} {sam_tag}")

    def toggle_roi_active(self, event=None):
        sel = self.lst_rois.curselection()
        if sel:
            idx = sel[0]
            self.ai_rois[idx]['active'] = not self.ai_rois[idx]['active']
            self.refresh_roi_list()

    def remove_ai_roi(self):
        sel = self.lst_rois.curselection()
        if sel:
            self.ai_rois.pop(sel[0])
            self.refresh_roi_list()

    def load_data(self):
        if not self.data_dir.get(): return
        if not self.output_dir.get(): self.output_dir.set(str(Path(self.data_dir.get())/"Output"))
        try:
            p = Path(self.data_dir.get())
            if not (p/'world_timestamps.csv').exists(): raise Exception("world_timestamps missing")
            self.btn_run.config(state=tk.NORMAL)
            self.btn_edit.config(state=tk.NORMAL)
            messagebox.showinfo("OK", "Ready.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def edit_events(self):
        if not self.working_dir:
             messagebox.showwarning("Info", "Data loaded directly from Data Folder for editing.")
             video_path = Path(self.data_dir.get()) / "external.mp4"
             events_path = Path(self.data_dir.get()) / "events.csv"
             ts_path = Path(self.data_dir.get()) / "world_timestamps.csv"
        else:
             video_path = self.working_dir / "external.mp4"
             events_path = self.working_dir / "events.csv"
             ts_path = self.working_dir / "world_timestamps.csv"

        if not video_path.exists(): messagebox.showerror("Err", "Video not found"); return
        df_ev = pd.read_csv(events_path); df_ts = pd.read_csv(ts_path)
        editor = AdvancedEventEditor(self.root, video_path, df_ev, df_ts)
        self.root.wait_window(editor)
        if editor.saved_df is not None:
            if 'selected' in editor.saved_df.columns: editor.saved_df = editor.saved_df.drop(columns=['selected'])
            editor.saved_df.to_csv(events_path, index=False)
            self.log_box.insert(tk.END, "Updated events saved.\n")

    def run_video_only(self):
        threading.Thread(target=self._worker_video_only, daemon=True).start()

    def _worker_video_only(self):
        try:
            self.gui_queue.put(('log', "--- GENERATING AI VIDEO ONLY ---"))
            data_p = Path(self.data_dir.get()); out_p = Path(self.output_dir.get())
            man_paths = [Path(r['path'].get()) for r in self.enrichment_rows if r['path'].get()]
            work_dir, warns = prepare_working_directory(data_p, man_paths, out_p)
            for w in warns: self.gui_queue.put(('log', f"WARN: {w}"))
            
            processed_ai_rois = []
            if self.ai_rois:
                self.gui_queue.put(('log', "Running Tracking for Active ROIs..."))
                vid_path = work_dir / "external.mp4"; ts_path = work_dir / "world_timestamps.csv"
                processed_ai_rois = run_ai_extraction(vid_path, ts_path, self.ai_rois, work_dir, self.gui_queue)
                
                g_raw = pd.read_csv(work_dir/"gaze_raw.csv")
                world_ts = pd.read_csv(ts_path)
                
                def check_poly(df, xc, yc, tc, polys):
                    df = pd.merge_asof(df.sort_values(tc), world_ts[[tc]].reset_index().rename(columns={'index':'fidx'}), on=tc, direction='nearest')
                    ins = []
                    for _,r in df.iterrows():
                        fid = int(r['fidx'])
                        val = False
                        if fid in polys:
                            if cv2.pointPolygonTest(polys[fid], (r[xc], r[yc]), False) >= 0: val = True
                        ins.append(val)
                    return ins

                for roi in processed_ai_rois:
                    suf = f"ROI_{roi['name']}"
                    shutil.copy(roi['path']/'surface_positions.csv', work_dir/f"surface_positions_{suf}.csv")
                    if not g_raw.empty:
                        g_raw['gaze detected on surface'] = check_poly(g_raw, 'gaze x [px]', 'gaze y [px]', 'timestamp [ns]', roi['polygons'])
                        g_raw.to_csv(work_dir/f"gaze_{suf}.csv", index=False)

            self.gui_queue.put(('log', "Rendering Video..."))
            layers = []
            for roi in processed_ai_rois:
                layers.append({'name': roi['name'], 'file_suffix': f"ROI_{roi['name']}"})
            
            ev = pd.read_csv(work_dir / "events.csv") 
            generate_full_video_layered(work_dir, out_p/"ai_tracking_video.mp4", ev, layers)
            self.gui_queue.put(('log', f"Video saved: {out_p/'ai_tracking_video.mp4'}"))
            self.gui_queue.put(('info', "Video Generation Completed!"))

        except Exception as e:
            self.gui_queue.put(('error', str(e))); print(e)

    def run_process(self):
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            self.gui_queue.put(('log', "--- STARTING FULL ANALYSIS ---"))
            data_p = Path(self.data_dir.get()); out_p = Path(self.output_dir.get())
            
            man_paths = [Path(r['path'].get()) for r in self.enrichment_rows if r['path'].get()]
            work_dir, warns = prepare_working_directory(data_p, man_paths, out_p)
            self.working_dir = work_dir 
            for w in warns: self.gui_queue.put(('log', f"WARN: {w}"))
            
            processed_ai_rois = []
            if self.ai_rois:
                self.gui_queue.put(('log', "--- RUNNING AI TRACKING ---"))
                vid_path = work_dir / "external.mp4"; ts_path = work_dir / "world_timestamps.csv"
                processed_ai_rois = run_ai_extraction(vid_path, ts_path, self.ai_rois, work_dir, self.gui_queue)
                
                for roi in processed_ai_rois:
                    suf = f"ROI_{roi['name']}"
                    shutil.copy(roi['path']/'surface_positions.csv', work_dir/f"surface_positions_{suf}.csv")
                    
                    g_raw = pd.read_csv(work_dir/"gaze_raw.csv")
                    f_raw = pd.read_csv(work_dir/"fixations_raw.csv")
                    world_ts = pd.read_csv(ts_path)
                    
                    def check_poly(df, xc, yc, tc, polys):
                        df = pd.merge_asof(df.sort_values(tc), world_ts[[tc]].reset_index().rename(columns={'index':'fidx'}), on=tc, direction='nearest')
                        ins = []
                        for _,r in df.iterrows():
                            fid = int(r['fidx'])
                            val = False
                            if fid in polys:
                                if cv2.pointPolygonTest(polys[fid], (r[xc], r[yc]), False) >= 0: val = True
                            ins.append(val)
                        return ins

                    if not g_raw.empty:
                        g_raw['gaze detected on surface'] = check_poly(g_raw, 'gaze x [px]', 'gaze y [px]', 'timestamp [ns]', roi['polygons'])
                        g_raw.to_csv(work_dir/f"gaze_{suf}.csv", index=False)
                    
                    if not f_raw.empty:
                        f_raw['fixation detected on surface'] = check_poly(f_raw, 'fixation x [px]', 'fixation y [px]', 'start timestamp [ns]', roi['polygons'])
                        f_raw.to_csv(work_dir/f"fixations_{suf}.csv", index=False)


            self.gui_queue.put(('log', "--- METRICS ---"))
            ev = pd.read_csv(work_dir / "events.csv").sort_values('timestamp [ns]')
            ev = ev[~ev['name'].isin(['recording.begin', 'recording.end'])]
            
            res = (1280, 720)
            fix_raw = pd.read_csv(work_dir / "fixations_raw.csv")
            gaze_raw = pd.read_csv(work_dir / "gaze_raw.csv")
            pupil = pd.read_csv(work_dir / "3d_eye_states.csv")
            blinks = pd.read_csv(work_dir / "blinks.csv")
            sacc = pd.read_csv(work_dir / "saccades.csv")

            all_metrics = []; roi_detailed_metrics = []

            for i in range(len(ev)):
                t_start = ev.iloc[i]['timestamp [ns]']
                t_end = ev.iloc[i+1]['timestamp [ns]'] if i < len(ev)-1 else gaze_raw['timestamp [ns]'].max()
                evt_name = ev.iloc[i]['name']
                self.gui_queue.put(('log', f"Event: {evt_name}"))
                
                seg_raw = {
                    'fixations': fix_raw[(fix_raw['start timestamp [ns]'] >= t_start) & (fix_raw['start timestamp [ns]'] < t_end)],
                    'gaze': gaze_raw[(gaze_raw['timestamp [ns]'] >= t_start) & (gaze_raw['timestamp [ns]'] < t_end)],
                    'pupil': pupil[(pupil['timestamp [ns]'] >= t_start) & (pupil['timestamp [ns]'] < t_end)],
                    'blinks': blinks[(blinks['start timestamp [ns]'] >= t_start) & (blinks['start timestamp [ns]'] < t_end)],
                    'saccades': sacc[(sacc['start timestamp [ns]'] >= t_start) & (sacc['start timestamp [ns]'] < t_end)]
                }
                m_main = calculate_metrics(seg_raw, res)
                m_main = {k+"_RAW": v for k,v in m_main.items()}
                m_main['event_name'] = evt_name

                # Manual
                for idx, r in enumerate(self.enrichment_rows):
                    if (work_dir/f"gaze_enr_{idx}.csv").exists():
                        g_enr = pd.read_csv(work_dir/f"gaze_enr_{idx}.csv")
                        fx_enr = pd.read_csv(work_dir/f"fixations_enr_{idx}.csv") if (work_dir/f"fixations_enr_{idx}.csv").exists() else pd.DataFrame()
                        seg_enr = seg_raw.copy()
                        seg_enr['gaze'] = g_enr[(g_enr['timestamp [ns]'] >= t_start) & (g_enr['timestamp [ns]'] < t_end)]
                        if not fx_enr.empty: seg_enr['fixations'] = fx_enr[(fx_enr['start timestamp [ns]'] >= t_start) & (fx_enr['start timestamp [ns]'] < t_end)]
                        m_enr = calculate_metrics(seg_enr, res)
                        m_main.update({k+f"_MAN_{idx}": v for k,v in m_enr.items()})

                # AI ROIs Details
                for roi in processed_ai_rois:
                    r_name = roi['name']
                    g_roi = pd.read_csv(work_dir/f"gaze_ROI_{r_name}.csv")
                    f_roi = pd.read_csv(work_dir/f"fixations_ROI_{r_name}.csv")
                    
                    seg_roi = seg_raw.copy()
                    seg_roi['gaze'] = g_roi[(g_roi['timestamp [ns]'] >= t_start) & (g_roi['timestamp [ns]'] < t_end)]
                    seg_roi['fixations'] = f_roi[(f_roi['start timestamp [ns]'] >= t_start) & (f_roi['start timestamp [ns]'] < t_end)]
                    
                    m_roi = calculate_metrics(seg_roi, res)
                    
                    # NORMALIZE METRICS
                    ts_range = pd.read_csv(work_dir/"world_timestamps.csv")
                    ts_range = ts_range[(ts_range['timestamp [ns]'] >= t_start) & (ts_range['timestamp [ns]'] < t_end)]
                    frame_idxs = ts_range.index.tolist() 
                    
                    frames_vis_event = sum(1 for f in frame_idxs if f in roi['polygons'])
                    sec_vis_event = frames_vis_event / 30.0 

                    roi_entry = {
                        'Event': evt_name,
                        'Object_ID': r_name,
                        'Frames_Visible': frames_vis_event,
                        'Seconds_Visible': sec_vis_event,
                        'N_Fixations': m_roi.get('ROI_n_fixations', 0),
                        'Mean_Fix_Dur': m_roi.get('ROI_mean_fix_duration', np.nan),
                        'Std_Fix_Dur': m_roi.get('ROI_std_fix_duration', np.nan),
                        'Mean_Pupil_Left': m_roi.get('ROI_mean_pupil_L', np.nan),
                        'Std_Pupil_Left': m_roi.get('ROI_std_pupil_L', np.nan),
                        'Mean_Pupil_Right': m_roi.get('ROI_mean_pupil_R', np.nan),
                        'Std_Pupil_Right': m_roi.get('ROI_std_pupil_R', np.nan),
                        'Fixations_Per_Sec_Vis': m_roi.get('ROI_n_fixations', 0)/sec_vis_event if sec_vis_event > 0 else 0
                    }
                    roi_detailed_metrics.append(roi_entry)

                all_metrics.append(m_main)

            pd.DataFrame(all_metrics).to_excel(out_p / "Speed_Lite_Results.xlsx", index=False)
            if roi_detailed_metrics: pd.DataFrame(roi_detailed_metrics).to_excel(out_p / "ROI_Analysis_Results.xlsx", index=False)

            layers = []
            for i, r in enumerate(self.enrichment_rows):
                if r['video'].get(): layers.append({'name': f"Man {i}", 'file_suffix': f"enr_{i}"})
            for roi in processed_ai_rois:
                layers.append({'name': roi['name'], 'file_suffix': f"ROI_{roi['name']}"})
            
            generate_full_video_layered(work_dir, out_p/"final_video.mp4", ev, layers)
            self.gui_queue.put(('log', "DONE.")); self.gui_queue.put(('enable', None))
            self.gui_queue.put(('info', "Analysis Completed"))
            
        except Exception as e:
            self.gui_queue.put(('error', str(e))); print(e); self.gui_queue.put(('enable', None))

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeedLiteApp(root)
    root.mainloop()