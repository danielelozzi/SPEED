# video_generator.py
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import traceback, logging

# --- Constants ---
NS_TO_S = 1e9 # Conversion factor from nanoseconds to seconds

# --- Drawing Constants ---
GAZE_COLOR = (0, 0, 255)  # Red in BGR
GAZE_RADIUS = 15
GAZE_THICKNESS = 2
PIP_SCALE = 0.25

# --- YOLO Drawing Constants ---
YOLO_BOX_COLOR = (0, 255, 255) # Cyan in BGR
YOLO_TEXT_COLOR = (0, 255, 255)
YOLO_THICKNESS = 2

# --- Pupil Plot Constants ---
PUPIL_PLOT_HISTORY = 200
PUPIL_PLOT_WIDTH = 350
PUPIL_PLOT_HEIGHT = 150
PUPIL_BG_COLOR = (80, 80, 80)
PUPIL_COLORS = {"Left": (80, 80, 255), "Right": (80, 255, 80), "Mean": (255, 255, 255)}
BLINK_TEXT_COLOR = (0, 0, 255)

# --- Fragmentation Plot Constants ---
FRAG_PLOT_HISTORY = 200
FRAG_PLOT_WIDTH = 350
FRAG_PLOT_HEIGHT = 150
FRAG_BG_COLOR = (80, 80, 80)
FRAG_LINE_COLOR = (255, 200, 100) # Light Blue

# --- Event Text Constants ---
EVENT_TEXT_COLOR = (255, 255, 255) # White
EVENT_BG_COLOR = (0, 0, 0) # Black

# --- On Surface Text Constants (NEW) ---
ON_SURFACE_TEXT_COLOR = (120, 255, 120) # Light Green

def _overlay_transparent(background, overlay, x, y):
    """
    Sovrappone un'immagine (overlay) con canale alpha su uno sfondo.
    """
    background_width = background.shape[1]
    background_height = background.shape[0]
    h, w = overlay.shape[0], overlay.shape[1]

    if x >= background_width or y >= background_height:
        return background

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]
    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        return background

    alpha = overlay[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    b, g, r = overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2]
    bgr = np.dstack([b, g, r])

    background_subsection = background[y:y+h, x:x+w]
    composite = bgr * alpha + background_subsection * (1.0 - alpha)
    background[y:y+h, x:x+w] = composite
    return background

def _prepare_data(data_dir: Path, un_enriched_mode: bool, options: dict):
    """
    Loads and synchronizes all necessary DataFrames for video generation.
    MODIFIED: Merges enriched gaze data if needed for 'on surface' text.
    """
    try:
        world_timestamps = pd.read_csv(data_dir / 'world_timestamps.csv').sort_values('timestamp [ns]')
        gaze_df = pd.read_csv(data_dir / 'gaze.csv').sort_values('timestamp [ns]')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Essential file not found: {e}. Cannot generate video.")

    merged_data = pd.merge_asof(
        world_timestamps,
        gaze_df[['timestamp [ns]', 'gaze x [px]', 'gaze y [px]']],
        on='timestamp [ns]',
        direction='nearest',
        tolerance=pd.Timedelta('50ms').value
    )
    
    # --- NUOVA LOGICA: Aggiungi dati enriched se necessario ---
    if not un_enriched_mode and (options.get('overlay_on_surface_text') or options.get('crop_and_correct_perspective')):
        try:
            gaze_enr_df = pd.read_csv(data_dir / 'gaze_enriched.csv').sort_values('timestamp [ns]')
            
            cols_to_merge_enr = []
            if 'gaze detected on surface' in gaze_enr_df.columns:
                cols_to_merge_enr.append('gaze detected on surface')
            if 'aoi_name' in gaze_enr_df.columns:
                cols_to_merge_enr.append('aoi_name')

            if cols_to_merge_enr:
                 merged_data = pd.merge_asof(
                    merged_data,
                    gaze_enr_df[['timestamp [ns]'] + cols_to_merge_enr],
                    on='timestamp [ns]',
                    direction='backward' # backward fill per lo stato 'on surface'
                 )
        except FileNotFoundError:
             print("WARNING: 'gaze_enriched.csv' not found. 'On Surface' text overlay disabled.")
             options['overlay_on_surface_text'] = False
    # --- FINE NUOVA LOGICA ---

    # Calculate Fragmentation (Gaze Speed) for the video
    if options.get('overlay_fragmentation_plot'):
        merged_data['gaze_speed_px_per_s'] = np.sqrt(merged_data['gaze x [px]'].diff()**2 + merged_data['gaze y [px]'].diff()**2) / (merged_data['timestamp [ns]'].diff() / NS_TO_S)

    if options.get('overlay_pupil_plot'):
        try:
            pupil_df = pd.read_csv(data_dir / '3d_eye_states.csv').sort_values('timestamp [ns]')
            cols_to_merge = []
            if 'pupil diameter left [mm]' in pupil_df.columns:
                cols_to_merge.append('pupil diameter left [mm]')
            if 'pupil diameter right [mm]' in pupil_df.columns:
                cols_to_merge.append('pupil diameter right [mm]')
            
            if cols_to_merge:
                pupil_df['pupil_diameter_mean'] = pupil_df[cols_to_merge].mean(axis=1)
                cols_to_merge.append('pupil_diameter_mean')

                merged_data = pd.merge_asof(
                    merged_data,
                    pupil_df[['timestamp [ns]'] + cols_to_merge],
                    on='timestamp [ns]',
                    direction='backward'
                )
        except FileNotFoundError:
            print("WARNING: '3d_eye_states.csv' not found. Pupil plot disabled.")

    try:
        blinks_df = pd.read_csv(data_dir / 'blinks.csv')
        merged_data['is_blinking'] = False
        for _, row in blinks_df.iterrows():
            merged_data.loc[(merged_data['timestamp [ns]'] >= row['start timestamp [ns]']) & (merged_data['timestamp [ns]'] <= row['end timestamp [ns]']), 'is_blinking'] = True
    except FileNotFoundError:
        print("WARNING: 'blinks.csv' not found. Blink overlay disabled.")

    if options.get('crop_and_correct_perspective'):
        try:
            surface_df = pd.read_csv(data_dir / 'surface_positions.csv').sort_values('timestamp [ns]')
            corner_cols = ['tl x [px]', 'tl y [px]', 'tr x [px]', 'tr y [px]', 
                           'br x [px]', 'br y [px]', 'bl x [px]', 'bl y [px]']
            merged_data = pd.merge_asof(
                merged_data,
                surface_df[['timestamp [ns]'] + corner_cols],
                on='timestamp [ns]',
                direction='backward'
            )
        except FileNotFoundError:
            print("WARNING: Perspective option is active, but 'surface_positions.csv' not found. Option disabled.")
            options['crop_and_correct_perspective'] = False

    return merged_data

def _draw_generic_plot(frame: np.ndarray, data_points: list, min_val: float, max_val: float, width: int, height: int, position: tuple, title: str, color: tuple, bg_color: tuple):
    """Generic function to draw a single-line plot on the frame."""
    if not data_points or max_val == min_val:
        return frame

    x_pos, y_pos = position
    plot_area = frame[y_pos:y_pos+height, x_pos:x_pos+width]
    bg = np.full(plot_area.shape, bg_color, dtype=np.uint8)
    res = cv2.addWeighted(plot_area, 0.5, bg, 0.5, 0)
    frame[y_pos:y_pos+height, x_pos:x_pos+width] = res

    cv2.putText(frame, title, (x_pos + 5, y_pos + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if len(data_points) < 2:
        return frame
    
    points = [
        (
            x_pos + int((i / (len(data_points) -1)) * width),
            y_pos + height - int(((val - min_val) / (max_val - min_val)) * (height - 25)) - 10
        )
        for i, val in enumerate(data_points) if pd.notna(val)
    ]

    if len(points) > 1:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False, color=color, thickness=2)

    return frame


def _draw_pupil_plot(frame: np.ndarray, plot_data_dict: dict, min_val: float, max_val: float, width: int, height: int, position: tuple):
    """Draws a multi-line graph for pupil data with a legend."""
    if not plot_data_dict or max_val == min_val:
        return frame
        
    x_pos, y_pos = position
    plot_area = frame[y_pos:y_pos+height, x_pos:x_pos+width]
    
    bg = np.full(plot_area.shape, PUPIL_BG_COLOR, dtype=np.uint8)
    res = cv2.addWeighted(plot_area, 0.5, bg, 0.5, 0)
    frame[y_pos:y_pos+height, x_pos:x_pos+width] = res

    # Draw the legend
    legend_y = y_pos + 15
    for name, color in PUPIL_COLORS.items():
        if name in plot_data_dict:
            cv2.putText(frame, name, (x_pos + 5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            legend_y += 15

    # Draw each plot line
    for name, data_points in plot_data_dict.items():
        if len(data_points) < 2: continue
        
        points = [
            (
                x_pos + int((i / (PUPIL_PLOT_HISTORY - 1)) * width),
                y_pos + height - int(((val - min_val) / (max_val - min_val)) * (height - 50)) - 10 # 50px margin for legend
            )
            for i, val in enumerate(data_points) if pd.notna(val)
        ]
        
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False, color=PUPIL_COLORS[name], thickness=2)

    return frame

def generate_concatenated_video(data_dir: Path, viv_events_df: pd.DataFrame) -> Path:
    """
    Creates a new base video ('external.mp4') by concatenating screen recordings
    synchronized with eye-tracking events. This video is intended to be used
    as the primary scene video for subsequent analyses.
    MODIFIED: Reads a DataFrame with video paths and optional start/end frames.

    Args:
        data_dir (Path): The working directory (e.g., SPEED_workspace) containing 'events.csv'.
        viv_events_df (pd.DataFrame): A DataFrame containing event info, video paths,
                                      and optional start/end frames for trimming.

    Returns:
        Path: The path to the newly created concatenated video.
    """
    # --- MODIFICA: Nuovo nome per il video di output ---
    video_out_path = data_dir / 'external-video-in-video.mp4'
    print(f"Starting concatenated video generation. Output will be saved to: {video_out_path}")

    # --- 1. Load and Prepare Data ---
    # Usa il DataFrame passato direttamente, che è già stato salvato come events.csv
    # nella cartella di lavoro dalla funzione chiamante.
    try:
        events_df = pd.read_csv(data_dir / 'events.csv').sort_values('timestamp [ns]')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Essential file not found: {e}. Cannot generate concatenated video.")

    # --- 2. Setup Video Timeline and Segments ---
    FPS = 30
    WIDTH, HEIGHT = 1280, 720  # Standard output resolution
    GRAY_BG = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)

    # Determine total video duration from the events
    min_ts = events_df['timestamp [ns]'].min()
    # Find the end of the last event. We approximate it by adding a reasonable duration (e.g., 30s)
    # or by finding the max duration of mapped videos. For simplicity, we add a fixed buffer.
    last_event_ts = events_df['timestamp [ns]'].max()
    max_ts = last_event_ts + pd.Timedelta('30s').value # Add 30s buffer

    total_duration_ns = max_ts - min_ts
    total_frames = int((total_duration_ns / NS_TO_S) * FPS)

    # Create a DataFrame representing every frame of the output video
    timeline_df = pd.DataFrame({
        'timestamp [ns]': np.linspace(min_ts, max_ts, total_frames, dtype=np.int64),
        'frame_idx': np.arange(total_frames)
    })

    # Merge event data onto our timeline
    # --- MODIFICA: Unisci tutte le colonne necessarie dal DataFrame degli eventi ---
    cols_to_merge = ['timestamp [ns]', 'name', 'video_path', 'start_frame', 'end_frame']
    sync_data = pd.merge_asof(
        timeline_df,
        viv_events_df[cols_to_merge].rename(columns={'name': 'event_name'}),
        on='timestamp [ns]',
        direction='backward'
    )
    sync_data['event_name'].fillna(method='bfill', inplace=True)
    sync_data.fillna(method='ffill', inplace=True)

    # --- 3. Initialize Video Writer ---
    writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))

    # --- 4. Main Video Generation Loop ---
    video_caps = {path: cv2.VideoCapture(path) for path in viv_events_df['video_path'].unique()}
    event_start_times = pd.Series(events_df['timestamp [ns]'].values, index=events_df['name']).to_dict()

    try:
        with tqdm(total=total_frames, desc="Generating Concatenated Video") as pbar:
            for idx, frame_data in sync_data.iterrows():
                current_event = frame_data.get('event_name')
                
                # --- MODIFICA: Logica per determinare il frame da mostrare ---
                video_path = frame_data.get('video_path')
                if pd.notna(video_path) and video_path in video_caps:
                    cap = video_caps[video_path]
                    event_start_ts = event_start_times[current_event]
                    
                    # Time elapsed since the event started
                    time_since_event_start_ns = frame_data['timestamp [ns]'] - event_start_ts
                    time_since_event_start_s = time_since_event_start_ns / NS_TO_S
                    
                    source_video_fps = cap.get(cv2.CAP_PROP_FPS)
                    if source_video_fps == 0: source_video_fps = 30 # Fallback

                    # Calcola l'indice del frame nel video sorgente
                    source_frame_idx = int(time_since_event_start_s * source_video_fps)

                    # Applica il taglio se start_frame è specificato
                    start_frame_offset = frame_data.get('start_frame', 0)
                    if pd.isna(start_frame_offset): start_frame_offset = 0
                    target_frame_idx = int(start_frame_offset + source_frame_idx)
                    
                    # Controlla se il frame è oltre end_frame (se specificato)
                    end_frame_limit = frame_data.get('end_frame')
                    if pd.notna(end_frame_limit) and target_frame_idx > end_frame_limit:
                        frame = GRAY_BG.copy() # Il segmento tagliato è terminato
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
                        ret, frame = cap.read()

                    if not ret:
                        frame = GRAY_BG.copy() # Video ended, show gray
                else:
                    # No video associated with this event segment, or before the first event
                    frame = GRAY_BG.copy()

                # Resize frame to standard output size
                if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
                    frame = cv2.resize(frame, (WIDTH, HEIGHT))

                writer.write(frame)
                pbar.update(1)

    except Exception as e:
        print(f"An error occurred during concatenated video generation: {e}")
        traceback.print_exc()
    finally:
        print("Finalizing and releasing resources...")
        writer.release()
        for cap in video_caps.values():
            cap.release()
        print("Concatenated video generation complete!")

    # Also create a dummy world_timestamps.csv for this new video
    world_timestamps_df = pd.DataFrame({
        'timestamp [ns]': np.linspace(min_ts, max_ts, total_frames, dtype=np.int64)
    })
    world_timestamps_df.to_csv(data_dir / 'world_timestamps_viv.csv', index=False)

    return video_out_path

def create_custom_video(data_dir: Path, output_dir: Path, subj_name: str, options: dict, un_enriched_mode: bool, selected_events: list = None):
    """
    Main function for creating the video with selected overlays.
    MODIFIED: Can trim video to selected events.
    """
    video_out_path = output_dir / options.get('output_filename', f'video_output_{subj_name}.mp4')
    # Nuovo: percorso temporaneo per il video senza audio 
    temp_video_path = output_dir / f'temp_no_audio_{subj_name}.mp4'
    
    print("Loading and synchronizing data...")
    try:
        sync_data = _prepare_data(data_dir, un_enriched_mode, options)
        # Assicurati che l'indice sia utile per ricerche veloci
        sync_data.set_index('timestamp [ns]', inplace=True, drop=False)
        sync_data.sort_index(inplace=True)
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}. Cannot generate video.")
        return

    external_vid_path = data_dir / 'external.mp4'
    cap_ext = cv2.VideoCapture(str(external_vid_path))
    if not cap_ext.isOpened():
        print(f"ERROR: Cannot open external video: {external_vid_path}")
        return

    cap_int = None
    fps_int = 0 # Inizializza la variabile per il frame rate del video interno
    if options.get('include_internal_cam'):
        internal_vid_path = data_dir / 'internal.mp4'
        if internal_vid_path.exists():
            cap_int = cv2.VideoCapture(str(internal_vid_path))
            if not cap_int.isOpened():
                print("WARNING: Cannot open internal video, PiP disabled.")
                options['include_internal_cam'] = False
            else:
                # Leggi il frame rate del video interno
                fps_int = cap_int.get(cv2.CAP_PROP_FPS)
                if fps_int == 0: # Fallback se i metadati sono mancanti
                    fps_int = 200
                    logging.warning("Could not read internal video FPS, falling back to 200 Hz.")
        else:
            print("WARNING: Internal video not found, PiP disabled.")
            options['include_internal_cam'] = False



    total_frames = int(cap_ext.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_ext.get(cv2.CAP_PROP_FPS)
    original_w = int(cap_ext.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap_ext.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_w, out_h = (1280, 720) if options.get('crop_and_correct_perspective') else (original_w, original_h)

    # Scrittura del video su un file temporaneo senza audio
    writer = cv2.VideoWriter(str(temp_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    print(f"The output video will be saved to: {video_out_path}")

    # Setup for Event Text & Trimming
    events_df = pd.DataFrame()
    try:
        events_df = pd.read_csv(data_dir / 'events.csv').sort_values('timestamp [ns]')
        events_df['name'] = events_df['name'].astype(str).str.replace(r'[\\/]', '_', regex=True)
    except FileNotFoundError:
        print("WARNING: 'events.csv' not found. Event text/trimming overlay disabled.")
        options['overlay_event_text'] = False
        options['trim_to_events'] = False

    current_event_name = ""
    event_idx = 0
    
    # Attempt to load YOLO detection data if the option is enabled.
    yolo_detections = pd.DataFrame()
    yolo_class_map = {}
    if options.get('overlay_yolo'):
        yolo_cache_path = output_dir / 'yolo_detections_cache.csv'
        try:
            yolo_detections = pd.read_csv(yolo_cache_path)
            # --- NUOVO: Applica i filtri se forniti ---
            if options.get('yolo_class_filter'):
                yolo_detections = yolo_detections[yolo_detections['class_name'].isin(options['yolo_class_filter'])]
            if options.get('yolo_id_filter'):
                yolo_detections = yolo_detections[yolo_detections['track_id'].isin(options['yolo_id_filter'])]
            print(f"YOLO overlay enabled. Loaded and filtered {len(yolo_detections)} detections.")
        except FileNotFoundError:
            print(f"WARNING: YOLO overlay is ON, but 'yolo_detections_cache.csv' was not found in {output_dir}")
            options['overlay_yolo'] = False
    
    # Setup for pupil plot
    pupil_plot_data = {"Left": [], "Right": [], "Mean": []}
    pupil_min, pupil_max = 0, 1
    pupil_cols = { "Left": "pupil diameter left [mm]", "Right": "pupil diameter right [mm]", "Mean": "pupil_diameter_mean" }
    if options.get('overlay_pupil_plot') and any(col in sync_data.columns for col in pupil_cols.values()):
        all_pupil_data = pd.concat([sync_data[col] for col in pupil_cols.values() if col in sync_data.columns]).dropna()
        if not all_pupil_data.empty: pupil_min, pupil_max = all_pupil_data.min(), all_pupil_data.max()

    # Setup for fragmentation plot
    frag_plot_data = []
    frag_min, frag_max = 0, 1
    if options.get('overlay_fragmentation_plot') and 'gaze_speed_px_per_s' in sync_data.columns:
        all_frag_data = sync_data['gaze_speed_px_per_s'].dropna()
        if not all_frag_data.empty: frag_min, frag_max = 0, all_frag_data.quantile(0.99)


    # Setup for Dynamic Heatmap
    gaze_history_for_heatmap = []
    heatmap_window_size_frames = 0
    if options.get('overlay_dynamic_heatmap'):
        # Il valore dello slider viene passato tramite il dizionario 'options'
        heatmap_window_seconds = options.get('heatmap_window_seconds', 2.0) 
        heatmap_window_size_frames = int(heatmap_window_seconds * fps)
    # --- FINE BLOCCO ---

    gaze_history_for_heatmap = []
    heatmap_window_size_frames = 0
    if options.get('overlay_dynamic_heatmap'):
        # Il valore dello slider viene passato tramite il dizionario 'options'
        heatmap_window_seconds = options.get('heatmap_window_seconds', 2.0) 
        heatmap_window_size_frames = int(heatmap_window_seconds * fps)

    # --- NUOVO: Inizializzazione per la scia dello sguardo ---
    gaze_path_history = []
    gaze_path_length = 10 # Memorizza gli ultimi 10 punti

    # --- NUOVA LOGICA: DEFINIZIONE DEI SEGMENTI DI FRAME DA PROCESSARE ---
    frame_segments = []
    if options.get('trim_to_events') and selected_events and not events_df.empty:
        print(f"Trimming video to {len(selected_events)} selected events.")
        
        # Filtra gli eventi per quelli selezionati
        events_to_process = events_df[events_df['name'].isin(selected_events)].copy()
        
        # Trova il timestamp di fine per ogni evento (è l'inizio del successivo)
        events_to_process['end_ts'] = events_to_process['timestamp [ns]'].shift(-1)
        # Per l'ultimo evento, la fine è l'ultimo timestamp disponibile nei dati
        events_to_process.fillna({'end_ts': sync_data['timestamp [ns]'].max()}, inplace=True)

        for _, event in events_to_process.iterrows():
            start_ts, end_ts = event['timestamp [ns]'], event['end_ts']
            
            # Trova gli indici dei frame corrispondenti nel dataframe sincronizzato
            segment_data = sync_data.loc[start_ts:end_ts]
            if not segment_data.empty:
                # 'frame' è la colonna originale da world_timestamps.csv che corrisponde all'indice del frame video
                start_frame = int(segment_data.iloc[0]['frame'])
                end_frame = int(segment_data.iloc[-1]['frame'])
                frame_segments.append((start_frame, end_frame))
        
        if not frame_segments:
            print("WARNING: Could not find any video frames for the selected events. Generating full video instead.")
            frame_segments = [(0, min(total_frames, len(sync_data)))]
    else:
        # Se il trim non è attivo, processa l'intero video
        frame_segments = [(0, min(total_frames, len(sync_data)))]
    # --- FINE NUOVA LOGICA ---

    try:
        # Loop principale attraverso i segmenti di frame (sarà uno solo se il trim non è attivo)
        with tqdm(total=sum(end - start for start, end in frame_segments), desc="Generating Video") as pbar:
            for start_frame, end_frame in frame_segments:
                # Posiziona la testina di lettura del video all'inizio del segmento
                cap_ext.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                for frame_idx in range(start_frame, end_frame):
                    ret_ext, frame = cap_ext.read()
                    if not ret_ext: break
                    
                    frame_data = sync_data.iloc[frame_idx]
                    current_ts = frame_data['timestamp [ns]']

                    # --- MODIFICA CHIAVE: Disegno Heatmap con Trasparenza ---
                    if options.get('overlay_dynamic_heatmap'):
                        if pd.notna(frame_data.get('gaze x [px]')):
                            gaze_history_for_heatmap.append((int(frame_data['gaze x [px]']), int(frame_data['gaze y [px]'])))
                        if len(gaze_history_for_heatmap) > heatmap_window_size_frames:
                            gaze_history_for_heatmap.pop(0)

                        if len(gaze_history_for_heatmap) > 1:
                            intensity_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            for point in gaze_history_for_heatmap:
                                cv2.circle(intensity_map, point, radius=25, color=50, thickness=-1)
                            
                            intensity_map = cv2.blur(intensity_map, (91, 91))
                            heatmap_color = cv2.applyColorMap(intensity_map, cv2.COLORMAP_JET)
                            heatmap_rgba = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
                            heatmap_rgba[:, :, 3] = intensity_map
                            frame = _overlay_transparent(frame, heatmap_rgba, 0, 0)
                    # --- FINE MODIFICA CHIAVE ---

                    # --- AGGIUNGI QUESTO BLOCCO PRIMA DI TUTTI GLI ALTRI OVERLAY ---
                    # Overlay Heatmap Dinamica
                    if options.get('overlay_dynamic_heatmap'):
                        # Aggiungi il punto di sguardo corrente alla cronologia
                        if pd.notna(frame_data.get('gaze x [px]')):
                            gaze_history_for_heatmap.append((int(frame_data['gaze x [px]']), int(frame_data['gaze y [px]'])))
                        
                        # Mantieni la cronologia entro la dimensione della finestra
                        if len(gaze_history_for_heatmap) > heatmap_window_size_frames:
                            gaze_history_for_heatmap.pop(0)

                        if len(gaze_history_for_heatmap) > 1:
                            heatmap_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                            for point in gaze_history_for_heatmap:
                                cv2.circle(heatmap_img, point, radius=20, color=20, thickness=-1)
                            
                            heatmap_img = cv2.blur(heatmap_img, (81, 81))
                            heatmap_color = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
                            frame = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
                    # --- FINE BLOCCO ---
                    M = None
                    if options.get('crop_and_correct_perspective') and pd.notna(frame_data.get('tl x [px]')):
                        src_pts = np.float32([[frame_data[c] for c in [f'{p} x [px]', f'{p} y [px]']] for p in ['tl','tr','br','bl']])
                        dst_pts = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
                        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                        frame = cv2.warpPerspective(frame, M, (out_w, out_h))
                    elif frame.shape[1] != out_w or frame.shape[0] != out_h:
                        frame = cv2.resize(frame, (out_w, out_h))

                    # --- OVERLAYS ---

                    # Check for current event
                    if options.get('overlay_event_text') and event_idx < len(events_df):
                        # Trova l'ultimo evento iniziato prima del timestamp corrente
                        active_events = events_df[events_df['timestamp [ns]'] <= current_ts]
                        if not active_events.empty:
                            current_event_name = active_events.iloc[-1]['name']

                    if options.get('include_internal_cam') and cap_int is not None:
                        # --- NUOVA LOGICA DI SINCRONIZZAZIONE TEMPORALE ---
                        # Calcola il frame interno corrispondente in base al tempo del video esterno
                        current_time_sec = frame_idx / fps
                        target_int_frame_idx = int(current_time_sec * fps_int)
                        cap_int.set(cv2.CAP_PROP_POS_FRAMES, target_int_frame_idx)
                        ret_int, frame_int = cap_int.read()
                        if ret_int:
                            # Disegna il frame interno come Picture-in-Picture
                            pip_h = int(out_h * PIP_SCALE)
                            pip_w = int(frame_int.shape[1] * (pip_h / frame_int.shape[0]))
                            frame[10:10+pip_h, 10:10+pip_w] = cv2.resize(frame_int, (pip_w, pip_h))

                    # Draw YOLO object detection overlays
                    if options.get('overlay_yolo') and not yolo_detections.empty:
                        detections_for_frame = yolo_detections[yolo_detections['frame_idx'] == frame_idx]
                        for _, det in detections_for_frame.iterrows():
                            x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                            if M is not None:
                                pts = np.float32([[x1, y1], [x2, y2]]).reshape(-1, 1, 2)
                                transformed_pts = cv2.perspectiveTransform(pts, M)
                                if transformed_pts is not None:
                                    x1, y1 = int(transformed_pts[0][0][0]), int(transformed_pts[0][0][1])
                                    x2, y2 = int(transformed_pts[1][0][0]), int(transformed_pts[1][0][1])
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), YOLO_BOX_COLOR, YOLO_THICKNESS)
                            track_id = int(det['track_id'])
                            cv2.putText(frame, f"{det['class_name']}:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YOLO_TEXT_COLOR, 2)

                    if options.get('overlay_gaze') and pd.notna(frame_data.get('gaze x [px]')):
                        gaze_x, gaze_y = frame_data['gaze x [px]'], frame_data['gaze y [px]']
                        px, py = (int(gaze_x), int(gaze_y))
                        if M is not None:
                            gaze_pt_transformed = cv2.perspectiveTransform(np.array([[[gaze_x, gaze_y]]], dtype=np.float32), M)
                            if gaze_pt_transformed is not None:
                                px, py = int(gaze_pt_transformed[0][0][0]), int(gaze_pt_transformed[0][0][1])
                        if 0 <= px < out_w and 0 <= py < out_h:
                            cv2.circle(frame, (px, py), GAZE_RADIUS, GAZE_COLOR, GAZE_THICKNESS, cv2.LINE_AA)

                            # --- NUOVO: Logica per la scia dello sguardo ---
                            gaze_path_history.append((px, py))
                            if len(gaze_path_history) > gaze_path_length:
                                gaze_path_history.pop(0)
                            
                            if options.get('overlay_gaze_path', True) and len(gaze_path_history) > 1:
                                for i in range(1, len(gaze_path_history)):
                                    thickness = int(np.ceil((i / len(gaze_path_history)) * (GAZE_THICKNESS)))
                                    cv2.line(frame, gaze_path_history[i-1], gaze_path_history[i], GAZE_COLOR, thickness, cv2.LINE_AA)
                            # --- FINE BLOCCO ---
                    
                    if frame_data.get('is_blinking', False):
                        cv2.putText(frame, "BLINK", (out_w - 150, out_h - 20), cv2.FONT_HERSHEY_TRIPLEX, 1.5, BLINK_TEXT_COLOR, 2)
                    
                    # --- NUOVO: Overlay testo 'On Surface' ---
                    if options.get('overlay_on_surface_text'):
                        # Priorità all'AOI
                        if pd.notna(frame_data.get('aoi_name')):
                            aoi_name = frame_data['aoi_name']
                            cv2.putText(frame, f"on AOI: {aoi_name}", (20, out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ON_SURFACE_TEXT_COLOR, 2)
                        # Altrimenti, controlla la superficie generica
                        elif frame_data.get('gaze detected on surface') == True:
                            cv2.putText(frame, "on enrichment area", (20, out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ON_SURFACE_TEXT_COLOR, 2)

                    if options.get('overlay_pupil_plot'):
                        for name, col in pupil_cols.items():
                            if col in frame_data and pd.notna(frame_data[col]): pupil_plot_data[name].append(frame_data[col])
                            if len(pupil_plot_data[name]) > PUPIL_PLOT_HISTORY: pupil_plot_data[name].pop(0)
                        frame = _draw_pupil_plot(frame, pupil_plot_data, pupil_min, pupil_max, PUPIL_PLOT_WIDTH, PUPIL_PLOT_HEIGHT, (out_w - PUPIL_PLOT_WIDTH - 10, 10))

                    if options.get('overlay_fragmentation_plot'):
                        if 'gaze_speed_px_per_s' in frame_data and pd.notna(frame_data['gaze_speed_px_per_s']): frag_plot_data.append(frame_data['gaze_speed_px_per_s'])
                        if len(frag_plot_data) > FRAG_PLOT_HISTORY: frag_plot_data.pop(0)
                        y_pos = (PUPIL_PLOT_HEIGHT + 20) if options.get('overlay_pupil_plot') else 10
                        frame = _draw_generic_plot(frame, frag_plot_data, frag_min, frag_max, FRAG_PLOT_WIDTH, FRAG_PLOT_HEIGHT, (out_w - FRAG_PLOT_WIDTH - 10, y_pos), "Fragmentation", FRAG_LINE_COLOR, FRAG_BG_COLOR)
                    
                    # Draw Event Text Overlay
                    if options.get('overlay_event_text') and current_event_name:
                        font_scale = 1.0; font_thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(current_event_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        text_x = (out_w - text_width) // 2; text_y = 40
                        
                        rect_start = (text_x - 10, text_y - text_height - 5); rect_end = (text_x + text_width + 10, text_y + baseline)
                        sub_img = frame[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
                        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                        res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
                        frame[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]] = res
                        cv2.putText(frame, current_event_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, EVENT_TEXT_COLOR, font_thickness, cv2.LINE_AA)

                    writer.write(frame)
                    pbar.update(1) # Aggiorna la barra di avanzamento

    except Exception as e:
        print(f"An error occurred during video generation: {e}")
        traceback.print_exc()
    finally:
        print("Finalizing and releasing resources...")
        cap_ext.release()
        if cap_int:
            cap_int.release()
        writer.release()
        
        # --- NUOVA LOGICA: Aggiunta dell'audio ---
        if external_vid_path.exists():
            try:
                print("Adding audio to the generated video...")
                from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
                video_clip = VideoFileClip(str(temp_video_path))
                original_audio_clip = AudioFileClip(str(external_vid_path))

                # Se il video è stato tagliato, taglia anche l'audio per farlo corrispondere
                if options.get('trim_to_events') and frame_segments and sum(end - start for start, end in frame_segments) < total_frames:
                    # Crea una lista di clip audio tagliati e concatenali
                    audio_segments = [original_audio_clip.subclip(start/fps, end/fps) for start, end in frame_segments]
                    final_audio = CompositeAudioClip(audio_segments)
                else: # Altrimenti usa l'audio completo
                    final_audio = original_audio_clip

                final_clip = video_clip.set_audio(final_audio)

                # Salva il video finale
                final_clip.write_videofile(str(video_out_path), codec='libx264', audio_codec='aac', logger=None)
                
                # Chiudi tutte le clip
                final_clip.close(); video_clip.close(); original_audio_clip.close()
                
                # Rimuovi il file temporaneo senza audio
                temp_video_path.unlink(missing_ok=True)
                
                print("Video creation with audio completed!")
            except Exception as e:
                print(f"WARNING: An error occurred while adding audio. The video will be saved without audio. Error: {e}")
                # Rinominare il file temporaneo nel nome finale se l'aggiunta audio fallisce
                if temp_video_path.exists():
                    temp_video_path.rename(video_out_path)
                print("Video creation process completed, but without audio.")

def classify_detections(cap: cv2.VideoCapture, detections_df: pd.DataFrame, classify_model) -> list:
    """
    Esegue la classificazione sulle immagini ritagliate dalle bounding box delle detection.

    Args:
        cap (cv2.VideoCapture): L'oggetto VideoCapture del video sorgente.
        detections_df (pd.DataFrame): DataFrame contenente le detection da classificare.
                                      Deve avere le colonne ['frame_idx', 'track_id', 'x1', 'y1', 'x2', 'y2'].
        classify_model: Il modello di classificazione YOLO caricato.

    Returns:
        list: Una lista di dizionari, ognuno con 'frame_idx', 'track_id', 
              'classification_class', e 'confidence'.
    """
    if detections_df.empty:
        return []

    results = []
    
    # Raggruppa per frame per leggere ogni frame una sola volta
    grouped_by_frame = detections_df.groupby('frame_idx')
    
    with tqdm(total=len(grouped_by_frame), desc="Classifying Detections") as pbar:
        for frame_idx, group in grouped_by_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            for _, row in group.iterrows():
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                
                # Ritaglia l'immagine dalla bounding box
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    cls_results = classify_model(crop, verbose=False)
                    top1 = cls_results[0].probs.top1
                    top1_conf = cls_results[0].probs.top1conf.item()
                    class_name = classify_model.names[top1]
                    
                    results.append({
                        'frame_idx': frame_idx,
                        'track_id': row['track_id'],
                        'classification_class': class_name,
                        'confidence': top1_conf
                    })
            pbar.update(1)
            
    return results
