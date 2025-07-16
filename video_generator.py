import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import traceback

# --- Drawing Constants ---
GAZE_COLOR = (0, 0, 255)  # Red in BGR
GAZE_RADIUS = 15
GAZE_THICKNESS = 2

PIP_SCALE = 0.25

PUPIL_PLOT_HISTORY = 200
PUPIL_PLOT_WIDTH = 350 # Slightly increased for the legend
PUPIL_PLOT_HEIGHT = 150
PUPIL_BG_COLOR = (80, 80, 80)
# Colors for the different plot lines
PUPIL_COLORS = {
    "Left": (80, 80, 255),   # Red
    "Right": (80, 255, 80),  # Green
    "Mean": (255, 255, 255) # White
}
BLINK_TEXT_COLOR = (0, 0, 255)

def _prepare_data(data_dir: Path, un_enriched_mode: bool, options: dict):
    """
    Loads and synchronizes all necessary DataFrames for video generation.
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
    
    # Load both pupil data and calculate the mean
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
            start_ts, end_ts = row['start timestamp [ns]'], row['end timestamp [ns]']
            merged_data.loc[(merged_data['timestamp [ns]'] >= start_ts) & (merged_data['timestamp [ns]'] <= end_ts), 'is_blinking'] = True
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

def _draw_pupil_plot(frame: np.ndarray, plot_data_dict: dict, min_val: float, max_val: float, width: int, height: int, position: tuple):
    """
    Draws a multi-line graph for pupil data with a legend.
    """
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
                x_pos + int((i / (PUPIL_PLOT_HISTORY -1)) * width),
                y_pos + height - int(((val - min_val) / (max_val - min_val)) * (height - 50)) - 10 # 50px margin for legend
            )
            for i, val in enumerate(data_points) if pd.notna(val)
        ]
        
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], isClosed=False, color=PUPIL_COLORS[name], thickness=2)

    return frame


def create_custom_video(data_dir: Path, output_dir: Path, subj_name: str, options: dict, un_enriched_mode: bool):
    """
    Main function for creating the video with the new overlay layout.
    """
    video_out_path = output_dir / options.get('output_filename', f'video_output_{subj_name}.mp4')
    
    print("Loading and synchronizing data...")
    try:
        sync_data = _prepare_data(data_dir, un_enriched_mode, options)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}. Cannot generate video.")
        return

    external_vid_path = data_dir / 'external.mp4'
    cap_ext = cv2.VideoCapture(str(external_vid_path))
    if not cap_ext.isOpened():
        print(f"ERROR: Cannot open external video: {external_vid_path}")
        return

    # Open the internal camera video if the option is active
    cap_int = None
    if options.get('include_internal_cam'):
        internal_vid_path = data_dir / 'internal.mp4'
        if internal_vid_path.exists():
            cap_int = cv2.VideoCapture(str(internal_vid_path))
            if not cap_int.isOpened():
                print("WARNING: Cannot open internal video, PiP disabled.")
                options['include_internal_cam'] = False
        else:
            print("WARNING: Internal video not found, PiP disabled.")
            options['include_internal_cam'] = False


    total_frames = int(cap_ext.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_ext.get(cv2.CAP_PROP_FPS)
    original_w = int(cap_ext.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap_ext.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_w, out_h = (1280, 720) if options.get('crop_and_correct_perspective') else (original_w, original_h)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (out_w, out_h))
    print(f"The output video will be saved to: {video_out_path}")

    # Setup for the three pupil plot lines
    pupil_plot_data = {"Left": [], "Right": [], "Mean": []}
    pupil_min, pupil_max = 0, 1
    pupil_cols = {
        "Left": "pupil diameter left [mm]",
        "Right": "pupil diameter right [mm]",
        "Mean": "pupil_diameter_mean"
    }
    if options.get('overlay_pupil_plot'):
        all_pupil_data = pd.concat([sync_data[col] for col in pupil_cols.values() if col in sync_data.columns]).dropna()
        if not all_pupil_data.empty:
            pupil_min = all_pupil_data.min()
            pupil_max = all_pupil_data.max()

    try:
        for frame_idx in tqdm(range(min(total_frames, len(sync_data))), desc="Generating Video"):
            ret_ext, frame = cap_ext.read()
            if not ret_ext:
                break
            
            frame_data = sync_data.iloc[frame_idx]
            
            M = None
            if options.get('crop_and_correct_perspective') and pd.notna(frame_data.get('tl x [px]')):
                src_pts = np.float32([[frame_data[c] for c in ['tl x [px]', 'tl y [px]']], [frame_data[c] for c in ['tr x [px]', 'tr y [px]']], [frame_data[c] for c in ['br x [px]', 'br y [px]']], [frame_data[c] for c in ['bl x [px]', 'bl y [px]']]])
                dst_pts = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]])
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                frame = cv2.warpPerspective(frame, M, (out_w, out_h))
            elif frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))

            # --- OVERLAYS ---
            
            # Logic for the internal camera (PiP) in the top-left corner
            if options.get('include_internal_cam') and cap_int is not None:
                ret_int, frame_int = cap_int.read()
                if ret_int:
                    pip_h = int(out_h * PIP_SCALE)
                    pip_w = int(frame_int.shape[1] * (pip_h / frame_int.shape[0]))
                    frame_int_resized = cv2.resize(frame_int, (pip_w, pip_h))
                    # Position in the top-left with a 10px margin
                    frame[10:10+pip_h, 10:10+pip_w] = frame_int_resized

            if options.get('overlay_gaze') and pd.notna(frame_data.get('gaze x [px]')):
                gaze_x, gaze_y = frame_data['gaze x [px]'], frame_data['gaze y [px]']
                if M is not None:
                    gaze_pt_transformed = cv2.perspectiveTransform(np.array([[[gaze_x, gaze_y]]], dtype=np.float32), M)
                    if gaze_pt_transformed is not None: px, py = int(gaze_pt_transformed[0][0][0]), int(gaze_pt_transformed[0][0][1])
                else:
                    px, py = int(gaze_x), int(gaze_y)
                if 0 <= px < out_w and 0 <= py < out_h:
                    cv2.circle(frame, (px, py), GAZE_RADIUS, GAZE_COLOR, GAZE_THICKNESS, cv2.LINE_AA)
            
            # BLINK text in the bottom-right
            if frame_data.get('is_blinking', False):
                cv2.putText(frame, "BLINK", (out_w - 150, out_h - 20), cv2.FONT_HERSHEY_TRIPLEX, 1.5, BLINK_TEXT_COLOR, 2)
            
            # Logic for the pupil plot in the top-right corner
            if options.get('overlay_pupil_plot'):
                for name, col in pupil_cols.items():
                    if col in frame_data:
                        pupil_plot_data[name].append(frame_data[col])
                        if len(pupil_plot_data[name]) > PUPIL_PLOT_HISTORY:
                            pupil_plot_data[name].pop(0)
                
                # Position in the top-right with a 10px margin
                frame = _draw_pupil_plot(frame, pupil_plot_data, pupil_min, pupil_max, PUPIL_PLOT_WIDTH, PUPIL_PLOT_HEIGHT, (out_w - PUPIL_PLOT_WIDTH - 10, 10))

            writer.write(frame)

    except Exception as e:
        print(f"An error occurred during video generation: {e}")
        traceback.print_exc()
    finally:
        print("Finalizing and releasing resources...")
        cap_ext.release()
        if cap_int:
            cap_int.release()
        writer.release()
        print("Video creation process completed!")