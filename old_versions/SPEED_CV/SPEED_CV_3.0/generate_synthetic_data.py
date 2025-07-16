# generate_synthetic_data.py
import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "synthetic_data_for_speed"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
DURATION_SECONDS = 30
FPS = 30
NUM_FRAMES = DURATION_SECONDS * FPS

def create_dummy_video(filename, width, height, num_frames, fps, surface_corners_over_time, gaze_points_over_time):
    """Creates a dummy MP4 video with a moving surface and gaze point."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in tqdm(range(num_frames), desc=f"Generating {Path(filename).name}"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw the moving surface
        corners = surface_corners_over_time[i]
        if corners is not None:
            cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(255, 255, 255), thickness=2)

        # Draw the gaze point
        gaze = gaze_points_over_time[i]
        if gaze is not None:
            cv2.drawMarker(frame, (int(gaze[0]), int(gaze[1])), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

        out.write(frame)
    out.release()

def create_internal_video(filename, width, height, num_frames, fps, pupil_diameters):
    """Creates a dummy internal video with a changing pupil size."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in tqdm(range(num_frames), desc=f"Generating {Path(filename).name}"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = int(pupil_diameters[i] * 5) # Scale diameter for visibility
        cv2.circle(frame, center, radius, (200, 200, 200), -1)
        out.write(frame)
    out.release()


def generate_synthetic_data():
    """Main function to generate all synthetic files."""
    print(f"Creating output directory: {OUTPUT_DIR}")
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # --- 1. Time and Event Generation ---
    start_time_ns = int(pd.Timestamp.now().timestamp() * 1e9)
    timestamps_ns = start_time_ns + np.arange(NUM_FRAMES) * int(1e9 / FPS)
    
    events_data = []
    num_events = 4
    for i in range(num_events):
        event_time = timestamps_ns[i * NUM_FRAMES // num_events]
        events_data.append({'recording id': 'synthetic_rec_01', 'name': f'Task_{i+1}', 'timestamp [ns]': event_time})
    events_df = pd.DataFrame(events_data)
    events_df.to_csv(output_path / "events.csv", index=False)
    print("Generated events.csv")

    # --- 2. Surface and Gaze Movement ---
    surface_corners_over_time = []
    gaze_points_over_time = []
    surface_data = []
    
    surface_w, surface_h = 400, 300
    
    for i in range(NUM_FRAMES):
        # Surface movement (sinusoidal path)
        center_x = VIDEO_WIDTH / 2 + np.sin(i / 100) * 200
        center_y = VIDEO_HEIGHT / 2 + np.cos(i / 70) * 100
        
        tl = (center_x - surface_w/2, center_y - surface_h/2)
        tr = (center_x + surface_w/2, center_y - surface_h/2)
        bl = (center_x - surface_w/2, center_y + surface_h/2)
        br = (center_x + surface_w/2, center_y + surface_h/2)
        # FIX: Ensure the dtype is float32 for cv2.pointPolygonTest
        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        surface_corners_over_time.append(corners)
        
        surface_data.append({
            'world_index': i, 'timestamp [ns]': timestamps_ns[i],
            'surface_name': 'synthetic_surface', 'is_detected': True,
            'tl x [px]': tl[0], 'tl y [px]': tl[1],
            'tr x [px]': tr[0], 'tr y [px]': tr[1],
            'bl x [px]': bl[0], 'bl y [px]': bl[1],
            'br x [px]': br[0], 'br y [px]': br[1],
        })

        # Gaze movement
        on_surface = (i // (NUM_FRAMES // 5)) % 2 == 0 # Alternate on/off surface
        if on_surface:
            gaze_x = center_x + np.random.randn() * 20
            gaze_y = center_y + np.random.randn() * 20
        else:
            gaze_x = np.random.randint(0, VIDEO_WIDTH)
            gaze_y = np.random.randint(0, VIDEO_HEIGHT)
        gaze_points_over_time.append((gaze_x, gaze_y))

    surface_df = pd.DataFrame(surface_data)
    surface_df.to_csv(output_path / "surface_positions.csv", index=False)
    print("Generated surface_positions.csv")

    # --- 3. Generate Gaze, Fixations, and Pupil Data ---
    gaze_list, gaze_enriched_list, fixations_list, fixations_enriched_list = [], [], [], []
    blinks_list, saccades_list, pupil_list = [], [], []
    fixation_id_counter = 0
    current_fixation = None

    for i in range(NUM_FRAMES):
        ts = timestamps_ns[i]
        gaze_x, gaze_y = gaze_points_over_time[i]
        corners = surface_corners_over_time[i]
        
        # Gaze Data
        gaze_list.append({'timestamp [ns]': ts, 'gaze x [px]': gaze_x, 'gaze y [px]': gaze_y, 'fixation id': -1})
        
        # Enriched Gaze Data
        gaze_on_surface = cv2.pointPolygonTest(corners, (gaze_x, gaze_y), False) >= 0
        gaze_x_norm, gaze_y_norm = np.nan, np.nan
        if gaze_on_surface:
            gaze_x_norm = (gaze_x - corners[0][0]) / surface_w
            gaze_y_norm = (gaze_y - corners[0][1]) / surface_h
        gaze_enriched_list.append({
            'timestamp [ns]': ts, 'gaze detected on surface': gaze_on_surface,
            'gaze position on surface x [normalized]': gaze_x_norm,
            'gaze position on surface y [normalized]': gaze_y_norm,
            'fixation id': -1
        })

        # Pupil Data
        pupil_diameter = 4.0 + np.sin(i / 50) * 0.5 + np.random.randn() * 0.1
        pupil_list.append({'timestamp [ns]': ts, 'pupil diameter left [mm]': pupil_diameter, 'pupil diameter right [mm]': pupil_diameter + 0.1})

        # Blinks and Saccades (randomly)
        if np.random.rand() < 0.01:
            blinks_list.append({'start timestamp [ns]': ts, 'duration [ms]': np.random.randint(100, 300)})
        if np.random.rand() < 0.05:
            saccades_list.append({'start timestamp [ns]': ts, 'duration [ms]': np.random.randint(30, 80), 'amplitude [deg]': np.random.uniform(1, 15)})

    # Create DataFrames
    gaze_df = pd.DataFrame(gaze_list)
    gaze_enriched_df = pd.DataFrame(gaze_enriched_list)
    pupil_df = pd.DataFrame(pupil_list)
    blinks_df = pd.DataFrame(blinks_list)
    saccades_df = pd.DataFrame(saccades_list)
    
    # Save CSVs
    gaze_df.to_csv(output_path / "gaze.csv", index=False)
    gaze_enriched_df.to_csv(output_path / "gaze_enriched.csv", index=False)
    pupil_df.to_csv(output_path / "3d_eye_states.csv", index=False)
    blinks_df.to_csv(output_path / "blinks.csv", index=False)
    saccades_df.to_csv(output_path / "saccades.csv", index=False)
    # Dummy fixation files (can be improved with more complex logic)
    gaze_df.rename(columns={'gaze x [px]': 'fixation x [px]', 'gaze y [px]': 'fixation y [px]'}).to_csv(output_path / "fixations.csv", index=False)
    gaze_enriched_df.rename(columns={'gaze position on surface x [normalized]': 'fixation x [normalized]', 'gaze position on surface y [normalized]': 'fixation y [normalized]'}).to_csv(output_path / "fixations_enriched.csv", index=False)
    print("Generated gaze, pupil, blink, saccade, and dummy fixation CSVs.")

    # --- 4. Generate Timestamps and Videos ---
    world_timestamps_df = pd.DataFrame({'timestamp [ns]': timestamps_ns})
    world_timestamps_df.to_csv(output_path / "world_timestamps.csv", index=False)
    print("Generated world_timestamps.csv")

    create_dummy_video(str(output_path / "external.mp4"), VIDEO_WIDTH, VIDEO_HEIGHT, NUM_FRAMES, FPS, surface_corners_over_time, gaze_points_over_time)
    create_internal_video(str(output_path / "internal.mp4"), 320, 240, NUM_FRAMES, FPS, pupil_df['pupil diameter left [mm]'])
    
    print("\n--- Synthetic Data Generation Complete! ---")
    print(f"All files saved in: {output_path.resolve()}")

if __name__ == "__main__":
    generate_synthetic_data()
