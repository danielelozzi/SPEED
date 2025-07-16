import pandas as pd
import numpy as np
import cv2
import os
import uuid
from pathlib import Path
from tqdm import tqdm
import time

# --- CONFIGURATION ---
OUTPUT_DIR = "synthetic_data"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
DURATION_SECONDS = 30  # Recording duration in seconds
FPS = 60  # Frames per second for videos and gaze data
NUM_FRAMES = DURATION_SECONDS * FPS

def create_dummy_video(filename, width, height, num_frames, fps, surface_corners_over_time, gaze_points_over_time, eye_image=False):
    """Creates a dummy video with a moving surface and a gaze point."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Eye image if requested
    eye_img_template = None
    if eye_image:
        eye_img_template = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.ellipse(eye_img_template, (width//2, height//2), (width//3, height//4), 0, 0, 360, (255, 255, 255), -1)


    for i in tqdm(range(num_frames), desc=f"Generating {Path(filename).name}"):
        if eye_image:
            frame = eye_img_template.copy()
            gaze = gaze_points_over_time[i]
            if gaze is not None:
                # Draw the pupil
                pupil_x = width // 2 + int((gaze[0] / VIDEO_WIDTH - 0.5) * 100)
                pupil_y = height // 2 + int((gaze[1] / VIDEO_HEIGHT - 0.5) * 100)
                cv2.circle(frame, (pupil_x, pupil_y), 30, (0,0,0), -1)
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Draw the moving surface
            corners = surface_corners_over_time[i]
            if corners is not None:
                cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(255, 255, 255), thickness=2)

            # Draw the gaze point
            gaze = gaze_points_over_time[i]
            if gaze is not None:
                cv2.drawMarker(frame, (int(gaze[0]), int(gaze[1])), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        out.write(frame)
    out.release()

def generate_data():
    """Generates all synthetic data files."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    # --- 1. Generation of Base IDs and Timestamps ---
    section_id = str(uuid.uuid4())
    recording_id = str(uuid.uuid4())
    start_time_ns = int(time.time() * 1e9)
    timestamps_ns = np.arange(start_time_ns, start_time_ns + DURATION_SECONDS * 1e9, 1e9 / FPS, dtype=np.int64)
    num_samples = len(timestamps_ns)

    print(f"Generating {num_samples} data samples...")

    # --- 2. Simulation of Gaze and Surface Movement ---
    # Surface movement (a rectangle moving slowly)
    surface_w, surface_h = 600, 400
    sx = VIDEO_WIDTH / 2 + np.sin(np.linspace(0, 2 * np.pi, num_samples)) * 100
    sy = VIDEO_HEIGHT / 2 + np.cos(np.linspace(0, 2 * np.pi, num_samples)) * 50
    surface_corners_over_time = [
        np.array([
            [x - surface_w/2, y - surface_h/2], [x + surface_w/2, y - surface_h/2],
            [x + surface_w/2, y + surface_h/2], [x - surface_w/2, y + surface_h/2]
        ]) for x, y in zip(sx, sy)
    ]

    # Gaze movement (a random but smooth path)
    gaze_x = pd.Series(np.random.randn(num_samples)).cumsum()
    gaze_y = pd.Series(np.random.randn(num_samples)).cumsum()
    gaze_x = np.interp(gaze_x, (gaze_x.min(), gaze_x.max()), (100, VIDEO_WIDTH - 100))
    gaze_y = np.interp(gaze_y, (gaze_y.min(), gaze_y.max()), (100, VIDEO_HEIGHT - 100))
    gaze_points_over_time = list(zip(gaze_x, gaze_y))

    # --- 3. Generation of Events: Blinks, Fixations, and Saccades ---
    # Blinks
    num_blinks = DURATION_SECONDS * 1 # Approximately 1 blink per second
    blink_starts = np.random.choice(num_samples, num_blinks, replace=False)
    blink_durations_samples = np.random.randint(int(0.1 * FPS), int(0.4 * FPS), num_blinks)
    is_blinking = np.zeros(num_samples, dtype=bool)
    blink_ids = np.full(num_samples, np.nan)
    
    blinks_list = []
    for i, start_idx in enumerate(blink_starts):
        end_idx = min(start_idx + blink_durations_samples[i], num_samples - 1)
        is_blinking[start_idx:end_idx] = True
        blink_ids[start_idx:end_idx] = i + 1
        blinks_list.append({
            'section id': section_id, 'recording id': recording_id, 'blink id': i + 1,
            'start timestamp [ns]': timestamps_ns[start_idx],
            'end timestamp [ns]': timestamps_ns[end_idx],
            'duration [ms]': (timestamps_ns[end_idx] - timestamps_ns[start_idx]) / 1e6
        })
    blinks_df = pd.DataFrame(blinks_list) if blinks_list else pd.DataFrame(columns=['section id', 'recording id', 'blink id', 'start timestamp [ns]', 'end timestamp [ns]', 'duration [ms]'])

    # Fixations and Saccades
    is_saccade = np.zeros(num_samples, dtype=bool)
    # Simulate saccades occasionally
    saccade_starts = np.sort(np.random.choice(np.where(~is_blinking)[0], DURATION_SECONDS * 3, replace=False))
    fixation_id_counter = 1
    fixation_ids = np.full(num_samples, np.nan)
    
    fixations_list = []
    saccades_list = []
    last_event_end = 0

    for start_idx in saccade_starts:
        if start_idx <= last_event_end:
            continue

        # Fixation period before the saccade
        fix_start_idx = last_event_end
        fix_end_idx = start_idx -1
        if fix_end_idx > fix_start_idx:
            fixation_ids[fix_start_idx:fix_end_idx] = fixation_id_counter
            fixations_list.append({
                'section id': section_id, 'recording id': recording_id, 'fixation id': fixation_id_counter,
                'start timestamp [ns]': timestamps_ns[fix_start_idx],
                'end timestamp [ns]': timestamps_ns[fix_end_idx],
                'duration [ms]': (timestamps_ns[fix_end_idx] - timestamps_ns[fix_start_idx]) / 1e6,
                'fixation x [px]': gaze_x[fix_start_idx:fix_end_idx].mean(),
                'fixation y [px]': gaze_y[fix_start_idx:fix_end_idx].mean(),
                'azimuth [deg]': np.random.uniform(-15, 15),
                'elevation [deg]': np.random.uniform(-15, 15)
            })
            fixation_id_counter += 1

        # Saccade period
        saccade_duration_samples = np.random.randint(int(0.02 * FPS), int(0.1 * FPS))
        saccade_end_idx = min(start_idx + saccade_duration_samples, num_samples - 1)
        is_saccade[start_idx:saccade_end_idx] = True

        start_point = np.array([gaze_x[start_idx], gaze_y[start_idx]])
        end_point = np.array([gaze_x[saccade_end_idx], gaze_y[saccade_end_idx]])
        px_dist = np.linalg.norm(end_point - start_point)
        duration_s = (timestamps_ns[saccade_end_idx] - timestamps_ns[start_idx]) / 1e9

        saccades_list.append({
            'section id': section_id, 'recording id': recording_id, 'saccade id': len(saccades_list) + 1,
            'start timestamp [ns]': timestamps_ns[start_idx],
            'end timestamp [ns]': timestamps_ns[saccade_end_idx],
            'duration [ms]': duration_s * 1000,
            'amplitude [px]': px_dist,
            'amplitude [deg]': px_dist / 50.0, # Rough estimate
            'mean velocity [px/s]': px_dist / duration_s if duration_s > 0 else 0,
            'peak velocity [px/s]': (px_dist / duration_s if duration_s > 0 else 0) * np.random.uniform(1.2, 1.8)
        })
        last_event_end = saccade_end_idx

    fixations_df = pd.DataFrame(fixations_list) if fixations_list else pd.DataFrame(columns=['section id', 'recording id', 'fixation id', 'start timestamp [ns]', 'end timestamp [ns]', 'duration [ms]', 'fixation x [px]', 'fixation y [px]', 'azimuth [deg]', 'elevation [deg]'])
    saccades_df = pd.DataFrame(saccades_list) if saccades_list else pd.DataFrame(columns=['section id', 'recording id', 'saccade id', 'start timestamp [ns]', 'end timestamp [ns]', 'duration [ms]', 'amplitude [px]', 'amplitude [deg]', 'mean velocity [px/s]', 'peak velocity [px/s]'])


    # --- 4. Creation of CSV DataFrames ---
    
    # world_timestamps.csv
    world_timestamps_df = pd.DataFrame({
        'section id': section_id, 'recording id': recording_id, 'timestamp [ns]': timestamps_ns
    })
    
    # events.csv
    events_df = pd.DataFrame([
        {'recording id': recording_id, 'timestamp [ns]': timestamps_ns[0], 'name': 'recording.begin', 'type': 'recording'},
        {'recording id': recording_id, 'timestamp [ns]': timestamps_ns[-1], 'name': 'recording.end', 'type': 'recording'}
    ])
    
    # gaze.csv
    gaze_df = pd.DataFrame({
        'section id': section_id, 'recording id': recording_id, 'timestamp [ns]': timestamps_ns,
        'gaze x [px]': gaze_x, 'gaze y [px]': gaze_y, 'worn': True,
        'fixation id': fixation_ids, 'blink id': blink_ids,
        'azimuth [deg]': np.random.uniform(-15, 15, num_samples),
        'elevation [deg]': np.random.uniform(-15, 15, num_samples)
    })
    gaze_df.loc[is_blinking, ['gaze x [px]', 'gaze y [px]', 'fixation id']] = np.nan

    # 3d_eye_states.csv
    eye_states_df = pd.DataFrame({
        'section id': section_id, 'recording id': recording_id, 'timestamp [ns]': timestamps_ns,
        'pupil diameter left [mm]': np.random.normal(3.5, 0.5, num_samples),
        'pupil diameter right [mm]': np.random.normal(3.5, 0.5, num_samples),
        'eyeball center left x [mm]': np.random.normal(-31, 1, num_samples),
        'eyeball center left y [mm]': np.random.normal(13, 1, num_samples),
        'eyeball center left z [mm]': np.random.normal(-34, 1, num_samples),
        'eyeball center right x [mm]': np.random.normal(31, 1, num_samples),
        'eyeball center right y [mm]': np.random.normal(15, 1, num_samples),
        'eyeball center right z [mm]': np.random.normal(-36, 1, num_samples),
        'optical axis left x': np.random.normal(-0.1, 0.05, num_samples),
        'optical axis left y': np.random.normal(0.1, 0.05, num_samples),
        'optical axis left z': np.random.normal(0.98, 0.01, num_samples),
        'optical axis right x': np.random.normal(-0.2, 0.05, num_samples),
        'optical axis right y': np.random.normal(0.1, 0.05, num_samples),
        'optical axis right z': np.random.normal(0.96, 0.01, num_samples),
        'eyelid angle top left [rad]': np.nan, 'eyelid angle bottom left [rad]': np.nan,
        'eyelid aperture left [mm]': np.nan, 'eyelid angle top right [rad]': np.nan,
        'eyelid angle bottom right [rad]': np.nan, 'eyelid aperture right [mm]': np.nan
    })

    # surface_positions.csv
    surface_pos_list = []
    for i, corners in enumerate(surface_corners_over_time):
        surface_pos_list.append({
            'section id': section_id, 'recording id': recording_id, 'timestamp [ns]': timestamps_ns[i],
            'detected marker IDs': '"1;2;3;4"', 'tl x [px]': corners[0,0], 'tl y [px]': corners[0,1],
            'tr x [px]': corners[1,0], 'tr y [px]': corners[1,1], 'br x [px]': corners[2,0],
            'br y [px]': corners[2,1], 'bl x [px]': corners[3,0], 'bl y [px]': corners[3,1]
        })
    surface_positions_df = pd.DataFrame(surface_pos_list)

    # gaze_enriched.csv & fixations_enriched.csv
    gaze_on_surface = []
    gaze_norm_x = []
    gaze_norm_y = []
    for i in range(num_samples):
        surface = surface_corners_over_time[i]
        gaze = gaze_points_over_time[i]
        min_x, min_y = surface.min(axis=0)
        max_x, max_y = surface.max(axis=0)
        
        on_surf = (min_x <= gaze[0] <= max_x) and (min_y <= gaze[1] <= max_y)
        gaze_on_surface.append(on_surf)

        surf_w = max_x - min_x
        surf_h = max_y - min_y
        gaze_norm_x.append((gaze[0] - min_x) / surf_w if surf_w > 0 else 0)
        gaze_norm_y.append((gaze[1] - min_y) / surf_h if surf_h > 0 else 0)
        
    gaze_enriched_df = pd.DataFrame({
        'section id': section_id, 'recording id': recording_id, 'timestamp [ns]': timestamps_ns,
        'gaze detected on surface': gaze_on_surface,
        'gaze position on surface x [normalized]': gaze_norm_x,
        'gaze position on surface y [normalized]': gaze_norm_y,
        'fixation id': fixation_ids
    })
    
    fixations_enriched_df = fixations_df.copy()
    fixations_enriched_df = fixations_enriched_df.rename(columns={
        'fixation x [px]': 'fixation x [normalized]',
        'fixation y [px]': 'fixation y [normalized]'
    })
    fixations_enriched_df['fixation detected on surface'] = False # Simplification
    fixations_enriched_df[['fixation x [normalized]', 'fixation y [normalized]']] = np.random.rand(len(fixations_enriched_df), 2)
    fixations_enriched_df = fixations_enriched_df[['section id', 'recording id', 'fixation id', 'start timestamp [ns]', 'end timestamp [ns]', 'duration [ms]', 'fixation detected on surface', 'fixation x [normalized]', 'fixation y [normalized]']]


    # --- 5. Saving the CSV Files ---
    gaze_df.to_csv(output_path / "gaze.csv", index=False)
    blinks_df.to_csv(output_path / "blinks.csv", index=False)
    fixations_df.to_csv(output_path / "fixations.csv", index=False)
    saccades_df.to_csv(output_path / "saccades.csv", index=False)
    eye_states_df.to_csv(output_path / "3d_eye_states.csv", index=False)
    world_timestamps_df.to_csv(output_path / "world_timestamps.csv", index=False)
    events_df.to_csv(output_path / "events.csv", index=False)
    surface_positions_df.to_csv(output_path / "surface_positions.csv", index=False)
    gaze_enriched_df.to_csv(output_path / "gaze_enriched.csv", index=False)
    fixations_enriched_df.to_csv(output_path / "fixations_enriched.csv", index=False)
    print(f"All CSV files have been generated in the '{output_path}' folder")

    # --- 6. Generating Videos ---
    create_dummy_video(str(output_path / "world.mp4"), VIDEO_WIDTH, VIDEO_HEIGHT, NUM_FRAMES, FPS, surface_corners_over_time, gaze_points_over_time)
    create_dummy_video(str(output_path / "eye.mp4"), 320, 240, NUM_FRAMES, FPS, [], gaze_points_over_time, eye_image=True)
    print("Videos generated.")


if __name__ == "__main__":
    generate_data()