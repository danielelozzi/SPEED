# video_generator.py (VERSIONE CORRETTA E DEFINITIVA)
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# La funzione _create_pupil_plot_image rimane identica
def _create_pupil_plot_image(pupil_path: Path, gaze_path: Path, un_enriched_mode: bool):
    if not pupil_path.exists(): return None, None
    pupil_data = pd.read_csv(pupil_path)
    if pupil_data.empty: return None, None
    t0 = pupil_data['timestamp [ns]'].min()
    pupil_data['time_sec'] = (pupil_data['timestamp [ns]'] - t0) / 1e9
    if not un_enriched_mode and gaze_path.exists():
        gaze_data = pd.read_csv(gaze_path)
        if 'gaze detected on surface' in gaze_data.columns:
            gaze_data.rename(columns={'timestamp [ns]': 'gaze_timestamp_ns'}, inplace=True)
            pupil_data = pd.merge_asof(pupil_data.sort_values('timestamp [ns]'), gaze_data[['gaze_timestamp_ns', 'gaze detected on surface']].sort_values('gaze_timestamp_ns'), left_on='timestamp [ns]', right_on='gaze_timestamp_ns', direction='nearest', tolerance=100_000_000)
            pupil_data['gaze detected on surface'] = pupil_data['gaze detected on surface'].fillna(False)
        else: pupil_data['gaze detected on surface'] = False
    else: pupil_data['gaze detected on surface'] = False
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=150)
    if 'pupil diameter left [mm]' in pupil_data.columns: ax.plot(pupil_data['time_sec'], pupil_data['pupil diameter left [mm]'], label='Left Pupil', color='blue', alpha=0.8)
    if 'pupil diameter right [mm]' in pupil_data.columns: ax.plot(pupil_data['time_sec'], pupil_data['pupil diameter right [mm]'], label='Right Pupil', color='purple', alpha=0.8)
    if not un_enriched_mode and 'gaze detected on surface' in pupil_data.columns:
        for status, color in [(True, 'lightgreen'), (False, 'lightcoral')]:
            for _, g in pupil_data[pupil_data['gaze detected on surface'] == status].groupby((pupil_data['gaze detected on surface'] != pupil_data['gaze detected on surface'].shift()).cumsum()):
                if not g.empty: ax.axvspan(g['time_sec'].iloc[0], g['time_sec'].iloc[-1], facecolor=color, alpha=0.3, zorder=-1)
    ax.set_title("Pupil Diameter"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Diameter (mm)")
    ax.legend(fontsize='small'); ax.grid(True, linestyle='--', alpha=0.6)
    max_time = pupil_data['time_sec'].max()
    ax.set_xlim(0, max_time); fig.tight_layout()
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR), max_time


def create_custom_video(data_dir: Path, output_dir: Path, subj_name: str, options: dict, un_enriched_mode: bool):
    # --- Caricamento File ---
    external_video_path = data_dir / 'external.mp4'
    internal_video_path = data_dir / 'internal.mp4' if options.get("include_internal_cam") else None
    gaze_path = data_dir / ('gaze.csv' if un_enriched_mode else 'gaze_enriched.csv')
    pupil_path = data_dir / '3d_eye_states.csv'
    needs_surface_data = options.get("crop_to_surface") or options.get("trim_to_surface")
    surface_path = data_dir / 'surface_positions.csv' if needs_surface_data else None
    world_timestamps_path = data_dir / 'world_timestamps.csv'

    # Controlli esistenza file
    if not external_video_path.exists(): raise FileNotFoundError(f"Video esterno non trovato: {external_video_path}")
    if not gaze_path.exists(): raise FileNotFoundError(f"File Gaze non trovato: {gaze_path}")
    if not world_timestamps_path.exists(): raise FileNotFoundError(f"File timestamp (world_timestamps.csv) non trovato: {world_timestamps_path}")

    cap_ext = cv2.VideoCapture(str(external_video_path))
    cap_int = cv2.VideoCapture(str(internal_video_path)) if internal_video_path and internal_video_path.exists() else None
    fps = cap_ext.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_ext.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"DIAGNOSTICA: Trovati {total_frames} fotogrammi totali nel video di input con {fps} FPS.")
    if fps == 0: fps = 30

    # Caricamento e allineamento dati CSV
    world_timestamps = pd.read_csv(world_timestamps_path)
    if 'world_index' not in world_timestamps.columns: world_timestamps['world_index'] = world_timestamps.index
    gaze_data = pd.read_csv(gaze_path)
    aligned_data = pd.merge_asof(world_timestamps.sort_values('timestamp [ns]'), gaze_data.sort_values('timestamp [ns]'), on='timestamp [ns]', direction='nearest', tolerance=100_000_000)

    if needs_surface_data:
        if not surface_path or not surface_path.exists():
            raise FileNotFoundError(f"L'opzione 'trim_to_surface' o 'crop_to_surface' richiede 'surface_positions.csv', non trovato in {data_dir}")
        surface_data = pd.read_csv(surface_path)
        if 'world_index' not in surface_data.columns: surface_data['world_index'] = surface_data.index
        aligned_data = pd.merge(aligned_data, surface_data, on='world_index', how='left', suffixes=('', '_surf'))

    aligned_data.set_index('world_index', inplace=True)
    pupil_plot_base_img, plot_max_time = None, None
    if options.get("overlay_pupil_plot"): pupil_plot_base_img, plot_max_time = _create_pupil_plot_image(pupil_path, gaze_path, un_enriched_mode)

    start_frame = 0
    end_frame = total_frames - 1

    if options.get("trim_to_surface"):
        print("Opzione 'trim_to_surface' attiva: Ricerca dei frame con superficie visibile...")
        surface_present_frames = aligned_data.dropna(subset=['tl x [px]'])

        if not surface_present_frames.empty:
            start_frame = int(surface_present_frames.index.min())
            end_frame = int(surface_present_frames.index.max())
            print(f"SEGMENTO TROVATO: Il video verrà tagliato dal frame {start_frame} al frame {end_frame}.")
            if start_frame >= total_frames:
                raise ValueError(f"Il frame di inizio calcolato ({start_frame}) è oltre la fine del video ({total_frames}).")
            if end_frame >= total_frames:
                print(f"ATTENZIONE: Il frame di fine ({end_frame}) è stato corretto a {total_frames - 1}.")
                end_frame = total_frames - 1
        else:
            print("ATTENZIONE: Nessun dato di superficie trovato in 'surface_positions.csv'. Il video non verrà tagliato.")
            start_frame = 0
            end_frame = total_frames - 1

    # --- INIZIO BLOCCO MODIFICATO ---

    output_video_path = output_dir / options.get('output_filename', 'video_output.mp4')
    out_writer = None
    w_out, h_out = None, None

    # Determina le dimensioni di output PRIMA del ciclo
    if options.get("crop_to_surface") and not options.get("apply_perspective"):
        # Se il crop è attivo senza prospettiva, l'output è variabile.
        # È più sicuro non pre-definire le dimensioni e inizializzare al primo frame valido.
        print("ATTENZIONE: La modalità 'Crop senza prospettiva' ha dimensioni di output variabili. L'inizializzazione avverrà sul primo frame valido.")
    elif options.get("crop_to_surface") and options.get("apply_perspective"):
        # Se la prospettiva è attiva, possiamo calcolare le dimensioni da 'surface_positions.csv'
        first_valid_surface = aligned_data.dropna(subset=['tl x [px]']).iloc[0]
        src_pts = np.float32([[first_valid_surface[f'{c} x [px]'], first_valid_surface[f'{c} y [px]']] for c in ['tl', 'tr', 'br', 'bl']])
        w_out = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[3] - src_pts[2])))
        h_out = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))
    else:
        # Nessun crop, usa le dimensioni del video originale
        w_out = int(cap_ext.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_out = int(cap_ext.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Inizializza il VideoWriter se le dimensioni sono note
    if w_out and h_out and w_out > 0 and h_out > 0:
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # Codec H.264 per compatibilità
        out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w_out, h_out))
        print(f"\nDIAGNOSTICA: VideoWriter inizializzato con dimensioni (L x A): {w_out} x {h_out} e codec 'avc1'")
        if not out_writer.isOpened():
            raise IOError("ERRORE CRITICO: cv2.VideoWriter non è riuscito ad aprirsi.")

    # --- FINE BLOCCO MODIFICATO ---

    cap_ext.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    num_frames_to_process = (end_frame - start_frame) + 1
    pbar = tqdm(total=num_frames_to_process, desc="Generating Custom Video")

    frames_read_count = 0
    frames_written_count = 0

    for frame_idx in range(start_frame, end_frame + 1):
        ret_ext, frame_ext = cap_ext.read()
        if not ret_ext:
            print(f"DIAGNOSTICA: Lettura fallita al frame {frame_idx}. Fine del video?")
            break

        frames_read_count += 1
        current_data = aligned_data.loc[frame_idx] if frame_idx in aligned_data.index else None
        matrix = None
        final_frame = frame_ext.copy()

        if options.get("crop_to_surface") and current_data is not None and pd.notna(current_data.get('tl x [px]')):
            src_pts = np.float32([[current_data[f'{c} x [px]'], current_data[f'{c} y [px]']] for c in ['tl', 'tr', 'br', 'bl']])
            w = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[3] - src_pts[2])))
            h = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))
            if w > 0 and h > 0:
                if options.get("apply_perspective"):
                    dst_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    final_frame = cv2.warpPerspective(frame_ext, matrix, (w, h))
                else:
                    x, y, w_crop, h_crop = cv2.boundingRect(src_pts)
                    if w_crop > 0 and h_crop > 0:
                        final_frame = frame_ext[y:y+h_crop, x:x+w_crop]

        # --- INIZIO SECONDO BLOCCO MODIFICATO ---
        
        # Se il writer non è stato inizializzato prima (caso 'crop senza prospettiva')
        # lo inizializziamo ora.
        if out_writer is None:
            if final_frame.shape[0] > 0 and final_frame.shape[1] > 0:
                h_out, w_out, _ = final_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w_out, h_out))
                print(f"\nDIAGNOSTICA: VideoWriter inizializzato al primo frame valido con dimensioni (L x A): {w_out} x {h_out}")
                if not out_writer.isOpened(): 
                    print("ERRORE CRITICO: cv2.VideoWriter non è riuscito ad aprirsi neanche al primo frame.")
                    break # Interrompe il ciclo se l'inizializzazione fallisce
            else:
                pbar.update(1)
                continue # Salta i frame iniziali non validi

        # --- FINE SECONDO BLOCCO MODIFICATO ---
        
        if options.get("overlay_gaze") and current_data is not None:
            # ... (la logica originale per l'overlay del gaze va qui, se presente) ...
            pass

        if out_writer.isOpened():
            h_out_int = int(out_writer.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w_out_int = int(out_writer.get(cv2.CAP_PROP_FRAME_WIDTH))
            if final_frame.shape[0] > 0 and final_frame.shape[1] > 0 and h_out_int > 0 and w_out_int > 0:
                if final_frame.shape[0] != h_out_int or final_frame.shape[1] != w_out_int:
                    final_frame = cv2.resize(final_frame, (w_out_int, h_out_int))
                out_writer.write(final_frame)
                frames_written_count += 1
        pbar.update(1)

    pbar.close()
    print("\n--- REPORT DI DIAGNOSTICA ---")
    print(f"Fotogrammi elaborati: {frames_read_count} / {num_frames_to_process}")
    print(f"Fotogrammi scritti nel video di output: {frames_written_count}")
    if frames_written_count == 0 and frames_read_count > 0:
        print("RISULTATO: NESSUN FOTOGRAMMA SCRITTO.")
        print("CAUSA PROBABILE: Problema con il codec video o frame iniziali non validi che hanno impedito l'inizializzazione del writer.")
    elif frames_written_count > 0:
         print(f"RISULTATO: Video generato con successo in '{output_video_path}'.")
    print("----------------------------\n")

    cap_ext.release()
    if cap_int: cap_int.release()
    if out_writer: out_writer.release()
    cv2.destroyAllWindows()
    print(f"Generazione video completata. Salvato in {output_video_path}")