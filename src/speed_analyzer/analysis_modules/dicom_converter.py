# src/speed_analyzer/analysis_modules/dicom_converter.py
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
import numpy as np
from pathlib import Path
import logging
import datetime
import time

def convert_to_dicom(unenriched_dir: Path, output_dicom_path: Path, patient_info: dict):
    """
    Converte i dati di eye-tracking in un singolo file DICOM usando il Waveform IOD.
    """
    logging.info(f"--- AVVIO CONVERSIONE DICOM ---")

    # 1. Carica e prepara i dati
    gaze_df = pd.read_csv(unenriched_dir / "gaze.csv")
    pupil_df = pd.read_csv(unenriched_dir / "3d_eye_states.csv")
    events_df = pd.read_csv(unenriched_dir / "events.csv") if (unenriched_dir / "events.csv").exists() else pd.DataFrame()

    merged_df = pd.merge_asof(
        gaze_df.sort_values('timestamp [ns]'),
        pupil_df[['timestamp [ns]', 'pupil diameter left [mm]']].sort_values('timestamp [ns]'),
        on='timestamp [ns]', direction='nearest', tolerance=pd.Timedelta('50ms').value
    ).dropna(subset=['gaze x [px]', 'gaze y [px]', 'pupil diameter left [mm]'])

    if merged_df.empty:
        raise ValueError("Nessun dato valido di sguardo o pupilla trovato dopo l'unione dei file.")
        
    start_time_ns = merged_df['timestamp [ns]'].min()
    merged_df['time_sec'] = (merged_df['timestamp [ns]'] - start_time_ns) / 1e9

    # 2. Imposta i metadati del file DICOM
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.9.1.1' # Waveform Storage SOP Class
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian 

    # 3. Crea il dataset DICOM principale
    ds = FileDataset(str(output_dicom_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    now = datetime.datetime.now()
    ds.PatientName = patient_info.get("name", "Anonymous")
    ds.PatientID = patient_info.get("id", "NoID")
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.StudyDate = now.strftime('%Y%m%d')
    ds.StudyTime = now.strftime('%H%M%S.%f')
    ds.ContentDate = ds.StudyDate
    ds.ContentTime = ds.StudyTime
    ds.Modality = 'WAV'

    # 4. Crea la Waveform Sequence usando i tag numerici corretti
    waveform = Dataset()
    # --- CORREZIONE FINALE: Utilizzo dei tag numerici espliciti ---
    waveform.add_new((0x003A, 0x0005), 'US', 3)  # Number of Channels
    waveform.add_new((0x003A, 0x0010), 'UL', len(merged_df)) # Number of Samples
    sampling_freq = 1 / merged_df['time_sec'].diff().mean()
    waveform.add_new((0x003A, 0x001A), 'DS', f"{sampling_freq:.6f}") # Sampling Frequency
    waveform.add_new((0x5400, 0x1004), 'US', 16) # Waveform Bits Allocated
    waveform.add_new((0x5400, 0x1006), 'CS', 'SS') # Waveform Sample Interpretation
    # --- FINE CORREZIONE FINALE ---

    channel_definition_list = []
    channel_defs = [("Gaze X", "pixels"), ("Gaze Y", "pixels"), ("Pupil Diameter", "mm")]
    for i, (name, units) in enumerate(channel_defs):
        ch_def = Dataset()
        source_ds = Dataset()
        source_ds.CodeValue = str(i)
        source_ds.CodingSchemeDesignator = "L"
        source_ds.CodeMeaning = name
        ch_def.ChannelSourceSequence = Sequence([source_ds])
        channel_definition_list.append(ch_def)
    waveform.ChannelDefinitionSequence = Sequence(channel_definition_list)
    
    # Prepara e inserisce i dati della waveform
    gaze_x_int = (merged_df['gaze x [px]'] * 100).astype(np.int16)
    gaze_y_int = (merged_df['gaze y [px]'] * 100).astype(np.int16)
    pupil_int = (merged_df['pupil diameter left [mm]'] * 100).astype(np.int16)

    waveform_data = np.empty((len(merged_df) * 3,), dtype=np.int16)
    waveform_data[0::3] = gaze_x_int
    waveform_data[1::3] = gaze_y_int
    waveform_data[2::3] = pupil_int
    
    waveform.WaveformData = waveform_data.tobytes()
    ds.WaveformSequence = Sequence([waveform])

    # 5. Aggiungi gli eventi
    if not events_df.empty:
        annotation_list = []
        for _, event in events_df.iterrows():
            annotation = Dataset()
            event_time_sec = (event['timestamp [ns]'] - start_time_ns) / 1e9
            time_obj = datetime.datetime(1900, 1, 1) + datetime.timedelta(seconds=event_time_sec)
            annotation.AnnotationTime = time_obj.strftime('%H%M%S.%f')
            annotation.UnformattedTextValue = event['name']
            annotation_list.append(annotation)
        ds.AnnotationSequence = Sequence(annotation_list)

    # 6. Salva il file
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.save_as(str(output_dicom_path), write_like_original=False)
    logging.info(f"File DICOM salvato con successo in: {output_dicom_path}")


def load_from_dicom(dicom_path: Path) -> Path:
    logging.info(f"--- CARICAMENTO DATI DICOM da: {dicom_path} ---")
    ds = pydicom.dcmread(dicom_path, force=True)

    temp_unenriched_dir = dicom_path.parent / "speed_temp_dicom_import"
    temp_unenriched_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Cartella temporanea 'un-enriched' creata in: {temp_unenriched_dir}")

    if 'WaveformSequence' in ds:
        waveform = ds.WaveformSequence[0]
        num_samples = int(waveform.NumberOfSamples)
        
        waveform_data_bytes = waveform.WaveformData
        waveform_data = np.frombuffer(waveform_data_bytes, dtype=np.int16)
        
        gaze_x = waveform_data[0::3].astype(float) / 100.0
        gaze_y = waveform_data[1::3].astype(float) / 100.0
        pupil = waveform_data[2::3].astype(float) / 100.0
        
        sampling_freq = float(waveform.SamplingFrequency)
        time_sec = np.arange(num_samples) / sampling_freq
        timestamps_ns = (time_sec * 1e9).astype('int64')

        gaze_df = pd.DataFrame({'timestamp [ns]': timestamps_ns, 'gaze x [px]': gaze_x, 'gaze y [px]': gaze_y})
        pupil_df = pd.DataFrame({'timestamp [ns]': timestamps_ns, 'pupil diameter left [mm]': pupil, 'pupil diameter right [mm]': pupil})
        
        gaze_df.to_csv(temp_unenriched_dir / "gaze.csv", index=False)
        pupil_df.to_csv(temp_unenriched_dir / "3d_eye_states.csv", index=False)
        logging.info("Convertiti gaze.csv e 3d_eye_states.csv")

    if 'AnnotationSequence' in ds:
        events = []
        start_ts = timestamps_ns[0] if 'timestamps_ns' in locals() else 0
        for annotation in ds.AnnotationSequence:
            time_val = annotation.AnnotationTime
            time_str = str(time_val) if time_val else "000000"
            try:
                dt_obj = datetime.datetime.strptime(time_str, '%H%M%S.%f')
            except ValueError:
                dt_obj = datetime.datetime.strptime(time_str, '%H%M%S')
            
            total_seconds = dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second + dt_obj.microsecond / 1e6

            events.append({
                'name': str(annotation.AnnotationText),
                'timestamp [ns]': start_ts + int(total_seconds * 1e9),
                'recording id': 'dicom_import'
            })
        events_df = pd.DataFrame(events)
        events_df.to_csv(temp_unenriched_dir / "events.csv", index=False)
        logging.info("Convertito events.csv")
        
    if 'gaze_df' in locals() and not gaze_df.empty:
        world_ts_df = pd.DataFrame({'timestamp [ns]': gaze_df['timestamp [ns]']})
        world_ts_df.to_csv(temp_unenriched_dir / 'world_timestamps.csv', index=False)

    pd.DataFrame(columns=['fixation id', 'start timestamp [ns]', 'duration [ms]', 'fixation x [px]', 'fixation y [px]']).to_csv(temp_unenriched_dir / 'fixations.csv', index=False)
    pd.DataFrame(columns=['saccade id', 'start timestamp [ns]', 'duration [ms]']).to_csv(temp_unenriched_dir / 'saccades.csv', index=False)
    pd.DataFrame(columns=['blink id', 'start timestamp [ns]', 'duration [ms]']).to_csv(temp_unenriched_dir / 'blinks.csv', index=False)
    logging.info("Creati file comportamentali fittizi.")

    logging.info(f"--- CARICAMENTO DICOM COMPLETATO. Dati pronti in: {temp_unenriched_dir} ---")
    return temp_unenriched_dir