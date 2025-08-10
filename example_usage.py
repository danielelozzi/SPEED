# example_usage.py
import pandas as pd
from pathlib import Path

# Per far funzionare questo esempio in locale, assicurati di aver
# installato il package con: pip install -e .
from src.speed_analyzer import run_full_analysis

# --- 1. IMPOSTA I PERCORSI E I PARAMETRI ---
BASE_PATH = Path("./synthetic_data_output")
if not BASE_PATH.exists():
    print(f"Cartella dati sintetici non trovata. Esegui prima 'python generate_synthetic_data.py'")
    exit()

RAW_DIR = BASE_PATH / "RAW"
UNENRICHED_DIR = BASE_PATH / "un-enriched"
ENRICHED_DIR = BASE_PATH / "enriched"
OUTPUT_DIR = Path("./analysis_results_package_test")
PARTICIPANT_NAME = "synthetic_test_package"

# --- 2. GESTIONE DEGLI EVENTI (VIA PANDAS) ---
events_data = {
    'name': ['Intro_Start', 'Task_1_End', 'Conclusion_Start'],
    'timestamp [ns]': [1672531201000000000, 1672531215000000000, 1672531230000000000],
    'recording id': ['rec_001', 'rec_001', 'rec_001']
}
events_dataframe = pd.DataFrame(events_data)

# --- 3. CONFIGURA LE OPZIONI DI ANALISI ---
plot_config = {"heatmaps": True, "path_plots": True, "pupillometry": True}
video_config = {"output_filename": f"{PARTICIPANT_NAME}_video.mp4", "overlay_gaze": True}

# --- 4. ESEGUI L'ANALISI COMPLETA ---
print("Avvio dell'analisi con il package speed_analyzer...")
try:
    risultati_path = run_full_analysis(
        raw_data_path=str(RAW_DIR), unenriched_data_path=str(UNENRICHED_DIR),
        enriched_data_path=str(ENRICHED_DIR), output_path=str(OUTPUT_DIR),
        subject_name=PARTICIPANT_NAME, events_df=events_dataframe,
        run_yolo=True, yolo_model_path="yolov8n.pt",
        generate_plots=True, plot_selections=plot_config,
        generate_video=True, video_options=video_config
    )
    print(f"\nAnalisi completata con successo! Risultati in: {risultati_path}")
except Exception as e:
    print(f"\nERRORE durante l'analisi: {e}")