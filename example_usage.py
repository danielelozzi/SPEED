# example_usage.py
import pandas as pd
from pathlib import Path
from speed_analyzer import run_full_analysis

# --- 1. IMPOSTA I PERCORSI E I PARAMETRI ---
BASE_PATH = Path("./synthetic_data_output")
if not BASE_PATH.exists():
    print(f"Cartella dati sintetici non trovata. Esegui prima 'python generate_synthetic_data.py'")
    exit()

RAW_DIR = BASE_PATH / "RAW"
UNENRICHED_DIR = BASE_PATH / "un-enriched"
# NOTA: Non specifichiamo ENRICHED_DIR, lasceremo che il software lo crei.
OUTPUT_DIR_BASE = Path("./analysis_results_package_test")
PARTICIPANT_NAME = "synthetic_test_package"

# --- 2. GESTIONE DEGLI EVENTI (VIA PANDAS) ---
events_data = {
    'name': ['Intro_Start', 'Task_1_End', 'Conclusion_Start'],
    'timestamp [ns]': [1672531201000000000, 1672531215000000000, 1672531230000000000],
    'recording id': ['rec_001', 'rec_001', 'rec_001']
}
events_dataframe = pd.DataFrame(events_data)

# --- 3. ESECUZIONE DELLE DIVERSE MODALITÀ DI ANALISI ---

def run_analysis_with_static_aoi():
    """Esempio con un'area di interesse statica definita da coordinate."""
    print("\n--- ESECUZIONE ANALISI CON AOI STATICA ---")
    output_path = OUTPUT_DIR_BASE / "static_aoi_test"
    
    # Definisci le coordinate del rettangolo (in pixel)
    static_aoi_coords = {'x1': 300, 'y1': 200, 'x2': 900, 'y2': 600}
    
    run_full_analysis(
        raw_data_path=str(RAW_DIR), unenriched_data_path=str(UNENRICHED_DIR),
        output_path=str(output_path), subject_name=f"{PARTICIPANT_NAME}_static",
        events_df=events_dataframe,
        run_yolo=True, # YOLO può comunque essere utile per altre analisi
        aoi_coordinates=static_aoi_coords # Passa le coordinate
    )
    print(f"Analisi con AOI statica completata. Risultati in: {output_path.resolve()}")

def run_analysis_with_dynamic_aoi_auto():
    """Esempio con un'AOI dinamica che traccia un oggetto rilevato da YOLO."""
    print("\n--- ESECUZIONE ANALISI CON AOI DINAMICA (YOLO TRACKING) ---")
    output_path = OUTPUT_DIR_BASE / "dynamic_aoi_auto_test"
    
    # Specifica l'ID dell'oggetto da tracciare (es. 1).
    # Questo ID viene dall'output di YOLO. Per dati reali, andrebbe ispezionato.
    track_id_to_follow = 1
    
    run_full_analysis(
        raw_data_path=str(RAW_DIR), unenriched_data_path=str(UNENRICHED_DIR),
        output_path=str(output_path), subject_name=f"{PARTICIPANT_NAME}_dynamic_auto",
        events_df=events_dataframe,
        run_yolo=True, # Obbligatorio per questa modalità
        aoi_track_id=track_id_to_follow # Passa l'ID da tracciare
    )
    print(f"Analisi con AOI dinamica (auto) completata. Risultati in: {output_path.resolve()}")

def run_analysis_with_dynamic_aoi_manual():
    """Esempio con un'AOI dinamica definita da keyframe manuali."""
    print("\n--- ESECUZIONE ANALISI CON AOI DINAMICA (KEYFRAMES MANUALI) ---")
    output_path = OUTPUT_DIR_BASE / "dynamic_aoi_manual_test"
    
    # Definisci i keyframe: dizionario {frame_index: (x1, y1, x2, y2)}
    manual_keyframes = {
        0: (100, 100, 400, 400),      # All'inizio, l'AOI è in alto a sinistra
        1500: (500, 300, 800, 600),   # A metà, si è spostata e ingrandita
        2700: (200, 150, 500, 450)    # Alla fine, è tornata indietro rimpicciolendosi
    }
    
    run_full_analysis(
        raw_data_path=str(RAW_DIR), unenriched_data_path=str(UNENRICHED_DIR),
        output_path=str(output_path), subject_name=f"{PARTICIPANT_NAME}_dynamic_manual",
        events_df=events_dataframe,
        run_yolo=False, # Non necessario se non si vogliono altre analisi YOLO
        aoi_keyframes=manual_keyframes # Passa i keyframe
    )
    print(f"Analisi con AOI dinamica (manuale) completata. Risultati in: {output_path.resolve()}")


if __name__ == "__main__":
    try:
        # Esegui le tre diverse analisi una dopo l'altra
        run_analysis_with_static_aoi()
        run_analysis_with_dynamic_aoi_auto()
        run_analysis_with_dynamic_aoi_manual()
        
    except FileNotFoundError as e:
        print(f"\nERRORE: Un file o una cartella necessari non sono stati trovati. Controlla i percorsi.")
        print(f"Dettagli: {e}")
    except Exception as e:
        print(f"\nSi è verificato un errore inaspettato durante l'analisi.")
        print(f"Dettagli: {e}")
