# realtime_cli.py
import time
import argparse
from src.speed_analyzer.analysis_modules.realtime_analyzer import RealtimeNeonAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Get real-time gazed object classification from a Pupil Labs Neon device.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO model file.")
    args = parser.parse_args()

    analyzer = RealtimeNeonAnalyzer(model_path=args.model)

    if not analyzer.connect():
        return

    print("\nStarting real-time gaze classification. Press Ctrl+C to stop.")
    print("-" * 50)
    
    try:
        while True:
            # Aggiorna i dati in background
            analyzer.get_latest_frames_and_gaze()
            
            # Ottieni l'oggetto guardato
            gazed_object = analyzer.get_gazed_object()
            
            # Stampa l'output in modo pulito sulla stessa riga
            print(f"\rGazing at: {gazed_object.ljust(30)}", end="")
            
            time.sleep(0.1) # Aggiorna 10 volte al secondo
    
    except KeyboardInterrupt:
        print("\nStopping analysis.")
    finally:
        analyzer.close()

if __name__ == "__main__":
    main()