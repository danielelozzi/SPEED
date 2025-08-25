# realtime_cli.py
import time
import argparse
import threading
from pathlib import Path
from src.speed_analyzer.analysis_modules.realtime_analyzer import RealtimeNeonAnalyzer
from generate_synthetic_stream import MockNeonDevice

def main():
    parser = argparse.ArgumentParser(description="Get real-time gaze object classification from a Pupil Labs Neon device.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO model file.")
    parser.add_argument("--record", action="store_true", help="Enable recording of video and data streams.")
    parser.add_argument("--output", type=str, default="./realtime_recording_cli", help="Folder to save recording files.")
    parser.add_argument("--use-mock", action="store_true", help="Use a simulated device for testing.")
    
    # --- NOVITÀ: Argomento per aggiungere AOI ---
    parser.add_argument("--aoi", action="append", help="Define a static AOI. Format: 'Name,x1,y1,x2,y2'. Can be used multiple times.")

    args = parser.parse_args()
    analyzer = RealtimeNeonAnalyzer(model_path=args.model)

    # --- NOVITÀ: Aggiungi le AOI definite da riga di comando ---
    if args.aoi:
        for aoi_string in args.aoi:
            try:
                name, x1, y1, x2, y2 = aoi_string.split(',')
                analyzer.add_static_aoi(name, [int(x1), int(y1), int(x2), int(y2)])
            except ValueError:
                print(f"Invalid AOI format: '{aoi_string}'. Please use 'Name,x1,y1,x2,y2'.")

    mock_device = MockNeonDevice() if args.use_mock else None
    if not analyzer.connect(mock_device=mock_device):
        return

    if args.record:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        analyzer.start_recording(str(output_path))
        print("-" * 50); print("RECORDING IS ACTIVE."); print("Press ENTER to add a generic event.")
        input_thread = threading.Thread(target=listen_for_events, args=(analyzer,), daemon=True)
        input_thread.start()

    print("\nStarting real-time analysis. Press Ctrl+C to stop.")
    print("-" * 50)
    
    try:
        while True:
            # Da codice, tutti gli overlay sono attivi di default
            frame = analyzer.process_and_visualize() 
            # Per mostrare il video, si potrebbe usare cv2.imshow qui
            # cv2.imshow("Real-time Stream", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Manteniamo l'output testuale per la CLI
            rec_indicator = "REC ● | " if analyzer.is_recording else ""
            print(f"\r{rec_indicator}Gazing at: {analyzer.last_gazed_object.ljust(30)}", end="")
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\nStopping analysis.")
    finally:
        # cv2.destroyAllWindows()
        analyzer.close()

def listen_for_events(analyzer: RealtimeNeonAnalyzer):
    event_counter = 1
    while analyzer.is_recording:
        input()
        if analyzer.is_recording:
            event_name = f"CLI_Event_{event_counter}"
            analyzer.add_event(event_name)
            event_counter += 1

if __name__ == "__main__":
    main()