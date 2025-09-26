 # desktop_app/nsi_calculator.py
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
import logging

class NsiCalculatorWindow(tk.Toplevel):
    """
    Una finestra interattiva per definire finestre temporali e calcolare
    il Normalized Switching Index (NSI) tra le AOI definite.
    """
    def __init__(self, parent, output_dir: Path, defined_aois: list):
        super().__init__(parent)
        self.title("Normalized Switching Index (NSI) Calculator")
        self.geometry("1300x800")
        self.transient(parent)
        self.grab_set()

        self.output_dir = output_dir
        self.defined_aois = defined_aois
        self.aoi_names = [aoi['name'] for aoi in self.defined_aois]

        # Caricamento dati
        self.video_path = next(self.output_dir.glob('SPEED_workspace/external.mp4'), None)
        self.gaze_enriched_path = self.output_dir / 'enriched_from_AOIs' / 'gaze_enriched.csv'
        
        if not self.video_path or not self.video_path.exists():
            messagebox.showerror("Errore", "Video 'external.mp4' non trovato nella cartella di analisi.", parent=self)
            self.destroy()
            return
        if not self.gaze_enriched_path.exists():
            messagebox.showerror("Errore", "File 'gaze_enriched.csv' non trovato. L'analisi potrebbe non averlo generato.", parent=self)
            self.destroy()
            return

        self.gaze_df = pd.read_csv(self.gaze_enriched_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.original_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_h = int(self.cap.get(cv2.CAP_PROP_HEIGHT))

        # Stato UI
        self.current_frame_idx = 0
        self.is_updating_slider = False
        self.time_windows = [] # Lista di tuple (start_frame, end_frame)
        self.drawing_window = False
        self.temp_rect_id = None

        # --- Layout ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_canvas = tk.Canvas(main_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        self.timeline_canvas = tk.Canvas(main_frame, height=60, bg='lightgrey')
        self.timeline_canvas.pack(fill=tk.X, pady=5)
        self.timeline_canvas.bind("<ButtonPress-1>", self.on_timeline_press)
        self.timeline_canvas.bind("<B1-Motion>", self.on_timeline_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.on_timeline_release)

        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)
        self.frame_scale = ttk.Scale(controls_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(controls_frame, text="Frame: 0 / 0", width=20)
        self.time_label.pack(side=tk.RIGHT)

        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(bottom_frame, text="Trascina sulla timeline per definire una finestra temporale. Clicca su una finestra per rimuoverla.").pack(side=tk.LEFT)
        tk.Button(bottom_frame, text="Calculate NSI for Windows", command=self.calculate_nsi, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(side=tk.RIGHT)

        self.after(50, lambda: self.update_frame(0))
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.cap.release()
        self.destroy()

    def seek_frame(self, frame_idx_str):
        if self.is_updating_slider: return
        self.update_frame(int(float(frame_idx_str)))

    def update_frame(self, frame_idx):
        self.current_frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        
        self.is_updating_slider = True
        self.frame_scale.set(self.current_frame_idx)
        self.is_updating_slider = False
        self.time_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            # Qui potremmo disegnare le AOI, ma per ora lo lasciamo semplice
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            canvas_w, canvas_h = self.video_canvas.winfo_width(), self.video_canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
            
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.draw_timeline()

    def draw_timeline(self):
        self.timeline_canvas.delete("all")
        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1: return

        # Disegna le finestre temporali
        for start_frame, end_frame in self.time_windows:
            start_x = (start_frame / self.total_frames) * canvas_width
            end_x = (end_frame / self.total_frames) * canvas_width
            self.timeline_canvas.create_rectangle(start_x, 10, end_x, 50, fill="lightblue", outline="blue", tags="time_window")

        # Disegna il cursore
        cursor_x = (self.current_frame_idx / self.total_frames) * canvas_width
        self.timeline_canvas.create_line(cursor_x, 0, cursor_x, 60, fill='dark green', width=2)

    def on_timeline_press(self, event):
        # Controlla se si clicca su una finestra esistente per rimuoverla
        clicked_items = self.timeline_canvas.find_overlapping(event.x, event.y, event.x, event.y)
        if any("time_window" in self.timeline_canvas.gettags(item) for item in clicked_items):
            canvas_width = self.timeline_canvas.winfo_width()
            clicked_frame = (event.x / canvas_width) * self.total_frames
            
            window_to_remove = None
            for start, end in self.time_windows:
                if start <= clicked_frame <= end:
                    window_to_remove = (start, end)
                    break
            if window_to_remove:
                self.time_windows.remove(window_to_remove)
                self.draw_timeline()
            return

        # Altrimenti, inizia a disegnare una nuova finestra
        self.drawing_window = True
        self.start_x = event.x
        self.temp_rect_id = self.timeline_canvas.create_rectangle(self.start_x, 10, self.start_x, 50, outline="green", width=2)

    def on_timeline_drag(self, event):
        if not self.drawing_window: return
        self.timeline_canvas.coords(self.temp_rect_id, self.start_x, 10, event.x, 50)

    def on_timeline_release(self, event):
        if not self.drawing_window: return
        self.drawing_window = False
        self.timeline_canvas.delete(self.temp_rect_id)

        canvas_width = self.timeline_canvas.winfo_width()
        start_frame = int((min(self.start_x, event.x) / canvas_width) * self.total_frames)
        end_frame = int((max(self.start_x, event.x) / canvas_width) * self.total_frames)

        if end_frame > start_frame:
            self.time_windows.append((start_frame, end_frame))
            self.draw_timeline()

    def calculate_nsi(self):
        if not self.time_windows:
            messagebox.showwarning("Attenzione", "Nessuna finestra temporale definita. Trascina sulla timeline per crearne una.", parent=self)
            return

        all_results = []
        logging.info(f"Avvio calcolo NSI per {len(self.time_windows)} finestre temporali...")

        # Pre-filtra il dataframe di sguardi per avere solo quelli con un'AOI mappata
        gaze_on_aois = self.gaze_df.dropna(subset=['aoi_name']).copy()

        for i, (start_frame, end_frame) in enumerate(self.time_windows):
            # Trova i timestamp di inizio e fine della finestra
            start_ts = self.gaze_df.iloc[start_frame]['timestamp [ns]']
            end_ts = self.gaze_df.iloc[end_frame]['timestamp [ns]']

            # Filtra i dati di sguardo per la finestra temporale corrente
            window_gaze_df = gaze_on_aois[
                (gaze_on_aois['timestamp [ns]'] >= start_ts) &
                (gaze_on_aois['timestamp [ns]'] <= end_ts)
            ].copy()

            if window_gaze_df.empty:
                logging.warning(f"Finestra {i+1}: Nessun punto di sguardo trovato su AOI. NSI = 0.")
                nsi = 0
                k = 0
                l_in = 0
            else:
                # Calcola la sequenza di transizioni (V)
                v_sequence = []
                last_aoi = None
                for aoi in window_gaze_df['aoi_name']:
                    if aoi != last_aoi:
                        v_sequence.append(aoi)
                        last_aoi = aoi
                
                k = len(v_sequence)
                l_in = len(window_gaze_df)

                # Calcola NSI
                if l_in <= 1:
                    nsi = 0.0
                else:
                    nsi = max(0, k - 1) / (l_in - 1)

            all_results.append({
                'aoi_list': self.aoi_names,
                'nsi_value': nsi,
                'window_start_timestamp_ns': start_ts,
                'window_end_timestamp_ns': end_ts
            })
            logging.info(f"Finestra {i+1}: NSI = {nsi:.4f} (K={k}, L_in={l_in})")

        # Salva i risultati
        results_df = pd.DataFrame(all_results)
        save_path = self.output_dir / 'nsi_results.csv'
        results_df.to_csv(save_path, index=False)
        
        messagebox.showinfo("Successo", f"Calcolo NSI completato. Risultati salvati in:\n{save_path}", parent=self)
        self.on_close()