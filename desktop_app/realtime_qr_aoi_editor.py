# desktop_app/realtime_qr_aoi_editor.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np

class RealtimeQRAoiEditor(tk.Toplevel):
    """
    Una finestra di dialogo per definire una AOI basata su QR code in tempo reale.
    Mostra un frame statico, rileva i QR code e permette all'utente di mapparli.
    """
    def __init__(self, parent, analyzer, current_frame: np.ndarray):
        super().__init__(parent)
        self.title("Define Real-time QR-based AOI")
        self.geometry("900x700")
        self.transient(parent)
        self.grab_set()

        self.analyzer = analyzer
        self.frame = current_frame

        # Risultati
        self.aoi_name = None
        self.qr_data_list = None
        self.result = None # Dizionario con i mapping

        self.corner_mappings = {
            'tl': tk.StringVar(value="Not Set"), 'tr': tk.StringVar(value="Not Set"),
            'br': tk.StringVar(value="Not Set"), 'bl': tk.StringVar(value="Not Set")
        }

        # --- GUI ---
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_frame = tk.Frame(main_pane)
        main_pane.add(video_frame, stretch="always")
        self.video_canvas = tk.Canvas(video_frame)
        self.video_canvas.pack(pady=5, expand=True)

        controls_panel = tk.Frame(main_pane, width=300)
        controls_panel.pack_propagate(False)
        main_pane.add(controls_panel)

        qr_frame = tk.LabelFrame(controls_panel, text="QR Code to Corner Mapping", padx=10, pady=10)
        qr_frame.pack(fill=tk.X, pady=10)

        tk.Label(qr_frame, text="Detected QR Data on this frame:").pack(anchor='w')
        self.detected_qr_listbox = tk.Listbox(qr_frame, height=8)
        self.detected_qr_listbox.pack(fill=tk.X, expand=True, pady=5)

        mapping_grid = tk.Frame(qr_frame)
        mapping_grid.pack(fill=tk.X, pady=10)
        positions = {'Top-Left': 'tl', 'Top-Right': 'tr', 'Bottom-Right': 'br', 'Bottom-Left': 'bl'}
        for i, (label, key) in enumerate(positions.items()):
            row, col = divmod(i, 2)
            f = tk.Frame(mapping_grid)
            f.grid(row=row, column=col, padx=5, pady=5, sticky='ew')
            tk.Button(f, text=f"Set {label}", command=lambda k=key: self.set_corner_from_selection(k)).pack(side=tk.LEFT)
            tk.Label(f, textvariable=self.corner_mappings[key], width=8).pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self, text="Pause the main stream on a frame where all QR codes are visible.")
        self.status_label.pack(pady=5)

        self.save_button = tk.Button(self, text="Save AOI Definition", command=self.save_and_close, font=('Helvetica', 10, 'bold'))
        self.save_button.pack(pady=10)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(50, self.detect_and_draw_qrs)

    def detect_and_draw_qrs(self):
        """Rileva i QR code sul frame fornito e aggiorna la UI."""
        frame_copy = self.frame.copy()
        ok, decoded_info, points, _ = self.analyzer.qr_detector.detectAndDecodeMulti(frame_copy)

        self.detected_qr_listbox.delete(0, tk.END)
        if ok and points is not None:
            for i, info in enumerate(decoded_info):
                if info:
                    self.detected_qr_listbox.insert(tk.END, info)
                    qr_points = points[i].astype(int)
                    cv2.polylines(frame_copy, [qr_points], True, (0, 255, 0), 2)
                    center_x = int(np.mean(qr_points[:, 0]))
                    cv2.putText(frame_copy, info, (center_x, qr_points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        img = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
        img.thumbnail((800, 600))
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_canvas.config(width=img.width, height=img.height)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def set_corner_from_selection(self, corner_key):
        selection = self.detected_qr_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a QR code from the list first.", parent=self)
            return
        qr_data = self.detected_qr_listbox.get(selection[0])
        for key, var in self.corner_mappings.items():
            if var.get() == qr_data and key != corner_key:
                messagebox.showerror("Duplicate Assignment", f"QR data '{qr_data}' is already assigned to {key.upper()}.", parent=self)
                return
        self.corner_mappings[corner_key].set(qr_data)

    def save_and_close(self):
        aoi_name = simpledialog.askstring("AOI Name", "Enter a unique name for this AOI:", parent=self)
        if not aoi_name: return

        mappings = {}
        all_corners_set = True
        for key, var in self.corner_mappings.items():
            val = var.get()
            if val == "Not Set":
                all_corners_set = False
                break
            mappings[key] = val

        if not all_corners_set:
            messagebox.showerror("Incomplete Mapping", "All four corners must be assigned to a QR code.", parent=self)
            return

        self.aoi_name = aoi_name
        # Salva la lista nell'ordine corretto per il disegno del poligono
        self.qr_data_list = [
            self.corner_mappings['tl'].get(),
            self.corner_mappings['tr'].get(),
            self.corner_mappings['br'].get(),
            self.corner_mappings['bl'].get()
        ]
        self.result = mappings
        self.destroy()