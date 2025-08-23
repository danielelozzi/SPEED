# desktop_app/aoi_editor.py
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class AoiEditor(tk.Toplevel):
    """
    Una finestra per definire un'Area di Interesse (AOI) in tre modalità:
    1. Statica: Un rettangolo fisso.
    2. Dinamica (Auto): Tracciando un oggetto rilevato da YOLO.
    3. Dinamica (Manuale): Impostando keyframe manualmente.
    """
    def __init__(self, parent, video_path):
        super().__init__(parent)
        self.title("Define Area of Interest (AOI)")
        self.geometry("1000x800")
        self.transient(parent)
        self.grab_set()

        if not video_path or not video_path.exists():
            messagebox.showerror("Error", f"Video file not found:\n{video_path}", parent=self)
            self.destroy()
            return
        
        self.cap = cv2.VideoCapture(str(video_path))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Risultati
        self.result = None 
        self.result_type = None

        # Stato del disegno/selezione
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.detected_objects = []
        self.selected_track_id = None

        # --- GUI Setup ---
        self.mode_var = tk.StringVar(value="static")
        
        top_frame = tk.Frame(self, pady=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Radiobutton(top_frame, text="Static AOI", variable=self.mode_var, value="static", command=self.update_ui_for_mode).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(top_frame, text="Dynamic AOI (Object Tracking)", variable=self.mode_var, value="dynamic_auto", command=self.update_ui_for_mode).pack(side=tk.LEFT, padx=10)
        
        self.video_canvas = tk.Canvas(self)
        self.video_canvas.pack(pady=10, expand=True)
        self.video_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.status_label = tk.Label(self, text="Draw a rectangle for the static AOI.")
        self.status_label.pack(pady=5)

        self.save_button = tk.Button(self, text="Save AOI", command=self.save_and_close, font=('Helvetica', 10, 'bold'))
        self.save_button.pack(pady=10)

        self.load_first_frame()
        self.update_ui_for_mode()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.cap.release()
        self.destroy()

    def load_first_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.original_frame = frame
            self.display_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.display_image.thumbnail((960, 720))
            
            self.img_display_width, self.img_display_height = self.display_image.size
            self.scale_x = self.video_width / self.img_display_width
            self.scale_y = self.video_height / self.img_display_height

            self.photo = ImageTk.PhotoImage(image=self.display_image)
            self.video_canvas.config(width=self.img_display_width, height=self.img_display_height)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        else:
            messagebox.showerror("Error", "Could not read the first frame of the video.", parent=self)
            self.on_close()

    def update_ui_for_mode(self):
        mode = self.mode_var.get()
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.rect = None
        self.selected_track_id = None

        if mode == "static":
            self.status_label.config(text="Draw a rectangle on the image to define the static AOI.")
            self.save_button.config(state=tk.NORMAL)
        elif mode == "dynamic_auto":
            self.status_label.config(text="Detecting objects... Please wait.")
            self.save_button.config(state=tk.DISABLED)
            self.after(100, self.detect_objects)

    def detect_objects(self):
        if YOLO is None:
            messagebox.showerror("Error", "YOLO (ultralytics) is not installed. Object tracking is not available.", parent=self)
            self.mode_var.set("static")
            self.update_ui_for_mode()
            return

        try:
            model = YOLO('yolov8n.pt')
            results = model.track(self.original_frame, persist=True, verbose=False)
            
            self.detected_objects = []
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                class_ids = results[0].boxes.cls.int().cpu().numpy()
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    class_name = model.names[class_id]
                    self.detected_objects.append({
                        "box": box, "track_id": track_id, "name": f"{class_name}_{track_id}"
                    })
                    
                    # Disegna sulla canvas
                    display_box = (box[0] / self.scale_x, box[1] / self.scale_y, 
                                   box[2] / self.scale_x, box[3] / self.scale_y)
                    self.video_canvas.create_rectangle(display_box, outline='cyan', width=2, tags="object_box")
                    self.video_canvas.create_text(display_box[0], display_box[1] - 10, text=f"{class_name}_{track_id}", fill='cyan', anchor='sw')

            if not self.detected_objects:
                self.status_label.config(text="No objects detected. Try Static or Manual AOI.")
            else:
                self.status_label.config(text="Click on an object to select it for tracking.")

        except Exception as e:
            logging.error(f"YOLO detection failed: {e}")
            messagebox.showerror("YOLO Error", f"Object detection failed: {e}", parent=self)
            self.status_label.config(text="Object detection failed.")

    def on_button_press(self, event):
        mode = self.mode_var.get()
        if mode == "static":
            self.start_x = event.x
            self.start_y = event.y
            if self.rect:
                self.video_canvas.delete(self.rect)
            self.rect = self.video_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)
        elif mode == "dynamic_auto":
            self.select_object_at(event.x, event.y)

    def on_mouse_drag(self, event):
        if self.mode_var.get() == "static":
            cur_x, cur_y = (event.x, event.y)
            self.video_canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        pass # Azione gestita al click o alla selezione

    def select_object_at(self, x, y):
        # Converte le coordinate del click in coordinate del video originale
        click_x_orig = x * self.scale_x
        click_y_orig = y * self.scale_y

        self.selected_track_id = None
        
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj["box"]
            if x1 <= click_x_orig <= x2 and y1 <= click_y_orig <= y2:
                self.selected_track_id = obj["track_id"]
                break
        
        # Evidenzia l'oggetto selezionato
        self.video_canvas.delete("object_box")
        for obj in self.detected_objects:
            display_box = (obj["box"][0] / self.scale_x, obj["box"][1] / self.scale_y,
                           obj["box"][2] / self.scale_x, obj["box"][3] / self.scale_y)
            
            color = 'magenta' if obj["track_id"] == self.selected_track_id else 'cyan'
            self.video_canvas.create_rectangle(display_box, outline=color, width=3 if color=='magenta' else 2, tags="object_box")
            self.video_canvas.create_text(display_box[0], display_box[1] - 10, text=obj["name"], fill=color, anchor='sw')

        if self.selected_track_id:
            self.status_label.config(text=f"Selected object with Track ID: {self.selected_track_id}. Click Save.")
            self.save_button.config(state=tk.NORMAL)
        else:
            self.status_label.config(text="Click on an object to select it for tracking.")
            self.save_button.config(state=tk.DISABLED)


    def save_and_close(self):
        mode = self.mode_var.get()
        if mode == "static":
            if not self.rect:
                messagebox.showwarning("Warning", "Please draw a rectangle on the video first.", parent=self)
                return
            
            coords = self.video_canvas.coords(self.rect)
            original_coords = {
                "x1": int(min(coords[0], coords[2]) * self.scale_x),
                "y1": int(min(coords[1], coords[3]) * self.scale_y),
                "x2": int(max(coords[0], coords[2]) * self.scale_x),
                "y2": int(max(coords[1], coords[3]) * self.scale_y)
            }
            self.result = original_coords
            self.result_type = 'static'

        elif mode == "dynamic_auto":
            if self.selected_track_id is None:
                messagebox.showwarning("Warning", "Please select an object to track first.", parent=self)
                return
            self.result = self.selected_track_id
            self.result_type = 'dynamic_auto'
        
        self.on_close()
