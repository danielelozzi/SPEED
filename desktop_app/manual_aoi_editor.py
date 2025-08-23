# desktop_app/manual_aoi_editor.py
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

class ManualAoiEditor(tk.Toplevel):
    """
    Una finestra di editor avanzata per definire un'AOI dinamica
    impostando manualmente dei keyframe sulla timeline del video.
    """
    def __init__(self, parent, video_path):
        super().__init__(parent)
        self.title("Manual Dynamic AOI Editor (Keyframes)")
        self.geometry("1000x800")
        self.transient(parent)
        self.grab_set()

        if not video_path or not video_path.exists():
            messagebox.showerror("Error", f"Video file not found:\n{video_path}", parent=self)
            self.destroy()
            return

        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Store keyframes: {frame_index: (x1, y1, x2, y2)}
        self.keyframes = {}
        self.saved_keyframes = None

        # Drawing state
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.current_frame_idx = 0
        self.is_updating = False

        # --- GUI Setup ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_canvas = tk.Canvas(main_frame)
        self.video_canvas.pack(pady=10)
        self.video_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.video_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.timeline_canvas = tk.Canvas(main_frame, height=60, bg='lightgrey')
        self.timeline_canvas.pack(fill=tk.X, padx=10, side=tk.BOTTOM)
        self.timeline_canvas.bind("<Button-1>", self.handle_timeline_click)

        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, padx=10, side=tk.BOTTOM)
        
        self.frame_scale = ttk.Scale(controls_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(controls_frame, text="Frame: 0 / 0", width=20)
        self.time_label.pack(side=tk.RIGHT, padx=5)

        action_frame = tk.Frame(main_frame)
        action_frame.pack(side=tk.BOTTOM, pady=10)
        tk.Button(action_frame, text="Set/Update Keyframe", command=self.set_keyframe, bg='#c8e6c9').pack(side=tk.LEFT, padx=10)
        tk.Button(action_frame, text="Remove Keyframe", command=self.remove_keyframe, bg='#ffcdd2').pack(side=tk.LEFT, padx=10)
        tk.Button(action_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        
        self.update_frame(0)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        self.cap.release()
        self.destroy()

    def seek_frame(self, frame_idx_str):
        if self.is_updating: return
        self.update_frame(int(float(frame_idx_str)))

    def update_frame(self, frame_idx):
        if self.is_updating: return
        self.is_updating = True

        self.current_frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        self.frame_scale.set(self.current_frame_idx)
        self.time_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.display_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.display_image.thumbnail((900, 650))
            
            self.img_display_width, self.img_display_height = self.display_image.size
            self.scale_x = self.video_width / self.img_display_width
            self.scale_y = self.video_height / self.img_display_height

            self.photo = ImageTk.PhotoImage(image=self.display_image)
            self.video_canvas.config(width=self.img_display_width, height=self.img_display_height)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            self.draw_current_rect()
            self.draw_timeline()

        self.is_updating = False

    def draw_current_rect(self):
        # Clear previous rectangles
        self.video_canvas.delete("rect")
        
        # Check if current frame is a keyframe
        if self.current_frame_idx in self.keyframes:
            coords = self.keyframes[self.current_frame_idx]
            # Scale from original video coords to display coords
            display_coords = (coords[0] / self.scale_x, coords[1] / self.scale_y, 
                              coords[2] / self.scale_x, coords[3] / self.scale_y)
            self.video_canvas.create_rectangle(display_coords, outline='green', width=2, tags="rect")
        else: # Interpolate
            prev_frames = [f for f in self.keyframes if f < self.current_frame_idx]
            next_frames = [f for f in self.keyframes if f > self.current_frame_idx]
            
            if prev_frames and next_frames:
                prev_f = max(prev_frames)
                next_f = min(next_frames)
                
                # Interpolation factor
                factor = (self.current_frame_idx - prev_f) / (next_f - prev_f)
                
                prev_coords = np.array(self.keyframes[prev_f])
                next_coords = np.array(self.keyframes[next_f])
                
                interp_coords = prev_coords + (next_coords - prev_coords) * factor
                
                display_coords = (interp_coords[0] / self.scale_x, interp_coords[1] / self.scale_y,
                                  interp_coords[2] / self.scale_x, interp_coords[3] / self.scale_y)
                self.video_canvas.create_rectangle(display_coords, outline='yellow', dash=(4, 2), width=2, tags="rect")

    def draw_timeline(self):
        self.timeline_canvas.delete("all")
        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1: return

        # Draw keyframes
        for frame_idx in self.keyframes:
            x_pos = (frame_idx / self.total_frames) * canvas_width
            self.timeline_canvas.create_line(x_pos, 10, x_pos, 40, fill='green', width=2)
            self.timeline_canvas.create_text(x_pos, 45, text=str(frame_idx), anchor=tk.N)

        # Draw current position cursor
        cursor_x = (self.current_frame_idx / self.total_frames) * canvas_width
        self.timeline_canvas.create_line(cursor_x, 0, cursor_x, 60, fill='red', width=2)

    def handle_timeline_click(self, event):
        new_frame = int((event.x / self.timeline_canvas.winfo_width()) * self.total_frames)
        self.update_frame(new_frame)

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.video_canvas.delete(self.rect)
        self.rect = self.video_canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.video_canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        pass # The rectangle is temporary until "Set Keyframe" is clicked

    def set_keyframe(self):
        if not self.rect:
            messagebox.showwarning("Warning", "Please draw a rectangle on the video first.", parent=self)
            return
        
        coords = self.video_canvas.coords(self.rect)
        # Scale back to original video dimensions
        original_coords = (
            int(coords[0] * self.scale_x), int(coords[1] * self.scale_y),
            int(coords[2] * self.scale_x), int(coords[3] * self.scale_y)
        )
        
        # Ensure x1 < x2 and y1 < y2
        x1 = min(original_coords[0], original_coords[2])
        y1 = min(original_coords[1], original_coords[3])
        x2 = max(original_coords[0], original_coords[2])
        y2 = max(original_coords[1], original_coords[3])

        self.keyframes[self.current_frame_idx] = (x1, y1, x2, y2)
        self.video_canvas.delete(self.rect)
        self.rect = None
        self.draw_current_rect()
        self.draw_timeline()

    def remove_keyframe(self):
        if self.current_frame_idx in self.keyframes:
            if messagebox.askyesno("Confirm", f"Remove keyframe at frame {self.current_frame_idx}?", parent=self):
                del self.keyframes[self.current_frame_idx]
                self.draw_current_rect()
                self.draw_timeline()
        else:
            messagebox.showinfo("Info", "The current frame is not a keyframe.", parent=self)

    def save_and_close(self):
        if len(self.keyframes) < 2:
            messagebox.showerror("Error", "You must define at least two keyframes to create a dynamic AOI.", parent=self)
            return
        
        # Sort keyframes by frame number and save
        self.saved_keyframes = dict(sorted(self.keyframes.items()))
        self.on_close()
