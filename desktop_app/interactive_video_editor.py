# interactive_video_editor.py
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import pandas as pd
import cv2
from PIL import Image, ImageTk

class InteractiveVideoEditor(tk.Toplevel):
    """
    Una finestra di editor video interattiva per aggiungere, rimuovere e modificare
    eventi direttamente su una timeline visuale.
    """
    def __init__(self, parent, video_path, events_df: pd.DataFrame, world_timestamps_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Interactive Video Event Editor")
        self.geometry("1000x800")
        self.transient(parent)
        self.grab_set()

        if not video_path or not video_path.exists():
            messagebox.showerror("Error", f"Video file not found:\n{video_path}", parent=self)
            self.destroy()
            return
            
        self.events_df = events_df.copy().reset_index(drop=True)
        if 'timestamp [ns]' not in self.events_df.columns and not self.events_df.empty:
            messagebox.showerror("Error", "Events DataFrame must have a 'timestamp [ns]' column.", parent=self)
            self.destroy()
            return
            
        self.world_ts = world_timestamps_df
        self.saved_df = None
        self.cap = cv2.VideoCapture(str(video_path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        self.is_playing = False
        self.current_frame_idx = 0
        self.root = parent
        
        # --- CORREZIONE: Aggiunto flag per prevenire loop ---
        self.is_updating_slider = False

        # ... (Il resto del __init__ è invariato)
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.video_label = tk.Label(main_frame)
        self.video_label.pack(pady=10)

        self.timeline_canvas = tk.Canvas(main_frame, height=80, bg='lightgrey')
        self.timeline_canvas.pack(fill=tk.X, padx=10, side=tk.BOTTOM)
        self.timeline_canvas.bind("<Button-1>", self.handle_timeline_click)
        self.timeline_canvas.bind("<B1-Motion>", self.handle_timeline_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.handle_timeline_release)
        self.dragged_event_index = None

        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, padx=10, side=tk.BOTTOM)
        self.play_pause_btn = tk.Button(controls_frame, text="▶ Play", command=self.toggle_play, width=10)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.frame_scale = ttk.Scale(controls_frame, from_=0, to=self.total_frames - 1, orient=tk.HORIZONTAL, command=self.seek_frame)
        self.frame_scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=5)
        self.time_label = tk.Label(controls_frame, text="Frame: 0 / 0", width=20)
        self.time_label.pack(side=tk.RIGHT, padx=5)

        action_frame = tk.Frame(main_frame)
        action_frame.pack(side=tk.BOTTOM, pady=10)
        tk.Button(action_frame, text="Add Event at Current Frame", command=self.add_event_at_frame, bg='#c8e6c9').pack(side=tk.LEFT, padx=10)
        tk.Button(action_frame, text="Remove Selected Event", command=self.remove_selected_event, bg='#ffcdd2').pack(side=tk.LEFT, padx=10)
        tk.Button(action_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        
        self.selected_event_index = None
        self.update_frame(self.current_frame_idx)
        self.after(100, lambda: self.draw_timeline())
        self.protocol("WM_DELETE_WINDOW", self.on_close)


    def on_close(self):
        self.is_playing = False
        self.cap.release()
        self.destroy()

    def seek_frame(self, frame_idx_str):
        # --- CORREZIONE: Controlla il flag ---
        if self.is_updating_slider or self.is_playing:
            return
        self.update_frame(int(float(frame_idx_str)))

    def update_frame(self, frame_idx):
        self.current_frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        
        # --- CORREZIONE: Usa il flag ---
        self.is_updating_slider = True
        self.frame_scale.set(self.current_frame_idx)
        self.is_updating_slider = False
        
        self.time_label.config(text=f"Frame: {self.current_frame_idx} / {self.total_frames}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((900, 650))
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
            self.draw_timeline()

    # ... (Il resto delle funzioni rimane invariato)
    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="❚❚ Pause" if self.is_playing else "▶ Play")
        if self.is_playing: self.play_video()

    def play_video(self):
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            self.update_frame(self.current_frame_idx + 1)
            self.after(int(1000 / self.fps), self.play_video)
        else:
            self.is_playing = False
            self.play_pause_btn.config(text="▶ Play")

    def get_frame_from_ts(self, ts):
        if self.world_ts.empty: return 0
        return (self.world_ts['timestamp [ns]'] - ts).abs().idxmin()

    def draw_timeline(self):
        self.timeline_canvas.delete("all")
        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1: return
        
        color_map = {'default': 'red', 'optional': 'purple', 'manual': 'green'}
        
        for index, event in self.events_df.iterrows():
            frame_idx = self.get_frame_from_ts(event['timestamp [ns]'])
            x_pos = (frame_idx / self.total_frames) * canvas_width
            base_color = color_map.get(event.get('source', 'manual'), 'black')
            final_color = "blue" if index == self.selected_event_index else base_color
            
            self.timeline_canvas.create_line(x_pos, 10, x_pos, 50, fill=final_color, width=3 if final_color=="blue" else 2, tags=f"event_{index}")
            self.timeline_canvas.create_text(x_pos, 60, text=event['name'], anchor=tk.N, fill=final_color, tags=f"event_{index}")

        cursor_x = (self.current_frame_idx / self.total_frames) * canvas_width
        self.timeline_canvas.create_line(cursor_x, 0, cursor_x, 80, fill='dark green', width=2)
    
    def get_event_at_pos(self, x):
        canvas_width = self.timeline_canvas.winfo_width()
        for index, event in self.events_df.iterrows():
            if abs(x - ((self.get_frame_from_ts(event['timestamp [ns]']) / self.total_frames) * canvas_width)) < 5:
                return index
        return None

    def handle_timeline_click(self, event):
        clicked_event = self.get_event_at_pos(event.x)
        self.selected_event_index = clicked_event
        self.dragged_event_index = clicked_event
        if clicked_event is None:
            self.update_frame((event.x / self.timeline_canvas.winfo_width()) * self.total_frames)
        self.draw_timeline()

    def handle_timeline_drag(self, event):
        if self.dragged_event_index is not None:
            canvas_width = self.timeline_canvas.winfo_width()
            new_frame = max(0, min(int((event.x / canvas_width) * self.total_frames), self.total_frames - 1))
            if new_frame < len(self.world_ts):
                self.events_df.loc[self.dragged_event_index, 'timestamp [ns]'] = self.world_ts.iloc[new_frame]['timestamp [ns]']
                self.update_frame(new_frame)
            
    def handle_timeline_release(self, event):
        if self.dragged_event_index is not None:
            self.dragged_event_index = None
            self.events_df.sort_values('timestamp [ns]', inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.selected_event_index = None
            self.draw_timeline()

    def add_event_at_frame(self):
        name = simpledialog.askstring("Add Event", "Enter event name:", parent=self)
        if name and self.current_frame_idx < len(self.world_ts):
            ts = self.world_ts.iloc[self.current_frame_idx]['timestamp [ns]']
            new_row = {'name': name, 'timestamp [ns]': ts, 'selected': True, 'source': 'manual', 'recording id': 'rec_001'}
            self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
            self.events_df.sort_values('timestamp [ns]', inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.draw_timeline()

    def remove_selected_event(self):
        if self.selected_event_index is None:
            messagebox.showinfo("Info", "Click on an event on the timeline to select it first.", parent=self)
            return
        event_name = self.events_df.loc[self.selected_event_index, 'name']
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to remove the event '{event_name}'?", parent=self):
            self.events_df.drop(self.selected_event_index, inplace=True)
            self.events_df = self.events_df.reset_index(drop=True)
            self.selected_event_index = None
            self.draw_timeline()

    def save_and_close(self):
        self.is_playing = False
        cols_to_save = ['name', 'timestamp [ns]', 'recording id', 'selected', 'source']
        self.saved_df = self.events_df[[col for col in cols_to_save if col in self.events_df.columns]]
        self.on_close()