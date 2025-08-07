# GUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import traceback
import json
import pandas as pd
import logging
import time
import cv2
from PIL import Image, ImageTk

try:
    import main_analyzer_advanced as main_analyzer
except ImportError as e:
    messagebox.showerror("Critical Error", f"Missing library: {e}. Make sure you have installed all requirements.")
    exit()

class EventManagerWindow(tk.Toplevel):
    """
    Una finestra per visualizzare, selezionare, modificare, aggiungere, unire e rimuovere eventi
    in una vista tabellare.
    """
    def __init__(self, parent, events_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Event Manager (Table View)")
        self.geometry("800x600")
        self.transient(parent)
        self.grab_set()

        self.events_df = events_df.copy()
        if 'selected' not in self.events_df.columns:
            self.events_df['selected'] = True
        self.events_df.sort_values('timestamp [ns]', inplace=True)
        self.saved_df = None

        frame = tk.Frame(self, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        cols = ("Selected", "Event Name", "Timestamp (s)", "Source")
        self.tree = ttk.Treeview(frame, columns=cols, show='headings', selectmode='extended')
        for col in cols:
            self.tree.heading(col, text=col)
        self.tree.column("Selected", width=80, anchor=tk.CENTER)
        self.tree.column("Event Name", width=350)
        self.tree.column("Timestamp (s)", width=150, anchor=tk.CENTER)
        self.tree.column("Source", width=100, anchor=tk.CENTER)

        self.populate_tree()
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self.on_double_click)
        self.tree.bind("<Button-1>", self.on_click)
        self.tree.bind('<<TreeviewSelect>>', self.on_selection_change)

        button_frame = tk.Frame(self, pady=10)
        button_frame.pack(fill=tk.X)

        tk.Button(button_frame, text="Sort by Time", command=self.sort_events).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Add Event", command=self.add_event).pack(side=tk.LEFT, padx=5)
        self.merge_button = tk.Button(button_frame, text="Merge Selected", command=self.merge_events, state=tk.DISABLED)
        self.merge_button.pack(side=tk.LEFT, padx=5)
        self.remove_button = tk.Button(button_frame, text="Remove Selected", command=self.remove_selected_events, state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for index, row in self.events_df.iterrows():
            selected_text = "Yes" if row.get('selected', True) else "No"
            timestamp_sec = row['timestamp [ns]'] / 1e9
            source = row.get('source', 'manual')
            self.tree.insert("", "end", iid=str(index), values=(selected_text, row['name'], f"{timestamp_sec:.4f}", source))

    def sort_events(self):
        self.events_df.sort_values('timestamp [ns]', inplace=True)
        self.populate_tree()

    def on_selection_change(self, event):
        num_selected = len(self.tree.selection())
        self.remove_button.config(state=tk.NORMAL if num_selected > 0 else tk.DISABLED)
        self.merge_button.config(state=tk.NORMAL if num_selected >= 2 else tk.DISABLED)

    def on_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        column = self.tree.identify_column(event.x)
        if region == "cell" and column == "#1":
            item_id = self.tree.identify_row(event.y)
            if item_id:
                df_index = int(item_id)
                self.events_df.loc[df_index, 'selected'] = not self.events_df.loc[df_index, 'selected']
                self.populate_tree()

    def on_double_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell": return
        column_id = self.tree.identify_column(event.x)
        item_id = self.tree.identify_row(event.y)
        df_index = int(item_id)
        
        if column_id not in ("#2", "#3"): return

        x, y, width, height = self.tree.bbox(item_id, column_id)
        entry = ttk.Entry(self.tree)
        current_values = self.tree.item(item_id, 'values')
        col_index = int(column_id.replace('#', '')) - 1
        entry.insert(0, current_values[col_index])
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def save_edit(evt):
            new_value = entry.get()
            try:
                if column_id == "#2":
                    self.events_df.loc[df_index, 'name'] = new_value
                elif column_id == "#3":
                    new_timestamp_sec = float(new_value)
                    self.events_df.loc[df_index, 'timestamp [ns]'] = int(new_timestamp_sec * 1e9)
                self.populate_tree()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for the timestamp.")
            finally:
                entry.destroy()
        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def add_event(self):
        name = simpledialog.askstring("Add Event", "Enter the new event name:", parent=self)
        if not name: return
        ts_str = simpledialog.askstring("Add Event", f"Enter timestamp in seconds for '{name}':", parent=self)
        if not ts_str: return
        try:
            ts_sec = float(ts_str)
            new_row = {'name': name, 'timestamp [ns]': int(ts_sec * 1e9), 'selected': True, 'source': 'manual', 'recording id': 'rec_001'}
            self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
            self.sort_events()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for the timestamp.")

    def merge_events(self):
        selected_items = self.tree.selection()
        if len(selected_items) < 2: return
        new_name = simpledialog.askstring("Merge Events", "Enter the name for the new merged event:", parent=self)
        if not new_name: return
        indices = [int(item_id) for item_id in selected_items]
        selected_df = self.events_df.loc[indices]
        first_timestamp_ns = selected_df['timestamp [ns]'].min()
        new_row = {'name': new_name, 'timestamp [ns]': first_timestamp_ns, 'selected': True, 'source': 'manual', 'recording id': 'rec_001'}
        self.events_df.drop(indices, inplace=True)
        self.events_df = pd.concat([self.events_df, pd.DataFrame([new_row])], ignore_index=True)
        self.sort_events()

    def remove_selected_events(self):
        selected_items = self.tree.selection()
        if not selected_items: return
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to remove {len(selected_items)} event(s)?"):
            indices_to_drop = [int(item_id) for item_id in selected_items]
            self.events_df.drop(indices_to_drop, inplace=True)
            self.populate_tree()

    def save_and_close(self):
        self.saved_df = self.events_df
        self.destroy()

class InteractiveVideoEditor(tk.Toplevel):
    """
    Editor video interattivo per eventi.
    Gestisce eventi da più fonti e li distingue per colore.
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
        self.time_label = tk.Label(controls_frame, text="00:00.000 / 00:00.000", width=20)
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
        if not self.is_playing:
            self.update_frame(int(float(frame_idx_str)))

    def format_time(self, frame_idx):
        total_seconds = frame_idx / self.fps
        minutes = int(total_seconds / 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:06.3f}"

    def update_frame(self, frame_idx):
        self.current_frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        self.frame_scale.set(self.current_frame_idx)
        self.time_label.config(text=f"{self.format_time(self.current_frame_idx)} / {self.format_time(self.total_frames)}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((900, 650))
            self.photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.photo)
            self.draw_timeline()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_pause_btn.config(text="❚❚ Pause" if self.is_playing else "▶ Play")
        if self.is_playing: self.play_video()

    def play_video(self):
        if self.is_playing and self.current_frame_idx < self.total_frames - 1:
            self.update_frame(self.current_frame_idx + 1)
            self.root.after(int(1000 / self.fps), self.play_video)
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
            
            self.timeline_canvas.create_line(x_pos, 10, x_pos, 50, fill=final_color, width=3 if final_color=="blue" else 2)
            self.timeline_canvas.create_text(x_pos, 60, text=event['name'], anchor=tk.N, fill=final_color)

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

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v3.5")
        self.root.geometry("850x850")

        self.raw_dir_var = tk.StringVar()
        self.unenriched_dir_var = tk.StringVar()
        self.enriched_dir_var = tk.StringVar()
        self.external_event_file_var = tk.StringVar()
        self.plot_vars = {}
        self.video_vars = {}
        self.events_df = pd.DataFrame()
        self.world_timestamps_df = pd.DataFrame()

        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        main_frame = self.scrollable_frame

        # --- Sezione 1: Setup ---
        setup_frame = tk.LabelFrame(main_frame, text="1. Project Setup", padx=10, pady=10)
        setup_frame.pack(fill=tk.X, pady=10, padx=10)
        name_frame = tk.Frame(setup_frame); name_frame.pack(fill=tk.X, pady=2)
        tk.Label(name_frame, text="Participant Name:", width=20, anchor='w').pack(side=tk.LEFT)
        self.participant_name_var = tk.StringVar(); self.participant_name_var.trace_add("write", self.update_output_dir_default)
        tk.Entry(name_frame, textvariable=self.participant_name_var).pack(fill=tk.X, expand=True)
        output_frame = tk.Frame(setup_frame); output_frame.pack(fill=tk.X, pady=2)
        tk.Label(output_frame, text="Output Folder:", width=20, anchor='w').pack(side=tk.LEFT)
        self.output_dir_entry = tk.Entry(output_frame); self.output_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(output_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.RIGHT)

        # --- Sezione 2: Input Folders ---
        folders_frame = tk.LabelFrame(main_frame, text="2. Input Folders", padx=10, pady=10)
        folders_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_dir_var.trace_add("write", lambda *args: self.load_data_for_editors())
        raw_frame = tk.Frame(folders_frame); raw_frame.pack(fill=tk.X, pady=2)
        tk.Label(raw_frame, text="RAW Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(raw_frame, textvariable=self.raw_dir_var).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(raw_frame, text="Browse...", command=lambda: self.select_folder(self.raw_dir_var, "Select RAW Data Folder")).pack(side=tk.RIGHT)
        unenriched_frame = tk.Frame(folders_frame); unenriched_frame.pack(fill=tk.X, pady=2)
        tk.Label(unenriched_frame, text="Un-enriched Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(unenriched_frame, textvariable=self.unenriched_dir_var).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(unenriched_frame, text="Browse...", command=lambda: self.select_folder(self.unenriched_dir_var, "Select Un-enriched Data Folder")).pack(side=tk.RIGHT)
        enriched_frame = tk.Frame(folders_frame); enriched_frame.pack(fill=tk.X, pady=2)
        tk.Label(enriched_frame, text="Enriched Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(enriched_frame, textvariable=self.enriched_dir_var).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(enriched_frame, text="Browse...", command=lambda: self.select_folder(self.enriched_dir_var, "Select Enriched Data Folder")).pack(side=tk.RIGHT)
        
        # --- Sezione 2.5: Event Management ---
        event_frame = tk.LabelFrame(main_frame, text="2.5 Event Management", padx=10, pady=10)
        event_frame.pack(fill=tk.X, pady=5, padx=10)
        ext_event_file_frame = tk.Frame(event_frame)
        ext_event_file_frame.pack(fill=tk.X, pady=2)
        tk.Label(ext_event_file_frame, text="Optional Events File:", width=25, anchor='w').pack(side=tk.LEFT)
        self.external_event_file_var.trace_add("write", lambda *args: self.load_data_for_editors())
        tk.Entry(ext_event_file_frame, textvariable=self.external_event_file_var).pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(ext_event_file_frame, text="Browse...", command=self.select_event_file).pack(side=tk.RIGHT)
        event_buttons_frame = tk.Frame(event_frame, pady=5)
        event_buttons_frame.pack(fill=tk.X)
        self.event_summary_label = tk.Label(event_buttons_frame, text="Load data to manage events.")
        self.event_summary_label.pack(side=tk.LEFT, pady=5)
        self.edit_video_btn = tk.Button(event_buttons_frame, text="Edit on Video", command=self.open_event_manager_video, state=tk.DISABLED)
        self.edit_video_btn.pack(side=tk.RIGHT, padx=5)
        self.edit_events_btn = tk.Button(event_buttons_frame, text="Edit in Table", command=self.open_event_manager_table, state=tk.DISABLED)
        self.edit_events_btn.pack(side=tk.RIGHT, padx=5)

        # --- Sezione 3: Analisi ---
        analysis_frame = tk.LabelFrame(main_frame, text="3. Run Core Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Analyze un-enriched data only", variable=self.unenriched_var).pack(anchor='w')
        self.yolo_var = tk.BooleanVar(value=True)
        tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (GPU Recommended)", variable=self.yolo_var).pack(anchor='w')
        tk.Button(analysis_frame, text="RUN CORE ANALYSIS", command=self.run_core_analysis, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5)

        # --- Sezioni 4, 5, 6 (Tabs) ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        plot_tab = tk.Frame(notebook); notebook.add(plot_tab, text='4. Generate Plots')
        video_tab = tk.Frame(notebook); notebook.add(video_tab, text='5. Generate Videos')
        yolo_tab = tk.Frame(notebook); notebook.add(yolo_tab, text='6. YOLO Results')
        self.setup_plot_tab(plot_tab)
        self.setup_video_tab(video_tab)
        self.setup_yolo_tab(yolo_tab)
        
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def setup_plot_tab(self, parent_tab):
        plot_options_frame = tk.LabelFrame(parent_tab, text="Plot Options", padx=10, pady=10)
        plot_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plot_types = {"path_plots": "Path Plots", "heatmaps": "Density Heatmaps", "histograms": "Duration Histograms", "pupillometry": "Pupillometry", "advanced_timeseries": "Advanced Time Series", "fragmentation": "Gaze Fragmentation Plot"}
        for key, text in plot_types.items():
            self.plot_vars[key] = tk.BooleanVar(value=True); tk.Checkbutton(plot_options_frame, text=text, variable=self.plot_vars[key]).pack(anchor='w')
        tk.Button(parent_tab, text="GENERATE SELECTED PLOTS", command=self.run_plot_generation, font=('Helvetica', 10, 'bold'), bg='#90caf9').pack(pady=10)

    def setup_video_tab(self, parent_tab):
        video_options_frame = tk.LabelFrame(parent_tab, text="Video Options", padx=10, pady=10)
        video_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_opts = {"trim_to_events": "Trim video to selected events only", "crop_and_correct_perspective": "Crop & Correct Perspective", "overlay_yolo": "Overlay YOLO detections", "overlay_gaze": "Overlay gaze point", "overlay_pupil_plot": "Overlay pupillometry plot", "overlay_fragmentation_plot": "Overlay gaze fragmentation plot", "overlay_event_text": "Overlay event name text", "overlay_on_surface_text": "Overlay 'On Surface' text", "include_internal_cam": "Include internal camera (PiP)"}
        for key, text in video_opts.items():
            self.video_vars[key] = tk.BooleanVar(value=False); tk.Checkbutton(video_options_frame, text=text, variable=self.video_vars[key]).pack(anchor='w')
        self.video_vars['overlay_gaze'].set(True); self.video_vars['overlay_event_text'].set(True)
        tk.Label(video_options_frame, text="\nOutput Filename:").pack(anchor='w')
        self.video_filename_var = tk.StringVar(value="video_output_1.mp4")
        tk.Entry(video_options_frame, textvariable=self.video_filename_var).pack(fill=tk.X, pady=5)
        tk.Button(parent_tab, text="GENERATE VIDEO", command=self.run_video_generation, font=('Helvetica', 10, 'bold'), bg='#ef9a9a').pack(pady=10)

    def setup_yolo_tab(self, parent_tab):
        tk.Button(parent_tab, text="Load/Refresh YOLO Results", command=self.load_yolo_results, font=('Helvetica', 10, 'bold'), bg='#ffcc80').pack(pady=10)
        class_frame = tk.LabelFrame(parent_tab, text="Results per Class", padx=10, pady=10); class_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.class_treeview = ttk.Treeview(class_frame, show='headings'); self.class_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        instance_frame = tk.LabelFrame(parent_tab, text="Results per Instance", padx=10, pady=10); instance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.instance_treeview = ttk.Treeview(instance_frame, show='headings'); self.instance_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def select_folder(self, var, title):
        dir_path = filedialog.askdirectory(title=title)
        if dir_path: var.set(dir_path)

    def select_event_file(self):
        filepath = filedialog.askopenfilename(title="Select Custom Event File", filetypes=[("CSV files", "*.csv")])
        if filepath:
            self.external_event_file_var.set(filepath)

    def load_data_for_editors(self):
        unenriched_path_str = self.unenriched_dir_var.get()
        ext_event_file_str = self.external_event_file_var.get()
        
        event_dfs = []
        
        if unenriched_path_str:
            default_events_path = Path(unenriched_path_str) / 'events.csv'
            if default_events_path.exists():
                try:
                    df_default = pd.read_csv(default_events_path)
                    df_default['source'] = 'default'
                    event_dfs.append(df_default)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not read default events.csv:\n{e}")

        if ext_event_file_str:
            optional_events_path = Path(ext_event_file_str)
            if optional_events_path.exists():
                try:
                    df_optional = pd.read_csv(optional_events_path)
                    df_optional['source'] = 'optional'
                    event_dfs.append(df_optional)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not read optional events file:\n{e}")

        if event_dfs:
            self.events_df = pd.concat(event_dfs, ignore_index=True).sort_values('timestamp [ns]').reset_index(drop=True)
            if 'selected' not in self.events_df.columns:
                self.events_df['selected'] = True
        else:
            self.events_df = pd.DataFrame()

        if unenriched_path_str:
            world_ts_path = Path(unenriched_path_str) / 'world_timestamps.csv'
            if world_ts_path.exists():
                try:
                    self.world_timestamps_df = pd.read_csv(world_ts_path)
                    if 'frame' not in self.world_timestamps_df.columns:
                        self.world_timestamps_df['frame'] = self.world_timestamps_df.index
                except Exception:
                    self.world_timestamps_df = pd.DataFrame()
            else:
                self.world_timestamps_df = pd.DataFrame()
        else:
            self.world_timestamps_df = pd.DataFrame()
            
        self.update_event_summary_display()

    def update_event_summary_display(self):
        if not self.events_df.empty:
            selected_count = self.events_df['selected'].sum() if 'selected' in self.events_df.columns else len(self.events_df)
            self.event_summary_label.config(text=f"{selected_count} of {len(self.events_df)} events loaded.")
            self.edit_events_btn.config(state=tk.NORMAL)
        else:
            self.event_summary_label.config(text="Load data to manage events.")
            self.edit_events_btn.config(state=tk.DISABLED)

        video_path = None
        if self.unenriched_dir_var.get():
            try:
                video_path = next(Path(self.unenriched_dir_var.get()).glob('*.mp4'))
            except StopIteration:
                video_path = None

        self.edit_video_btn.config(state=tk.NORMAL if video_path and video_path.exists() else tk.DISABLED)

    def open_event_manager_table(self):
        if self.events_df.empty:
            messagebox.showwarning("Warning", "No events loaded to edit.")
            return
        manager = EventManagerWindow(self.root, self.events_df)
        self.root.wait_window(manager)
        if manager.saved_df is not None:
            self.events_df = manager.saved_df.reset_index(drop=True)
            self.update_event_summary_display()
            logging.info("Event list updated via table editor.")

    def open_event_manager_video(self):
        video_path = next(Path(self.unenriched_dir_var.get()).glob('*.mp4'))
        manager = InteractiveVideoEditor(self.root, video_path, self.events_df, self.world_timestamps_df)
        self.root.wait_window(manager)
        if manager.saved_df is not None:
            self.events_df = manager.saved_df.reset_index(drop=True)
            self.update_event_summary_display()
            logging.info("Event list updated via video editor.")

    def run_core_analysis(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not (output_dir and subj_name and self.raw_dir_var.get() and self.unenriched_dir_var.get()):
            messagebox.showerror("Error", "Participant Name, Output Folder, RAW, and Un-enriched folders are mandatory.")
            return
            
        if self.events_df.empty and not messagebox.askyesno("Confirmation", "No events loaded. Analysis will run on the entire recording without segmentation. Continue?"):
            return
            
        try:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            final_events_df = self.events_df.copy()
            cols_to_save = ['name', 'timestamp [ns]', 'recording id']
            final_events_df = final_events_df[[col for col in cols_to_save if col in final_events_df.columns]]
            if 'recording id' not in final_events_df.columns and not final_events_df.empty:
                final_events_df['recording id'] = 'rec_001'

            modified_events_path = output_dir_path / 'modified_events.csv'
            final_events_df.to_csv(modified_events_path, index=False)
            logging.info(f"Final user-modified event list saved to: {modified_events_path}")

            selected_event_names = self.events_df[self.events_df['selected']]['name'].tolist() if 'selected' in self.events_df.columns and not self.events_df.empty else []

            messagebox.showinfo("In Progress", "Starting core analysis...")
            
            main_analyzer.run_core_analysis(
                subj_name=subj_name, output_dir_str=output_dir, raw_dir_str=self.raw_dir_var.get(),
                unenriched_dir_str=self.unenriched_dir_var.get(), enriched_dir_str=self.enriched_dir_var.get(),
                un_enriched_mode=self.unenriched_var.get(), run_yolo=self.yolo_var.get(),
                selected_events=selected_event_names,
                custom_event_path=str(modified_events_path) 
            )
            messagebox.showinfo("Success", f"Core analysis completed.\nResults in: {output_dir}")
        except Exception as e:
            logging.error(f"Core Analysis Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Analysis Error", f"An error occurred: {e}\n\nSee log for details.")

    def update_output_dir_default(self, *args):
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(Path(f'./analysis_results_{subj_name}').resolve()))

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(title="Select Output Folder")
        if dir_path: self.output_dir_entry.delete(0, tk.END); self.output_dir_entry.insert(0, dir_path)

    def _get_common_paths(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not output_dir or not subj_name:
            messagebox.showerror("Error", "Please enter participant name and output folder.")
            return None
        return Path(output_dir), subj_name

    def run_plot_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths
        plot_selections = {key: var.get() for key, var in self.plot_vars.items()}
        try:
            messagebox.showinfo("In Progress", "Generating plots...")
            main_analyzer.generate_selected_plots(output_dir_str=str(output_dir_path), subj_name=subj_name, plot_selections=plot_selections)
            messagebox.showinfo("Success", f"Plots generated in {output_dir_path / 'plots'}")
        except Exception as e:
            logging.error(f"Plotting Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Plotting Error", f"An error occurred: {e}\n\nSee log for details.")

    def run_video_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths
        video_options = {key: var.get() for key, var in self.video_vars.items()}
        video_options['output_filename'] = self.video_filename_var.get().strip()
        if not video_options['output_filename']:
            messagebox.showerror("Error", "Please specify an output video filename.")
            return
        try:
            messagebox.showinfo("In Progress", "Generating video...")
            main_analyzer.generate_custom_video(output_dir_str=str(output_dir_path), subj_name=subj_name, video_options=video_options)
            messagebox.showinfo("Success", f"Video saved to {output_dir_path / video_options['output_filename']}")
        except Exception as e:
            logging.error(f"Video Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Video Error", f"An error occurred: {e}\n\nSee log for details.")

    def load_yolo_results(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, _ = common_paths
        try:
            class_csv = output_dir_path / 'stats_per_class.csv'
            if class_csv.exists(): self._populate_treeview(self.class_treeview, pd.read_csv(class_csv))
            else: messagebox.showinfo("Info", "stats_per_class.csv not found. Run Core Analysis with YOLO enabled.")
            instance_csv = output_dir_path / 'stats_per_instance.csv'
            if instance_csv.exists(): self._populate_treeview(self.instance_treeview, pd.read_csv(instance_csv))
            else: messagebox.showinfo("Info", "stats_per_instance.csv not found. Run Core Analysis with YOLO enabled.")
        except Exception as e:
            logging.error(f"Could not read YOLO results: {e}")
            messagebox.showerror("Read Error", f"Could not read YOLO results: {e}")

    def _populate_treeview(self, treeview, dataframe):
        treeview.delete(*treeview.get_children())
        treeview["columns"] = list(dataframe.columns)
        for col in dataframe.columns:
            treeview.heading(col, text=col)
            treeview.column(col, width=120, anchor='center')
        for index, row in dataframe.iterrows():
            treeview.insert("", "end", values=list(row))

if __name__ == '__main__':
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"speed_log_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    logging.info("Application started.")
    root = tk.Tk()
    app = SpeedApp(root)
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except (ImportError, RuntimeError):
        pass
    root.mainloop()
    logging.info("Application closed.")