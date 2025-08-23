# desktop_app/GUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import traceback
import json
import pandas as pd
import logging
import time
import sys

# --- MODIFICA CHIAVE PER RISOLVERE L'ERRORE ---
# Aggiungiamo la cartella radice del progetto al percorso di ricerca di Python.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Importa gli editor e la funzione di analisi
from desktop_app.interactive_video_editor import InteractiveVideoEditor
from desktop_app.aoi_editor import AoiEditor
from desktop_app.manual_aoi_editor import ManualAoiEditor
from src.speed_analyzer import run_full_analysis

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

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v3.7")
        self.root.geometry("850x850")

        self.raw_dir_var = tk.StringVar()
        self.unenriched_dir_var = tk.StringVar()
        self.enriched_dir_var = tk.StringVar()
        self.external_event_file_var = tk.StringVar()
        self.plot_vars = {}
        self.video_vars = {}
        self.events_df = pd.DataFrame()
        self.world_timestamps_df = pd.DataFrame()
        
        # --- NUOVO: Variabili per gestire l'AOI definita dall'utente ---
        self.user_defined_aoi = None
        self.user_defined_aoi_type = None


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
        self.enriched_dir_var.trace_add("write", lambda *args: self.update_aoi_button_state())

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
        
        # --- NUOVO: Sezione AOI ---
        aoi_frame = tk.Frame(folders_frame)
        aoi_frame.pack(fill=tk.X, pady=5)
        self.aoi_status_label = tk.Label(aoi_frame, text="No Enriched data. You can define an AOI.", fg="blue")
        self.aoi_status_label.pack(side=tk.LEFT, padx=5)
        self.define_aoi_btn = tk.Button(aoi_frame, text="Define AOI...", command=self.open_aoi_editor, state=tk.DISABLED)
        self.define_aoi_btn.pack(side=tk.RIGHT)
        
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
        analysis_frame = tk.LabelFrame(main_frame, text="3. Run Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5, padx=10)
        self.yolo_var = tk.BooleanVar(value=True)
        tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (Required for Dynamic AOI, GPU Recommended)", variable=self.yolo_var).pack(anchor='w')
        tk.Button(analysis_frame, text="RUN FULL ANALYSIS", command=self.run_full_analysis_wrapper, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5)

        # --- Sezioni 4, 5, 6 (Tabs) ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        plot_tab = tk.Frame(notebook); notebook.add(plot_tab, text='4. Generate Plots')
        video_tab = tk.Frame(notebook); notebook.add(video_tab, text='5. Generate Videos')
        yolo_tab = tk.Frame(notebook); notebook.add(yolo_tab, text='6. YOLO Results')
        self.setup_plot_tab(plot_tab)
        self.setup_video_tab(video_tab)
        self.setup_yolo_tab(yolo_tab)
        
        self.update_aoi_button_state()
        
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
        self.update_aoi_button_state()
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

    def update_aoi_button_state(self):
        unenriched_ok = Path(self.unenriched_dir_var.get()).is_dir()
        enriched_present = Path(self.enriched_dir_var.get()).is_dir()
        
        if enriched_present:
            self.define_aoi_btn.config(state=tk.DISABLED)
            self.aoi_status_label.config(text="Enriched data folder provided.", fg="black")
            self.user_defined_aoi = None # Reset if user selects an enriched folder
            self.user_defined_aoi_type = None
        elif unenriched_ok:
            self.define_aoi_btn.config(state=tk.NORMAL)
            if self.user_defined_aoi is not None:
                if self.user_defined_aoi_type == 'static':
                    self.aoi_status_label.config(text="Static AOI defined.", fg="green")
                elif self.user_defined_aoi_type == 'dynamic_auto':
                     self.aoi_status_label.config(text=f"Dynamic AOI (Track ID: {self.user_defined_aoi}) defined.", fg="green")
                elif self.user_defined_aoi_type == 'dynamic_manual':
                     self.aoi_status_label.config(text=f"Dynamic AOI ({len(self.user_defined_aoi)} Keyframes) defined.", fg="green")
            else:
                self.aoi_status_label.config(text="No Enriched data. You can define an AOI.", fg="blue")
        else:
            self.define_aoi_btn.config(state=tk.DISABLED)
            self.aoi_status_label.config(text="Select Un-enriched folder to define an AOI.", fg="grey")

    def open_aoi_editor(self):
        # --- NUOVA FINESTRA DI SCELTA ---
        choice_dialog = tk.Toplevel(self.root)
        choice_dialog.title("Choose AOI Definition Method")
        choice_dialog.geometry("450x200")
        choice_dialog.transient(self.root)
        choice_dialog.grab_set()
        
        tk.Label(choice_dialog, text="How would you like to define the Area of Interest?", pady=10).pack()
        
        aoi_mode = tk.StringVar(value="static")
        
        ttk.Radiobutton(choice_dialog, text="Static AOI (Fixed Rectangle)", variable=aoi_mode, value="static").pack(anchor='w', padx=20)
        ttk.Radiobutton(choice_dialog, text="Dynamic AOI (Automatic Object Tracking)", variable=aoi_mode, value="dynamic_auto").pack(anchor='w', padx=20)
        ttk.Radiobutton(choice_dialog, text="Dynamic AOI (Manual Keyframes)", variable=aoi_mode, value="dynamic_manual").pack(anchor='w', padx=20)

        def on_proceed():
            mode = aoi_mode.get()
            choice_dialog.destroy()
            self.launch_specific_aoi_editor(mode)

        tk.Button(choice_dialog, text="Proceed", command=on_proceed, font=('Helvetica', 10, 'bold')).pack(pady=20)

    def launch_specific_aoi_editor(self, mode):
        try:
            video_path = next(Path(self.unenriched_dir_var.get()).glob('*.mp4'))
        except StopIteration:
            messagebox.showerror("Error", "No .mp4 video file found in the Un-enriched folder.")
            return

        editor = None
        if mode == 'static' or mode == 'dynamic_auto':
            editor = AoiEditor(self.root, video_path)
            # Pre-seleziona la modalità corretta
            editor.mode_var.set(mode)
            editor.update_ui_for_mode()
        elif mode == 'dynamic_manual':
            editor = ManualAoiEditor(self.root, video_path)

        if editor:
            self.root.wait_window(editor)
            
            if mode == 'static' or mode == 'dynamic_auto':
                if editor.result is not None:
                    self.user_defined_aoi = editor.result
                    self.user_defined_aoi_type = editor.result_type
                    logging.info(f"AOI defined: type='{self.user_defined_aoi_type}', data='{self.user_defined_aoi}'")
            elif mode == 'dynamic_manual':
                 if editor.saved_keyframes:
                    self.user_defined_aoi = editor.saved_keyframes
                    self.user_defined_aoi_type = 'dynamic_manual'
                    logging.info(f"Manual AOI defined with {len(self.user_defined_aoi)} keyframes.")

            self.update_aoi_button_state()


    def run_full_analysis_wrapper(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not (output_dir and subj_name and self.raw_dir_var.get() and self.unenriched_dir_var.get()):
            messagebox.showerror("Error", "Participant Name, Output Folder, RAW, and Un-enriched folders are mandatory.")
            return
        
        try:
            output_dir_path = Path(output_dir)
            final_events_df = self.events_df.copy()
            if not final_events_df.empty:
                cols_to_save = ['name', 'timestamp [ns]', 'recording id', 'selected']
                df_to_save = final_events_df[[col for col in cols_to_save if col in final_events_df.columns]]
                modified_events_path = output_dir_path / 'modified_events.csv'
                df_to_save.to_csv(modified_events_path, index=False)
                logging.info(f"Final user-modified event list saved to: {modified_events_path}")

            messagebox.showinfo("In Progress", "Starting full analysis...")
            
            # --- NUOVO: Passa i parametri AOI alla funzione di analisi ---
            aoi_params = {}
            if self.user_defined_aoi_type == 'static':
                aoi_params['aoi_coordinates'] = self.user_defined_aoi
            elif self.user_defined_aoi_type == 'dynamic_auto':
                aoi_params['aoi_track_id'] = self.user_defined_aoi
            elif self.user_defined_aoi_type == 'dynamic_manual':
                aoi_params['aoi_keyframes'] = self.user_defined_aoi

            run_full_analysis(
                raw_data_path=self.raw_dir_var.get(),
                unenriched_data_path=self.unenriched_dir_var.get(),
                enriched_data_path=self.enriched_dir_var.get() or None,
                output_path=output_dir,
                subject_name=subj_name,
                events_df=final_events_df,
                run_yolo=self.yolo_var.get(),
                yolo_model_path="yolov8n.pt",
                **aoi_params # Passa i parametri AOI
            )

            messagebox.showinfo("Success", f"Full analysis completed.\nResults in: {output_dir}")
        except Exception as e:
            logging.error(f"Full Analysis Error: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Analysis Error", f"An error occurred: {e}\n\nSee log for details.")

    def update_output_dir_default(self, *args):
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(project_root / f'analysis_results_{subj_name}'))

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
        messagebox.showinfo("Info", "Questa funzione andrebbe ricollegata al nuovo package.")

    def run_video_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        messagebox.showinfo("Info", "Questa funzione andrebbe ricollegata al nuovo package.")

    def load_yolo_results(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, _ = common_paths
        try:
            class_csv = output_dir_path / 'stats_per_class.csv'
            if class_csv.exists(): self._populate_treeview(self.class_treeview, pd.read_csv(class_csv))
            else: messagebox.showinfo("Info", "stats_per_class.csv not found. Run Analysis with YOLO enabled.")
            instance_csv = output_dir_path / 'stats_per_instance.csv'
            if instance_csv.exists(): self._populate_treeview(self.instance_treeview, pd.read_csv(instance_csv))
            else: messagebox.showinfo("Info", "stats_per_instance.csv not found. Run Analysis with YOLO enabled.")
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
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"speed_log_{time.strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    logging.info("Application started.")
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()
    logging.info("Application closed.")
