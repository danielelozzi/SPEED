# GUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import traceback
import json
import pandas as pd
import logging
import time

# Import the main orchestrator functions
try:
    import main_analyzer_advanced as main_analyzer
except ImportError as e:
    messagebox.showerror("Critical Error", f"Missing library: {e}. Make sure you have installed all requirements.")
    exit()

class EventManagerWindow(tk.Toplevel):
    """A window for viewing, selecting, editing, adding, merging, and removing events."""
    def __init__(self, parent, events_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Event Manager")
        self.geometry("800x600") # Aumentata leggermente la larghezza per i nuovi pulsanti
        self.transient(parent)
        self.grab_set()

        self.events_df = events_df.copy()
        self.events_df.sort_values('timestamp [ns]', inplace=True)
        self.saved_df = None

        frame = tk.Frame(self, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        cols = ("Selected", "Event Name", "Timestamp (ns)")
        self.tree = ttk.Treeview(frame, columns=cols, show='headings', selectmode='browse') # 'browse' permette solo una selezione singola
        for col in cols:
            self.tree.heading(col, text=col)
        self.tree.column("Selected", width=80, anchor=tk.CENTER)
        self.tree.column("Event Name", width=400)
        self.tree.column("Timestamp (ns)", width=150, anchor=tk.CENTER)
        
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
        
        # --- NUOVO: Pulsante per modificare il timestamp ---
        self.modify_ts_button = tk.Button(button_frame, text="Modify Timestamp", command=self.modify_selected_timestamp, state=tk.DISABLED)
        self.modify_ts_button.pack(side=tk.LEFT, padx=5)

        self.merge_button = tk.Button(button_frame, text="Merge Selected", command=self.merge_events, state=tk.DISABLED)
        self.merge_button.pack(side=tk.LEFT, padx=5)
        self.remove_button = tk.Button(button_frame, text="Remove Selected", command=self.remove_selected_events, state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def populate_tree(self):
        # Salva la selezione corrente per ripristinarla dopo l'aggiornamento
        selected_item = self.tree.selection()
        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for index, row in self.events_df.iterrows():
            selected_text = "Yes" if row['selected'] else "No"
            timestamp_sec = row['timestamp [ns]'] / 1e9
            self.tree.insert("", "end", iid=str(index), values=(selected_text, row['name'], f"{timestamp_sec:.4f}"))
        
        # Ripristina la selezione se esiste ancora
        if selected_item and self.tree.exists(selected_item[0]):
            self.tree.selection_set(selected_item)

    def sort_events(self):
        self.events_df.sort_values('timestamp [ns]', inplace=True)
        self.populate_tree()

    def on_selection_change(self, event):
        num_selected = len(self.tree.selection())
        # Abilita il pulsante di modifica solo se è selezionato UN solo evento
        self.modify_ts_button.config(state=tk.NORMAL if num_selected == 1 else tk.DISABLED)
        
        # Logica per gli altri pulsanti
        self.remove_button.config(state=tk.NORMAL if num_selected > 0 else tk.DISABLED)
        self.merge_button.config(state=tk.NORMAL if num_selected >= 2 else tk.DISABLED)

    def on_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        column = self.tree.identify_column(event.x)
        if region == "cell" and column == "#1":
            item_id = self.tree.identify_row(event.y)
            if item_id:
                # Usa 'browse' per la selezione singola, quindi non serve deselezionare
                df_index = int(item_id)
                self.events_df.loc[df_index, 'selected'] = not self.events_df.loc[df_index, 'selected']
                self.populate_tree()

    def on_double_click(self, event):
        # Permette di modificare solo il nome con il doppio clic
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell": return
        column_id = self.tree.identify_column(event.x)
        item_id = self.tree.identify_row(event.y)
        
        # Abilitato solo per la colonna del nome (#2)
        if column_id != "#2": return

        x, y, width, height = self.tree.bbox(item_id, column_id)
        entry = ttk.Entry(self.tree)
        current_values = self.tree.item(item_id, 'values')
        entry.insert(0, current_values[1]) # Colonna nome
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def save_name_edit(event):
            new_value = entry.get()
            df_index = int(item_id)
            self.events_df.loc[df_index, 'name'] = new_value
            self.populate_tree()
            entry.destroy()
        entry.bind("<Return>", save_name_edit)
        entry.bind("<FocusOut>", save_name_edit)

    def add_event(self):
        name = simpledialog.askstring("Add Event", "Enter the new event name:", parent=self)
        if not name: return
        ts_str = simpledialog.askstring("Add Event", f"Enter timestamp in seconds for '{name}':", parent=self)
        if not ts_str: return
        try:
            ts_sec = float(ts_str)
            new_index = self.events_df.index.max() + 1 if not self.events_df.empty else 0
            new_row = {'name': name, 'timestamp [ns]': int(ts_sec * 1e9), 'selected': True}
            self.events_df = pd.concat([self.events_df, pd.DataFrame(new_row, index=[new_index])])
            self.populate_tree()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for the timestamp.")

    # --- NUOVO: Metodo chiamato dal pulsante "Modify Timestamp" ---
    def modify_selected_timestamp(self):
        selected_items = self.tree.selection()
        if not selected_items:
            return
        
        item_id = selected_items[0]
        df_index = int(item_id)
        
        current_name = self.events_df.loc[df_index, 'name']
        current_ts_sec = self.events_df.loc[df_index, 'timestamp [ns]'] / 1e9

        # Chiede all'utente il nuovo valore
        prompt_title = "Modify Timestamp"
        prompt_text = f"Enter new timestamp in seconds for event:\n'{current_name}'"
        
        new_ts_str = simpledialog.askstring(prompt_title, prompt_text, initialvalue=f"{current_ts_sec:.4f}", parent=self)

        if new_ts_str:
            try:
                new_ts_sec = float(new_ts_str)
                # Aggiorna il DataFrame
                self.events_df.loc[df_index, 'timestamp [ns]'] = int(new_ts_sec * 1e9)
                # Aggiorna la tabella nella GUI
                self.populate_tree()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for the timestamp.", parent=self)

    def merge_events(self):
        # Per unire eventi, è meglio permettere selezioni multiple. Modifichiamo temporaneamente la modalità.
        self.tree.config(selectmode="extended")
        messagebox.showinfo("Merge Events", "Select two or more events to merge using Ctrl-Click or Shift-Click, then click 'Merge Selected' again.", parent=self)
        
        # Questo è un trucco per gestire il doppio click sul pulsante. 
        # L'utente clicca una volta per attivare la modalità, seleziona, e poi clicca di nuovo.
        if len(self.tree.selection()) < 2: return
        
        selected_items = self.tree.selection()
        new_name = simpledialog.askstring("Merge Events", "Enter the name for the new merged event:", parent=self)
        if not new_name:
            self.tree.config(selectmode="browse") # Ripristina la modalità
            return
            
        indices = [int(item_id) for item_id in selected_items]
        selected_df = self.events_df.loc[indices]
        first_timestamp_ns = selected_df['timestamp [ns]'].min()
        new_index = self.events_df.index.max() + 1
        new_row = {'name': new_name, 'timestamp [ns]': first_timestamp_ns, 'selected': True}
        self.events_df = pd.concat([self.events_df, pd.DataFrame(new_row, index=[new_index])])
        self.populate_tree()
        messagebox.showinfo("Success", f"Event '{new_name}' created from merge.", parent=self)
        self.tree.config(selectmode="browse") # Ripristina la modalità di selezione singola

    def remove_selected_events(self):
        selected_items = self.tree.selection()
        if not selected_items: return
        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to remove {len(selected_items)} event(s)?", parent=self):
            indices_to_drop = [int(item_id) for item_id in selected_items]
            self.events_df.drop(indices_to_drop, inplace=True)
            self.populate_tree()

    def save_and_close(self):
        self.saved_df = self.events_df
        self.destroy()
        
    """A window for viewing, selecting, editing, adding, merging, and removing events."""
    def __init__(self, parent, events_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Event Manager")
        self.geometry("750x600")
        self.transient(parent)
        self.grab_set()

        self.events_df = events_df.copy()
        # Sorts the data only once, when the window is opened
        self.events_df.sort_values('timestamp [ns]', inplace=True)
        self.saved_df = None

        frame = tk.Frame(self, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        cols = ("Selected", "Event Name", "Timestamp (ns)")
        self.tree = ttk.Treeview(frame, columns=cols, show='headings', selectmode='extended')
        for col in cols:
            self.tree.heading(col, text=col)
        self.tree.column("Selected", width=80, anchor=tk.CENTER)
        self.tree.column("Event Name", width=350)
        self.tree.column("Timestamp (ns)", width=150, anchor=tk.CENTER)
        
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
        
        tk.Button(button_frame, text="Sort by Time", command=self.sort_events).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Add Event", command=self.add_event).pack(side=tk.LEFT, padx=5)
        self.merge_button = tk.Button(button_frame, text="Merge Selected", command=self.merge_events, state=tk.DISABLED)
        self.merge_button.pack(side=tk.LEFT, padx=5)
        self.remove_button = tk.Button(button_frame, text="Remove Selected", command=self.remove_selected_events, state=tk.DISABLED)
        self.remove_button.pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save & Close", command=self.save_and_close, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def populate_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for index, row in self.events_df.iterrows():
            selected_text = "Yes" if row['selected'] else "No"
            timestamp_sec = row['timestamp [ns]'] / 1e9
            self.tree.insert("", "end", iid=str(index), values=(selected_text, row['name'], f"{timestamp_sec:.4f}"))

    def sort_events(self):
        """Sorts the events by timestamp and repopulates the tree."""
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
                self.tree.selection_set(())
                df_index = int(item_id)
                self.events_df.loc[df_index, 'selected'] = not self.events_df.loc[df_index, 'selected']
                self.populate_tree()

    def on_double_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell": return
        column_id = self.tree.identify_column(event.x)
        item_id = self.tree.identify_row(event.y)
        if column_id not in ("#2", "#3"): return

        x, y, width, height = self.tree.bbox(item_id, column_id)
        entry = ttk.Entry(self.tree)
        current_values = self.tree.item(item_id, 'values')
        col_index = int(column_id.replace('#', '')) - 1
        entry.insert(0, current_values[col_index])
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def save_edit(event):
            new_value = entry.get()
            df_index = int(item_id)
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
            new_index = self.events_df.index.max() + 1 if not self.events_df.empty else 0
            new_row = {'name': name, 'timestamp [ns]': int(ts_sec * 1e9), 'selected': True}
            self.events_df = pd.concat([self.events_df, pd.DataFrame(new_row, index=[new_index])])
            self.populate_tree()
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
        new_index = self.events_df.index.max() + 1
        new_row = {'name': new_name, 'timestamp [ns]': first_timestamp_ns, 'selected': True}
        self.events_df = pd.concat([self.events_df, pd.DataFrame(new_row, index=[new_index])])
        self.populate_tree()
        messagebox.showinfo("Success", f"Event '{new_name}' created from merge.")

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
        self.root.title("SPEED v3.4")
        self.root.geometry("800x750")

        self.raw_dir_var = tk.StringVar()
        self.unenriched_dir_var = tk.StringVar()
        self.enriched_dir_var = tk.StringVar()
        self.plot_vars = {}
        self.video_vars = {}
        self.events_df = pd.DataFrame()

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

        folders_frame = tk.LabelFrame(main_frame, text="2. Input Folders", padx=10, pady=10)
        folders_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_dir_var.trace_add("write", lambda *args: self.load_events_from_file())
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
        
        self.event_selection_frame = tk.LabelFrame(main_frame, text="2.5. Event Management", padx=10, pady=10)
        self.event_selection_frame.pack(fill=tk.X, pady=5, padx=10)
        event_controls_frame = tk.Frame(self.event_selection_frame)
        event_controls_frame.pack(fill=tk.X)
        self.event_summary_label = tk.Label(event_controls_frame, text="Select un-enriched folder to load events.")
        self.event_summary_label.pack(side=tk.LEFT, pady=5)
        self.modify_events_button = tk.Button(event_controls_frame, text="Edit Events", command=self.open_event_manager)
        self.modify_events_button.pack(side=tk.RIGHT, padx=5)
        self.modify_events_button.config(state=tk.DISABLED)

        analysis_frame = tk.LabelFrame(main_frame, text="3. Run Core Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Analyze un-enriched data only", variable=self.unenriched_var).pack(anchor='w')
        self.yolo_var = tk.BooleanVar(value=True)
        tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (GPU Recommended)", variable=self.yolo_var).pack(anchor='w')
        tk.Button(analysis_frame, text="RUN CORE ANALYSIS", command=self.run_core_analysis, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5)

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

    def load_events_from_file(self):
        self.modify_events_button.config(state=tk.DISABLED)
        unenriched_path = self.unenriched_dir_var.get()
        if not unenriched_path:
            self.events_df = pd.DataFrame()
            self.update_event_summary_display()
            return
        events_file = Path(unenriched_path) / 'events.csv'
        if not events_file.exists():
            messagebox.showerror("Error", "events.csv not found.")
            self.events_df = pd.DataFrame(); self.update_event_summary_display()
            return
        try:
            self.events_df = pd.read_csv(events_file)
            # --- CORREZIONE CHIAVE ---
            # Pulisce i nomi degli eventi subito dopo il caricamento per coerenza
            if 'name' in self.events_df.columns:
                self.events_df['name'] = self.events_df['name'].astype(str).str.replace(r'[\\/]', '_', regex=True)
            # --- FINE CORREZIONE ---
            self.events_df['selected'] = True
            self.update_event_summary_display()
        except Exception as e:
            messagebox.showerror("Error", f"Could not read events.csv:\n{e}")
            self.events_df = pd.DataFrame(); self.update_event_summary_display()

    def open_event_manager(self):
        if self.events_df.empty:
            messagebox.showwarning("Warning", "No events loaded.")
            return
        manager = EventManagerWindow(self.root, self.events_df)
        self.root.wait_window(manager)
        if manager.saved_df is not None:
            self.events_df = manager.saved_df
            self.update_event_summary_display()
            logging.info("Event list updated by user.")

    def update_event_summary_display(self):
        total_events = len(self.events_df)
        if total_events > 0:
            selected_count = self.events_df['selected'].sum()
            self.event_summary_label.config(text=f"{selected_count} of {total_events} events selected.")
            self.modify_events_button.config(state=tk.NORMAL)
        else:
            self.event_summary_label.config(text="Select un-enriched folder to load events.")
            self.modify_events_button.config(state=tk.DISABLED)

    def run_core_analysis(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not (output_dir and subj_name and self.raw_dir_var.get() and self.unenriched_dir_var.get()):
            messagebox.showerror("Error", "Participant Name, Output Folder, RAW, and Un-enriched folders are mandatory.")
            return
        if self.events_df.empty and not messagebox.askyesno("Confirmation", "No events loaded. Continue?"):
            return
        try:
            working_data_dir = Path(output_dir) / 'eyetracking_file'
            working_data_dir.mkdir(parents=True, exist_ok=True)
            # Questa chiamata assicura che i file di base siano copiati prima di sovrascrivere events.csv
            main_analyzer._prepare_eyetracking_files(
                 Path(output_dir), Path(self.raw_dir_var.get()),
                 Path(self.unenriched_dir_var.get()),
                 Path(self.enriched_dir_var.get()) if self.enriched_dir_var.get() else Path(),
                 self.unenriched_var.get()
            )
            df_to_save = self.events_df.copy()
            output_cols = [col for col in ['name', 'timestamp [ns]'] if col in df_to_save.columns]
            df_to_save = df_to_save[output_cols]
            events_output_path = working_data_dir / 'events.csv'
            df_to_save.to_csv(events_output_path, index=False)
            logging.info(f"Saved/updated user-modified event list to {events_output_path}")
            selected_event_names = self.events_df[self.events_df['selected']]['name'].tolist()
            messagebox.showinfo("In Progress", "Starting core analysis...")
            main_analyzer.run_core_analysis(
                subj_name=subj_name, output_dir_str=output_dir, raw_dir_str=self.raw_dir_var.get(),
                unenriched_dir_str=self.unenriched_dir_var.get(), enriched_dir_str=self.enriched_dir_var.get(),
                un_enriched_mode=self.unenriched_var.get(), run_yolo=self.yolo_var.get(),
                selected_events=selected_event_names
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
            instance_csv = output_dir_path / 'stats_per_instance.csv'
            if instance_csv.exists(): self._populate_treeview(self.instance_treeview, pd.read_csv(instance_csv))
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