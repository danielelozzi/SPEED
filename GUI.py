# GUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import traceback
import json
import pandas as pd
import logging # NUOVA IMPORTAZIONE
import time # NUOVA IMPORTAZIONE

# Import the main orchestrator functions
try:
    import main_analyzer_advanced as main_analyzer
except ImportError as e:
    messagebox.showerror("Critical Error", f"Missing library: {e}. Make sure you have installed all requirements (e.g., pandas).")
    exit()


class EventSelectionWindow:
    """A separate window for selecting events to analyze."""
    def __init__(self, parent, title, event_names, event_vars):
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.geometry("600x700")
        self.event_vars = event_vars

        # Make the window modal
        self.top.transient(parent)
        self.top.grab_set()

        # --- WIDGETS ---
        main_frame = tk.Frame(self.top, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="Select the events you want to include in the analysis:", font=('Helvetica', 10, 'bold')).pack(pady=(0, 10), anchor='w')

        # --- Scrollable Checkbox Area ---
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Populate with checkboxes
        for name, var in self.event_vars.items():
            tk.Checkbutton(scrollable_frame, text=name, variable=var).pack(anchor='w', padx=5)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Button Area ---
        button_frame = tk.Frame(self.top, pady=10)
        button_frame.pack(fill=tk.X)

        tk.Button(button_frame, text="Select All", command=self.select_all).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Deselect All", command=self.deselect_all).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Confirm Selection", command=self.confirm, font=('Helvetica', 10, 'bold')).pack(side=tk.RIGHT, padx=10)

    def select_all(self):
        for var in self.event_vars.values():
            var.set(True)

    def deselect_all(self):
        for var in self.event_vars.values():
            var.set(False)

    def confirm(self):
        self.top.destroy()


class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v3.4 - Enhanced Video & Plots") # Titolo modificato
        self.root.geometry("800x850")

        # Variables for folder paths
        self.raw_dir_var = tk.StringVar()
        self.unenriched_dir_var = tk.StringVar()
        self.enriched_dir_var = tk.StringVar()

        self.plot_vars = {}
        self.video_vars = {}
        self.event_vars = {} # To hold event checkbox variables

        # --- Main Scrollable Canvas Setup ---
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

        # --- Setup Section ---
        setup_frame = tk.LabelFrame(main_frame, text="1. Project Setup", padx=10, pady=10)
        setup_frame.pack(fill=tk.X, pady=10, padx=10)

        name_frame = tk.Frame(setup_frame); name_frame.pack(fill=tk.X, pady=2)
        tk.Label(name_frame, text="Participant Name:", width=20, anchor='w').pack(side=tk.LEFT)
        self.participant_name_var = tk.StringVar(); self.participant_name_var.trace_add("write", self.update_output_dir_default)
        self.name_entry = tk.Entry(name_frame, textvariable=self.participant_name_var); self.name_entry.pack(fill=tk.X, expand=True)

        output_frame = tk.Frame(setup_frame); output_frame.pack(fill=tk.X, pady=2)
        tk.Label(output_frame, text="Output Folder:", width=20, anchor='w').pack(side=tk.LEFT)
        self.output_dir_entry = tk.Entry(output_frame); self.output_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(output_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.RIGHT)

        # --- Input Folders Section ---
        folders_frame = tk.LabelFrame(main_frame, text="2. Input Folders", padx=10, pady=10)
        folders_frame.pack(fill=tk.X, pady=5, padx=10)

        raw_frame = tk.Frame(folders_frame); raw_frame.pack(fill=tk.X, pady=2)
        tk.Label(raw_frame, text="RAW Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        self.raw_dir_entry = tk.Entry(raw_frame, textvariable=self.raw_dir_var); self.raw_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(raw_frame, text="Browse...", command=lambda: self.select_folder(self.raw_dir_var, "Select RAW Data Folder")).pack(side=tk.RIGHT)

        unenriched_frame = tk.Frame(folders_frame); unenriched_frame.pack(fill=tk.X, pady=2)
        tk.Label(unenriched_frame, text="Un-enriched Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        self.unenriched_dir_entry = tk.Entry(unenriched_frame, textvariable=self.unenriched_dir_var); self.unenriched_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(unenriched_frame, text="Browse...", command=lambda: self.select_folder(self.unenriched_dir_var, "Select Un-enriched Data Folder")).pack(side=tk.RIGHT)

        enriched_frame = tk.Frame(folders_frame); enriched_frame.pack(fill=tk.X, pady=2)
        tk.Label(enriched_frame, text="Enriched Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        self.enriched_dir_entry = tk.Entry(enriched_frame, textvariable=self.enriched_dir_var); self.enriched_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(enriched_frame, text="Browse...", command=lambda: self.select_folder(self.enriched_dir_var, "Select Enriched Data Folder")).pack(side=tk.RIGHT)

        # --- Event Selection Display (MODIFIED) ---
        self.event_selection_frame = tk.LabelFrame(main_frame, text="2.5. Event Selection", padx=10, pady=10)
        self.event_selection_frame.pack(fill=tk.X, pady=5, padx=10)
        self.event_summary_label = tk.Label(self.event_selection_frame, text="Select un-enriched folder to load and choose events.")
        self.event_summary_label.pack(pady=5)


        # --- Core Analysis Section ---
        analysis_frame = tk.LabelFrame(main_frame, text="3. Run Core Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Analyze un-enriched data only (ignores enriched folder)", variable=self.unenriched_var).pack(anchor='w')
        self.yolo_var = tk.BooleanVar(value=True) # MODIFICATO: YOLO attivo di default
        tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (GPU Recommended)", variable=self.yolo_var).pack(anchor='w')
        tk.Button(analysis_frame, text="RUN CORE ANALYSIS", command=self.run_core_analysis, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5)

        # --- Output Notebook ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        plot_tab = tk.Frame(notebook); notebook.add(plot_tab, text='4. Generate Plots')
        video_tab = tk.Frame(notebook); notebook.add(video_tab, text='5. Generate Videos')
        yolo_tab = tk.Frame(notebook); notebook.add(yolo_tab, text='6. YOLO Results')

        self.setup_plot_tab(plot_tab)
        self.setup_video_tab(video_tab)
        self.setup_yolo_tab(yolo_tab)
        
        # --- Credits ---
        credits_frame = tk.LabelFrame(main_frame, text="Credits", padx=10, pady=10)
        credits_frame.pack(fill=tk.X, pady=(20, 10), padx=10)
        tk.Label(credits_frame, text="Cognitive and Behavioral Sciences Laboratory (labSCoC), University of L'Aquila", justify=tk.LEFT).pack(anchor='w')

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def setup_plot_tab(self, parent_tab):
        plot_options_frame = tk.LabelFrame(parent_tab, text="Plot Options", padx=10, pady=10)
        plot_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plot_types = {
            "path_plots": "Path Plots", "heatmaps": "Density Heatmaps", "histograms": "Duration Histograms",
            "pupillometry": "Pupillometry (with 'On Surface' highlight)", "advanced_timeseries": "Advanced Time Series", "fragmentation": "Gaze Fragmentation Plot"
        }
        for key, text in plot_types.items():
            self.plot_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(plot_options_frame, text=text, variable=self.plot_vars[key]).pack(anchor='w')
        tk.Button(parent_tab, text="GENERATE SELECTED PLOTS", command=self.run_plot_generation, font=('Helvetica', 10, 'bold'), bg='#90caf9').pack(pady=10)

    def setup_video_tab(self, parent_tab):
        video_options_frame = tk.LabelFrame(parent_tab, text="Video Composition Options", padx=10, pady=10)
        video_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_opts = {
            "trim_to_events": "Trim video to include selected events only", # NUOVA OPZIONE
            "crop_and_correct_perspective": "Crop & Correct Perspective to Surface", "overlay_yolo": "Overlay YOLO object detections",
            "overlay_gaze": "Overlay gaze point", "overlay_pupil_plot": "Overlay pupillometry plot",
            "overlay_fragmentation_plot": "Overlay gaze fragmentation plot", "overlay_event_text": "Overlay event name text",
            "overlay_on_surface_text": "Overlay 'On Surface' text", # NUOVA OPZIONE
            "include_internal_cam": "Include internal camera view (PiP)",
        }
        for key, text in video_opts.items():
            self.video_vars[key] = tk.BooleanVar(value=False)
            tk.Checkbutton(video_options_frame, text=text, variable=self.video_vars[key]).pack(anchor='w')
            
        # Imposta di default le opzioni più comuni
        self.video_vars['overlay_gaze'].set(True)
        self.video_vars['overlay_event_text'].set(True)
        
        tk.Label(video_options_frame, text="\nOutput Video Filename:").pack(anchor='w')
        self.video_filename_var = tk.StringVar(value="video_output_1.mp4")
        tk.Entry(video_options_frame, textvariable=self.video_filename_var).pack(fill=tk.X, pady=5)
        tk.Button(parent_tab, text="GENERATE VIDEO", command=self.run_video_generation, font=('Helvetica', 10, 'bold'), bg='#ef9a9a').pack(pady=10)

    def setup_yolo_tab(self, parent_tab):
        tk.Button(parent_tab, text="Load/Refresh YOLO Results", command=self.load_yolo_results, font=('Helvetica', 10, 'bold'), bg='#ffcc80').pack(pady=10)
        class_frame = tk.LabelFrame(parent_tab, text="Results per Class", padx=10, pady=10)
        class_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.class_treeview = ttk.Treeview(class_frame, show='headings')
        self.class_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        instance_frame = tk.LabelFrame(parent_tab, text="Results per Instance", padx=10, pady=10)
        instance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.instance_treeview = ttk.Treeview(instance_frame, show='headings')
        self.instance_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def select_folder(self, var, title):
        dir_path = filedialog.askdirectory(title=title)
        if dir_path:
            var.set(dir_path)
            if title == "Select Un-enriched Data Folder":
                self.load_and_display_events()

    def load_and_display_events(self):
        """Opens a new window to select events from events.csv."""
        self.event_summary_label.config(text="Loading events...")
        unenriched_path = self.unenriched_dir_var.get()
        if not unenriched_path:
            self.event_summary_label.config(text="Select un-enriched folder to load and choose events.")
            return

        events_file = Path(unenriched_path) / 'events.csv'
        if not events_file.exists():
            messagebox.showerror("Error", "events.csv not found in the selected folder.")
            self.event_summary_label.config(text="events.csv not found.")
            return

        try:
            events_df = pd.read_csv(events_file)
            event_names = events_df['name'].unique().tolist()
            if not event_names:
                messagebox.showinfo("Info", "No events found in events.csv.")
                self.event_summary_label.config(text="No events found.")
                return

            self.event_vars.clear()
            for name in event_names:
                self.event_vars[name] = tk.BooleanVar(value=True)

            selection_window = EventSelectionWindow(self.root, "Select Events", event_names, self.event_vars)
            self.root.wait_window(selection_window.top)

            self.update_event_summary_display()

        except Exception as e:
            messagebox.showerror("Error", f"Could not read or process events.csv:\n{e}")
            self.event_summary_label.config(text="Error reading events file.")

    def update_event_summary_display(self):
        """Updates the label in the main window to show how many events are selected."""
        total_events = len(self.event_vars)
        selected_count = sum(var.get() for var in self.event_vars.values())
        if total_events > 0:
            self.event_summary_label.config(text=f"{selected_count} of {total_events} events selected for analysis.")
        else:
            self.event_summary_label.config(text="No events loaded.")
            
    def run_core_analysis(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not (output_dir and subj_name and self.raw_dir_var.get() and self.unenriched_dir_var.get()):
            messagebox.showerror("Error", "Participant Name, Output Folder, RAW, and Un-enriched folders are mandatory.")
            return

        selected_events = [name for name, var in self.event_vars.items() if var.get()]
        if not self.event_vars or not selected_events:
            if not messagebox.askyesno("Confirmation", "No events are selected for analysis. Continue anyway?"):
                return

        try:
            messagebox.showinfo("In Progress", "Starting core analysis. This might take some time... Check the log file for details.")
            logging.info("="*50)
            logging.info(f"STARTING CORE ANALYSIS FOR: {subj_name}")
            main_analyzer.run_core_analysis(
                subj_name=subj_name, output_dir_str=output_dir, raw_dir_str=self.raw_dir_var.get(),
                unenriched_dir_str=self.unenriched_dir_var.get(), enriched_dir_str=self.enriched_dir_var.get(),
                un_enriched_mode=self.unenriched_var.get(), run_yolo=self.yolo_var.get(),
                selected_events=selected_events
            )
            messagebox.showinfo("Success", f"Core analysis completed.\nResults saved in: {output_dir}")
            logging.info(f"CORE ANALYSIS FOR {subj_name} COMPLETED SUCCESSFULLY.")
            logging.info("="*50)
        except Exception as e:
            logging.error(f"An error occurred during Core Analysis: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Analysis Error", f"An error occurred: {e}\n\nSee log file for full traceback.")
            
    def update_output_dir_default(self, *args):
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(Path(f'./analysis_results_{subj_name}').resolve()))

    def select_output_dir(self):
        directory_path = filedialog.askdirectory(title="Select Output Folder")
        if directory_path: self.output_dir_entry.delete(0, tk.END); self.output_dir_entry.insert(0, directory_path)
            
    def _get_common_paths(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not output_dir or not subj_name:
            messagebox.showerror("Error", "Please enter the participant name and the output folder.")
            return None
        return Path(output_dir), subj_name

    def run_plot_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths
        plot_selections = {key: var.get() for key, var in self.plot_vars.items()}
        try:
            messagebox.showinfo("In Progress", "Generating selected plots... This may take a while.")
            logging.info("--- Starting Plot Generation ---")
            main_analyzer.generate_selected_plots(output_dir_str=str(output_dir_path), subj_name=subj_name, plot_selections=plot_selections)
            messagebox.showinfo("Success", f"Plots generated in {output_dir_path / 'plots'}")
            logging.info("--- Plot Generation Finished ---")
        except Exception as e:
            logging.error(f"An error occurred during Plot Generation: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Plotting Error", f"An error occurred: {e}\n\nSee log file for full traceback.")

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
            messagebox.showinfo("In Progress", "Generating custom video. This process can be very slow...")
            logging.info("--- Starting Video Generation ---")
            main_analyzer.generate_custom_video(output_dir_str=str(output_dir_path), subj_name=subj_name, video_options=video_options)
            messagebox.showinfo("Success", f"Video saved to {output_dir_path / video_options['output_filename']}")
            logging.info("--- Video Generation Finished ---")
        except Exception as e:
            logging.error(f"An error occurred during Video Generation: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Video Error", f"An error occurred: {e}\n\nSee log file for full traceback.")

    def load_yolo_results(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, _ = common_paths
        class_csv = output_dir_path / 'stats_per_class.csv'
        instance_csv = output_dir_path / 'stats_per_instance.csv'
        try:
            if class_csv.exists(): self._populate_treeview(self.class_treeview, pd.read_csv(class_csv))
            else: logging.warning(f"Could not find {class_csv}")
            if instance_csv.exists(): self._populate_treeview(self.instance_treeview, pd.read_csv(instance_csv))
            else: logging.warning(f"Could not find {instance_csv}")
        except Exception as e:
            logging.error(f"Could not read the YOLO result CSV files: {e}")
            messagebox.showerror("Read Error", f"Could not read the YOLO result CSV files: {e}")

    def _populate_treeview(self, treeview, dataframe):
        treeview.delete(*treeview.get_children())
        treeview["columns"] = list(dataframe.columns)
        for col in dataframe.columns:
            treeview.heading(col, text=col)
            treeview.column(col, width=120, anchor='center')
        for index, row in dataframe.iterrows():
            treeview.insert("", "end", values=list(row))

if __name__ == '__main__':
    # --- NUOVO: CONFIGURAZIONE DEL LOGGING ---
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / f"speed_log_{time.strftime('%Y%m%d-%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler() # Per mostrare i log anche in console
        ]
    )
    logging.info("Application started.")
    # ------------------------------------

    root = tk.Tk()
    app = SpeedApp(root)
    
    # Questo è necessario su Windows per far funzionare correttamente il multiprocessing
    # con i file .py che creano una GUI.
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except (ImportError, RuntimeError):
        pass

    root.mainloop()
    logging.info("Application closed.")