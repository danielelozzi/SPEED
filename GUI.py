# GUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import traceback
import json

# Import the main orchestrator functions
try:
    import main_analyzer_advanced as main_analyzer
    import pandas as pd # Added to read YOLO result CSVs
except ImportError as e:
    messagebox.showerror("Critical Error", f"Missing library: {e}. Make sure you have installed all requirements (e.g., pandas).")
    exit()

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v3.1 - Folder-Based Input")
        self.root.geometry("800x850") # Slightly increased size
        
        # Variables for folder paths
        self.raw_dir_var = tk.StringVar()
        self.unenriched_dir_var = tk.StringVar()
        self.enriched_dir_var = tk.StringVar()

        self.plot_vars = {}
        self.video_vars = {}

        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # --- EDIT: CROSS-PLATFORM SCROLLING BINDING ---
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel) # For Linux
        self.canvas.bind_all("<Button-5>", self._on_mousewheel) # For Linux

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
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

        # --- Core Analysis Section ---
        analysis_frame = tk.LabelFrame(main_frame, text="3. Run Core Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Analyze un-enriched data only (ignores enriched folder)", variable=self.unenriched_var).pack(anchor='w')
        self.yolo_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (requires GPU)", variable=self.yolo_var).pack(anchor='w')
        tk.Button(analysis_frame, text="RUN CORE ANALYSIS", command=self.run_core_analysis, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5)

        # --- Output Notebook ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        plot_tab = tk.Frame(notebook); notebook.add(plot_tab, text='4. Generate Plots')
        video_tab = tk.Frame(notebook); notebook.add(video_tab, text='5. Generate Videos')
        
        # --- NEW TAB for YOLO results ---
        yolo_tab = tk.Frame(notebook); notebook.add(yolo_tab, text='6. YOLO Results')

        self.setup_plot_tab(plot_tab)
        self.setup_video_tab(video_tab)
        self.setup_yolo_tab(yolo_tab) # Call the new setup function

        # --- ADDED: CREDITS SECTION ---
        credits_frame = tk.LabelFrame(main_frame, text="Credits", padx=10, pady=10)
        credits_frame.pack(fill=tk.X, pady=(20, 10), padx=10)

        tk.Label(credits_frame, text="Cognitive and Behavioral Sciences Laboratory (labSCoC), University of L'Aquila", justify=tk.LEFT).pack(anchor='w')
        tk.Label(credits_frame, text="https://labscoc.wordpress.com/", justify=tk.LEFT).pack(anchor='w')
        tk.Label(credits_frame, text="").pack(anchor='w') # Spacer
        tk.Label(credits_frame, text="Dr. Daniele Lozzi (https://github.com/danielelozzi)", justify=tk.LEFT).pack(anchor='w')


    # --- EDIT: CROSS-PLATFORM SCROLLING FUNCTION ---
    def _on_mousewheel(self, event):
        """Handles mouse wheel scrolling in a cross-platform way."""
        # Windows and macOS use event.delta
        if hasattr(event, 'delta') and event.delta != 0:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        # Linux uses event.num
        elif event.num == 4: # Scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5: # Scroll down
            self.canvas.yview_scroll(1, "units")

    def setup_plot_tab(self, parent_tab):
        plot_options_frame = tk.LabelFrame(parent_tab, text="Plot Options", padx=10, pady=10)
        plot_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plot_types = {
            "path_plots": "Path Plots (Fixation and Gaze)", "heatmaps": "Density Heatmaps (Fixation and Gaze)",
            "histograms": "Duration Histograms (Fixations, Blinks, Saccades)", "pupillometry": "Pupillometry (Time Series and Spectral Analysis)",
            "advanced_timeseries": "Advanced Time Series (Saccades, Blinks)" # Added for new plots
        }
        for key, text in plot_types.items():
            self.plot_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(plot_options_frame, text=text, variable=self.plot_vars[key]).pack(anchor='w')
        tk.Button(parent_tab, text="GENERATE SELECTED PLOTS", command=self.run_plot_generation, font=('Helvetica', 10, 'bold'), bg='#90caf9').pack(pady=10)

    def setup_video_tab(self, parent_tab):
        video_options_frame = tk.LabelFrame(parent_tab, text="Video Composition Options", padx=10, pady=10)
        video_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- MODIFIED OPTION LOGIC ---
        # "crop_to_surface" and "apply_perspective" have been merged into a single option.
        self.video_vars['crop_and_correct_perspective'] = tk.BooleanVar(value=False)
        tk.Checkbutton(video_options_frame, text="Crop & Correct Perspective to Surface", variable=self.video_vars['crop_and_correct_perspective']).pack(anchor='w')
        
        # The other options remain unchanged
        video_opts = {
            "overlay_yolo": "Overlay YOLO object detections", "overlay_gaze": "Overlay gaze point",
            "overlay_pupil_plot": "Overlay blinks and pupillometry plot", "include_internal_cam": "Include internal camera view (PiP)",
        }
        for key, text in video_opts.items():
            self.video_vars[key] = tk.BooleanVar(value=False)
            tk.Checkbutton(video_options_frame, text=text, variable=self.video_vars[key]).pack(anchor='w')
        
        tk.Label(video_options_frame, text="\nOutput Video Filename:").pack(anchor='w')
        self.video_filename_var = tk.StringVar(value="video_output_1.mp4")
        tk.Entry(video_options_frame, textvariable=self.video_filename_var).pack(fill=tk.X, pady=5)
        tk.Button(parent_tab, text="GENERATE VIDEO", command=self.run_video_generation, font=('Helvetica', 10, 'bold'), bg='#ef9a9a').pack(pady=10)
    
    def setup_yolo_tab(self, parent_tab):
        """Sets up the tab to display YOLO analysis results."""
        tk.Button(parent_tab, text="Load/Refresh YOLO Results", command=self.load_yolo_results, font=('Helvetica', 10, 'bold'), bg='#ffcc80').pack(pady=10)

        # Frame for the "per class" table
        class_frame = tk.LabelFrame(parent_tab, text="Results per Class", padx=10, pady=10)
        class_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.class_treeview = ttk.Treeview(class_frame, show='headings')
        class_scroll = ttk.Scrollbar(class_frame, orient="vertical", command=self.class_treeview.yview)
        self.class_treeview.configure(yscrollcommand=class_scroll.set)
        self.class_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        class_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Frame for the "per instance" table
        instance_frame = tk.LabelFrame(parent_tab, text="Results per Instance", padx=10, pady=10)
        instance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.instance_treeview = ttk.Treeview(instance_frame, show='headings')
        instance_scroll = ttk.Scrollbar(instance_frame, orient="vertical", command=self.instance_treeview.yview)
        self.instance_treeview.configure(yscrollcommand=instance_scroll.set)
        self.instance_treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        instance_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _populate_treeview(self, treeview, dataframe):
        """Clears and populates a Treeview widget with a pandas DataFrame."""
        # Clear the view
        treeview.delete(*treeview.get_children())
        
        # Set columns
        treeview["columns"] = list(dataframe.columns)
        for col in dataframe.columns:
            treeview.heading(col, text=col)
            treeview.column(col, width=120, anchor='center')
        
        # Add data
        for index, row in dataframe.iterrows():
            treeview.insert("", "end", values=list(row))

    def load_yolo_results(self):
        """Loads YOLO result CSV files and populates the tables."""
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, _ = common_paths

        class_csv = output_dir_path / 'stats_per_class.csv'
        instance_csv = output_dir_path / 'stats_per_instance.csv'

        try:
            if class_csv.exists():
                df_class = pd.read_csv(class_csv)
                self._populate_treeview(self.class_treeview, df_class)
            else:
                messagebox.showinfo("Info", f"File not found: {class_csv.name}\nRun YOLO analysis first.")
            
            if instance_csv.exists():
                df_instance = pd.read_csv(instance_csv)
                self._populate_treeview(self.instance_treeview, df_instance)
            else:
                messagebox.showinfo("Info", f"File not found: {instance_csv.name}\nRun YOLO analysis first.")

        except Exception as e:
            messagebox.showerror("Read Error", f"Could not read the YOLO result CSV files: {e}")

    def update_output_dir_default(self, *args):
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(Path(f'./analysis_results_{subj_name}').resolve()))

    def select_output_dir(self):
        directory_path = filedialog.askdirectory(title="Select Output Folder")
        if directory_path: self.output_dir_entry.delete(0, tk.END); self.output_dir_entry.insert(0, directory_path)

    def select_folder(self, var, title):
        dir_path = filedialog.askdirectory(title=title)
        if dir_path:
            var.set(dir_path)

    def _get_common_paths(self):
        output_dir = self.output_dir_entry.get().strip()
        subj_name = self.participant_name_var.get().strip()
        if not output_dir or not subj_name:
            messagebox.showerror("Error", "Please enter the participant name and the output folder.")
            return None
        return Path(output_dir), subj_name

    def run_core_analysis(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths

        raw_dir = self.raw_dir_var.get().strip()
        unenriched_dir = self.unenriched_dir_var.get().strip()
        enriched_dir = self.enriched_dir_var.get().strip()

        if not (raw_dir and unenriched_dir):
            messagebox.showerror("Error", "The RAW and Un-enriched folders are mandatory.")
            return
        
        try:
            messagebox.showinfo("In Progress", "Starting core analysis. This might take some time...")
            # Call the updated function in main_analyzer
            main_analyzer.run_core_analysis(
                subj_name=subj_name, 
                output_dir_str=str(output_dir_path),
                raw_dir_str=raw_dir,
                unenriched_dir_str=unenriched_dir,
                enriched_dir_str=enriched_dir,
                un_enriched_mode=self.unenriched_var.get(),
                run_yolo=self.yolo_var.get()
            )
            success_msg = "Core analysis completed."
            if self.yolo_var.get():
                success_msg += "\nYou can now load the YOLO results in the 'YOLO Results' tab."
            messagebox.showinfo("Success", success_msg)
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred: {e}\n\n{traceback.format_exc()}")

    def run_plot_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths
        
        plot_selections = {key: var.get() for key, var in self.plot_vars.items()}
        try:
            messagebox.showinfo("In Progress", "Generating selected plots...")
            main_analyzer.generate_selected_plots(output_dir_str=str(output_dir_path), subj_name=subj_name, plot_selections=plot_selections)
            messagebox.showinfo("Success", f"Plots generated in {output_dir_path / 'plots'}")
        except Exception as e:
            messagebox.showerror("Plotting Error", f"An error occurred: {e}\n\n{traceback.format_exc()}")

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
            main_analyzer.generate_custom_video(output_dir_str=str(output_dir_path), subj_name=subj_name, video_options=video_options)
            messagebox.showinfo("Success", f"Video saved to {output_dir_path / video_options['output_filename']}")
        except Exception as e:
            messagebox.showerror("Video Error", f"An error occurred: {e}\n\n{traceback.format_exc()}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()