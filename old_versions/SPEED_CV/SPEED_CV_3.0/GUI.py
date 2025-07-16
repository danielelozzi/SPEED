# GUI.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import traceback
import json

# Importa le funzioni orchestratrici principali
try:
    import main_analyzer_advanced as main_analyzer
except ImportError:
    messagebox.showerror("Errore Critico", "File 'main_analyzer_advanced.py' non trovato.")
    exit()

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v3.1 - Folder-Based Input")
        self.root.geometry("720x800")
        
        # Variabili per i percorsi delle cartelle
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
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        main_frame = self.scrollable_frame 

        # --- Sezione Setup ---
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

        # --- NUOVA Sezione Cartelle di Input ---
        folders_frame = tk.LabelFrame(main_frame, text="2. Input Folders", padx=10, pady=10)
        folders_frame.pack(fill=tk.X, pady=5, padx=10)
        
        # Riga Cartella RAW
        raw_frame = tk.Frame(folders_frame); raw_frame.pack(fill=tk.X, pady=2)
        tk.Label(raw_frame, text="RAW Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        self.raw_dir_entry = tk.Entry(raw_frame, textvariable=self.raw_dir_var); self.raw_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(raw_frame, text="Browse...", command=lambda: self.select_folder(self.raw_dir_var, "Select RAW Data Folder")).pack(side=tk.RIGHT)
        
        # Riga Cartella Un-enriched
        unenriched_frame = tk.Frame(folders_frame); unenriched_frame.pack(fill=tk.X, pady=2)
        tk.Label(unenriched_frame, text="Un-enriched Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        self.unenriched_dir_entry = tk.Entry(unenriched_frame, textvariable=self.unenriched_dir_var); self.unenriched_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(unenriched_frame, text="Browse...", command=lambda: self.select_folder(self.unenriched_dir_var, "Select Un-enriched Data Folder")).pack(side=tk.RIGHT)

        # Riga Cartella Enriched
        enriched_frame = tk.Frame(folders_frame); enriched_frame.pack(fill=tk.X, pady=2)
        tk.Label(enriched_frame, text="Enriched Data Folder:", width=25, anchor='w').pack(side=tk.LEFT)
        self.enriched_dir_entry = tk.Entry(enriched_frame, textvariable=self.enriched_dir_var); self.enriched_dir_entry.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(0, 5))
        tk.Button(enriched_frame, text="Browse...", command=lambda: self.select_folder(self.enriched_dir_var, "Select Enriched Data Folder")).pack(side=tk.RIGHT)

        # --- Sezione Analisi Principale ---
        analysis_frame = tk.LabelFrame(main_frame, text="3. Run Core Analysis", padx=10, pady=10)
        analysis_frame.pack(fill=tk.X, pady=5, padx=10)
        self.unenriched_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Analyze un-enriched data only (ignores enriched folder)", variable=self.unenriched_var).pack(anchor='w')
        self.yolo_var = tk.BooleanVar(value=False)
        tk.Checkbutton(analysis_frame, text="Run YOLO Object Detection (requires GPU)", variable=self.yolo_var).pack(anchor='w')
        tk.Button(analysis_frame, text="RUN CORE ANALYSIS", command=self.run_core_analysis, font=('Helvetica', 10, 'bold'), bg='#c5e1a5').pack(pady=5)

        # --- Notebook per Output ---
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        plot_tab = tk.Frame(notebook); notebook.add(plot_tab, text='4. Generate Plots')
        video_tab = tk.Frame(notebook); notebook.add(video_tab, text='5. Generate Videos')

        self.setup_plot_tab(plot_tab)
        self.setup_video_tab(video_tab)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def setup_plot_tab(self, parent_tab):
        plot_options_frame = tk.LabelFrame(parent_tab, text="Plot Options", padx=10, pady=10)
        plot_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        plot_types = {
            "path_plots": "Path Plots (Fixation and Gaze)", "heatmaps": "Density Heatmaps (Fixation and Gaze)",
            "histograms": "Duration Histograms (Fixations, Blinks, Saccades)", "pupillometry": "Pupillometry (Time Series and Spectral Analysis)",
        }
        for key, text in plot_types.items():
            self.plot_vars[key] = tk.BooleanVar(value=True)
            tk.Checkbutton(plot_options_frame, text=text, variable=self.plot_vars[key]).pack(anchor='w')
        tk.Button(parent_tab, text="GENERATE SELECTED PLOTS", command=self.run_plot_generation, font=('Helvetica', 10, 'bold'), bg='#90caf9').pack(pady=10)

    def setup_video_tab(self, parent_tab):
        video_options_frame = tk.LabelFrame(parent_tab, text="Video Composition Options", padx=10, pady=10)
        video_options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_opts = {
            "crop_to_surface": "Crop video to marker-defined surface", "apply_perspective": "Apply perspective correction to surface",
            "overlay_yolo": "Overlay YOLO object detections", "overlay_gaze": "Overlay gaze point",
            "overlay_pupil_plot": "Overlay pupillometry plot", "include_internal_cam": "Include internal camera view (PiP)",
        }
        for key, text in video_opts.items():
            self.video_vars[key] = tk.BooleanVar(value=False)
            tk.Checkbutton(video_options_frame, text=text, variable=self.video_vars[key]).pack(anchor='w')
        tk.Label(video_options_frame, text="\nOutput Video Filename:").pack(anchor='w')
        self.video_filename_var = tk.StringVar(value="video_output_1.mp4")
        tk.Entry(video_options_frame, textvariable=self.video_filename_var).pack(fill=tk.X, pady=5)
        tk.Button(parent_tab, text="GENERATE VIDEO", command=self.run_video_generation, font=('Helvetica', 10, 'bold'), bg='#ef9a9a').pack(pady=10)

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
            messagebox.showerror("Errore", "Per favore, inserisci il nome del partecipante e la cartella di output.")
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
            messagebox.showerror("Errore", "Le cartelle RAW e Un-enriched sono obbligatorie.")
            return
        
        try:
            messagebox.showinfo("In corso", "Avvio dell'analisi principale. Questo potrebbe richiedere del tempo...")
            # Chiamata alla funzione aggiornata in main_analyzer
            main_analyzer.run_core_analysis(
                subj_name=subj_name, 
                output_dir_str=str(output_dir_path),
                raw_dir_str=raw_dir,
                unenriched_dir_str=unenriched_dir,
                enriched_dir_str=enriched_dir,
                un_enriched_mode=self.unenriched_var.get(),
                run_yolo=self.yolo_var.get()
            )
            messagebox.showinfo("Successo", "Analisi principale completata. Ora puoi generare grafici e video.")
        except Exception as e:
            messagebox.showerror("Errore di Analisi", f"Si è verificato un errore: {e}\n\n{traceback.format_exc()}")

    def run_plot_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths
        
        plot_selections = {key: var.get() for key, var in self.plot_vars.items()}
        try:
            messagebox.showinfo("In corso", "Generazione dei grafici selezionati...")
            main_analyzer.generate_selected_plots(output_dir_str=str(output_dir_path), subj_name=subj_name, plot_selections=plot_selections)
            messagebox.showinfo("Successo", f"Grafici generati in {output_dir_path / 'plots'}")
        except Exception as e:
            messagebox.showerror("Errore Grafici", f"Si è verificato un errore: {e}\n\n{traceback.format_exc()}")

    def run_video_generation(self):
        common_paths = self._get_common_paths()
        if not common_paths: return
        output_dir_path, subj_name = common_paths

        video_options = {key: var.get() for key, var in self.video_vars.items()}
        video_options['output_filename'] = self.video_filename_var.get().strip()
        if not video_options['output_filename']:
            messagebox.showerror("Errore", "Per favore, specifica un nome per il file video di output.")
            return
        try:
            messagebox.showinfo("In corso", "Generazione del video personalizzato. Questo processo può essere molto lento...")
            main_analyzer.generate_custom_video(output_dir_str=str(output_dir_path), subj_name=subj_name, video_options=video_options)
            messagebox.showinfo("Successo", f"Video salvato in {output_dir_path / video_options['output_filename']}")
        except Exception as e:
            messagebox.showerror("Errore Video", f"Si è verificato un errore: {e}\n\n{traceback.format_exc()}")

if __name__ == '__main__':
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()