import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import webbrowser
from pathlib import Path

# Importa la funzione di analisi principale dallo script refattorizzato
from speed_script_10_events import run_analysis # Import the main analysis function from the refactored script

# --- Configurazione ---
# --- Configuration ---
# Defines the standard file names that the analysis script expects
REQUIRED_FILES = {
    "events.csv": "Select the events CSV file",
    "gaze.csv": "Select the gaze CSV file",
    "gaze_not_enr.csv": "Select the un-enriched gaze CSV file",
    "3d_eye_states.csv": "Select the 3D eye states CSV file (pupil)",
    "fixations.csv": "Select the fixations CSV file",
    "blinks.csv": "Select the blinks CSV file",
    "saccades.csv": "Select the saccades CSV file",
    "internal.mp4": "Select the internal video (eye)",
    "external.mp4": "Select the external video (scene)",
}

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v 0.2 - Laboratorio di Scienze Cognitive e del Comportamento")
        
        self.file_entries = {}
        
        # --- GUI Setup ---
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Participant Name
        name_frame = tk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=5)
        name_label = tk.Label(name_frame, text="Participant Name:", width=20, anchor='w')
        name_label.pack(side=tk.LEFT)
        self.name_entry = tk.Entry(name_frame)
        self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # File Selection
        files_frame = tk.LabelFrame(main_frame, text="Select Data Files", padx=5, pady=5)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        for i, (std_name, description) in enumerate(REQUIRED_FILES.items()):
            row_frame = tk.Frame(files_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            label = tk.Label(row_frame, text=f"{std_name}:", width=20, anchor='w')
            label.pack(side=tk.LEFT)
            
            entry = tk.Entry(row_frame)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.file_entries[std_name] = entry

            button = tk.Button(row_frame, text="Browse...", command=lambda e=entry, d=description: self.select_file(e, d))
            button.pack(side=tk.RIGHT)

        # Start Button
        run_button = tk.Button(main_frame, text="Start Analysis", command=self.run_analysis_process, font=('Helvetica', 10, 'bold'))
        run_button.pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(main_frame, text="", fg="blue")
        self.status_label.pack(pady=5)

        # Lab Website Link
        lab_link = tk.Label(main_frame, text="https://labscoc.wordpress.com/", fg="blue", cursor="hand2") # Lab website link
        lab_link.pack(side=tk.BOTTOM, pady=(5, 10))
        lab_link.bind("<Button-1>", lambda e: self.open_link("https://labscoc.wordpress.com/"))

    def select_file(self, entry_widget, description):
        file_path = filedialog.askopenfilename(title=description)
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)

    def open_link(self, url): # Opens the specified URL in the default web browser.
        webbrowser.open_new(url)

    def run_analysis_process(self):
        subj_name = self.name_entry.get().strip()
        if not subj_name:
            messagebox.showerror("Error", "Please enter a participant name.")
            return

        # Collect file paths
        selected_files = {std_name: entry.get().strip() for std_name, entry in self.file_entries.items()}

        # Check that all files have been selected
        if any(not path for path in selected_files.values()):
            messagebox.showerror("Error", "Please select all required files.")
            return

        try:
            self.status_label.config(text=f"Preparing folders for {subj_name}...", fg="blue")
            self.root.update_idletasks()

            # Define output and data folders
            base_output_dir = Path(f'./analysis_results_{subj_name}')
            data_dir = base_output_dir / 'eyetracking_file'
            data_dir.mkdir(parents=True, exist_ok=True)

            self.status_label.config(text="Copying files...")
            self.root.update_idletasks()

            # Copy selected files to the data folder with standard names
            for std_name, source_path_str in selected_files.items():
                shutil.copy(Path(source_path_str), data_dir / std_name)


            self.status_label.config(text="Avvio dell'analisi... L'operazione potrebbe richiedere tempo.")
            self.root.update_idletasks()
            
            # Esegue lo script di analisi
            run_analysis(subj_name=subj_name, data_dir_str=str(data_dir), output_dir_str=str(base_output_dir))
            
            self.status_label.config(text="Analysis complete!", fg="green")
            messagebox.showinfo("Success", f"Analysis for {subj_name} has finished.\nResults are located in '{base_output_dir}'.")

        except Exception as e:
            self.status_label.config(text="An error occurred.", fg="red")
            messagebox.showerror("Error", f"An error occurred during the process:\n{e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()
