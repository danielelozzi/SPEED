import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import webbrowser
from pathlib import Path

# Import the main analysis function from the analysis script
from speed_script_events import run_analysis

# --- Configuration ---
# Defines the standard file names that the analysis script expects
REQUIRED_FILES = {
    "events.csv": "Select the events CSV file",
    "gaze_enriched.csv": "Select the gaze CSV file (enriched)",
    "fixations_enriched.csv": "Select the enriched fixations CSV file (with surface data)",
    "gaze.csv": "Select the un-enriched gaze CSV file",
    "fixations.csv": "Select the un-enriched fixations CSV file",
    "3d_eye_states.csv": "Select the 3D eye states CSV file (pupil)",
    "blinks.csv": "Select the blinks CSV file",
    "saccades.csv": "Select the saccades CSV file",
    "internal.mp4": "Select the internal video (eye)",
    "external.mp4": "Select the external video (scene)",
}

# Files that are optional when 'un-enriched data only' is selected
OPTIONAL_FOR_UNENRICHED = ["gaze_enriched.csv", "fixations_enriched.csv"] # Enriched files are optional

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v 0.4 - Cognitive and Behavioral Science Lab")
        
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

        # Un-enriched Data Checkbox
        self.unenriched_var = tk.BooleanVar()
        self.unenriched_checkbox = tk.Checkbutton(main_frame, text="Analyze un-enriched data only (gaze_enriched.csv and fixations_enriched.csv become optional)", variable=self.unenriched_var, command=self.toggle_file_requirements)
        self.unenriched_checkbox.pack(anchor='w', pady=5)

        # File Selection
        files_frame = tk.LabelFrame(main_frame, text="Select Data Files", padx=5, pady=5)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        for i, (std_name, description) in enumerate(REQUIRED_FILES.items()):
            row_frame = tk.Frame(files_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            label = tk.Label(row_frame, text=f"{std_name}:", width=25, anchor='w')
            label.pack(side=tk.LEFT)
            
            entry = tk.Entry(row_frame)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.file_entries[std_name] = entry

            button = tk.Button(row_frame, text="Browse...", command=lambda e=entry, d=description: self.select_file(e, d))
            button.pack(side=tk.RIGHT)
        
        self.toggle_file_requirements() # Set initial state

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

    def toggle_file_requirements(self):
        is_unenriched = self.unenriched_var.get()
        for std_name in OPTIONAL_FOR_UNENRICHED:
            entry_widget = self.file_entries[std_name]
            label_widget = entry_widget.master.winfo_children()[0] # Get the label in the same frame
            if is_unenriched:
                entry_widget.config(state=tk.DISABLED)
                label_widget.config(state=tk.DISABLED)
                entry_widget.delete(0, tk.END) # Clear content when disabled
            else:
                entry_widget.config(state=tk.NORMAL)
                label_widget.config(state=tk.NORMAL)


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

        is_unenriched = self.unenriched_var.get()
        selected_files = {std_name: entry.get().strip() for std_name, entry in self.file_entries.items()}

        # Check that all *required* files have been selected
        missing_files = []
        for std_name, path in selected_files.items():
            if not path:
                if is_unenriched and std_name in OPTIONAL_FOR_UNENRICHED:
                    continue # Skip if optional and un-enriched mode is active
                else:
                    missing_files.append(std_name)

        if missing_files:
            messagebox.showerror("Error", f"Please select all required files. Missing: {', '.join(missing_files)}")
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
                if source_path_str: # Only copy if a path was provided
                    shutil.copy(Path(source_path_str), data_dir / std_name)
                elif not is_unenriched and std_name in OPTIONAL_FOR_UNENRICHED:
                     pass # Already handled by the earlier check

            self.status_label.config(text="Starting analysis... This might take some time.", fg="blue")
            self.root.update_idletasks()
            
            # Pass the un_enriched_mode flag to the analysis script
            run_analysis(subj_name=subj_name, data_dir_str=str(data_dir), output_dir_str=str(base_output_dir), un_enriched_mode=is_unenriched)
            
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