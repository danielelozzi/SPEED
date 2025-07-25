import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import webbrowser
from pathlib import Path

# Import the main analysis function from the analysis script
from speed_script_events import run_analysis

# --- Configuration ---
REQUIRED_FILES = {
    "events.csv": "events.csv", "gaze_enriched.csv": "gaze_enriched.csv",
    "fixations_enriched.csv": "fixations_enriched.csv", "gaze.csv": "gaze.csv",
    "fixations.csv": "fixations.csv", "3d_eye_states.csv": "3d_eye_states.csv",
    "blinks.csv": "blinks.csv", "saccades.csv": "saccades.csv",
    "internal.mp4": "internal camera video", "external.mp4": "external camera video",
}
FILE_DESCRIPTIONS = {
    "events.csv": "Select the events CSV file", "gaze_enriched.csv": "Select the gaze CSV file (enriched)",
    "fixations_enriched.csv": "Select the enriched fixations CSV file (with surface data)", "gaze.csv": "Select the un-enriched gaze CSV file",
    "fixations.csv": "Select the un-enriched fixations CSV file", "3d_eye_states.csv": "Select the 3D eye states CSV file (pupil)",
    "blinks.csv": "Select the blinks CSV file", "saccades.csv": "Select the saccades CSV file",
    "internal.mp4": "Select the internal video (eye)", "external.mp4": "Select the external video (scene)",
}
OPTIONAL_FOR_UNENRICHED = ["gaze_enriched.csv", "fixations_enriched.csv"]

class SpeedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPEED v 1.0 - Cognitive and Behavioral Science Lab")
        self.file_entries = {}
        
        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Participant Name
        name_frame = tk.Frame(main_frame); name_frame.pack(fill=tk.X, pady=5)
        name_label = tk.Label(name_frame, text="Participant Name:", width=20, anchor='w'); name_label.pack(side=tk.LEFT)
        self.participant_name_var = tk.StringVar(); self.participant_name_var.trace_add("write", self.update_output_dir_default)
        self.name_entry = tk.Entry(name_frame, textvariable=self.participant_name_var); self.name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Output Directory
        output_frame = tk.Frame(main_frame); output_frame.pack(fill=tk.X, pady=5)
        output_label = tk.Label(output_frame, text="Output Folder:", width=20, anchor='w'); output_label.pack(side=tk.LEFT)
        self.output_dir_entry = tk.Entry(output_frame); self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        output_button = tk.Button(output_frame, text="Browse...", command=self.select_output_dir); output_button.pack(side=tk.RIGHT)

        # Analysis Options
        options_frame = tk.Frame(main_frame); options_frame.pack(fill=tk.X, pady=(5,0))
        self.unenriched_var = tk.BooleanVar()
        self.unenriched_checkbox = tk.Checkbutton(options_frame, text="Analyze un-enriched data only", variable=self.unenriched_var, command=self.toggle_file_requirements)
        self.unenriched_checkbox.pack(anchor='w')
        
        self.generate_video_var = tk.BooleanVar(value=True)
        self.video_checkbox = tk.Checkbutton(options_frame, text="Generate Analysis Video (slow process)", variable=self.generate_video_var)
        self.video_checkbox.pack(anchor='w')

        # File Selection
        files_frame = tk.LabelFrame(main_frame, text="Select Data Files", padx=5, pady=5)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        for std_name, display_label in REQUIRED_FILES.items():
            row_frame = tk.Frame(files_frame); row_frame.pack(fill=tk.X, pady=2)
            label = tk.Label(row_frame, text=f"{display_label}:", width=25, anchor='w'); label.pack(side=tk.LEFT)
            entry = tk.Entry(row_frame); entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.file_entries[std_name] = entry
            description = FILE_DESCRIPTIONS[std_name]
            button = tk.Button(row_frame, text="Browse...", command=lambda e=entry, d=description: self.select_file(e, d)); button.pack(side=tk.RIGHT)
        self.toggle_file_requirements()

        # Run Button & Status
        run_button = tk.Button(main_frame, text="Start Analysis", command=self.run_analysis_process, font=('Helvetica', 10, 'bold')); run_button.pack(pady=10)
        self.status_label = tk.Label(main_frame, text="", fg="blue"); self.status_label.pack(pady=5)
        lab_link = tk.Label(main_frame, text="https://labscoc.wordpress.com/", fg="blue", cursor="hand2"); lab_link.pack(side=tk.BOTTOM, pady=(5, 10))
        lab_link.bind("<Button-1>", lambda e: self.open_link("https://labscoc.wordpress.com/"))

        # GitHub Link 
        github_link = tk.Label(main_frame, text="https://github.com/danielelozzi/", fg="blue", cursor="hand2")
        github_link.pack(side=tk.BOTTOM, pady=(0, 10)) # Added pady to the bottom
        github_link.bind("<Button-1>", lambda e: self.open_link("https://github.com/danielelozzi/"))

    def update_output_dir_default(self, *args):
        subj_name = self.participant_name_var.get().strip()
        if subj_name:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, str(Path(f'./analysis_results_{subj_name}').resolve()))
        else:
            self.output_dir_entry.delete(0, tk.END)

    def select_output_dir(self):
        directory_path = filedialog.askdirectory(title="Select Output Folder")
        if directory_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, directory_path)

    def toggle_file_requirements(self):
        is_unenriched = self.unenriched_var.get()
        for std_name in OPTIONAL_FOR_UNENRICHED:
            entry_widget = self.file_entries[std_name]
            label_widget = entry_widget.master.winfo_children()[0]
            state = tk.DISABLED if is_unenriched else tk.NORMAL
            entry_widget.config(state=state); label_widget.config(state=state)
            if is_unenriched: entry_widget.delete(0, tk.END)

    def select_file(self, entry_widget, description):
        file_path = filedialog.askopenfilename(title=description)
        if file_path:
            entry_widget.delete(0, tk.END); entry_widget.insert(0, file_path)

    def open_link(self, url): webbrowser.open_new(url)

    def run_analysis_process(self):
        subj_name = self.name_entry.get().strip()
        output_dir_path = self.output_dir_entry.get().strip()
        if not subj_name or not output_dir_path:
            messagebox.showerror("Error", "Please enter a participant name and select an output folder."); return

        is_unenriched = self.unenriched_var.get()
        generate_video = self.generate_video_var.get()
        selected_files = {std_name: entry.get().strip() for std_name, entry in self.file_entries.items()}

        missing_files = [REQUIRED_FILES[name] for name, path in selected_files.items() if not path and not (is_unenriched and name in OPTIONAL_FOR_UNENRICHED)]
        if missing_files:
            messagebox.showerror("Error", f"Please select all required files. Missing: {', '.join(missing_files)}"); return

        try:
            self.status_label.config(text=f"Preparing folders for {subj_name}...", fg="blue"); self.root.update_idletasks()
            base_output_dir = Path(output_dir_path)
            data_dir = base_output_dir / 'eyetracking_file'
            data_dir.mkdir(parents=True, exist_ok=True)

            self.status_label.config(text="Copying files..."); self.root.update_idletasks()
            for std_name, source_path_str in selected_files.items():
                if source_path_str: shutil.copy(Path(source_path_str), data_dir / std_name)

            self.status_label.config(text="Starting analysis... This might take some time.", fg="blue"); self.root.update_idletasks()
            run_analysis(subj_name=subj_name, data_dir_str=str(data_dir), output_dir_str=str(base_output_dir), un_enriched_mode=is_unenriched, generate_video=generate_video)
            
            self.status_label.config(text="Analysis complete!", fg="green")
            messagebox.showinfo("Success", f"Analysis for {subj_name} has finished.\nResults are in '{base_output_dir}'.")
        except Exception as e:
            self.status_label.config(text="An error occurred.", fg="red")
            messagebox.showerror("Error", f"An error occurred during the process:\n{e}")
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    root = tk.Tk()
    app = SpeedApp(root)
    root.mainloop()