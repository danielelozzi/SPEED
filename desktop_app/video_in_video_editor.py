# desktop_app/video_in_video_editor.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from pathlib import Path
import pandas as pd
import logging


class VideoInVideoEditor(tk.Toplevel):
    """
    A window for creating a "video-in-video" by mapping screen-recording
    videos to specific events from an eye-tracking session.
    """
    def __init__(self, parent, events_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Video-in-Video Event Table Editor")
        self.geometry("900x700")
        self.transient(parent)
        self.grab_set()

        if events_df.empty:
            messagebox.showerror("No Events", "No events loaded. Please load a dataset with events first.", parent=self)
            self.destroy()
            return

        # --- MODIFICA: Usa un DataFrame come sorgente dati ---
        self.viv_df = events_df.copy()
        if 'video_path' not in self.viv_df.columns:
            self.viv_df['video_path'] = ""
        if 'start_frame' not in self.viv_df.columns:
            self.viv_df['start_frame'] = pd.NA
        if 'end_frame' not in self.viv_df.columns:
            self.viv_df['end_frame'] = pd.NA

        # Converte le colonne opzionali in tipi numerici che supportano NA
        self.viv_df['start_frame'] = pd.to_numeric(self.viv_df['start_frame'], errors='coerce')
        self.viv_df['end_frame'] = pd.to_numeric(self.viv_df['end_frame'], errors='coerce')

        self.result = None

        # --- Main Frame ---
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Instructions and Controls ---
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(controls_frame, text="Map events to video files. Double-click cells to edit.", justify=tk.LEFT).pack(anchor='w')
        
        load_button = tk.Button(controls_frame, text="Load Mapping from CSV...", command=self.load_from_csv)
        load_button.pack(side=tk.LEFT, pady=5, padx=(0, 10))

        # --- Treeview for Mapping ---
        tree_frame = tk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        # --- MODIFICA: Aggiunta di nuove colonne ---
        cols = ("Event", "Timestamp (s)", "Video Path", "Start Frame", "End Frame")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        self.tree.heading("Event", text="Event Name")
        self.tree.heading("Timestamp (s)", text="Timestamp (s)")
        self.tree.heading("Video Path", text="Path to Video File")
        self.tree.heading("Start Frame", text="Start Frame (Opt.)")
        self.tree.heading("End Frame", text="End Frame (Opt.)")
        self.tree.column("Event", width=250)
        self.tree.column("Video Path", width=400)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Double-1>", self.on_double_click)

        # --- Bottom Buttons ---
        bottom_frame = tk.Frame(self, pady=10)
        bottom_frame.pack(fill=tk.X)

        self.generate_button = tk.Button(bottom_frame, text="Save Mapping & Proceed", command=self.save_and_close, font=('Helvetica', 10, 'bold'))
        self.generate_button.pack(side=tk.RIGHT, padx=10)
        tk.Button(bottom_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)

        self.populate_tree()

    def populate_tree(self):
        """Refreshes the treeview with the current mapping data."""
        self.tree.delete(*self.tree.get_children())
        self.viv_df.sort_values('timestamp [ns]', inplace=True)
        for index, row in self.viv_df.iterrows():
            values = (
                row['name'],
                f"{row['timestamp [ns]'] / 1e9:.4f}",
                row.get('video_path', ''),
                "" if pd.isna(row.get('start_frame')) else int(row['start_frame']),
                "" if pd.isna(row.get('end_frame')) else int(row['end_frame'])
            )
            self.tree.insert("", "end", iid=str(index), values=values)

    def load_from_csv(self):
        """Loads event-video mappings from a user-selected CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select CSV with 'event' and 'path' columns",
            filetypes=[("CSV files", "*.csv")],
            parent=self
        )
        if not filepath:
            return

        try:
            df = pd.read_csv(filepath)
            if 'name' not in df.columns or 'timestamp [ns]' not in df.columns:
                messagebox.showerror("Invalid CSV", "The CSV file must contain at least 'name' and 'timestamp [ns]' columns.", parent=self)
                return

            # Unisci i dati basandoti sul timestamp per aggiornare i percorsi video
            self.viv_df = pd.merge(self.viv_df.drop(columns=['video_path', 'start_frame', 'end_frame'], errors='ignore'),
                                   df[['timestamp [ns]', 'video_path', 'start_frame', 'end_frame']],
                                   on='timestamp [ns]',
                                   how='left')
            self.viv_df['video_path'].fillna('', inplace=True)
            self.viv_df['start_frame'] = pd.to_numeric(self.viv_df['start_frame'], errors='coerce')
            self.viv_df['end_frame'] = pd.to_numeric(self.viv_df['end_frame'], errors='coerce')
            self.populate_tree()
            messagebox.showinfo("Success", "Successfully loaded mappings from CSV.", parent=self)

        except Exception as e:
            logging.error(f"Failed to load video mapping CSV: {e}")
            messagebox.showerror("Error", f"An error occurred while reading the CSV:\n{e}", parent=self)

    def add_mapping_row(self):
        """Adds a new, empty row to the mapping for manual entry."""
        pass # Questa funzionalità è ora gestita dall'editor di eventi principale

    def on_double_click(self, event):
        """Handles double-clicking a row to edit it."""
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        item_id = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        df_index = int(item_id)

        if column_id == "#3":  # Video Path column
            self.edit_video_path(df_index)
        elif column_id in ["#4", "#5"]: # Start/End Frame columns
            self.edit_frame_value(df_index, column_id)

    def edit_video_path(self, df_index):
        """Opens a file dialog to select a new video path."""
        event_name = self.viv_df.loc[df_index, 'name']
        filepath = filedialog.askopenfilename(
            title=f"Select video for event: {event_name}",
            filetypes=[("Media Files", "*.mp4 *.avi *.mov *.png *.jpg *.jpeg"), 
                       ("Video Files", "*.mp4 *.avi *.mov"),
                       ("Image Files", "*.png *.jpg *.jpeg"),
                       ("All files", "*.*")],
            parent=self
        )
        if filepath:
            self.viv_df.loc[df_index, 'video_path'] = filepath
            self.populate_tree()

    def edit_frame_value(self, df_index, column_id):
        """Opens an entry box to edit start/end frame values."""
        col_map = {"#4": "start_frame", "#5": "end_frame"}
        col_name = col_map[column_id]

        x, y, width, height = self.tree.bbox(str(df_index), column_id)
        entry = ttk.Entry(self.tree)
        
        current_val = self.viv_df.loc[df_index, col_name]
        if not pd.isna(current_val):
            entry.insert(0, str(int(current_val)))

        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def save_edit(evt):
            new_value = entry.get().strip()
            try:
                if new_value == "":
                    self.viv_df.loc[df_index, col_name] = pd.NA
                else:
                    self.viv_df.loc[df_index, col_name] = int(new_value)
                self.populate_tree()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer for the frame number, or leave it empty.", parent=self)
            finally:
                entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def save_and_close(self):
        """Validates the mapping and closes the window."""
        # Rimuovi righe dove il video path non è stato impostato
        valid_df = self.viv_df[self.viv_df['video_path'] != ""].copy()

        if valid_df.empty:
            messagebox.showwarning("No Mapping", "Please map at least one event to a video file.", parent=self)
            return
        
        # Final validation
        for index, row in valid_df.iterrows():
            path = Path(row['video_path'])
            if not path.exists():
                messagebox.showwarning("Invalid Path", f"The path for event '{row['name']}' is invalid and will be skipped:\n{row['video_path']}", parent=self)
                valid_df.drop(index, inplace=True)
                continue

        if valid_df.empty:
            messagebox.showerror("Error", "No valid mappings were found. Please correct the paths.", parent=self)
            return

        self.result = valid_df
        self.destroy()