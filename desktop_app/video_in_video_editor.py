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
    THIS EDITOR ONLY CREATES A MAPPING; IT DOES NOT MODIFY THE EVENTS THEMSELVES.
    """
    def __init__(self, parent, events_df: pd.DataFrame):
        super().__init__(parent)
        self.title("Video-in-Video Event to Media Mapper")
        self.geometry("900x700")
        self.transient(parent)
        self.grab_set()

        if events_df.empty:
            messagebox.showerror("No Events", "No events loaded. Please load a dataset with events first.", parent=self)
            self.destroy()
            return

        # --- MODIFIED: Use a DataFrame as the data source for mapping ---
        # This DataFrame is a copy and will not affect the original events_df
        self.viv_df = events_df.copy()
        if 'video_path' not in self.viv_df.columns:
            self.viv_df['video_path'] = ""       

        self.result = None

        # --- Main Frame ---
        main_frame = tk.Frame(self, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Instructions and Controls ---
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(controls_frame, text="Map events to video/image files. Double-click a cell in the 'Path' column to select a file.", justify=tk.LEFT).pack(anchor='w')
        
        load_button = tk.Button(controls_frame, text="Load Mapping from CSV...", command=self.load_from_csv)
        load_button.pack(side=tk.LEFT, pady=5, padx=(0, 10))

        # --- Treeview for Mapping ---
        tree_frame = tk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("Event Name", "Timestamp (s)", "Path to Media File")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        self.tree.heading("Event Name", text="Event Name")
        self.tree.heading("Timestamp (s)", text="Timestamp (s)")
        self.tree.heading("Path to Media File", text="Path to Media File")
        self.tree.column("Event Name", width=250)
        self.tree.column("Timestamp (s)", width=120, anchor=tk.CENTER)
        self.tree.column("Path to Media File", width=400)

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
                row.get('video_path', '')
            )
            self.tree.insert("", "end", iid=str(index), values=values)

    def load_from_csv(self):
        """Loads event-video mappings from a user-selected CSV file."""
        filepath = filedialog.askopenfilename(
            title="Select CSV with mapping data",
            filetypes=[("CSV files", "*.csv")],
            parent=self
        )
        if not filepath:
            return

        try:
            df = pd.read_csv(filepath)
            # Essential columns for mapping
            required_cols = ['timestamp [ns]', 'video_path']
            if not all(col in df.columns for col in required_cols):
                messagebox.showerror("Invalid CSV", "The CSV file must contain at least 'timestamp [ns]' and 'video_path' columns.", parent=self)
                return

            # Merge based on timestamp, updating only the mapping columns
            merge_cols = ['timestamp [ns]', 'video_path']
            # Drop old mapping columns and merge new ones
            self.viv_df = self.viv_df.drop(columns=['video_path'], errors='ignore')
            self.viv_df = pd.merge(self.viv_df, df[merge_cols], on='timestamp [ns]', how='left')
            
            # Clean up merged data
            self.viv_df['video_path'].fillna('', inplace=True)

            self.populate_tree()
            messagebox.showinfo("Success", "Successfully loaded mappings from CSV.", parent=self)

        except Exception as e:
            logging.error(f"Failed to load video mapping CSV: {e}")
            messagebox.showerror("Error", f"An error occurred while reading the CSV:\n{e}", parent=self)

    def on_double_click(self, event):
        """Handles double-clicking a cell to edit it."""
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        item_id = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        df_index = int(item_id)

        # Allow editing only for the 'Path to Media File' column
        if column_id == "#3":
            self.edit_video_path(df_index)
        else:
            pass

    def edit_video_path(self, df_index):
        """Opens a file dialog to select a new video or image path."""
        event_name = self.viv_df.loc[df_index, 'name']
        filepath = filedialog.askopenfilename(
            title=f"Select Media for Event: {event_name}",
            filetypes=[("Media Files", "*.mp4 *.avi *.mov *.png *.jpg *.jpeg"), 
                       ("Video Files", "*.mp4 *.avi *.mov"),
                       ("Image Files", "*.png *.jpg *.jpeg"),
                       ("All files", "*.*")],
            parent=self
        )
        if filepath:
            self.viv_df.loc[df_index, 'video_path'] = filepath
            self.populate_tree()

    def save_and_close(self):
        """Validates the mapping and closes the window, returning the mapping DataFrame."""
        # Return the entire DataFrame.
        # Rows without a mapping will have an empty 'video_path'.
        # The main application will handle validation.
        self.result = self.viv_df
        self.destroy()