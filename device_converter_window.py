# desktop_app/device_converter_window.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from src.speed_analyzer import convert_device_data


class DeviceConverterWindow(tk.Toplevel):
    """
    Una finestra per convertire i dati da dispositivi specifici (es. Tobii)
    in formati standard come BIDS o DICOM.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Device Data Converter")
        self.geometry("500x250")
        self.transient(parent)
        self.grab_set()

        self.source_folder_var = tk.StringVar()

        main_frame = tk.Frame(self, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Configurazione della griglia ---
        main_frame.columnconfigure(1, weight=1)

        # --- Selezione del Dispositivo ---
        tk.Label(main_frame, text="Device:").grid(row=0, column=0, sticky="w", pady=5)
        self.device_combo = ttk.Combobox(main_frame, values=["Tobii"], state="readonly")
        self.device_combo.grid(row=0, column=1, columnspan=2, sticky="ew", pady=5)
        self.device_combo.set("Tobii")

        # --- Cartella di Input ---
        tk.Label(main_frame, text="Input Folder:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(main_frame, textvariable=self.source_folder_var).grid(row=1, column=1, sticky="ew", pady=5)
        tk.Button(main_frame, text="Browse...", command=self.select_source_folder).grid(row=1, column=2, sticky="e", padx=(5, 0), pady=5)

        # --- Formato di Output ---
        tk.Label(main_frame, text="Output Format:").grid(row=2, column=0, sticky="w", pady=5)
        self.output_format_combo = ttk.Combobox(main_frame, values=["BIDS", "DICOM"], state="readonly")
        self.output_format_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=5)
        self.output_format_combo.set("BIDS")

        # --- Pulsante di Esecuzione ---
        run_button = tk.Button(main_frame, text="Run Conversion", command=self.run_conversion, font=('Helvetica', 10, 'bold'), bg='#FFD54F')
        run_button.grid(row=4, column=0, columnspan=3, pady=(20, 0), sticky="ew")

    def select_source_folder(self):
        """Apre una finestra di dialogo per selezionare la cartella di origine."""
        folder_path = filedialog.askdirectory(title="Select Source Data Folder")
        if folder_path:
            self.source_folder_var.set(folder_path)

    def run_conversion(self):
        """
        Esegue la conversione dei dati del dispositivo in base ai parametri selezionati dall'utente.
        """
        device = self.device_combo.get()
        input_folder = self.source_folder_var.get()
        output_format = self.output_format_combo.get()

        if not input_folder:
            messagebox.showwarning("Input Mancante", "Per favore, seleziona una cartella di input.", parent=self)
            return

        output_folder = filedialog.askdirectory(title="Select Destination Folder")
        if not output_folder:
            return  # L'utente ha annullato la selezione della cartella

        params = {}
        if output_format == "BIDS":
            params['subject_id'] = simpledialog.askstring("BIDS Info", "Enter Subject ID (e.g., sub-01):", parent=self)
            params['session_id'] = simpledialog.askstring("BIDS Info", "Enter Session ID (e.g., ses-01):", parent=self)
            params['task_name'] = simpledialog.askstring("BIDS Info", "Enter Task Name (e.g., task-mytask):", parent=self)
            if not all(params.values()):
                messagebox.showwarning("Input Mancante", "Tutti i campi BIDS sono obbligatori.", parent=self)
                return

        elif output_format == "DICOM":
            params['patient_name'] = simpledialog.askstring("DICOM Info", "Enter Patient Name:", parent=self)
            if not params['patient_name']:
                messagebox.showwarning("Input Mancante", "Il nome del paziente è obbligatorio.", parent=self)
                return

        try:
            convert_device_data(
                device=device,
                input_folder=input_folder,
                output_folder=output_folder,
                output_format=output_format,
                **params
            )
            messagebox.showinfo("Successo", f"Conversione in {output_format} completata con successo!", parent=self)
        except Exception as e:
            messagebox.showerror("Errore di Conversione", f"Si è verificato un errore: {e}", parent=self)
