import tkinter as tk
from tkinter import filedialog
import os
from speed_script_10_events import run_script  # Importa solo la funzione run_script dallo script speed_script.py

def select_folder():
    folder_path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def run_speed_script():
    folder_path = folder_entry.get()
    subj_name = name_entry.get()
    if folder_path and subj_name:
        print(folder_path,subj_name)
        run_script(folder_path, subj_name)  # Passa folder_path e subj_name alla funzione run_script
    else:
        print("Please select a folder and enter participant name.")

# Creazione della GUI
root = tk.Tk()
root.title("SPEED v 0.1, Laboratorio di Scienze Cognitive e del Comportamento")

# Etichetta e campo di inserimento per la cartella
folder_label = tk.Label(root, text="Select Folder:")
folder_label.grid(row=0, column=0, padx=5, pady=5)
folder_entry = tk.Entry(root, width=50)
folder_entry.grid(row=0, column=1, padx=5, pady=5)
browse_button = tk.Button(root, text="Browse", command=select_folder)
browse_button.grid(row=0, column=2, padx=5, pady=5)

# Etichetta e campo di inserimento per il nome del partecipante
name_label = tk.Label(root, text="Participant Name:")
name_label.grid(row=1, column=0, padx=5, pady=5)
name_entry = tk.Entry(root, width=50)
name_entry.grid(row=1, column=1, padx=5, pady=5)

# Pulsante per eseguire lo script
run_button = tk.Button(root, text="Run Script", command=run_script)
run_button.grid(row=2, column=1, padx=5, pady=5)

root.mainloop()

#%%
