import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector

class DataPlotterWindow(tk.Toplevel):
    """
    Una finestra interattiva per visualizzare i dati CSV dell'eye-tracking come serie temporali,
    con strumenti per l'analisi statistica su intervalli selezionati.
    """
    def __init__(self, parent, data_dir: Path):
        super().__init__(parent)
        self.title("CSV Data Plotter")
        self.geometry("1400x900")
        self.data_dir = data_dir
        self.dataframes = {}
        self.time_series_data = {}
        self.max_time = 0

        # --- Layout Principale ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pannello laterale per le statistiche
        stats_panel = tk.Frame(main_frame, width=300, relief=tk.SUNKEN, borderwidth=1)
        stats_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        stats_panel.pack_propagate(False)
        
        tk.Label(stats_panel, text="Statistiche sulla Selezione", font=('Helvetica', 12, 'bold')).pack(pady=10)
        self.stats_text = tk.Text(stats_panel, wrap="word", height=20, width=35)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Area per i grafici
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Caricamento Dati ---
        if not self._load_data():
            self.destroy()
            return
            
        self._prepare_time_series_data()

        # --- Creazione Grafici Matplotlib ---
        self.fig = plt.figure(figsize=(12, 10), dpi=100)
        self.axs = self.fig.subplots(5, 1, sharex=True, gridspec_kw={'height_ratios': [2, 2, 2, 2, 1]})
        self.fig.subplots_adjust(hspace=0.4) # Aggiunge spazio tra i subplot

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Aggiunta della toolbar per zoom/pan
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- NUOVO: Connessione dell'evento di scroll del mouse per lo zoom ---
        self.fig.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)
        
        # --- NUOVO: Scrollbar orizzontale ---
        self.scrollbar = ttk.Scrollbar(plot_frame, orient=tk.HORIZONTAL, command=self.on_scroll)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.axs[0].callbacks.connect('xlim_changed', self.on_xlim_changed)
        
        # --- NUOVO: Selettore rettangolare per le statistiche ---
        self.selector = RectangleSelector(
            self.axs[0], self.on_select,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True,
            props=dict(facecolor='red', edgecolor='red', alpha=0.2, fill=True)
        )
        self.selection_patches = []

        self._plot_data() # Disegna i dati
        self.setup_view() # Imposta la vista iniziale e la scrollbar

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_xlim_changed(self, ax):
        """Callback per quando l'utente zooma/pan, aggiorna la scrollbar."""
        # Le statistiche ora sono gestite dal selettore
        self.update_scrollbar()

    def on_close(self):
        plt.close(self.fig)
        self.destroy()

    def on_mouse_scroll(self, event):
        """Callback per lo zoom con la rotellina del mouse."""
        ax = event.inaxes
        if ax is None:
            return

        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min

        # Se il tasto Shift è premuto, esegui lo scorrimento (pan)
        if event.key == 'shift':
            # Fattore di scorrimento (più è piccolo, più veloce è lo scroll)
            pan_factor = 0.1
            pan_amount = x_range * pan_factor

            if event.button == 'up': # Scroll verso l'alto/destra
                new_x_min = x_min - pan_amount
                new_x_max = x_max - pan_amount
            else: # Scroll verso il basso/sinistra
                new_x_min = x_min + pan_amount
                new_x_max = x_max + pan_amount
        else: # Altrimenti, esegui lo zoom
            # Fattore di zoom
            zoom_factor = 1.1 if event.button == 'up' else 1 / 1.1
            
            # Posizione del cursore come punto focale dello zoom
            x_cursor = event.xdata
            
            # Calcola i nuovi limiti per lo zoom
            new_x_min = x_cursor - (x_cursor - x_min) / zoom_factor
            new_x_max = x_cursor + (x_max - x_cursor) / zoom_factor

        # Applica i nuovi limiti
        ax.set_xlim(new_x_min, new_x_max)
        
        self.canvas.draw_idle()

    def _load_data(self):
        """Carica tutti i file CSV necessari dalla cartella di output."""
        if not self.data_dir.exists():
            messagebox.showerror("Errore", f"La cartella dati specificata non esiste:\n{self.data_dir}", parent=self)
            return False

        # --- NUOVO: Logica per gestire sia le cartelle di output che quelle un-enriched ---
        # Controlla se esiste una sottocartella di workspace (tipica dell'output di analisi)
        workspace_dir = self.data_dir / 'SPEED_workspace'
        if workspace_dir.is_dir():
            data_source_dir = workspace_dir
        else:
            # Altrimenti, usa la cartella fornita direttamente (caso un-enriched)
            data_source_dir = self.data_dir

        files_to_load = {
            'pupil': '3d_eye_states.csv', 'gaze': 'gaze.csv', 'fixations': 'fixations.csv',
            'saccades': 'saccades.csv', 'blinks': 'blinks.csv', 'events': 'events.csv'
        }
        
        for name, filename in files_to_load.items():
            try:
                self.dataframes[name] = pd.read_csv(data_source_dir / filename)
            except FileNotFoundError:
                messagebox.showwarning("File Mancante", f"File '{filename}' non trovato. Il grafico corrispondente potrebbe essere vuoto.", parent=self)
                self.dataframes[name] = pd.DataFrame()
        return True

    def _prepare_time_series_data(self):
        """Converte i timestamp in secondi e prepara i dati per il plotting."""
        # Trova il timestamp di inizio globale per sincronizzare l'asse X
        min_ts = float('inf')
        for df in self.dataframes.values():
            for col in ['timestamp [ns]', 'start timestamp [ns]']:
                if col in df.columns and not df.empty:
                    min_ts = min(min_ts, df[col].min())
        
        if min_ts == float('inf'):
            min_ts = 0

        # Prepara ogni DataFrame con una colonna 'time_sec'
        for name, df in self.dataframes.items():
            if df.empty: continue
            
            ts_col = 'timestamp [ns]' if 'timestamp [ns]' in df.columns else 'start timestamp [ns]'
            if ts_col in df.columns:
                df['time_sec'] = (df[ts_col] - min_ts) / 1e9
                self.max_time = max(self.max_time, df['time_sec'].max())
                self.time_series_data[name] = df.sort_values('time_sec').reset_index(drop=True)

    def _plot_data(self):
        """Disegna i dati sui subplot."""
        ax_pupil, ax_gaze, ax_fix, ax_sacc, ax_events = self.axs
        
        def draw_shared_markers(ax):
            """Disegna marcatori di eventi e blink su un dato asse."""
            if 'events' in self.time_series_data:
                for _, row in self.time_series_data['events'].iterrows():
                    ax.axvline(x=row['time_sec'], color='red', linestyle='--', linewidth=1)
            
            if 'blinks' in self.time_series_data:
                df_blinks = self.time_series_data['blinks']
                if df_blinks.empty or 'time_sec' not in df_blinks.columns: return

                for _, row in self.time_series_data['blinks'].iterrows():
                    start_sec = row['time_sec']
                    end_sec = start_sec + (row['duration [ms]'] / 1000.0)
                    ax.axvspan(start_sec, end_sec, color='grey', alpha=0.3, zorder=0)

        # 1. Pupillometria
        if 'pupil' in self.time_series_data:
            df = self.time_series_data['pupil']
            if 'pupil diameter left [mm]' in df.columns:
                ax_pupil.plot(df['time_sec'], df['pupil diameter left [mm]'], color='blue', label='Diam. Sinistro', alpha=0.8)
            if 'pupil diameter right [mm]' in df.columns:
                ax_pupil.plot(df['time_sec'], df['pupil diameter right [mm]'], color='orange', label='Diam. Destro', alpha=0.8)
        ax_pupil.set_title('Pupillometria')
        ax_pupil.set_ylabel('Diametro [mm]')
        ax_pupil.legend(loc='upper right')
        draw_shared_markers(ax_pupil)

        # 2. Posizione Sguardo (Gaze)
        # Questo grafico ora cerca colonne specifiche per occhio sinistro e destro.
        # Se non le trova, non disegnerà nulla.
        if 'gaze' in self.time_series_data:
            df = self.time_series_data['gaze']
            plotted_gaze = False
            # Occhio Sinistro
            if 'gaze x left [px]' in df.columns:
                ax_gaze.plot(df['time_sec'], df['gaze x left [px]'], color='green', label='Gaze X (sx)', alpha=0.7)
                plotted_gaze = True
            if 'gaze y left [px]' in df.columns:
                ax_gaze.plot(df['time_sec'], df['gaze y left [px]'], color='darkgreen', label='Gaze Y (sx)', linestyle='--', alpha=0.7)
                plotted_gaze = True
            # Occhio Destro
            if 'gaze x right [px]' in df.columns:
                ax_gaze.plot(df['time_sec'], df['gaze x right [px]'], color='purple', label='Gaze X (dx)', alpha=0.7)
                plotted_gaze = True
            if 'gaze y right [px]' in df.columns:
                ax_gaze.plot(df['time_sec'], df['gaze y right [px]'], color='darkmagenta', label='Gaze Y (dx)', linestyle='--', alpha=0.7)
                plotted_gaze = True
            # Fallback a colonne generiche
            if not plotted_gaze:
                if 'gaze x [px]' in df.columns:
                    ax_gaze.plot(df['time_sec'], df['gaze x [px]'], color='green', label='Gaze X', alpha=0.8)
                if 'gaze y [px]' in df.columns:
                    ax_gaze.plot(df['time_sec'], df['gaze y [px]'], color='purple', label='Gaze Y', alpha=0.8)

        ax_gaze.set_title('Posizione Sguardo (Gaze)')
        ax_gaze.set_ylabel('Posizione [px]')
        ax_gaze.legend(loc='upper right')
        draw_shared_markers(ax_gaze)

        # 3. Fissazioni
        # Anche qui, cerchiamo dati specifici per occhio.
        if 'fixations' in self.time_series_data:
            df = self.time_series_data['fixations']
            plotted_fix = False
            for _, row in df.iterrows():
                start_time = row['time_sec']
                end_time = start_time + (row['duration [ms]'] / 1000.0)
                # Occhio Sinistro
                if 'fixation x left [px]' in row:
                    ax_fix.plot([start_time, end_time], [row['fixation x left [px]'], row['fixation x left [px]']], color='cyan', alpha=0.7)
                    plotted_fix = True
                if 'fixation y left [px]' in row:
                    ax_fix.plot([start_time, end_time], [row['fixation y left [px]'], row['fixation y left [px]']], color='deepskyblue', alpha=0.7)
                    plotted_fix = True
                # Occhio Destro
                if 'fixation x right [px]' in row:
                    ax_fix.plot([start_time, end_time], [row['fixation x right [px]'], row['fixation x right [px]']], color='magenta', alpha=0.7)
                    plotted_fix = True
                if 'fixation y right [px]' in row:
                    ax_fix.plot([start_time, end_time], [row['fixation y right [px]'], row['fixation y right [px]']], color='orchid', alpha=0.7)
                    plotted_fix = True
                # Fallback a colonne generiche
                if not plotted_fix:
                    if 'fixation x [px]' in row:
                        ax_fix.plot([start_time, end_time], [row['fixation x [px]'], row['fixation x [px]']], color='cyan', alpha=0.7)
                    if 'fixation y [px]' in row:
                        ax_fix.plot([start_time, end_time], [row['fixation y [px]'], row['fixation y [px]']], color='magenta', alpha=0.7)
            
            # Legenda
            ax_fix.plot([], [], color='cyan', label='Fiss. X (sx/gen)')
            ax_fix.plot([], [], color='deepskyblue', label='Fiss. Y (sx)')
            ax_fix.plot([], [], color='magenta', label='Fiss. X (dx/gen)')
            ax_fix.plot([], [], color='orchid', label='Fiss. Y (dx)')
        ax_fix.set_title('Fissazioni')
        ax_fix.set_ylabel('Posizione [px]')
        ax_fix.legend(loc='upper right')
        draw_shared_markers(ax_fix)

        # 4. Saccadi
        # Sdoppiato per occhio sinistro e destro.
        if 'saccades' in self.time_series_data:
            df = self.time_series_data['saccades']
            plotted_sacc = False
            # Occhio Sinistro
            if 'amplitude left [px]' in df.columns:
                ax_sacc.plot(df['time_sec'], df['amplitude left [px]'], color='red', label='Amp. (sx)', alpha=0.7, marker='o', linestyle='None')
                plotted_sacc = True
            # Occhio Destro
            if 'amplitude right [px]' in df.columns:
                ax_sacc.plot(df['time_sec'], df['amplitude right [px]'], color='orangered', label='Amp. (dx)', alpha=0.7, marker='^', linestyle='None')
                plotted_sacc = True
            # Fallback
            if not plotted_sacc and 'amplitude [px]' in df.columns:
                ax_sacc.plot(df['time_sec'], df['amplitude [px]'], color='red', label='Ampiezza', alpha=0.7, marker='o', linestyle='None')
            
            ax_sacc_twin = ax_sacc.twinx()
            if 'peak velocity left [px/s]' in df.columns:
                ax_sacc_twin.plot(df['time_sec'], df['peak velocity left [px/s]'], color='gold', label='Vel. Picco (sx)', alpha=0.7, marker='x', linestyle='None')
                plotted_sacc = True
            if 'peak velocity right [px/s]' in df.columns:
                ax_sacc_twin.plot(df['time_sec'], df['peak velocity right [px/s]'], color='goldenrod', label='Vel. Picco (dx)', alpha=0.7, marker='+', linestyle='None')
                plotted_sacc = True
            # Fallback
            if not plotted_sacc and 'peak velocity [px/s]' in df.columns:
                ax_sacc_twin.plot(df['time_sec'], df['peak velocity [px/s]'], color='gold', label='Velocità Picco', alpha=0.7, marker='x', linestyle='None')

            ax_sacc_twin.set_ylabel('Velocità [px/s]')
            # Combina le legende dei due assi Y
            lines, labels = ax_sacc.get_legend_handles_labels()
            lines2, labels2 = ax_sacc_twin.get_legend_handles_labels()
            ax_sacc_twin.legend(lines + lines2, labels + labels2, loc='upper right')

        ax_sacc.set_title('Saccadi')
        ax_sacc.set_ylabel('Ampiezza [px]')
        draw_shared_markers(ax_sacc)

        # 5. Grafico Eventi e Blink
        ax_events.set_title('Trigger (Eventi e Blink)')
        if 'events' in self.time_series_data:
            y_pos_alternator = 1.3 # Alterna l'altezza del testo per evitare sovrapposizioni
            df_events = self.time_series_data['events']
            ax_events.plot(df_events['time_sec'], [1] * len(df_events), color='red', marker='|', markersize=20, linestyle='None', label='Eventi')
            for _, row in df_events.iterrows():
                ax_events.text(row['time_sec'], y_pos_alternator, row['name'], rotation=45, ha='left', va='bottom', fontsize=9)
                y_pos_alternator = 1.6 if y_pos_alternator == 1.3 else 1.3 # Alterna altezza

        if 'blinks' in self.time_series_data:
            df_blinks = self.time_series_data['blinks']
            if not df_blinks.empty and 'time_sec' in df_blinks.columns:
                for _, row in df_blinks.iterrows():
                    start_sec = row['time_sec']
                    end_sec = start_sec + (row['duration [ms]'] / 1000.0)
                    ax_events.axvspan(start_sec, end_sec, color='grey', alpha=0.5, ymin=0.25, ymax=0.75, zorder=0)
        
        # Aggiungi una legenda fittizia per axvspan
        ax_events.bar(0, 0, color='grey', alpha=0.5, label='Blink')
        
        ax_events.set_ylim(0, 2)
        ax_events.get_yaxis().set_visible(False) # Nasconde l'asse Y e le sue etichette
        ax_events.legend(loc='upper right')
        ax_events.set_xlabel('Tempo (s)')

        self.canvas.draw()

    def on_select(self, eclick, erelease):
        """Callback per quando l'utente seleziona un'area con il mouse."""
        t_start, t_end = sorted([eclick.xdata, erelease.xdata])
        
        # Rimuovi le selezioni precedenti
        for patch in self.selection_patches:
            patch.remove()
        self.selection_patches.clear()

        # Disegna la nuova area di selezione su tutti i grafici
        for ax in self.axs:
            patch = ax.axvspan(t_start, t_end, color='red', alpha=0.2, zorder=-1)
            self.selection_patches.append(patch)
        
        self.update_stats_for_range(t_start, t_end)
        self.canvas.draw_idle()

    def setup_view(self):
        """Imposta la vista iniziale e la scrollbar."""
        initial_view_seconds = 60.0
        self.axs[0].set_xlim(0, min(initial_view_seconds, self.max_time))
        self.update_scrollbar()
        self.canvas.draw()

    def on_scroll(self, *args):
        """Callback per lo spostamento della scrollbar."""
        if args[0] == 'moveto':
            pos = float(args[1])
            current_xlim = self.axs[0].get_xlim()
            view_width = current_xlim[1] - current_xlim[0]
            new_start = pos * (self.max_time - view_width)
            self.axs[0].set_xlim(new_start, new_start + view_width)
            self.canvas.draw_idle()

    def update_scrollbar(self):
        """Aggiorna la posizione e la dimensione della scrollbar in base allo zoom."""
        current_xlim = self.axs[0].get_xlim()
        view_width = current_xlim[1] - current_xlim[0]
        low = max(0, current_xlim[0] / (self.max_time - view_width if self.max_time > view_width else 1))
        high = min(1, (current_xlim[0] + view_width) / (self.max_time if self.max_time > 0 else 1))
        self.scrollbar.set(low, high)

    def update_stats_for_range(self, t_start, t_end):
        stats_report = f"Statistiche per la selezione [{t_start:.2f}s - {t_end:.2f}s]:\n"
        stats_report += "="*35 + "\n"

        def calculate_and_format_stats(data_series, name):
            if data_series.empty:
                return f"** {name} **\nNessun dato in questo intervallo.\n\n"
            
            stats = {
                "Media": data_series.mean(),
                "Mediana": data_series.median(),
                "Dev. Standard": data_series.std(),
                "Curtosi": kurtosis(data_series, nan_policy='omit'),
                "Asimmetria": skew(data_series, nan_policy='omit')
            }
            report = f"** {name} **\n"
            for key, value in stats.items():
                report += f"{key}: {value:.4f}\n"
            return report + "\n"

        # Pupillometria
        if 'pupil' in self.time_series_data:
            df = self.time_series_data['pupil']
            selected_pupil = df[(df['time_sec'] >= t_start) & (df['time_sec'] <= t_end)]
            if 'pupil diameter left [mm]' in selected_pupil.columns:
                stats_report += calculate_and_format_stats(selected_pupil['pupil diameter left [mm]'].dropna(), "Diam. Pupilla Sinistra")
            if 'pupil diameter right [mm]' in selected_pupil.columns:
                stats_report += calculate_and_format_stats(selected_pupil['pupil diameter right [mm]'].dropna(), "Diam. Pupilla Destra")

        # Fixations
        if 'fixations' in self.time_series_data:
            df = self.time_series_data['fixations']
            selected_fix = df[(df['time_sec'] >= t_start) & (df['time_sec'] <= t_end)]
            stats_report += calculate_and_format_stats(selected_fix['duration [ms]'].dropna(), "Durata Fissazione")

        # Saccades
        if 'saccades' in self.time_series_data:
            df_sacc = self.time_series_data['saccades']
            selected_sacc = df_sacc[(df_sacc['time_sec'] >= t_start) & (df_sacc['time_sec'] <= t_end)]
            stats_report += calculate_and_format_stats(selected_sacc['amplitude [px]'].dropna(), "Ampiezza Saccade")
            stats_report += calculate_and_format_stats(selected_sacc['duration [ms]'].dropna(), "Durata Saccade")
            stats_report += calculate_and_format_stats(selected_sacc['peak velocity [px/s]'].dropna(), "Velocità Picco Saccade")

        self.stats_text.delete(1.0, tk.END)
        
        # Configura i tag per il grassetto
        self.stats_text.tag_configure("bold", font=('Helvetica', 10, 'bold'))
        
        # Inserisci il testo e applica i tag
        lines = stats_report.split('\n')
        for line in lines:
            if line.startswith("**") and line.endswith("**"):
                self.stats_text.insert(tk.END, line.strip('*') + '\n', "bold")
            else:
                self.stats_text.insert(tk.END, line + '\n')