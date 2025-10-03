# desktop_app/lsl_time_series_viewer.py
import tkinter as tk
from tkinter import messagebox
import threading
from collections import deque
import numpy as np

try:
    from pylsl import resolve_streams, StreamInlet
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    resolve_streams, StreamInlet, plt, FigureCanvasTkAgg = None, None, None, None

class LSLTimeSeriesViewer(tk.Toplevel):
    """
    Una finestra per visualizzare stream LSL di serie temporali in tempo reale.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.title("LSL Real-time Time Series Viewer")
        self.geometry("1200x800")

        if not plt:
            messagebox.showerror("Errore", "Matplotlib o pylsl non sono installati. Impossibile avviare il viewer.", parent=self)
            self.destroy()
            return

        self.is_streaming = False
        self.stream_threads = []
        self.inlets = {}
        self.data_buffers = {}
        self.plot_lines = {}
        self.event_lines = []
        self.time_window_sec = 10

        # --- Layout ---
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        self.stream_listbox = tk.Listbox(top_frame, selectmode=tk.MULTIPLE, height=5)
        self.stream_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.refresh_streams()

        btn_frame = tk.Frame(top_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)
        self.toggle_stream_btn = tk.Button(btn_frame, text="Start Streaming", command=self.toggle_streaming, font=('Helvetica', 10, 'bold'), bg='#a5d6a7')
        self.toggle_stream_btn.pack(fill=tk.X)
        tk.Button(btn_frame, text="Refresh List", command=self.refresh_streams).pack(fill=tk.X, pady=5)

        # --- Grafico Matplotlib ---
        self.fig, self.ax = plt.subplots()
        self.fig.tight_layout(pad=3)
        self.ax.set_title("LSL Real-time Data")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def refresh_streams(self):
        self.stream_listbox.delete(0, tk.END)
        try:
            self.available_streams = resolve_streams(wait_time=1.0)
            if not self.available_streams:
                self.stream_listbox.insert(tk.END, "No streams found on the network.")
            else:
                for i, stream in enumerate(self.available_streams):
                    self.stream_listbox.insert(tk.END, f"{i}: {stream.name()} ({stream.type()}, {stream.channel_count()} ch)")
        except Exception as e:
            messagebox.showerror("LSL Error", f"Could not resolve streams: {e}", parent=self)

    def toggle_streaming(self):
        if self.is_streaming:
            self.stop_streaming()
        else:
            self.start_streaming()

    def start_streaming(self):
        selected_indices = self.stream_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one stream to view.", parent=self)
            return

        self.ax.clear()
        self.ax.set_title("LSL Real-time Data")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Value")
        self.plot_lines.clear()
        self.data_buffers.clear()

        for i in selected_indices:
            stream_info = self.available_streams[i]
            inlet = StreamInlet(stream_info)
            self.inlets[stream_info.uid()] = inlet

            if stream_info.type().lower() == 'markers':
                self.data_buffers[stream_info.uid()] = deque()
            else:
                self.data_buffers[stream_info.uid()] = [deque(maxlen=int(stream_info.nominal_srate() * self.time_window_sec)) for _ in range(stream_info.channel_count())]
                self.data_buffers[f"{stream_info.uid()}_ts"] = deque(maxlen=int(stream_info.nominal_srate() * self.time_window_sec))
                
                for ch_idx in range(stream_info.channel_count()):
                    line, = self.ax.plot([], [], label=f"{stream_info.name()}_ch{ch_idx}")
                    self.plot_lines[f"{stream_info.uid()}_{ch_idx}"] = line

            thread = threading.Thread(target=self.stream_reader_thread, args=(inlet,), daemon=True)
            self.stream_threads.append(thread)
            thread.start()

        self.ax.legend(loc='upper left')
        self.is_streaming = True
        self.toggle_stream_btn.config(text="Stop Streaming", bg='#ffcdd2')
        self.animation = self.canvas.new_timer(interval=100, callbacks=[(self.update_plot, (), {})])
        self.animation.start()

    def stop_streaming(self):
        self.is_streaming = False
        if hasattr(self, 'animation'):
            self.animation.stop()
        self.stream_threads.clear()
        self.inlets.clear()
        self.toggle_stream_btn.config(text="Start Streaming", bg='#a5d6a7')

    def stream_reader_thread(self, inlet: StreamInlet):
        uid = inlet.info().uid()
        is_marker_stream = inlet.info().type().lower() == 'markers'
        while self.is_streaming:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                if is_marker_stream:
                    self.data_buffers[uid].append((timestamp, sample[0]))
                else:
                    self.data_buffers[f"{uid}_ts"].append(timestamp)
                    for ch_idx, value in enumerate(sample):
                        self.data_buffers[uid][ch_idx].append(value)

    def update_plot(self):
        if not self.is_streaming: return

        for uid, inlet in self.inlets.items():
            if inlet.info().type().lower() == 'markers':
                while self.data_buffers[uid]:
                    ts, label = self.data_buffers[uid].popleft()
                    line = self.ax.axvline(ts, color='red', linestyle='--', lw=1)
                    text = self.ax.text(ts, self.ax.get_ylim()[1], label, rotation=90, verticalalignment='top')
                    self.event_lines.append((line, text))
            else:
                timestamps = np.array(self.data_buffers[f"{uid}_ts"])
                for ch_idx in range(inlet.info().channel_count()):
                    line = self.plot_lines.get(f"{uid}_{ch_idx}")
                    if line:
                        line.set_data(timestamps, self.data_buffers[uid][ch_idx])
        
        all_ts = [ts for key, buf in self.data_buffers.items() if key.endswith('_ts') for ts in buf]
        if all_ts:
            min_t, max_t = min(all_ts), max(all_ts)
            self.ax.set_xlim(max(min_t, max_t - self.time_window_sec), max_t + 1)
            self.ax.relim()
            self.ax.autoscale_view(scalex=False, scaley=True)

            visible_xlim = self.ax.get_xlim()
            self.event_lines = [(line, text) for line, text in self.event_lines if visible_xlim[0] <= line.get_xdata()[0] <= visible_xlim[1]]

        self.canvas.draw()

    def on_close(self):
        self.stop_streaming()
        self.destroy()