# SPEED (v0.2) - Eye-Tracking Data Analysis Software

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🎯 Overview

SPEED (v0.2) is a comprehensive eye-tracking data analysis software developed by the [Laboratorio di Scienze Cognitive e del Comportamento](https://labscoc.wordpress.com/). It provides a user-friendly Graphical User Interface (GUI) for processing raw eye-tracking data, extracting meaningful features, generating various visualizations, and creating an integrated analysis video.

The software is designed to streamline the analysis workflow for researchers working with eye-tracking data, particularly from Pupil Labs devices (though adaptable if data formats match).

## ✨ Features

* **Intuitive GUI:** Easy selection of required eye-tracking data files (CSV and MP4).
* **Participant Management:** Specify a participant name for organized output.
* **Automated Data Preparation:** Copies and renames input files to standard formats in a dedicated `eyetracking_file` directory.
* **Event-Based Analysis:** Processes data segmented by events defined in `events.csv`.
* **Feature Extraction:** Calculates key metrics for:
    * Fixations (number, duration, position)
    * Blinks (number, duration)
    * Pupillometry (start, end, average, std diameter)
    * Gaze Movements (number, duration, displacement)
* **Comprehensive Visualizations (PDF):**
    * Pupil diameter periodograms and spectrograms.
    * Histograms for gaze elevation, pupil diameter, fixation duration, blink duration, and saccade duration.
    * Gaze path and fixation path plots.
    * Heatmaps of fixation density.
    * Movement path plots.
* **Integrated Analysis Video (MP4):** Combines internal (eye) and external (scene) video feeds with a real-time pupil diameter time series plot.
* **Summary Results:** Aggregates all calculated features into a single `summary_results_<participant_name>.csv` file.
* **Error Handling:** Provides informative messages for missing files or processing errors.

## 📁 Required Input Files

The software expects a specific set of files, which will be copied into the `eyetracking_file` subdirectory within your output folder and renamed to the standard names if they differ:

| Standard Filename    | Description                                   | Format |
| :------------------- | :-------------------------------------------- | :----- |
| `events.csv`         | Eye-tracking events data                      | CSV    |
| `gaze.csv`           | Enriched gaze data                            | CSV    |
| `gaze_not_enr.csv`   | Un-enriched gaze data (for plotting)          | CSV    |
| `3d_eye_states.csv`  | 3D eye states data (pupil diameter)           | CSV    |
| `fixations.csv`      | Detected fixations data                       | CSV    |
| `blinks.csv`         | Detected blinks data                          | CSV    |
| `saccades.csv`       | Detected saccades data                        | CSV    |
| `internal.mp4`       | Video feed from the internal (eye) camera     | MP4    |
| `external.mp4`       | Video feed from the external (scene) camera   | MP4    |

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Required Python libraries (install via pip):
    ```bash
    pip install pandas numpy matplotlib opencv-python scipy pathlib
    ```

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/your-username/SPEED.git](https://github.com/your-username/SPEED.git)
    cd SPEED
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt # (Assuming you create a requirements.txt from the above list)
    ```

### Usage

1.  **Run the GUI:**
    ```bash
    python SPEED_0_2_gui.py
    ```
2.  **Enter Participant Name:** In the GUI, type the name of the participant (e.g., `subj_001`).
3.  **Select Files:** Click "Browse..." next to each required file type (`events.csv`, `gaze.csv`, etc.) and select the corresponding file from your system.
4.  **Start Analysis:** Click the "Start Analysis" button.

The GUI will show status updates. Once completed, a success message will appear.

## 📊 Output

All results will be saved in a new directory named `analysis_results_<participant_name>` (e.g., `analysis_results_subj_001`) in the same location where you run the script.

This directory will contain:

* `eyetracking_file/`: A copy of all input data files, renamed to their standard names.
* `summary_results_<participant_name>.csv`: A CSV file containing all aggregated summary features for each event.
* Individual PDF plots for each event (e.g., `periodogram_subj_001_Event1.pdf`, `hist_fixations_subj_001_Event2.pdf`).
* `output_analysis_video.mp4`: The combined video showing internal view, external view, and pupil diameter over time.

## 🛠️ Project Structure

├── SPEED_0_2_gui.py           # The Graphical User Interface application

├── speed_script_10_events.py  # The core eye-tracking data analysis logic

├── README.md                  # This documentation file

└── requirements.txt           # List of Python dependencies

## 🧠 Core Logic (`speed_script_10_events.py`)

This script contains the main algorithms for processing eye-tracking data.

### Key Functions:

* `load_all_data()`: Loads all necessary CSV files.
* `filter_data_by_event()`: Filters data for a specific event based on timestamps.
* `process_gaze_movements()`: Identifies and quantifies periods of gaze movement (non-fixations).
* `calculate_summary_features()`: Computes various statistical measures for fixations, blinks, pupil diameter, and movements.
* `generate_plots()`: Creates and saves various types of plots (histograms, paths, heatmaps, spectral analysis).
* `downsample_video()`: Utility function to reduce video frame rate.
* `create_analysis_video()`: Combines video feeds and pupil time series into an integrated output video.
* `run_analysis()`: The orchestrator function that drives the entire analysis pipeline, called by the GUI.

## Authors

* Daniele Lozzi
* Ilaria Di Pompeo
* Martina Marcaccio
* Matias Ademaj
* Simone Migliore
* Giuseppe Curcio

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details. (Note: You'll need to create a `LICENSE` file if you don't have one).

## 📞 Contact

For questions or support, please visit the [Laboratorio di Scienze Cognitive e del Comportamento website](https://labscoc.wordpress.com/).

---
*Developed by Laboratorio di Scienze Cognitive e del Comportamento*
