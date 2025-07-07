# SPEED (labScoc Processing and Extraction of Eye tracking Data - v0.4)

*EyeTracking Data Analysis Software*

-----

## 🎯 Overview

SPEED (v0.4) is a comprehensive eye-tracking data analysis software developed by the [Cognitive and Behavioral Science Lab](https://labscoc.wordpress.com/). It provides a user-friendly Graphical User Interface (GUI) for processing raw eye-tracking data, extracting meaningful features, generating various visualizations, and creating an integrated analysis video.

The software is designed to streamline the analysis workflow for researchers working with eye-tracking data, particularly from Pupil Labs devices (though adaptable if data formats match).

**New Features in v0.4:**
* **Clearer Filenames**: Standard filenames for gaze and fixation data have been updated to be more intuitive (`gaze_enriched.csv`, `gaze.csv`, etc.).
* **Full English Interface**: All user-facing messages, plots, and outputs are now in English for broader accessibility.

-----

## ✨ Features

* **Intuitive GUI:** Easy selection of required eye-tracking data files (CSV and MP4).
* **Participant Management:** Specify a participant name for organized output.
* **Automated Data Preparation:** Copies and renames input files to standard formats in a dedicated `eyetracking_file` directory.
* **Event-Based Analysis:** Processes data segmented by events defined in `events.csv`.
* **Un-Enriched Data Support:** An option in the GUI to indicate "un-enriched" data analysis. When selected, `gaze_enriched.csv` and `fixations_enriched.csv` become optional. The analysis adapts to calculate only the metrics possible with the available data.
* **Feature Extraction:** Calculates key metrics for:
    * Fixations (number, duration, position) - *Available only with enriched data*.
    * Blinks (number, duration).
    * Pupillometry (start, end, average, std diameter).
    * Gaze Movements (number, duration, displacement) - *Available only with enriched data*.
* **Comprehensive Visualizations (PDF):**
    * Pupil diameter periodograms and spectrograms.
    * Histograms for gaze elevation, pupil diameter, fixation duration, blink duration, and saccade duration.
    * Gaze path and fixation path plots.
    * Heatmaps of fixation density.
    * Movement path plots.
* **Integrated Analysis Video (MP4):** Combines internal (eye) and external (scene) video feeds with a real-time pupil diameter time series plot.
* **Summary Results:** Aggregates all calculated features into a single `summary_results_<participant_name>.csv` file.

-----

## 📁 Required Input Files

The software expects a specific set of files. The GUI will prompt you to select your data, which will then be copied into an `eyetracking_file` subdirectory and renamed to the standard names listed below.

| Standard Filename        | Description                                 | Format | Required (enriched mode) | Required (un-enriched mode) |
| :----------------------- | :------------------------------------------ | :----- | :----------------------- | :-------------------------- |
| `events.csv`             | Eye-tracking events data                    | CSV    | Yes                      | Yes                         |
| `gaze_enriched.csv`      | Enriched gaze data                          | CSV    | Yes                      | No                          |
| `fixations_enriched.csv` | Enriched fixations data (with surface info) | CSV    | Yes                      | No                          |
| `gaze.csv`               | Un-enriched gaze data                       | CSV    | Yes                      | Yes                         |
| `fixations.csv`          | Un-enriched fixations data                  | CSV    | Yes                      | Yes                         |
| `3d_eye_states.csv`      | 3D eye states data (pupil diameter)         | CSV    | Yes                      | Yes                         |
| `blinks.csv`             | Detected blinks data                        | CSV    | Yes                      | Yes                         |
| `saccades.csv`           | Detected saccades data                      | CSV    | Yes                      | Yes                         |
| `internal.mp4`           | Video feed from the internal (eye) camera   | MP4    | Yes                      | Yes                         |
| `external.mp4`           | Video feed from the external (scene) camera | MP4    | Yes                      | Yes                         |

-----

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Required Python libraries:
    ```bash
    pip install pandas numpy matplotlib opencv-python scipy pathlib
    ```

### Usage

1.  **Run the GUI:**
    ```bash
    python SPEED_0_4_gui.py
    ```
2.  **Enter Participant Name:** In the GUI, type the name of the participant (e.g., `subj_001`).
3.  **Select Analysis Mode:**
    * **"Analyze un-enriched data only"**: Select this checkbox to run the analysis with only un-enriched data. The fields for `gaze_enriched.csv` and `fixations_enriched.csv` will be disabled.
    * **Unchecked (default)**: For a full analysis with enriched data, leave this unchecked. You must provide all required files, including the enriched ones.
4.  **Select Files:** Click "Browse..." next to each required file type and select the corresponding file from your system.
5.  **Start Analysis:** Click the "Start Analysis" button.

The GUI will show status updates. Once completed, a success message will appear.

-----

## 📊 Output

All results will be saved in a new directory named `analysis_results_<participant_name>` (e.g., `analysis_results_subj_001`).

This directory will contain:

* `eyetracking_file/`: A copy of all input data files, renamed to their standard names.
* `summary_results_<participant_name>.csv`: A CSV file containing all aggregated summary features for each event.
* Individual PDF plots for each event (e.g., `periodogram_subj_001_Event1.pdf`, `hist_fixations_subj_001_Event2.pdf`).
* `output_analysis_video.mp4`: The combined video showing internal view, external view, and pupil diameter over time.

-----

## 🛠️ Project Structure