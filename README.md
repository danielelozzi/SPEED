# SPEED (labScoc Processing and Extraction of Eye tracking Data - v0.4)

*Eye-Tracking Data Analysis Software*

-----

## 🎯 Overview

SPEED (v0.4) is a comprehensive eye-tracking data analysis software developed by the [Cognitive and Behavioral Science Lab](https://labscoc.wordpress.com/). It provides a user-friendly Graphical User Interface (GUI) for processing raw eye-tracking data, extracting meaningful features, generating various visualizations, and creating an integrated analysis video.

The software is designed to streamline the analysis workflow for researchers working with eye-tracking data, particularly from Pupil Labs devices (though adaptable if data formats match).

**New Features in v0.4:**
* **Clearer Filenames**: Standard filenames for gaze and fixation data have been updated to be more intuitive (`gaze_enriched.csv`, `gaze.csv`, etc.).
* **Full English Interface**: All user-facing messages, plots, and outputs are now in English for broader accessibility.
* **Custom Output Folder**: The GUI now allows users to specify a custom output directory for analysis results.
* **Improved GUI Labels**: Video input labels are now more descriptive ("internal camera video", "external camera video").
* **Enhanced Plots**: All histograms now include axis labels with units and have an improved aesthetic for better readability.
* **Adaptive Movement Analysis**: In "un-enriched" mode, the software uses `saccades.csv` to calculate movement metrics, providing a baseline analysis even without enriched data.
* **Fixation Coordinate Normalization**: Automatically converts fixation coordinates from pixels to a normalized format (0-1) using the scene video's (`external.mp4`) dimensions, if available.

-----

## ✨ Features

* **Intuitive GUI:** Easy selection of required eye-tracking data files (CSV and MP4).
* **Participant Management:** Specify a participant name for organized output.
* **Custom Output Directory:** Choose where to save the analysis results, with a smart default based on the participant's name.
* **Automated Data Preparation:** Copies and renames input files to standard formats in a dedicated `eyetracking_file` directory within your chosen output folder.
* **Event-Based Analysis:** Processes data segmented by events defined in `events.csv`.
* **Un-Enriched Data Support:** An option in the GUI to indicate "un-enriched" data analysis. When selected, `gaze_enriched.csv` and `fixations_enriched.csv` become optional. The analysis adapts to calculate only the metrics possible with the available data.
* **Feature Extraction:** Calculates key metrics for:
    * **Fixations**: Number, duration, and normalized position (x, y).
    * **Blinks**: Number and duration.
    * **Pupillometry**: Start, end, average, and standard deviation of pupil diameter.
    * **Gaze Movements**: Number, duration, and displacement. Calculated from enriched gaze data if available, otherwise estimated from saccade data.
* **Comprehensive Visualizations (PDF):** The software calculates the necessary metrics to generate periodograms, spectrograms, histograms (e.g., for fixation/saccade duration), gaze paths, and heatmaps.
* **Integrated Analysis Video (MP4):** The GUI includes an option to generate a video combining the internal (eye) and external (scene) video feeds with a real-time plot of the pupil diameter time series. 
* **Summary Results:** Aggregates all calculated features into a single `summary_results_<participant_name>.csv` file.

-----

## 📁 Required Input Files

The software expects a specific set of files. The GUI will prompt you to select your data files using descriptive labels. These files will then be copied into an `eyetracking_file` subdirectory (inside your chosen output folder) and renamed to the standard names listed below, which the analysis script uses internally.

| Standard Filename        | GUI Prompt Label          | Description                                 | Format | Required (enriched mode) | Required (un-enriched mode) |
| :----------------------- | :------------------------ | :------------------------------------------ | :----- | :----------------------- | :-------------------------- |
| `events.csv`             | `events.csv`              | Eye-tracking events data                    | CSV    | Yes                      | Yes                         |
| `gaze_enriched.csv`      | `gaze_enriched.csv`       | Enriched gaze data                          | CSV    | Yes                      | No                          |
| `fixations_enriched.csv` | `fixations_enriched.csv`  | Enriched fixations data (with surface info) | CSV    | Yes                      | No                          |
| `gaze.csv`               | `gaze.csv`                | Un-enriched gaze data                       | CSV    | Yes                      | Yes                         |
| `fixations.csv`          | `fixations.csv`           | Un-enriched fixations data                  | CSV    | Yes                      | Yes                         |
| `3d_eye_states.csv`      | `3d_eye_states.csv`       | 3D eye states data (pupil diameter)         | CSV    | Yes                      | Yes                         |
| `blinks.csv`             | `blinks.csv`              | Detected blinks data                        | CSV    | Yes                      | Yes                         |
| `saccades.csv`           | `saccades.csv`            | Detected saccades data                      | CSV    | Yes                      | Yes                         |
| `internal.mp4`           | `internal camera video`   | Video feed from the internal (eye) camera   | MP4    | Yes                      | Yes                         |
| `external.mp4`           | `external camera video`   | Video feed from the external (scene) camera | MP4    | Yes                      | Yes                         |

-----

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Required Python libraries:
    ```bash
    pip install pandas numpy matplotlib opencv-python scipy
    ```

### Usage

1.  **Run the GUI:**
    ```bash
    python SPEED_0_4_gui.py
    ```
2.  **Enter Participant Name:** Type the name of the participant (e.g., `subj_001`).
3.  **Select Output Folder:** An output folder path will be automatically suggested (e.g., `./analysis_results_subj_001`). You can keep it or click "Browse..." to choose a different location.
4.  **Select Analysis Mode:**
    * **"Analyze un-enriched data only"**: Select this checkbox to run the analysis with only un-enriched data. The fields for `gaze_enriched.csv` and `fixations_enriched.csv` will be disabled.
    * **Unchecked (default)**: For a full analysis with enriched data, leave this unchecked. You must provide all required files, including the enriched ones.
5.  **Select Files:** Click "Browse..." next to each required file type and select the corresponding file from your system.
6.  **Start Analysis:** Click the "Start Analysis" button.

The GUI will show status updates. Once completed, a success message will appear.

-----

## 📊 Output

All results will be saved in the output directory you specified (e.g., `analysis_results_subj_001`).

This directory will contain:

* `eyetracking_file/`: A copy of all input data files, renamed to their standard names.
* `summary_results_<participant_name>.csv`: A CSV file containing all aggregated summary features for each event.
* (Potentially) Individual PDF plots for each event (e.g., `periodogram_subj_001_Event1.pdf`), if their generation is enabled in the script.
* (Potentially) `output_analysis_video.mp4`: The combined video showing the internal view, external view, and pupil diameter over time, if the feature is implemented.

-----

## 🛠️ Project Structure

.
├── SPEED_0_4_gui.py           # The Graphical User Interface application
├── speed_script_events.py     # The core eye-tracking data analysis logic
└── README.md                  # This documentation file


-----

## 🧠 Core Logic (`speed_script_events.py`)

This script contains the main algorithms for processing eye-tracking data.

### Key Functions:

* `run_analysis()`: The main function that drives the entire analysis pipeline, called by the GUI.
* `load_all_data()`: Loads all necessary CSV files, adapting based on `un_enriched_mode`.
* `get_video_dimensions()`: Extracts width and height from the `external.mp4` file to use for coordinate normalization.
* `filter_data_by_segment()`: Filters data for a specific time segment based on events. This function was named `filter_data_by_event` in a previous version.
* `process_gaze_movements()`: Identifies and quantifies periods of gaze movement (non-fixations). *Skipped in `un_enriched_mode`*.
* `calculate_summary_features()`: Computes various statistical measures, adapting its logic based on `un_enriched_mode`. If enriched data is unavailable, it uses `saccades.csv` for movement metrics. It normalizes fixation coordinates if they are in pixels and video dimensions are known.
* `generate_plots()`: Would create and save all plots, selectively generating them based on data availability.
* `create_analysis_video()`: Would combine video feeds and the pupil time series into an integrated output video.

-----

## ✍️ Authors

* Dr. Daniele Lozzi
* Dr. Ilaria Di Pompeo
* Martina Marcaccio
* Matias Ademaj
* Dr. Simone Migliore
* Prof. Giuseppe Curcio

-----

## 📞 Contact

For questions or support, please visit the [Cognitive and Behavioral Science Lab website](https://labscoc.wordpress.com/).

-----

## ✍️ How to Cite

*If you use this script in your research or work, please cite the following publications:*

Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Ademaj, M.; Migliore, S.; Curcio, G. SPEED: A Graphical User Interface Software for Processing Eye Tracking Data. NeuroSci 2025, 6, 35. [https://doi.org/10.3390/neurosci6020035](https://doi.org/10.3390/neurosci6020035)

Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Alemanno, M.; Krüger, M.; Curcio, G.; Migliore, S. AI-Powered Analysis of Eye Tracker Data in Basketball Game. Sensors 2025, 25, 3572. [https://doi.org/10.3390/s25113572](https://doi.org/10.3390/s25113572)