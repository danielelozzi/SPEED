# SPEED (labScoc Processing and Extraction of Eye tracking Data - v0.5)

*Eye-Tracking Data Analysis Software*

-----

## 🎯 Overview

SPEED is a comprehensive software tool for analyzing eye-tracking data, developed by the [Cognitive and Behavioral Science Lab](https://labscoc.wordpress.com/). It provides an intuitive Graphical User Interface (GUI) to process raw eye-tracking data, calculate key metrics, and generate detailed visualizations.

The software is designed to streamline the analysis workflow for researchers. It intelligently handles both **enriched** (processed by Pupil Player with surface tracking) and **un-enriched** (raw export) data formats, providing a flexible and powerful analysis pipeline.

**New in This Version:**
* **Dual Analysis Mode:** The most significant new feature. If both enriched and un-enriched data files are provided, the software automatically performs two parallel analyses—one for each data type—allowing for direct comparison.
* **Targeted Output Naming:** All output files (CSV summaries and PDF plots) are now clearly labeled with `_enriched` or `_not_enriched` suffixes when running in Dual Analysis Mode.
* **Robust Data Handling:** The script is no longer dependent on rigid column names. It intelligently detects common variations (e.g., `duration` vs. `duration [ms]`, `fixation_id` vs. `fixation id`) to prevent errors.
* **Refined Path Plotting:** Path plots for fixations and gaze are now generated more reliably. The logic strictly follows the rule that un-enriched data uses pixel coordinates for plotting, while enriched data uses pre-normalized surface coordinates. Saccade path plots have been removed as requested.
* **Confirmed Surface Filtering:** The logic to filter enriched data for `fixation detected on surface == True` and `gaze detected on surface == True` is confirmed and central to the enriched analysis path.

-----

## ✨ Features

* **Intuitive GUI:** For easy selection of participant info, output directories, and data files.
* **Dual Analysis Mode:** Automatically generates a side-by-side comparative analysis of enriched vs. un-enriched data if all files are available.
* **Custom Output Directory:** Full control over where analysis results are saved.
* **Automated Data Preparation:** Copies and standardizes input files into a dedicated `eyetracking_file` directory for clean and repeatable analysis.
* **Event-Based Segmentation:** All metrics and plots are generated for time segments defined by `events.csv`.
* **Intelligent Un-Enriched Data Support:** Adheres to the rule that un-enriched data relies on pixel coordinates (e.g., `'fixation x [px]'`, `'gaze x [px]'`), which are then normalized using the scene video's dimensions.
* **Precise Enriched Data Analysis:** Uses pre-normalized data from enriched files and critically filters it to only include gaze and fixations that were successfully mapped to a surface (`... detected on surface == True`).
* **Path Visualizations:** Generates clear PDF plots for **fixation** and **gaze** paths for each event segment, with distinct outputs for enriched and un-enriched modes.
* **Summary Results:** Aggregates all calculated metrics into one or two clearly labeled CSV files (`summary_results_enriched_...` and/or `summary_results_not_enriched_...`).

-----

## 📁 Required Input Files

The software requires a specific set of files, which you select via the GUI.

| File               | Required (Enriched Mode) | Required (Un-enriched Only) | Notes                                                                   |
| :----------------- | :----------------------- | :-------------------------- | :---------------------------------------------------------------------- |
| `events.csv`       | **Yes** | **Yes** | Defines the start and end of analysis segments.                         |
| `fixations.csv`    | **Yes** | **Yes** | Base fixation data (used as fallback in enriched mode).                 |
| `gaze.csv`         | **Yes** | **Yes** | Base gaze data (used for un-enriched path plots).                       |
| `saccades.csv`     | **Yes** | **Yes** | Used for calculating movement statistics.                               |
| `blinks.csv`       | **Yes** | **Yes** | Blink event data.                                                       |
| `3d_eye_states.csv`| **Yes** | **Yes** | Pupillometry data.                                                      |
| `external.mp4`     | **Yes** | **Yes** | Scene camera video. **Crucial for normalizing pixel coordinates.** |
| `internal.mp4`     | **Yes** | **Yes** | Eye camera video (used for optional video generation).                  |
| `fixations_enriched.csv`| **Yes** | No                          | Enriched data with surface information. Triggers Dual Analysis Mode.    |
| `gaze_enriched.csv`| **Yes** | No                          | Enriched gaze data. Triggers Dual Analysis Mode.                        |

-----

## 🚀 Getting Started

1.  **Run the GUI:** `python SPEED_0_4_gui.py`
2.  **Enter Participant Name & Output Folder.**
3.  **Select Analysis Mode:**
    * **Full/Dual Analysis (Recommended):** Leave "Analyze un-enriched data only" **unchecked** and provide all files, including `_enriched.csv`.
    * **Un-enriched Only:** Check the "Analyze un-enriched data only" box. The `_enriched.csv` file fields will be disabled.
4.  **Select All Required Files** using the "Browse..." buttons.
5.  **Click "Start Analysis".**

-----

## 📊 Output

When running in **Dual Analysis Mode**, you will get two sets of outputs in your chosen folder, clearly labeled:

* **Enriched Outputs:**
    * `summary_results_enriched_SUBJECT.csv`: Summary statistics calculated from enriched data.
    * `path_fixations_enriched_SUBJECT_EVENT.pdf`: Fixation paths from enriched data (`...on surface == True`).
    * `path_gaze_enriched_SUBJECT_EVENT.pdf`: Gaze paths from enriched data (`...on surface == True`).
* **Un-enriched Outputs:**
    * `summary_results_not_enriched_SUBJECT.csv`: Summary statistics calculated from base data.
    * `path_fixations_not_enriched_SUBJECT_EVENT.pdf`: Fixation paths created from pixel coordinates in `fixations.csv`.
    * `path_gaze_not_enriched_SUBJECT_EVENT.pdf`: Gaze paths created from pixel coordinates in `gaze.csv`.

If running in "un-enriched only" mode, only the second set of files (without the `_not_enriched` suffix) will be generated.

-----

## 🧠 Core Logic (`speed_script_events.py`)

* `run_analysis()`: Orchestrates the entire process. It now checks if a **Dual Analysis** should be performed and calls the processing functions accordingly for both enriched and un-enriched passes.
* `calculate_summary_features()`: A robust function that dynamically checks for multiple possible column names (e.g., `duration` vs `duration [ms]`). It correctly uses either enriched or un-enriched data sources based on the analysis pass it's running.
* `generate_path_plots()`: This function has been rewritten for clarity and correctness. It strictly adheres to the rule of using pixel coordinates for un-enriched data and pre-normalized surface coordinates for enriched data. It no longer plots saccade paths.
* `process_segment()`: The core function that is called for each event segment. It manages the data filtering and calls the calculation and plotting functions for the specified mode (`_enriched` or `_not_enriched`).

-----

## ✍️ Authors & Citation

* Dr. Daniele Lozzi
* Dr. Ilaria Di Pompeo
* Martina Marcaccio
* Matias Ademaj
* Dr. Simone Migliore
* Prof. Giuseppe Curcio


*If you use this script in your research or work, please cite the following publications:*

Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Ademaj, M.; Migliore, S.; Curcio, G. SPEED: A Graphical User Interface Software for Processing Eye Tracking Data. NeuroSci 2025, 6, 35. [https://doi.org/10.3390/neurosci6020035](https://doi.org/10.3390/neurosci6020035)

Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Alemanno, M.; Krüger, M.; Curcio, G.; Migliore, S. AI-Powered Analysis of Eye Tracker Data in Basketball Game. Sensors 2025, 25, 3572. [https://doi.org/10.3390/s25113572](https://doi.org/10.3390/s25113572)