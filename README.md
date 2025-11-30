# SPEED-light

## Description

This project provides a graphical user interface (GUI) for processing and analyzing eye-tracking data from Pupil Labs devices. It allows users to load recording data, segment it based on events, calculate various metrics (related to fixations, blinks, pupil diameter, and saccades), generate plots, and create an overlay video with gaze points and event information.

## How It Works

The application uses Tkinter to create a user-friendly interface. Here's a breakdown of the main components and how they interact:

1.  **Data Preparation**:
    *   The user selects a `Data` folder (from the Pupil Labs recording) and an optional `Enrichment` folder.
    *   The application creates a unified `files` directory, intelligently merging files from both sources. For example, it prioritizes `gaze.csv` and `fixations.csv` from the enrichment folder if available.

2.  **Event Editing (Optional)**:
    *   A simple event editor allows the user to review the recording, add new event markers, or remove existing ones. This helps in refining the data segmentation.

3.  **Analysis & Output Generation**:
    *   The script segments the data based on the events defined in `events.csv`.
    *   For each segment, it calculates 16 different eye-tracking metrics (e.g., number of fixations, mean pupil diameter).
    *   It generates PDF plots for each event, including gaze/fixation heatmaps and pupillometry timeseries.
    *   It produces a final summary video (`final_video_overlay.mp4`) that shows the original recording overlaid with gaze points, gaze path, active events, and a live pupil diameter chart.
    *   All calculated metrics are saved to an Excel file (`Speed_Lite_Results.xlsx`).

### Data Structure

The application is designed to work with the data structure produced by Pupil Labs recording software. It primarily requires the following folders and files:

*   **Data Folder**: The main folder containing the raw recording data (e.g., `gaze.csv`, `fixations.csv`, `blinks.csv`, `world_timestamps.csv`, and the `external.mp4` video).
*   **Enrichment Folder (Optional)**: A folder containing enriched or post-processed data, such as a corrected `gaze.csv` or `fixations.csv`.

## Usage

### Prerequisites

Before you begin, ensure you have **Git** and **Anaconda** installed on your system.

*   **Git**: Puoi scaricarlo e installarlo da git-scm.com.
*   **Anaconda**: You can download and install the Anaconda Distribution (for Python 3.x) from anaconda.com/products/distribution.

`pip` is included with the Anaconda distribution, so a separate installation is not necessary.

### Installation

1.  **Clone the repository**
    Open a terminal (or Anaconda Prompt on Windows) and clone this repository to your local machine using the following command:
    ```bash
    git clone https://github.com/your-username/SPEED-light.git
    cd SPEED-light
    ```
    *(Replace `your-username` with the correct repository path if needed)*

2.  **Create and activate the Anaconda environment**
    Create a new virtual environment for this project to manage dependencies in isolation.
    ```bash
    conda create --name speedlight python=3.9
    ```
    Activate the new environment:
    ```bash
    conda activate speedlight
    ```

3.  **Install dependencies**
    With the environment activated, install the required Python libraries listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Running the Application**:
    *   Execute the `gui.py` script from your terminal:
        ```bash
        python gui.py
        ```
    *   The GUI window will appear.
    *   Use the "Browse" buttons to select the `Data`, `Enrichment` (optional), and `Output` folders.
    *   Click "1. Load and Prepare Data" to initialize the process.
    *   (Optional) Click "2. Edit Events" to open the event editor.
    *   Click "3. Extract Features, Plots & Video" to run the full analysis pipeline.

## Code Overview

The `gui.py` script contains the entire implementation. Key parts include:

*   **Tkinter Setup**: Creates the main application window and all GUI elements.
*   **Data Preparation**: `prepare_working_directory` function handles the logic for merging data sources.
*   **Metrics Calculation**: `calculate_metrics` computes the 16 features for each data segment.
*   **Plotting**: Functions like `generate_heatmap_pdf` and `generate_pupil_timeseries_pdf` create the visual outputs.
*   **Video Generation**: `generate_full_video` uses OpenCV to render the final video with overlays.
*   **Event Editor**: The `LiteEventEditor` class provides an interactive way to manage event markers.

## Additional Information

*   The script is designed to be easily customizable. You can modify the analysis parameters or add new metrics as needed.
*   The analysis runs in a separate thread to keep the GUI responsive.

## Citations
 
If you use this script in your research or work, please cite the following publications:

- Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Ademaj, M.; Migliore, S.; Curcio, G. SPEED: A Graphical User Interface Software for Processing Eye Tracking Data. NeuroSci 2025, 6, 35. [10.3390/neurosci6020035](https://doi.org/10.3390/neurosci6020035)

It is also requested to cite Pupil Labs publication, as requested on their website [https://docs.pupil-labs.com/neon/data-collection/publications-and-citation/](https://docs.pupil-labs.com/neon/data-collection/publications-and-citation/)

- Baumann, C., & Dierkes, K. (2023). Neon accuracy test report. Pupil Labs, 10. [10.5281/zenodo.10420388](https://doi.org/10.5281/zenodo.10420388)

---

## 💻 Artificial Intelligence disclosure

This code is written in Vibe Coding with Google Gemini Pro