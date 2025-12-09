# SPEED-light - Software Processing and Extraction of Eye tracking Data by LabSCoC (University of L'Aquila, Italy)

## Description

This project provides a graphical user interface (GUI) for processing and analyzing eye-tracking data from Pupil Labs devices. It allows users to load recording data, segment it based on events, calculate various metrics, generate plots, and create an overlay video. A key feature is its **AI-powered Region of Interest (ROI) tracking**, which uses YOLO for object detection and SAM (Segment Anything Model) for defining and tracking custom objects.

## How It Works

The application uses Tkinter to create a user-friendly interface. Here's a breakdown of the main components and how they interact:

1.  **Data Preparation**:
    *   The user selects a `Data` folder (from the Pupil Labs recording) and one or more optional `Manual Enrichment` folders.
    *   The application creates a unified `files` directory in the output folder, intelligently merging files from all sources. It distinguishes between raw data (e.g., `gaze_raw.csv`) and data from enrichment folders (e.g., `gaze_enr_0.csv`).

2.  **AI-Powered ROI Definition (Optional)**:
    *   The application integrates **YOLOv8** for object detection and tracking and **SAM** for precise segmentation.
    *   Users can define ROIs in multiple ways:
        *   **YOLO Class Tracking**: Add a standard object class (e.g., "person", "laptop") to track all instances of that object throughout the video.
        *   **Interactive AI Window**: Play the video with live YOLO detections and simply click on the objects you want to track.
        *   **Custom Object Definer (SAM)**: For objects not in the YOLO model, pause the video on any frame, click on the desired object, and SAM will segment it, allowing it to be tracked.

3.  **Advanced Event Editing (Optional)**:
    *   An advanced event editor allows the user to review the recording frame-by-frame.
    *   It displays the video alongside a table of events, enabling users to **add, delete, rename, move, or merge** event markers with precision.

4.  **Analysis & Output Generation**:
    *   The script segments the data based on the events defined in `events.csv`.
    *   For each segment, it calculates various eye-tracking metrics.
    *   If AI ROIs are defined, it runs the tracking and calculates specific metrics for each tracked object (e.g., number of fixations on the object, mean pupil diameter while looking at the object).
    *   It produces a final summary video (`final_video.mp4`) that shows the original recording overlaid with raw gaze points, gaze path, and the polygons of all tracked AI ROIs.
    *   All calculated metrics are saved to an Excel file (`Speed_Lite_Results.xlsx`).
    *   A detailed analysis for each AI-tracked object is saved in a separate file (`ROI_Analysis_Results.xlsx`).

### Data Structure

The application is designed to work with the data structure produced by Pupil Labs recording software. It primarily requires the following folders and files:

*   **Data Folder**: The main folder containing the raw recording data (e.g., `gaze.csv`, `fixations.csv`, `blinks.csv`, `world_timestamps.csv`, and the `external.mp4` video).
*   **Manual Enrichment Folder (Optional)**: A folder containing enriched or post-processed data, such as a corrected `gaze.csv` or `fixations.csv`.

## Usage

### Prerequisites

Before you begin, ensure you have **Anaconda** installed on your system.

*   **Anaconda**: You can download and install the Anaconda Distribution (for Python 3.x) from anaconda.com/products/distribution.

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
    conda install pip
    conda install git
    ```

3.  **Install dependencies**
    With the environment activated, install the required Python libraries listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    **Note for AI Features**: The AI functionalities depend on PyTorch and Ultralytics. The `requirements.txt` file includes `ultralytics`, which will attempt to install the necessary PyTorch version. If you encounter issues, especially with GPU support, you may need to install PyTorch manually by following the instructions on the official PyTorch website.
    ```bash
    # Example for CUDA 12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Running the Application**:
    *   Execute the `gui.py` script from your terminal:
        ```bash
        python gui.py
        ```
    *   The GUI window will appear.
    *   Use the "Browse" buttons to select the `Data`, `Enrichment` (optional), and `Output` folders.
    *   (Optional) Configure AI ROIs using the "AI ROI Tracker" section.
    *   Click "1. Load Data" to validate paths.
    *   (Optional) Click "2. Edit Events" to open the advanced event editor.
    *   Click "3. RUN FULL ANALYSIS" to run the complete pipeline, including AI tracking, metrics calculation, and video generation.

## Code Overview

The `gui.py` script contains the entire implementation. Key parts include:

*   **Tkinter Setup**: Creates the main application window and all GUI elements.
*   **AI Engine**: The `run_ai_extraction` function manages YOLO and SAM models to track objects and generate surface data.
*   **Interactive AI Windows**: The `AIInteractiveWindow` and `CustomObjectDefiner` classes provide intuitive ways to define ROIs.
*   **Data Preparation**: `prepare_working_directory` handles the logic for merging data from multiple sources.
*   **Metrics Calculation**: `calculate_metrics` computes features for each data segment and for each ROI.
*   **Video Generation**: `generate_full_video_layered` uses OpenCV to render the final video with multiple data overlays.
*   **Advanced Event Editor**: The `AdvancedEventEditor` class provides a comprehensive interface for managing event markers.

## Additional Information

*   The script is designed to be easily customizable. You can modify the analysis parameters or add new metrics as needed.
*   The analysis runs in a separate thread to keep the GUI responsive.

## Citations
 
If you use this script in your research or work, please cite the following publications:

- Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Ademaj, M.; Migliore, S.; Curcio, G. SPEED: A Graphical User Interface Software for Processing Eye Tracking Data. NeuroSci 2025, 6, 35. [10.3390/neurosci6020035](https://doi.org/10.3390/neurosci6020035)
- Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Alemanno, M.; Krüger, M.; Curcio, G.; Migliore, S. AI-Powered Analysis of Eye Tracker Data in Basketball Game. Sensors 2025, 25, 3572. [10.3390/s25113572](https://doi.org/10.3390/s25113572)

It is also requested to cite Pupil Labs publication, as requested on their website [https://docs.pupil-labs.com/neon/data-collection/publications-and-citation/](https://docs.pupil-labs.com/neon/data-collection/publications-and-citation/)

- Baumann, C., & Dierkes, K. (2023). Neon accuracy test report. Pupil Labs, 10. [10.5281/zenodo.10420388](https://doi.org/10.5281/zenodo.10420388)

---

## Artificial Intelligence disclosure

This code is written in Vibe Coding with Google Gemini Pro
