# SPEED v3.3 - labScoc Processing and Extraction of Eye tracking Data

*An Advanced Eye-Tracking Data Analysis Software*

SPEED is a Python-based tool with a graphical user interface (GUI) for processing, analyzing, and visualizing eye-tracking data from cognitive and behavioral experiments. 

## The Modular Workflow

SPEED v3.3 operates on a two-step workflow designed to save time and computational resources.

**Step 1: Run Core Analysis**
This is the main data processing stage. You run this step **only once** per participant for a given set of events. The software will:
1.  Load all necessary files from the specified input folders (RAW, Un-enriched, Enriched).
2.  Dynamically load events from `events.csv` into the GUI, allowing you to **select which events to analyze**.
3.  Segment the data based on your selection.
4.  Calculate all relevant statistics for each selected segment.
5.  Optionally run YOLO object detection on the video frames, saving the results to a cache to speed up future runs.
6.  Save the processed data (e.g., filtered dataframes for each event) and summary statistics into the output folder.

This step creates a `processed_data` directory containing intermediate files. Once this is complete, you do not need to run it again unless you want to analyze a different combination of events.

**Step 2: Generate Outputs On-Demand**
After the core analysis is complete, you can use the dedicated tabs in the GUI to generate as many plots and videos as you need, with any combination of settings, without re-processing the raw data.
* **Generate Plots:** Select which categories of plots you want to create for the analyzed events.
* **Generate Videos:** Compose highly customized videos by selecting different overlays, including the new **Event Name Overlay**.
* **View YOLO Results:** Load and view the quantitative results from the object detection analysis.

---

## Data Acquisition 📋

Before using this software, you need to acquire and prepare the data following a specific procedure with Pupil Labs tools.

* **Video Recording**: Use Pupil Labs Neon glasses to record the session.
* *(optional)* **Surface Definition (AprilTag)**: Place AprilTags at the four corners of a PC screen. These markers allow the Pupil Labs software to track the surface and map gaze coordinates onto it. For more details, see the official documentation: [**Pupil Labs Surface Tracker**](https://docs.pupil-labs.com/neon/neon-player/surface-tracker/).
* **Upload to Pupil Cloud**: Once the recording is complete, upload the data to the Pupil Cloud platform.
* *(optional)* **Enrichment with Marker Mapper**: Inside Pupil Cloud, start the "Marker Mapper" enrichment. This process analyzes the video, detects the AprilTags, generates the `surface_positions.csv` file (which contains the surface coordinates for each frame), and downloads all the data. Marker Mapper Usage Guide: [**Pupil Cloud Marker Mapper**](https://docs.pupil-labs.com/neon/pupil-cloud/enrichments/marker-mapper/#setup).

---

## Environment Setup ⚙️

To run the SPEED analysis tool, you'll need Python 3 and several scientific computing libraries. It's highly recommended to use a virtual environment to manage dependencies.

0.  **Install Anaconda**
    [Anaconda link](https://www.anaconda.com/docs/getting-started/anaconda/install)

1.  **Create a virtual environment:**

    open Anaconda Prompt

    ```bash
    conda create --name speed
    conda activate speed
    conda install pip
    ```

2.  **Install the required libraries:**
    The required libraries depend on the analysis you want to run. Create a `requirements.txt` file with the content below. For the optional YOLO analysis, you will need `torch` and `ultralytics`.
    ```
    pandas
    numpy
    matplotlib
    opencv-python
    scipy
    tqdm
    # Optional for YOLO analysis
    torch
    ultralytics
    ```
    OR, install them using pip:

    ```bash
    pip install -r requirements.txt
    ```
    * **Note on `Tkinter`**: This is part of the Python standard library and does not require a separate installation.
    * **Note on `torch`**: Installing PyTorch can be complex, especially if you want to use a GPU (highly recommended for YOLO). Please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions tailored to your system.
    * **Note on YOLO**: To use YOLO, you must download the pre-trained neural network weights from the following link: [**yolov8n.pt Download**](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt). Place this file in the same directory as the scripts.

---

## How to Use the Application 🚀

1.  **Launch the GUI**: Run the `GUI.py` script from your terminal.
    ```bash
    cd <drag and drop speed folder>
    python GUI.py
    ```

2.  **Sections 1-3: Setup and Core Analysis**
    * **Step 1**: In the top sections of the GUI, fill in the **Participant Name** and select the **Output Folder**.
    * **Step 2**: Use the "Browse..." buttons to select the required **Input Folders**: **RAW** and **Un-enriched**. The **Enriched** folder is optional.

    ![GUI - Setup](images/gui1.png)

    * **Step 2.5**: After selecting the "Un-enriched Data Folder", the **"Event Selection"** box will automatically populate with all events found in `events.csv`. Uncheck any events you wish to exclude from the analysis.

    ![GUI - Event Selection](images/gui_events.png)

    * **Step 3**: In the "Run Core Analysis" section, configure the analysis mode (e.g., `un-enriched only`, `Run YOLO`).
    * Click the **"RUN CORE ANALYSIS"** button and wait for the confirmation message. This completes the main data processing for the selected events.

3.  **Section 4: Generate Plots**
    * Switch to the "4. Generate Plots" tab.
    * Select the categories of plots you wish to generate. These plots will only be created for the events you analyzed in the previous step.
    * Click the **"GENERATE SELECTED PLOTS"** button.

    ![GUI - Plot Generation Tab](images/gui2.png)

4.  **Section 5: Generate Videos**
    * Switch to the "5. Generate Videos" tab.
    * **Configure Video Composition**: Select the desired options, such as **"Overlay event name text"**, to display the current event's name on the video.
    * **Set the Output Video Filename**.
    * Click the **"GENERATE VIDEO"** button.

    ![GUI - Video Generation Tab with Event Overlay option](images/gui3.png)

5.  **Section 6: YOLO Results**
    * Switch to the "6. YOLO Results" tab.
    * Click **"Load/Refresh YOLO Results"**.
    * The tables will be populated with statistics calculated during the Core Analysis.

    ![GUI - YOLO Results Tab](images/gui4.png)

---

## Input Folder Structure 📂

The application now expects a specific folder structure as input. You must organize the files exported from Pupil Cloud into three separate folders.

### 1. RAW Data Folder
This folder should contain the raw media files.
| Filename | Requirement |
|---|---|
| `Neon Sensor Module v1 ps1.mp4` | **Always** (will be renamed to `internal.mp4`) |

### 2. Un-enriched Data Folder
This folder contains the main gaze and event data in pixel coordinates, along with the scene video.
| Filename | Requirement |
|---|---|
| `external.mp4` (or any single `.mp4`) | **Always** (must be only one video file) |
| `events.csv` | **Always** |
| `gaze.csv` | **Always** |
| `fixations.csv` | **Always** |
| `blinks.csv` | **Always** |
| `saccades.csv` | **Always** |
| `3d_eye_states.csv` | **Always** |
| `world_timestamps.csv` | **Always** |

### 3. Enriched Data Folder
This folder contains data that has been "enriched" in Pupil Cloud, typically mapped to a defined surface.
| Filename | Requirement |
|---|---|
| `gaze.csv` | Required if "un-enriched only" is **unchecked** (will be used as `gaze_enriched.csv`) |
| `fixations.csv` | Required if "un-enriched only" is **unchecked** (will be used as `fixations_enriched.csv`) |
| `surface_positions.csv` | Required for video perspective cropping/correction |
---

## Output Files 📈

All outputs are saved within the specified `analysis_results_{participant_name}` folder.

* **`eyetracking_file/`**: Contains copies of all the input files used for the analysis.
* **`processed_data/`**: Contains intermediate data files (`.pkl`) for each **selected** event segment.
* **`plots/`**: Contains all the generated PDF plots for the analyzed events.
* **`config.json`**: A file saving the settings used for the Core Analysis, including the `selected_events` list for reproducibility.
* **`summary_results_{subj_name}.csv`**: A CSV file with the main quantitative outcomes of the analysis for the selected events.
* **`{video_name}.mp4`**: Each custom video you generate is saved in the main output folder.
* **YOLO Outputs**:
    * `yolo_detections_cache.csv`: A cache of the raw YOLO detections.
    * `stats_per_class.csv`: Aggregated statistics for each object class.
    * `stats_per_instance.csv`: Statistics for each individual tracked object instance.
    * `class_id_map.csv`: A utility file that maps tracking IDs to class names.

---
## 🧪 Synthetic Data Generator (`generate_synthetic_data.py`)

Included in this project is a utility script to create a full set of dummy eye-tracking data. This is extremely useful for testing the SPEED software without needing Pupil Labs hardware or actual recordings.

### Purpose
The script generates all the necessary `.csv` and `.mp4` files that mimic a real recording session, including gaze movements, blinks, saccades, a moving surface, and the corresponding scene and eye videos.

### How to Use
1.  Run the script from your terminal:
    ```bash
    python generate_synthetic_data.py
    ```
2.  The script will create a new folder named `synthetic_data` in the current directory.
3.  This folder will contain all the necessary files (`gaze.csv`, `fixations.csv`, `world.mp4`, etc.).
4.  In the SPEED GUI, you can now use the `synthetic_data` folder as input (e.g., for both the "Un-enriched Data Folder" and "Enriched Data Folder" paths) to run a full analysis pipeline.

---
## ✍️ Authors & Citation

* This tool is developed by the **Cognitive and Behavioral Science Lab (LabSCoC), University of L'Aquila** and **Dr. Daniele Lozzi**.
* If you use this script in your research or work, please cite the following publications:
    * Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Ademaj, M.; Migliore, S.; Curcio, G. SPEED: A Graphical User Interface Software for Processing Eye Tracking Data. NeuroSci 2025, 6, 35. <https://doi.org/10.3390/neurosci6020035>
    * Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Alemanno, M.; Krüger, M.; Curcio, G.; Migliore, S. AI-Powered Analysis of Eye Tracker Data in Basketball Game. Sensors 2025, 25, 3572. <https://doi.org/10.3390/s25113572>

* It is also requested to cite Pupil Labs publication, as requested on their website <https://docs.pupil-labs.com/neon/data-collection/publicatheir-and-citation/>
    * Baumann, C., & Dierkes, K. (2023). Neon accuracy test report. Pupil Labs, 10. <https://doi.org/10.5281/zenodo.10420388>

* If you also use the Computer Vision YOLO-based feature, please cite the following publication:
    * Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). <https://doi.org/10.1109/CVPR.2016.91>

## 💻 Artificial Intelligence disclosure

This code is partially written using Google Gemini.