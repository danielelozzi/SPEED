# SPEED v4.5 - labScoc Processing and Extraction of Eye tracking Data

Desktop App & Analysis Package

*An Advanced Eye-Tracking Data Analysis Software*

SPEED is a Python-based project for processing, analyzing, and visualizing eye-tracking data. Version 4.5 introduces a major restructuring, offering two distinct components:

1.  **SPEED Desktop App**: A user-friendly GUI application for running a full analysis pipeline, designed for end-users and researchers.
2.  **`speed-analyzer`**[![PyPI version](https://img.shields.io/pypi/v/speed-analyzer.svg)](https://pypi.org/project/speed-analyzer/): A Python package for developers who want to integrate the analysis logic into their own scripts.
   
This version supports GPU acceleration for YOLO analysis and also offers three powerful AOI definition methods:
1.  **Static AOI**: A fixed rectangle for stationary scenes.
2.  **Dynamic AOI (Object Tracking)**: An AOI that automatically follows a selected object detected by YOLO.
3.  **Dynamic AOI (Manual Keyframes)**: A user-defined AOI path created by setting its position and size at key moments in the video.
4.  **Real-time visualization**: A real-time visualization of external and internal camera with YOLO AOI and manual multiple AOI, allowing the visualization of blink, pupillometry, fragmentation and events management. 

---

## 1. SPEED Desktop Application (For End Users)

An application with a graphical user interface (GUI) for a complete, visually-driven analysis workflow.

### How to Use the Application
1.  **Download the latest version**: Go to the [Releases page](https://github.com/danielelozzi/SPEED/releases) and download the `.zip` file for your operating system (Windows or macOS).
2.  **Extract and Run**: Unzip the file and run the `SpeedApp` executable.
3.  **Follow the Instructions**: 
    - Use the interface to select your data folders (RAW, Un-enriched).
    - If you do not provide an "Enriched" data folder, a **"Define AOI..."** button will become active.
    - Click it to choose your preferred AOI method (Static, Dynamic Auto, or Dynamic Manual) and follow the on-screen instructions in the interactive editor.
    - Manage events, run the analysis, and generate outputs as before.


---

## 2. `speed-analyzer` (Python Package for Developers)

The core analysis engine of SPEED, now available as a reusable package. It's designed for automation and integration into custom data pipelines.

### Installation from PyPI
You can install the package directly from the Python Package Index (PyPI) using pip:
```bash
pip install speed-analyzer==4.5
```
### How to Use the Package
The package exposes a main function, `run_full_analysis`, that takes paths and options as arguments. See the `example_usage.py` file for a complete demonstration.

Here is a basic snippet:

```python
import pandas as pd
from speed_analyzer import run_full_analysis

 Define input and output paths
raw_path = "./data/raw"
unenriched_path = "./data/unenriched"
output_path = "./analysis_results"
subject_name = "participant_01"
```

### Choose Your AOI Strategy

The speed-analyzer package allows you to define Areas of Interest (AOIs) on-the-fly, directly in your code. This is the recommended workflow when you do not have a pre-existing enriched_data_path. The system is designed to handle a list of multiple, mixed-type AOIs in a single analysis run.

When you provide the defined_aois parameter, the software will automatically generate new enriched data files (gaze_enriched.csv, fixations_enriched.csv) where each gaze point and fixation is mapped to the name of the AOI it falls into. It will also compute the Normalized Switching Index (SI) based on the sequence of transitions between these AOIs.

You define AOIs by creating a list of Python dictionaries. Each dictionary must have three keys: name, type, and data.


```python
my_aois = [
    { "name": "AOI_Name_1", "type": "...", "data": ... },
    { "name": "AOI_Name_2", "type": "...", "data": ... },
]
```

#### AOI Type 1: Static AOI

Use this for a fixed rectangular region that does not move throughout the video. The data is a dictionary containing the pixel coordinates of the rectangle's corners.

```python
static_aoi = {
    "name": "Control_Panel",
    "type": "static",
    "data": {'x1': 100, 'y1': 150, 'x2': 800, 'y2': 600}
}
```

#### AOI Type 2: Dynamic AOI (Automatic Object Tracking)

Use this to have an AOI automatically follow an object detected by YOLO. This requires setting run_yolo=True. The data is the integer track_id of the object you want to follow. You would typically get the track_id from a preliminary YOLO analysis

```python
object_id_to_track = 1 

dynamic_auto_aoi = {
    "name": "Tracked_Ball",
    "type": "dynamic_auto",
    "data": object_id_to_track
}
```

#### AOI Type 3: Dynamic AOI (Manual Keyframes)

Use this to define a custom path for a moving and resizing AOI. You set the AOI's position and size at specific frames (keyframes), and the software will interpolate its position for all frames in between. The data is a dictionary where keys are frame indices and values are tuples of coordinates (x1, y1, x2, y2).

```python
manual_keyframes_aoi = {
    "name": "Animated_Focus_Area",
    "type": "dynamic_manual",
    "data": {
        0: (50, 50, 250, 250),      # Position at the start (frame 0)
        1000: (400, 300, 600, 500), # Position at frame 1000
        2000: (50, 50, 250, 250)     # Return to the start position at frame 2000
    }
}
```

#### Putting It All Together: Example with Multiple AOIs

You can combine any number of AOIs of any type into a single list and pass it to the analysis function.

```python
import pandas as pd
from speed_analyzer import run_full_analysis

# 1. Define paths
raw_path = "./data/raw"
unenriched_path = "./data/unenriched"
output_path = "./analysis_results_multi_aoi"
subject_name = "participant_02"

# 2. Define multiple, mixed-type AOIs
my_aois = [
    { "name": "Left_Monitor", "type": "static", "data": {'x1': 0, 'y1': 0, 'x2': 960, 'y2': 1080}},
    { "name": "Right_Monitor", "type": "static", "data": {'x1': 961, 'y1': 0, 'x2': 1920, 'y2': 1080}},
    { "name": "Moving_Target", "type": "dynamic_auto", "data": 3 }
]

# 3. Run the analysis
run_full_analysis(
    raw_data_path=raw_path,
    unenriched_data_path=unenriched_path,
    output_path=output_path,
    subject_name=subject_name,
    run_yolo=True, # Required for the 'dynamic_auto' AOI
    defined_aois=my_aois # Pass the complete list of AOIs
)
```

### Real-time

The real-time window provides a suite of interactive tools:

* **Live Data Overlays**: Toggle various visualizations on the fly:
    * **YOLO Detections**: See what objects the system is identifying in real-time.
    * **Gaze Point**: A circle indicating the current gaze position.
    * **Pupil & Fragmentation Plots**: Live graphs showing pupillometry and gaze speed.
    * **Blink Detector**: An on-screen indicator that appears during a blink.
    * **AOIs**: View your defined Areas of Interest overlaid on the video.
* **Recording**:
    * Start and stop recordings directly from the interface.
    * Data (gaze, events, video) is saved into a selected folder.
* **Event Management**:
    * Add timestamped events during a live recording by typing a name and clicking "Add Event".
* **On-the-fly AOI Definition**:
    * Pause the stream to draw, name, and add static rectangular AOIs directly on the video feed.
    * These AOIs are visualized instantly and used for analysis when the recording is stopped. At the end of the recording, a `gaze_in_aoi_results.csv` file is automatically generated.

#### Command-Line Interface (for Developers)

For more advanced use cases and automation, a command-line interface is also available.

```bash
# Example: Run real-time analysis with a specific YOLO model and define two static AOIs
python realtime_cli.py --model yolov8n.pt --record --aoi "Screen,100,100,800,600" --aoi "Panel,850,300,1200,500"
```

You can also use a simulated data stream for testing without a physical device:

```bash
# Run with a mock device for testing purposes
python realtime_cli.py --use-mock
```

---

## 3. Docker Container (For Maximum Reproducibility)

To ensure maximum scientific reproducibility and to eliminate any issues with installation or dependencies, we provide a pre-configured Docker image that contains the exact environment to run the `speed-analyzer` package.

### Prerequisites
You must have **Docker Desktop** installed on your computer. You can download it for free from the [official Docker website](https://www.docker.com/products/docker-desktop/).

### How to Use the Docker Image

1.  **Pull the Image (Download)**:
    Open a terminal and run this command to download the latest version of the image from the GitHub Container Registry (GHCR).
    ```bash
    docker pull ghcr.io/danielelozzi/speed:latest
    ```

2.  **Run the Analysis**:
    To launch an analysis, you need to use the `docker run` command. The most important part is to "mount" your local folders (containing the data and where to save the results) inside the container.

    Here is a complete example. Replace the `/path/to/...` placeholders with the actual absolute paths on your computer.
    ```bash
    docker run --rm \
      -v "/path/to/your/RAW/folder:/data/raw" \
      -v "/path/to/your/un-enriched/folder:/data/unenriched" \
      -v "/path/to/your/output/folder:/output" \
      ghcr.io/danielelozzi/speed:latest \
      python -c "from speed_analyzer import run_full_analysis; run_full_analysis(raw_data_path='/data/raw', unenriched_data_path='/data/unenriched', output_path='/output', subject_name='docker_test')"
    ```
    
    **Command Explanation:**
    * `docker run --rm`: Runs the container and automatically removes it when finished.
    * `-v "/local/path:/container/path"`: The `-v` (volume) option creates a bridge between a folder on your computer and a folder inside the container. We are mapping your data folders into `/data/` and your output folder into `/output` inside the container.
    * `ghcr.io/danielelozzi/speed:latest`: The name of the image to use.
    * `python -c "..."`: The command that is executed inside the container. In this case, we launch a Python script that imports and runs your `run_full_analysis` function, using the paths *internal* to the container (`/data/`, `/output/`).

This approach guarantees that your analysis is always executed in the same controlled environment, regardless of the host computer.

---

## The Modular Workflow (GUI)
SPEED v4.5 operates on a two-step workflow designed to save time and computational resources.

### Step 1: Run Core Analysis
This is the main data processing stage. You run this step only once per participant for a given set of events. The software will:

- Load all necessary files from the specified input folders (RAW, Un-enriched, Enriched).
- **If you don't have Enriched data**, use the **"Define AOI..."** feature to create it dynamically. This is the new, recommended workflow for analyzing specific parts of a video. You can also combine several AOI (static, dynamic, and dynamic based on YOLO.)
- Dynamically load events from `events.csv` into the GUI, allowing you to select which events to analyze.
- Segment the data based on your selection.
- Calculate all relevant statistics for each selected segment.
- Optionally run YOLO object detection on the video frames, saving the results to a cache to speed up future runs.
- Save the processed data (e.g., filtered dataframes for each event) and summary statistics into the output folder.

This step creates a `processed_data` directory containing intermediate files. Once this is complete, you do not need to run it again unless you want to analyze a different combination of events.

### Step 2: Generate Outputs On-Demand

After the core analysis is complete, you can use the dedicated tabs in the GUI to generate as many plots and videos as you need, with any combination of settings, without re-processing the raw data.

#### Generate Plots 📊

The "Generate Plots" tab allows you to create a wide range of visualizations for each event segment. All plots are saved in PDF format for high-quality figures suitable for publications. The available plot categories are:

* **Path Plots**: Visualize the sequence of gaze points and fixations directly on the scene. This is perfect for understanding the participant's visual exploration strategy. You can generate separate plots for raw gaze data and for fixations, both in pixel coordinates (un-enriched) and normalized coordinates (enriched).
* **Density Heatmaps**: Create heatmaps to reveal the areas that attracted the most visual attention. The intensity of the color corresponds to the amount of time the participant spent looking at that specific area.
* **Duration Histograms**: Analyze the distribution of event durations. You can generate histograms for the duration of fixations, saccades, and blinks to understand their statistical properties.
* **Pupillometry**: Plot the changes in pupil diameter over time for each event. This is a crucial tool for research related to cognitive load, arousal, and emotional response. The plot also visualizes periods when the gaze is on a defined surface versus off-surface.
* **Advanced Time Series**: Dive deeper with detailed time series plots, including:
    * Mean pupil diameter over time.
    * Saccade velocity and amplitude over time.
    * A binary plot showing the exact moments when blinks occurred.
* **Gaze Fragmentation Plot**: This plot displays the speed of gaze movement (in pixels per second) over time. High fragmentation can be an indicator of visual searching behavior or cognitive instability.

Simply select the desired plot types in the GUI and click "GENERATE SELECTED PLOTS". The software will use the pre-processed data to generate the figures for all selected events.

### Computer Vision Analysis with YOLO 🤖

SPEED integrates the powerful **YOLO (You Only Look Once)** object detection model to add a layer of semantic understanding to the eye-tracking data. When this option is enabled during the "Core Analysis" step, the software analyzes the video to detect and track objects frame by frame.

#### How It Works

1.  **Object Detection & Tracking**: SPEED processes the scene video to identify objects and assigns a unique `track_id` to each detected object throughout its appearance in the video. The results are saved in a cache file (`yolo_detections_cache.csv`) to avoid re-processing on subsequent runs.
2.  **Gaze Correlation**: The system then correlates the participant's gaze and fixation data with the bounding boxes of the detected objects. This allows you to know not just *where* the participant was looking, but also *what* they were looking at.
3.  **Quantitative Analysis**: After the analysis, you can go to the **"YOLO Results"** tab in the GUI to view detailed statistics, such as:
    * **Stats per Instance**: A table showing metrics for each individual tracked object (e.g., `person_1`, `car_3`), including the total number of fixations it received and the average pupil diameter when looking at it.
    * **Stats per Class**: An aggregated view showing the same metrics for each object category (e.g., `person`, `car`).

#### Key Outputs

* **Dynamic AOI (Object Tracking)**: The `track_id` generated by YOLO can be used to define a "Dynamic AOI", where the Area of Interest automatically follows a specific object.
* **Switching Index (SI)**: The software automatically calculates the Normalized Switching Index based on the sequence of objects the participant looked at. This metric is valuable for understanding visual attention switching behavior. The results are saved in `switching_index_yolo_results.csv`.
* **Video Overlays**: When generating a custom video, you can choose to overlay the YOLO detection boxes and their labels directly onto the video, providing a clear and intuitive visualization of the analysis.

This feature transforms raw gaze coordinates into meaningful interactions with the environment, opening up new possibilities for analyzing human behavior in complex scenes.

---

## Environment Setup (For Development) ⚙️
To run the project from source or contribute to development, you'll need Python 3 and several libraries.

1. **Install Anaconda**: [Link](https://www.anaconda.com/)
2. *(Optional)* Install CUDA Toolkit: For GPU acceleration with NVIDIA. [Link](https://developer.nvidia.com/cuda-downloads)
3. **Create a virtual environment**:
```bash
conda create --name speed
conda activate speed
conda install pip
```
4. **Install the required libraries**:
```bash
pip install -r requirements.txt
```

---

## How to Use the Application from Source 🚀
### Launch the GUI:
```bash
# Navigate to the desktop_app folder
cd desktop_app
python GUI.py
```
### Setup and Analysis:
- Fill in the Participant Name and select the Output Folder.
- Select the required Input Folders: RAW and Un-enriched.
- Use the Advanced Event Management section to load and edit events using the table or interactive video editor.
- Click **"RUN CORE ANALYSIS"**.
- Use the other tabs to generate plots, videos, and view YOLO results.

---

## 🧪 Synthetic Data Generator (`generate_synthetic_data.py`)
Included in this project is a utility script to create a full set of dummy eye-tracking data. This is extremely useful for testing the SPEED software without needing Pupil Labs hardware or actual recordings.

### How to Use
Run the script from your terminal:
```bash
python generate_synthetic_data.py
```
The script will create a new folder named `synthetic_data_output` in the current directory.

This folder will contain all the necessary files (`gaze.csv`, `fixations.csv`, `external.mp4`, etc.), ready to be used as input for the GUI application or the `speed-analyzer` package.

It is also possible to generate a synthetic streaming for realtime with GUI using:
```bash
python generate_synthetic_stream.py
```

or for LSL testing:

```bash
python lsl_stream_simulator.py
```

---

## Export to BIDS Format

SPEED 4.5 introduces a new feature to convert processed eye-tracking data into a format compatible with the **Brain Imaging Data Structure (BIDS)**, following the [BEP020 for Eye Tracking](https://bids.neuroimaging.io/extensions/beps/bep_020.html) guidelines. This facilitates data sharing and standardization for the research community.

### Use via Desktop App

1. After setting the input folders (specifically the **Un-enriched** folder), a new section "4. Data Export" will be available.
2. Click the **"CONVERT TO BIDS FORMAT"** button.
3. A dialog box will open asking you for the metadata required for the BIDS structure:
* **Subject ID**: The participant's identifier (e.g., `01`).
* **Session ID**: The session identifier (e.g., `01`).
* **Task Name**: The task name (e.g., `reading`, `visualsearch`).
4. Select an empty output folder where the BIDS structure will be created.
5. Click "Start Conversion" to begin the process.

### Usage via the Python `speed-analyzer` package

This functionality is also available via the `convert_to_bids` function.

```python
from pathlib import Path
from speed_analyzer import convert_to_bids

# Define input and output paths
unenriched_path = Path("./data/unenriched")
bids_output_path = Path("./bids_dataset")

# Define BIDS metadata
subject = "01"
session = "01"
task = "visualsearch"

# Perform the conversion
convert_to_bids( 
unenriched_dir=unenriched_path, 
output_bids_dir=bids_output_path, 
subject_id=subject, 
session_id=session, 
task_name=task
)


---

## ✍️ Authors & Citation
This tool is developed by the Cognitive and Behavioral Science Lab (LabSCoC), University of L'Aquila and Dr. Daniele Lozzi.

If you use this script in your research or work, please cite the following publications:

- Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Ademaj, M.; Migliore, S.; Curcio, G. SPEED: A Graphical User Interface Software for Processing Eye Tracking Data. NeuroSci 2025, 6, 35. [https://doi.org/10.3390/neurosci6020035](https://doi.org/10.3390/neurosci6020035)
- Lozzi, D.; Di Pompeo, I.; Marcaccio, M.; Alemanno, M.; Krüger, M.; Curcio, G.; Migliore, S. AI-Powered Analysis of Eye Tracker Data in Basketball Game. Sensors 2025, 25, 3572. [https://doi.org/10.3390/s25113572](https://doi.org/10.3390/s25113572)

It is also requested to cite Pupil Labs publication, as requested on their website [https://docs.pupil-labs.com/neon/data-collection/publication-and-citation/](https://docs.pupil-labs.com/neon/data-collection/publication-and-citation/)

- Baumann, C., & Dierkes, K. (2023). Neon accuracy test report. Pupil Labs, 10. [https://doi.org/10.5281/zenodo.10420388](https://doi.org/10.5281/zenodo.10420388)

If you also use the Computer Vision YOLO-based feature, please cite the following publication:

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788). [https://doi.org/10.1109/CVPR.2016.91](https://doi.org/10.1109/CVPR.2016.91)

---

## 💻 Artificial Intelligence disclosure

This code is written in Vibe Coding with Google Gemini 2.5 Pro
