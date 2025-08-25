# SPEED v4 - Desktop App & Analysis Package

*An Advanced Eye-Tracking and Real-Time Data Analysis Software*

SPEED is a Python-based project for processing, analyzing, and visualizing eye-tracking data.  
Version 4 introduces a major restructuring, offering two distinct components:

1.  **SPEED Desktop App**: A user-friendly GUI application for running a full analysis pipeline, designed for end-users and researchers.
2.  **`speed-analyzer`** [![PyPI version](https://img.shields.io/pypi/v/speed-analyzer.svg)](https://pypi.org/project/speed-analyzer/):  
    A Python package for developers who want to integrate the analysis logic into their own scripts.

---

## 🔑 Key Features

- GPU acceleration for YOLO-based analysis
- Three powerful AOI (Area of Interest) definition methods:
  1. **Static AOI** – fixed rectangle for stationary scenes
  2. **Dynamic AOI (Object Tracking)** – AOI follows objects detected by YOLO
  3. **Dynamic AOI (Manual Keyframes)** – user-defined AOI path via keyframes
- **Realtime usage**: run live eye-tracking analysis with YOLO
- **Synthetic data & stream generators** for testing and validation
- **Advanced Analyzer** (`main_analyzer_advanced.py`) for research workflows
- **Docker container** for maximum reproducibility

---

## 1. SPEED Desktop Application (For End Users)

An application with a graphical user interface (GUI) for a complete, visually-driven analysis workflow.

### How to Use the Application
1.  **Download the latest version**: From the [Releases page](https://github.com/danielelozzi/SPEED/releases).
2.  **Extract and Run**: Unzip and run the `SpeedApp` executable.
3.  **Follow the Instructions**:
    - Select your data folders (RAW, Un-enriched).
    - If no "Enriched" folder is provided, use **"Define AOI..."**.
    - Choose AOI method (Static, Dynamic Auto, or Dynamic Manual).
    - Manage events, run analysis, and generate outputs.

---

## 2. `speed-analyzer` (Python Package for Developers)

The core analysis engine of SPEED, designed for automation and integration.

### Installation from PyPI
```bash
pip install speed-analyzer==4
```

### Quick Example
```python
from speed_analyzer import run_full_analysis

run_full_analysis(
    raw_data_path="./data/raw",
    unenriched_data_path="./data/unenriched",
    output_path="./analysis_results",
    subject_name="participant_01"
)
```

### AOI Definition Strategies
SPEED allows AOIs to be defined programmatically (static, dynamic_auto, dynamic_manual).  
[See detailed examples in the docs.](index.md#choose-your-aoi-strategy)

---

## 3. Command-Line Tools

SPEED includes several CLI scripts for direct usage:

- **Realtime CLI Analyzer**  
  ```bash
  python realtime_cli.py --input 0
  ```
  Runs live analysis using the default webcam.

- **Synthetic Data Generator**  
  ```bash
  python generate_synthetic_data.py
  ```
  Produces dummy gaze/fixation data (`synthetic_data_output`).

- **Synthetic Stream Generator**  
  ```bash
  python generate_synthetic_stream.py --duration 60
  ```
  Generates a continuous artificial stream for 60 seconds.

- **Advanced Analyzer**  
  ```bash
  python main_analyzer_advanced.py --config config.yaml
  ```
  Runs complex analysis workflows with custom settings.

- **Example Usage**  
  ```bash
  python example_usage.py
  ```

---

## 4. Docker Container

Pre-configured container ensures reproducibility.  

```bash
docker pull ghcr.io/danielelozzi/speed:latest
```

[See full Docker usage in index.md](index.md#3-docker-container-for-maximum-reproducibility).

---

## 5. Project Structure

```
SPEED/
├── desktop_app/                # GUI application
├── src/speed_analyzer/         # Core Python package
│   └── analysis_modules/       # Analysis modules
│       ├── realtime_analyzer.py
│       ├── yolo_analyzer.py
│       ├── video_generator.py
│       └── speed_script_events.py
├── example_usage.py
├── realtime_cli.py
├── generate_synthetic_data.py
├── generate_synthetic_stream.py
├── main_analyzer_advanced.py
├── yolov8n.pt                  # Pretrained YOLO model
├── requirements.txt
├── Dockerfile
├── setup.py
└── index.md
```

---

## 6. Development Setup ⚙️

- Python 3.9+ recommended  
- Install dependencies:
```bash
pip install -r requirements.txt
```
- (Optional) Install CUDA for GPU acceleration

---

## 7. Contributing

1. Fork the repo  
2. Create a new branch  
3. Submit a Pull Request  

---

## ✍️ Authors & Citation

Developed by **Cognitive and Behavioral Science Lab (LabSCoC), University of L'Aquila** and **Dr. Daniele Lozzi**.  

[Full citation list available in docs.](index.md#-authors--citation)

---

## 💻 AI Disclosure

This code was partially generated using **Google Gemini 2.5 Pro** and refined manually.
