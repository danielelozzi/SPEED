---
layout: default
title: SPEED - Eye-Tracking Analysis Software
---

# Welcome to the Official Page for SPEED v4

*An Advanced Eye-Tracking and Real-Time Data Analysis Software for Researchers*

SPEED provides two main components: a **Desktop App** and the **`speed-analyzer` Python package**.

---

## 🚀 Quick Links

- **[Download the App](#1-speed-desktop-application-for-end-users)**
- **[Install the Python Package](#2-speed-analyzer-python-package-for-developers)**
- **[Use with Docker](#3-docker-container-for-maximum-reproducibility)**
- **[CLI Tools](#4-command-line-tools)**
- **[GitHub Project](https://www.github.com/danielelozzi/SPEED/)**
- **[Documentation Website](https://danielelozzi.github.io/SPEED/)**

---

## 🔑 Core Features

- Modular Analysis: process once, generate unlimited plots/videos later  
- Advanced Event Editor: edit via table or interactive timeline  
- Computer Vision Integration: YOLOv8 for gaze-object mapping  
- Rich Outputs: heatmaps, gaze paths, pupillometry, overlays  
- Realtime Usage: live eye-tracker + YOLO analysis  
- Synthetic Data/Stream Generators  
- Docker support for reproducibility  

---

## 1. SPEED Desktop Application (For End Users)

Download, unzip, run `SpeedApp`, and follow the GUI workflow.  

Supports **RAW**, **Un-enriched**, and **Enriched** folders with dynamic AOI creation.  
See screenshots in the README for details.

---

## 2. `speed-analyzer` (Python Package for Developers)

Install from PyPI:
```bash
pip install speed-analyzer==4
```

Use in scripts:
```python
from speed_analyzer import run_full_analysis
```

Supports multiple AOI strategies (static, dynamic_auto, dynamic_manual).  
[See AOI examples in README.md](README.md#2-speed-analyzer-python-package-for-developers).

---

## 3. Docker Container (For Maximum Reproducibility)

```bash
docker pull ghcr.io/danielelozzi/speed:latest
```

Run with volume mounts for your data:
```bash
docker run --rm -v "/path/raw:/data/raw" -v "/path/output:/output" ghcr.io/danielelozzi/speed:latest ...
```

---

## 4. Command-Line Tools

- **Realtime Analyzer** (`realtime_cli.py`)  
- **Synthetic Data Generator** (`generate_synthetic_data.py`)  
- **Synthetic Stream Generator** (`generate_synthetic_stream.py`)  
- **Advanced Analyzer** (`main_analyzer_advanced.py`)  
- **Example Usage** (`example_usage.py`)  

Each script can be launched from terminal with standard arguments.  
[See README.md for examples.](README.md#3-command-line-tools)

---

## 5. Modular Workflow (GUI)

1. **Run Core Analysis** – preprocess and cache data  
2. **Generate Outputs On-Demand** – plots, videos, enriched data  

This ensures efficiency: heavy computation only once.

---

## 6. Environment Setup (Development)

- Use Anaconda or venv  
- Python 3.9+ recommended  
- Install requirements:  
  ```bash
  pip install -r requirements.txt
  ```  

Optional: CUDA Toolkit for GPU acceleration.

---

## 7. Authors & Citation

Developed by **Cognitive and Behavioral Science Lab (LabSCoC), University of L'Aquila** and **Dr. Daniele Lozzi**.  

Full citations are provided in the README.

---

## 8. Contributing & Roadmap

We welcome contributions!  

### Roadmap:
- Extend event scripting language  
- Enhance GUI integration  
- Add support for new ML models  
- Cloud deployment support  

---

## 💻 AI Disclosure

This project was partially developed using **Google Gemini 2.5 Pro** and refined manually.
