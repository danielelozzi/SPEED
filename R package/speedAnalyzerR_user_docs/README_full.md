
# speedAnalyzerR

`speedAnalyzerR` is an **R command-line tool** inspired by the Python package `speed-analyzer`. It provides basic functionality for organizing eye-tracking data into a lightweight BIDS-like structure, generating plots, and optionally creating simple video overlays of gaze positions.

## Features
- Convert raw CSV/video data into a **minimal BIDS-like structure**.
- Load datasets from this structure for analysis.
- Generate summary **plots**:
  - Gaze speed histogram.
  - Gaze density heatmap (2D KDE).
  - Fixation duration histogram.
  - Events timeline.
- (Optional) Create a **video overlay** of gaze points on top of the scene video (requires FFmpeg + ImageMagick).

## Installation
1. Install dependencies in R:
   ```r
   install.packages("data.table")
   install.packages("ggplot2")
   install.packages("optparse")
   install.packages("jsonlite")
   install.packages("MASS")
   # Optional for video overlay
   install.packages("av")
   install.packages("magick")
   ```

2. Install the package from source:
   ```r
   install.packages("speedAnalyzerR_0.1.0.tar.gz", repos = NULL, type = "source")
   ```

## Command-Line Usage
Once installed, the package provides a command-line executable:

### General help
```
speed-analyzer-r help
```

### Analyze data
```
speed-analyzer-r analyze --data ./data --out ./results --subject S01
```
Options:
- `--bids` : Interpret `--data` as a BIDS-like root directory.
- `--video-overlay` : Also generate a video overlay.

### Convert raw data to BIDS-like structure
```
speed-analyzer-r bids-convert --input ./raw --out ./bids --subject S01 --session 01
```

### Create only a video overlay
```
speed-analyzer-r video --data ./data --out ./results --subject S01 --video ./scene.mp4
```

## Input Data Format
The following files are recognized if present:
- events.csv
- fixations.csv
- gaze.csv
- 3d_eye_states.csv
- blinks.csv
- scene_video.mp4

## Output
- Plots are saved as `.png` files in the output directory.
- If video overlay is enabled, an `.mp4` file with gaze points drawn will be created.

## Example Workflow
1. Convert raw data:
   ```bash
   speed-analyzer-r bids-convert --input ./raw --out ./bids --subject S01 --session 01
   ```

2. Run analysis:
   ```bash
   speed-analyzer-r analyze --data ./bids --out ./results --subject S01 --bids --video-overlay
   ```

3. Check plots and video overlay in ./results/.

## Notes
- The video overlay is intentionally simple and may be slow on long videos.
- No object detection or deep learning is included (unlike the Python tool). The focus is on lightweight R-only analysis.
- The package can be extended with custom metrics, reports, or AOI-based analyses.

---
Developed in R for CLI-only usage, without Python dependencies.
