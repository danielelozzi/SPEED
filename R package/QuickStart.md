
# speedAnalyzerR – Quick Start

This is a minimal guide to get started with `speedAnalyzerR`.

## Install dependencies in R
```r
install.packages(c("data.table", "ggplot2", "optparse", "jsonlite", "MASS"))
# Optional for video overlay
install.packages(c("av", "magick"))
```

## Install the package
```r
install.packages("speedAnalyzerR_0.1.0.tar.gz", repos = NULL, type = "source")
```

## Run analysis from CLI
```bash
# Analyze raw data (CSV + video)
speed-analyzer-r analyze --data ./data --out ./results --subject S01

# Convert to BIDS-like layout and analyze
speed-analyzer-r bids-convert --input ./raw --out ./bids --subject S01 --session 01
speed-analyzer-r analyze --data ./bids --out ./results --subject S01 --bids --video-overlay
```

## Outputs
- Plots as `.png` in the output folder.
- Optional gaze-overlay video as `.mp4`.

