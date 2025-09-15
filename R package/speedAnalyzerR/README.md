# speedAnalyzerR

An experimental, command-line-only R package inspired by the `speed-analyzer` Python package.
It focuses on eye-tracking datasets: converting to a lightweight BIDS-like structure,
basic summaries/plots, and a simple video overlay option (if FFmpeg is available via the `av` package).

## CLI usage (after installing the package)
```
speed-analyzer-r analyze --data ./path/to/data --out ./out --subject S01
speed-analyzer-r bids-convert --input ./raw --out ./bids --subject S01 --session 01
speed-analyzer-r video --data ./data --out ./out --subject S01 --video ./scene.mp4
```

On Windows the entrypoint is `speed-analyzer-r.bat`.