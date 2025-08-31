# Plot generation using ggplot2

#' Generate basic plots (speed hist, heatmap, time series)
#' @param data_dir path with CSVs (events.csv, fixations.csv, gaze.csv, etc.) OR a BIDS-like root if bids = TRUE
#' @param output_dir where to write PNGs
#' @param subject character id to annotate
#' @param bids logical if true, use load_from_bids
#' @return vector of file paths created
generate_plots <- function(data_dir, output_dir, subject = "S01", bids = FALSE) {
  data_dir <- normalizePath(data_dir, mustWork = TRUE)
  ensure_dir(output_dir)
  suppressPackageStartupMessages({
    library(data.table); library(ggplot2); library(MASS)
  })

  if (bids) {
    lst <- load_from_bids(data_dir, subject = subject)
    events <- lst$events; fix <- lst$fix; gaze <- lst$gaze
  } else {
    events <- read_csv_safe(file.path(data_dir, "events.csv"))
    fix <- read_csv_safe(file.path(data_dir, "fixations.csv"))
    gaze <- read_csv_safe(file.path(data_dir, "gaze.csv"))
  }

  out_files <- character(0)

  # 1) Gaze speed histogram (if vx, vy, or x,y with time)
  if (!is.null(gaze)) {
    dt <- data.table::as.data.table(gaze)
    # Infer speed from successive points if not present
    if (!("speed" %in% names(dt))) {
      needed <- c("x","y","timestamp")
      if (all(needed %in% names(dt))) {
        dt <- dt[order(timestamp)]
        dt[, dx := c(NA, diff(x))]
        dt[, dy := c(NA, diff(y))]
        dt[, dtm := c(NA, diff(timestamp))]
        dt[, speed := sqrt(dx^2 + dy^2) / dtm]
      } else {
        dt[, speed := NA_real_]
      }
    }
    p <- ggplot(dt[is.finite(speed)], aes(x = speed)) + 
      geom_histogram(bins = 60) + 
      ggtitle(paste0("Gaze speed histogram - ", subject)) + xlab("px/ms (approx)") + ylab("Count")
    f <- file.path(output_dir, "plot_gaze_speed_hist.png")
    ggplot2::ggsave(f, p, dpi = 120, width = 8, height = 5)
    out_files <- c(out_files, f)

    # Heatmap via KDE if x,y exist
    if (all(c("x","y") %in% names(dt))) {
      dd <- dt[is.finite(x) & is.finite(y)]
      if (nrow(dd) > 10) {
        kd <- MASS::kde2d(dd$x, dd$y, n = 100)
        df <- data.frame(expand.grid(x = kd$x, y = kd$y), z = as.vector(kd$z))
        p2 <- ggplot(df, aes(x, y, fill = z)) + 
          geom_raster(interpolate = TRUE) + 
          ggtitle(paste0("Gaze density (KDE) - ", subject)) + 
          xlab("x") + ylab("y")
        f2 <- file.path(output_dir, "plot_gaze_density_kde.png")
        ggplot2::ggsave(f2, p2, dpi = 120, width = 7, height = 6)
        out_files <- c(out_files, f2)
      }
    }
  }

  # 2) Fixation duration histogram
  if (!is.null(fix) && ("duration" %in% names(fix))) {
    ff <- data.table::as.data.table(fix)
    p3 <- ggplot(ff[is.finite(duration)], aes(x = duration)) + 
      geom_histogram(bins = 50) + 
      ggtitle(paste0("Fixation duration - ", subject)) + xlab("ms") + ylab("Count")
    f3 <- file.path(output_dir, "plot_fixation_duration.png")
    ggplot2::ggsave(f3, p3, dpi = 120, width = 8, height = 5)
    out_files <- c(out_files, f3)
  }

  # 3) Events timeline (if onset, duration, event exist)
  if (!is.null(events) && all(c("onset","duration","event") %in% names(events))) {
    ee <- data.table::as.data.table(events)
    ee[, t0 := onset]
    ee[, t1 := onset + duration]
    p4 <- ggplot(ee, aes(y = event)) + 
      geom_segment(aes(x = t0, xend = t1, yend = event)) + 
      ggtitle(paste0("Events timeline - ", subject)) + xlab("time (ms)") + ylab("event")
    f4 <- file.path(output_dir, "plot_events_timeline.png")
    ggplot2::ggsave(f4, p4, dpi = 120, width = 9, height = 6)
    out_files <- c(out_files, f4)
  }

  out_files
}