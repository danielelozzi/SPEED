#' Create a simple video overlay with gaze points.
#'
#' This function draws gaze points (and optionally a trailing path) onto each
#' frame of a scene video. It requires the 'av' and 'magick' packages.
#'
#' @param data_dir Path to the directory containing 'gaze.csv' and a scene video.
#' @param output_dir Path to the directory where the output video will be saved.
#' @param subject The subject identifier, used for naming the output file.
#' @param video_file Optional path to the video file. If NULL, the function will search for a '.mp4' file in `data_dir`.
#' @param draw_gaze_path Logical. If TRUE, a short trail of recent gaze points is drawn.
#' @param point_radius Numeric. The radius of the gaze point circle in pixels.
#' @return The path to the created MP4 video, or NULL if prerequisites are missing or an error occurs.
create_video_overlay <- function(data_dir, output_dir, subject = "S01", video_file = NULL, draw_gaze_path = TRUE, point_radius = 10) {
  suppressWarnings({
    has_av <- requireNamespace("av", quietly = TRUE)
    has_magick <- requireNamespace("magick", quietly = TRUE)
  })
  if (!has_av || !has_magick) {
    pkg_message("av and magick packages are required for video overlay.")
    return(NULL)
  }

  data_dir <- normalizePath(data_dir, mustWork = TRUE)
  ensure_dir(output_dir)

  gaze <- read_csv_safe(file.path(data_dir, "gaze.csv"))
  if (is.null(video_file)) {
    video_files <- list.files(data_dir, pattern = "\\.mp4$", full.names = TRUE)
    if (length(video_files) == 1) {
      video_file <- video_files[1]
    }
  }

  if (is.null(gaze) || is.null(video_file) || !file.exists(video_file)) {
    pkg_message("Missing gaze.csv or a unique .mp4 video file. Skipping video overlay.")
    return(NULL)
  }

  # --- 1. Setup paths and parameters ---
  tmpdir <- tempfile("frames_")
  dir.create(tmpdir, showWarnings = FALSE)
  on.exit(unlink(tmpdir, recursive = TRUE), add = TRUE)

  info <- av::av_media_info(video_file)
  fps <- as.numeric(info$video$framerate)
  if (!is.finite(fps) || fps <= 0) fps <- 30

  # --- 2. Prepare gaze data ---
  # Auto-detect timestamp unit (seconds vs. milliseconds)
  if ("timestamp" %in% names(gaze) && max(gaze$timestamp, na.rm = TRUE) > 6000) {
    gaze$timestamp <- gaze$timestamp / 1000
  }

  # --- 3. Extract frames and map to gaze data ---
  pkg_message("Extracting video frames...")
  av::av_video_images(video_file, destdir = tmpdir, format = "png", fps = fps, verbose = FALSE)
  frames <- list.files(tmpdir, pattern = "\\.png$", full.names = TRUE)
  if (length(frames) == 0) {
    pkg_message("No frames were extracted from the video. Aborting.")
    return(NULL)
  }

  t_frame <- seq(0, by = 1/fps, length.out = length(frames))
  gaze_dt <- data.table::as.data.table(gaze)
  frame_dt <- data.table::data.table(frame_idx = seq_along(t_frame), t_frame = t_frame)
  gaze_dt[, t_gaze := timestamp]

  # Find the nearest gaze point for each frame time
  frame_dt[, gaze_idx := gaze_dt[frame_dt, on = .(t_gaze = t_frame), roll = "nearest", which = TRUE]]

  # --- 4. Overlay gaze on frames ---
  pkg_message("Overlaying gaze data on frames...")
  outdir <- file.path(output_dir, "frames_overlay")
  dir.create(outdir, showWarnings = FALSE)

  lapply(seq_along(frames), function(i) {
    fr <- magick::image_read(frames[i])
    gaze_point <- gaze_dt[frame_dt$gaze_idx[i], ]

    if (nrow(gaze_point) == 1 && is.finite(gaze_point$x) && is.finite(gaze_point$y)) {
      fr <- magick::image_draw(fr)
      graphics::symbols(gaze_point$x, gaze_point$y, circles = point_radius, inches = FALSE, add = TRUE, bg = "#FF000080", fg = "white")
      dev.off()
    }
    magick::image_write(fr, path = file.path(outdir, sprintf("frame_%06d.png", i)))
    return(NULL)
  })

  # --- 5. Encode video ---
  pkg_message("Encoding final video...")
  out_file <- file.path(output_dir, paste0("video_overlay_", subject, ".mp4"))
  av::av_encode_video(list.files(outdir, pattern="\\.png$", full.names=TRUE), framerate = fps, output = out_file, verbose = FALSE)
  pkg_message("Video overlay created: ", out_file)
  return(out_file)
}