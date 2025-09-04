# Video overlay (very simple): requires av and optionally magick

#' Create a simple overlay video: draw gaze as small circles over frames
#' @param data_dir path with gaze.csv and a scene video
#' @param output_dir destination
#' @param subject subject id
#' @param video_file optional path to video; if NULL, tries data_dir/scene_video.mp4
#' @param point_radius numeric radius in pixels
#' @return output video path (mp4) or NULL if prerequisites missing
create_video_overlay <- function(data_dir, output_dir, subject = "S01", video_file = NULL, point_radius = 10) {
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
    guess <- file.path(data_dir, "scene_video.mp4")
    video_file <- if (file.exists(guess)) guess else NULL
  }
  if (is.null(gaze) || is.null(video_file) || !file.exists(video_file)) {
    pkg_message("Missing gaze.csv or scene_video.mp4. Skipping video overlay.")
    return(NULL)
  }
  # Extract frames, overlay, re-encode (this is best-effort and may be slow)
  tmpdir <- tempfile("frames_")
  dir.create(tmpdir, showWarnings = False <- FALSE)
  on.exit(unlink(tmpdir, recursive = TRUE), add = TRUE)
  info <- av::av_media_info(video_file)
  fps <- as.numeric(info$video$framerate)
  if (!is.finite(fps) || fps <= 0) fps <- 30
  # timestamps in gaze assumed in seconds or ms? Try to auto-detect: if max > 6e3, assume ms
  ts <- gaze$timestamp
  if (max(ts, na.rm=TRUE) > 6000) ts <- ts / 1000
  # Extract frames as images
  av::av_video_images(video_file, destdir = tmpdir, format = "png", fps = fps, verbose = FALSE)
  frames <- list.files(tmpdir, pattern = "\\.png$", full.names = TRUE)
  # Build a simple nearest timestamp index per frame
  t_frame <- seq(0, by = 1/fps, length.out = length(frames))
  # For speed, use nearest join
  which_nearest <- function(x, y) y[max(1, which.min(abs(y - x)))]
  idx <- vapply(t_frame, function(t) {
    which.min(abs(ts - t))
  }, integer(1))
  # Overlay
  outdir <- file.path(output_dir, "frames_overlay")
  dir.create(outdir, showWarnings = FALSE)
  for (i in seq_along(frames)) {
    fr <- magick::image_read(frames[i])
    j <- idx[i]
    if (is.finite(gaze$x[j]) && is.finite(gaze$y[j])) {
      fr <- magick::image_draw(fr)
      symbols(gaze$x[j], gaze$y[j], circles = point_radius, inches = FALSE, add = TRUE)
      dev.off()
    }
    magick::image_write(fr, path = file.path(outdir, sprintf("frame_%06d.png", i)))
  }
  out_file <- file.path(output_dir, paste0("video_overlay_", subject, ".mp4"))
  av::av_encode_video(list.files(outdir, pattern="\\.png$", full.names=TRUE), framerate = fps, output = out_file, verbose = FALSE)
  out_file
}