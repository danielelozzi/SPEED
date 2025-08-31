# High-level pipeline

#' Run full analysis: read CSVs, generate plots, optionally video
#' @param working_dir path
#' @param output_dir path
#' @param subject_name id
#' @param generate_plots logical
#' @param generate_video logical
#' @param bids logical interpret working_dir as BIDS-like root
#' @param video_file optional video path for overlay
#' @return the output directory (invisible)
run_full_analysis <- function(
  working_dir,
  output_dir = file.path(working_dir, "speed_out"),
  subject_name = "S01",
  generate_plots = TRUE,
  generate_video = FALSE,
  bids = FALSE,
  video_file = NULL
) {
  ensure_dir(output_dir)
  if (generate_plots) {
    pkg_message("--- START PLOTS ---")
    pf <- try(generate_plots(working_dir, output_dir, subject = subject_name, bids = bids), silent = TRUE)
    if (inherits(pf, "try-error")) pkg_message("Plot generation failed: ", as.character(pf))
    pkg_message("--- END PLOTS ---")
  }
  if (generate_video) {
    pkg_message("--- START VIDEO ---")
    vf <- try(create_video_overlay(working_dir, output_dir, subject = subject_name, video_file = video_file), silent = TRUE)
    if (inherits(vf, "try-error") || is.null(vf)) {
      pkg_message("Video overlay failed or skipped.")
    } else {
      pkg_message("Video written: ", vf)
    }
    pkg_message("--- END VIDEO ---")
  }
  invisible(normalizePath(output_dir))
}