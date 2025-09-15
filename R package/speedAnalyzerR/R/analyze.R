# High-level pipeline

#' Run full analysis: read CSVs, generate plots, optionally video
#' @param working_dir Path to the data directory (can be un-enriched or BIDS root).
#' @param output_dir Path for saving results. Defaults to a 'speed_out' folder inside `working_dir`.
#' @param subject_name The subject identifier for annotating plots and files.
#' @param enriched_dirs A character vector of paths to "enriched" data folders, each corresponding to an AOI.
#' @param generate_plots Logical. If TRUE, summary plots will be generated.
#' @param generate_video Logical. If TRUE, a simple video overlay is created.
#' @param bids Logical. If TRUE, `working_dir` is interpreted as a BIDS root directory.
#' @param draw_gaze_path Logical. If TRUE, the video overlay will include a trail for the gaze.
#' @param video_file Optional path to a specific video file for the overlay.
#' @return the output directory (invisible)
run_full_analysis <- function(
  working_dir,
  output_dir = file.path(working_dir, "speed_out"),
  subject_name = "S01",
  enriched_dirs = NULL, # NUOVO PARAMETRO
  generate_plots = TRUE,
  generate_video = FALSE,
  bids = FALSE,
  draw_gaze_path = TRUE,
  video_file = NULL
) {
  ensure_dir(output_dir)
  
  if (generate_plots) {
    pkg_message("--- START PLOTS ---")
    # --- MODIFICA: Passa enriched_dirs alla funzione di plotting ---
    pf <- try(generate_plots(
        data_dir = working_dir, 
        output_dir = output_dir, 
        subject = subject_name, 
        bids = bids, 
        enriched_dirs = enriched_dirs
    ), silent = TRUE)
    # --- FINE MODIFICA ---
    if (inherits(pf, "try-error")) pkg_message("Plot generation failed: ", as.character(pf))
    pkg_message("--- END PLOTS ---")
  }
  
  if (generate_video) {
    pkg_message("--- START VIDEO ---")
    vf <- try(create_video_overlay(
        data_dir = working_dir, 
        output_dir = output_dir, 
        subject = subject_name, 
        video_file = video_file,
        draw_gaze_path = draw_gaze_path
    ), silent = TRUE)
    if (inherits(vf, "try-error") || is.null(vf)) {
      pkg_message("Video overlay failed or skipped.")
    } else {
      pkg_message("Video written: ", vf)
    }
    pkg_message("--- END VIDEO ---")
  }
  
  invisible(normalizePath(output_dir))
}