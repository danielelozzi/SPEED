# High-level pipeline

#' Run full analysis: read CSVs, generate plots, optionally video
#' @param working_dir path to un-enriched data
#' @param output_dir path for results
#' @param subject_name id for annotation
#' @param enriched_dirs character vector of paths to enriched data folders (AOIs)
#' @param generate_plots logical, if TRUE plots will be generated
#' @param generate_video logical, if TRUE a video overlay is created
#' @param bids logical, interpret working_dir as BIDS-like root
#' @param video_file optional video path for overlay
#' @return the output directory (invisible)
run_full_analysis <- function(
  working_dir,
  output_dir = file.path(working_dir, "speed_out"),
  subject_name = "S01",
  enriched_dirs = NULL, # NUOVO PARAMETRO
  generate_plots = TRUE,
  generate_video = FALSE,
  bids = FALSE,
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