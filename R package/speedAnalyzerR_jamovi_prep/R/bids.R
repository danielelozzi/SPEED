# BIDS-like conversion / loading

#' Convert raw input into a minimal BIDS-like tree
#' @param input_dir character path with raw files (CSV/MP4)
#' @param output_dir character path for BIDS-like structure
#' @param subject character subject id, e.g., "S01"
#' @param session character session id, e.g., "01"
#' @return invisibly, the created path
convert_to_bids <- function(input_dir, output_dir, subject, session = "01") {
  input_dir <- normalizePath(input_dir, mustWork = TRUE)
  output_dir <- normalizePath(output_dir, mustWork = FALSE)
  subj <- paste0("sub-", subject)
  ses  <- paste0("ses-", session)
  base <- file.path(output_dir, subj, ses, "eeg") # using eeg/behavior folder for eye-tracking CSVs
  ensure_dir(base)
  ensure_dir(file.path(output_dir, subj, ses, "beh"))

  # Copy known files if present
  known <- c("events.csv","fixations.csv","gaze.csv","3d_eye_states.csv","blinks.csv","scene_video.mp4")
  for (f in known) {
    src <- file.path(input_dir, f)
    if (file.exists(src)) {
      file.copy(src, file.path(base, f), overwrite = TRUE)
    }
  }
  # Minimal dataset_description.json
  desc <- list(Name = "speedAnalyzerR BIDS-like", BIDSVersion = "1.9.0-like", GeneratedBy = list(list(Name="speedAnalyzerR", Version="0.1.0")))
  jsonlite::write_json(desc, file.path(output_dir, "dataset_description.json"), auto_unbox = TRUE, pretty = TRUE)
  invisible(base)
}

#' Load from BIDS-like structure
#' @param bids_root path to root
#' @param subject e.g., "S01"
#' @param session e.g., "01"
#' @return a named list of data.tables
load_from_bids <- function(bids_root, subject, session = "01") {
  subj <- paste0("sub-", subject)
  ses  <- paste0("ses-", session)
  base <- file.path(bids_root, subj, ses, "eeg")
  out <- list(
    events   = read_csv_safe(file.path(base, "events.csv")),
    fix      = read_csv_safe(file.path(base, "fixations.csv")),
    gaze     = read_csv_safe(file.path(base, "gaze.csv")),
    pupil    = read_csv_safe(file.path(base, "3d_eye_states.csv")),
    blinks   = read_csv_safe(file.path(base, "blinks.csv")),
    video    = if (file.exists(file.path(base,"scene_video.mp4"))) file.path(base,"scene_video.mp4") else NULL
  )
  out
}