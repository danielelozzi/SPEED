#' Convert unenriched data to a minimal BIDS-compliant structure.
#'
#' This function takes a directory of "un-enriched" data (as produced by
#' Pupil Player) and organizes it into a BIDS-compliant structure for
#' eye-tracking data.
#'
#' @param input_dir Path to the directory with raw files (e.g., gaze.csv, events.csv).
#' @param output_dir Path for the BIDS output structure.
#' @param subject The subject identifier (e.g., "01").
#' @param session The session identifier (e.g., "01").
#' @param task The task name (e.g., "visualsearch").
#' @return Invisibly, the path to the created session directory.
#' @export
convert_to_bids <- function(input_dir, output_dir, subject, session = "01", task = "eyetracking") {
  input_dir <- normalizePath(input_dir, mustWork = TRUE)
  output_dir <- normalizePath(output_dir, mustWork = FALSE)
  
  # --- 1. Create BIDS directory structure ---
  session_dir <- file.path(output_dir, paste0("sub-", subject), paste0("ses-", session), "eyetrack")
  dir.create(session_dir, recursive = TRUE, showWarnings = FALSE)
  pkg_message("Created BIDS directory: ", session_dir)
  
  base_name <- paste0("sub-", subject, "_ses-", session, "_task-", task)

  # --- 2. Write dataset_description.json ---
  dataset_desc_path <- file.path(output_dir, "dataset_description.json")
  if (!file.exists(dataset_desc_path)) {
    desc <- list(
      Name = "speedAnalyzerR BIDS Export",
      BIDSVersion = "1.8.0",
      DatasetType = "raw",
      GeneratedBy = list(list(Name = "speedAnalyzerR", Version = "0.1.0"))
    )
    jsonlite::write_json(desc, dataset_desc_path, auto_unbox = TRUE, pretty = TRUE)
    pkg_message("Created dataset_description.json")
  }

  # --- 3. Copy and rename relevant files ---
  # This is a simplified conversion. A more robust one would process the files.
  # For now, we copy them with BIDS-compliant names.
  file_map <- c(
    "gaze.csv" = "_gaze.csv", # Not standard, but for completeness
    "fixations.csv" = "_fixations.csv", # Not standard
    "events.csv" = "_events.tsv" # This one is standard
  )
  
  for (source_name in names(file_map)) {
    src_path <- file.path(input_dir, source_name)
    if (file.exists(src_path)) {
      dest_name <- paste0(base_name, file_map[[source_name]])
      file.copy(src_path, file.path(session_dir, dest_name), overwrite = TRUE)
    }
  }
  
  invisible(session_dir)
}

#' Load from BIDS-like structure
#' @param bids_root path to root
#' @param subject e.g., "01"
#' @param session e.g., "01"
#' @return a named list of data.tables
load_from_bids <- function(bids_root, subject, session = "01") {
  subj <- paste0("sub-", subject)
  ses  <- paste0("ses-", session)
  base <- file.path(bids_root, subj, ses, "eyetrack")
  out <- list(
    events   = read_csv_safe(file.path(base, "events.csv")),
    fix      = read_csv_safe(file.path(base, "fixations.csv")),
    gaze     = read_csv_safe(file.path(base, "gaze.csv")),
    pupil    = read_csv_safe(file.path(base, "3d_eye_states.csv")),
    blinks   = read_csv_safe(file.path(base, "blinks.csv")),
    video    = list.files(base, pattern = "\\.mp4$", full.names = TRUE)[1]
  )
  out
}