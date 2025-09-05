# Internal function to convert Tobii data to BIDS format.
# This function is not exported to the user.
#
# @param source_dir Path to the directory containing Tobii data (a single .tsv and a .mp4 video).
# @param bids_root_dir Path to the root of the BIDS output directory.
# @param subject_id The subject identifier (e.g., "01").
# @param session_id The session identifier (e.g., "01").
# @param task_name The task name (e.g., "visualsearch").
# @return Invisibly returns TRUE on success, or stops with an error.
# @import data.table
# @import jsonlite
# @import R.utils
#
convert_tobii_to_bids_r <- function(source_dir, bids_root_dir, subject_id, session_id, task_name) {

  # --- 1. Find and read the source TSV file ---
  tsv_file <- list.files(source_dir, pattern = "\\.tsv$", full.names = TRUE)
  if (length(tsv_file) != 1) {
    stop("Expected exactly one .tsv file in the source directory: ", source_dir)
  }
  pkg_message("Reading source data from: ", basename(tsv_file))
  dt <- data.table::fread(tsv_file, na.strings = c("", "NA", "-"))

  # --- 2. Create BIDS directory structure ---
  session_dir <- file.path(bids_root_dir, paste0("sub-", subject_id), paste0("ses-", session_id), "eyetrack")
  dir.create(session_dir, recursive = TRUE, showWarnings = FALSE)
  pkg_message("Created BIDS directory: ", session_dir)

  base_name <- paste0("sub-", subject_id, "_ses-", session_id, "_task-", task_name)

  # --- 3. Write dataset_description.json ---
  dataset_desc <- list(
    "Name" = "SPEED Eye-Tracking Dataset (Converted from Tobii)",
    "BIDSVersion" = "1.8.0",
    "DatasetType" = "raw",
    "Authors" = list("Dr. Daniele Lozzi, LabSCoC")
  )
  jsonlite::write_json(dataset_desc, file.path(bids_root_dir, "dataset_description.json"), auto_unbox = TRUE, pretty = TRUE)
  pkg_message("Created dataset_description.json")

  # --- 4. Prepare and write _eyetrack.tsv.gz ---
  # Assuming Tobii column names. These may need adjustment.
  # We select the first timestamp as the reference start time.
  start_time_ms <- dt[!is.na(RecordingTimestamp), min(RecordingTimestamp)]

  eyetrack_dt <- data.table(
    time = (dt$RecordingTimestamp - start_time_ms) / 1000, # Convert to seconds
    eye1_x_coordinate = dt$`GazePointX (ADCSpx)`,
    eye1_y_coordinate = dt$`GazePointY (ADCSpx)`,
    # Take the mean of left and right pupil, ignoring NAs
    eye1_pupil_size = rowMeans(dt[, .(`PupilLeft`, `PupilRight`)], na.rm = TRUE)
  )
  # Replace NaN from rowMeans (when both are NA) with NA
  eyetrack_dt[is.nan(eye1_pupil_size), eye1_pupil_size := NA]

  eyetrack_tsv_path <- file.path(session_dir, paste0(base_name, "_eyetrack.tsv"))
  data.table::fwrite(eyetrack_dt, file = eyetrack_tsv_path, sep = "\t", na = "n/a")

  # Gzip the file and remove the original
  R.utils::gzip(eyetrack_tsv_path, destname = paste0(eyetrack_tsv_path, ".gz"), remove = TRUE)
  pkg_message("Created _eyetrack.tsv.gz")

  # --- 5. Write _eyetrack.json sidecar ---
  # Assuming a common sampling frequency for Tobii Pro Glasses.
  # This should be verified from the device specifications.
  sampling_freq <- 100
  eyetrack_json <- list(
    "SamplingFrequency" = sampling_freq,
    "StartTime" = 0,
    "Columns" = c("time", "eye1_x_coordinate", "eye1_y_coordinate", "eye1_pupil_size"),
    "eye1_x_coordinate" = list("Units" = "pixels"),
    "eye1_y_coordinate" = list("Units" = "pixels"),
    "eye1_pupil_size" = list("Units" = "mm")
  )
  jsonlite::write_json(eyetrack_json, file.path(session_dir, paste0(base_name, "_eyetrack.json")), auto_unbox = TRUE, pretty = TRUE)
  pkg_message("Created _eyetrack.json")

  # --- 6. Prepare and write _events.tsv ---
  # Assuming events are in a column named 'Event'
  if ("Event" %in% names(dt)) {
    events_dt <- dt[!is.na(Event) & Event != "", .(
      onset = (RecordingTimestamp - start_time_ms) / 1000,
      duration = 0,
      trial_type = Event
    )]

    if (nrow(events_dt) > 0) {
      data.table::fwrite(events_dt, file = file.path(session_dir, paste0(base_name, "_events.tsv")), sep = "\t")
      pkg_message("Created _events.tsv")

      # --- 7. Write _events.json sidecar ---
      events_json <- list(
        "onset" = list("Description" = "Onset of the event in seconds relative to the start of the eyetracking recording."),
        "duration" = list("Description" = "Duration of the event in seconds (0 for instantaneous)."),
        "trial_type" = list("Description" = "Type of event, as recorded by the device.")
      )
      jsonlite::write_json(events_json, file.path(session_dir, paste0(base_name, "_events.json")), auto_unbox = TRUE, pretty = TRUE)
      pkg_message("Created _events.json")
    } else {
      pkg_message("No events found in the 'Event' column. Skipping _events.tsv generation.")
    }
  } else {
    pkg_message("No 'Event' column found. Skipping _events.tsv generation.")
  }

  # --- 8. Copy the video file ---
  video_file <- list.files(source_dir, pattern = "\\.mp4$", full.names = TRUE)
  if (length(video_file) == 1) {
    dest_video_path <- file.path(session_dir, paste0(base_name, "_recording.mp4"))
    file.copy(video_file, dest_video_path, overwrite = TRUE)
    pkg_message("Copied video file to BIDS directory.")
  } else {
    warning("Could not find a unique .mp4 video file in the source directory. Video not copied.")
  }

  pkg_message("Tobii to BIDS conversion completed successfully!")
  return(invisible(TRUE))
}

#' A helper function to print messages with a package prefix.
#'
#' @param ... Items to be printed.
#' @keywords internal
pkg_message <- function(...) {
  message("[speedAnalyzerR] ", ...)
}

#' Convert data from a specific eye-tracking device to a standard format.
#'
#' This function acts as a dispatcher to call the appropriate internal conversion
#' function based on the device name and desired output format.
#'
#' @param device_name The name of the source device (e.g., "tobii").
#' @param source_dir The path to the directory containing the source data.
#' @param output_dir The path to the root output directory.
#' @param output_format The desired output format (e.g., "bids").
#' @param bids_info A list containing BIDS-specific information, required if
#'   `output_format` is "bids". Must include `subject_id`, `session_id`, and
#'   `task_name`.
#' @return Invisibly returns TRUE on success, or stops with an error.
#' @export
run_device_conversion <- function(device_name, source_dir, output_dir, output_format, bids_info = NULL) {

  if (tolower(device_name) == "tobii") {
    if (tolower(output_format) == "bids") {
      if (is.null(bids_info) || !is.list(bids_info) || !all(c("subject_id", "session_id", "task_name") %in% names(bids_info))) {
        stop("Le informazioni BIDS (una lista con 'subject_id', 'session_id', 'task_name') sono obbligatorie per l'output BIDS.")
      }
      # Call the internal Tobii to BIDS converter
      convert_tobii_to_bids_r(
        source_dir = source_dir,
        bids_root_dir = output_dir,
        subject_id = bids_info$subject_id,
        session_id = bids_info$session_id,
        task_name = bids_info$task_name
      )
    } # Future output formats for Tobii could be added here
  } else {
    stop("Dispositivo non supportato: '", device_name, "'. Attualmente è supportato solo 'tobii'.")
  }

  return(invisible(TRUE))
}