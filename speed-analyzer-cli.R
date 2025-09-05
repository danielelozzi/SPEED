#!/usr/bin/env Rscript

# Import specific functions to make them available without the package prefix
#' @importFrom speedAnalyzerR run_device_conversion

# Main Command Line Interface for the speedAnalyzerR package

# Load required libraries
if (!require("optparse", quietly = TRUE)) {
  install.packages("optparse", repos = "http://cran.us.r-project.org")
  library(optparse)
}

# Load the package itself. This assumes the package is installed.
if (!require("speedAnalyzerR", quietly = TRUE)) {
  stop("The 'speedAnalyzerR' package is not installed. Please install it first.")
}

subcommands <- c("analyze", "convert")

args <- commandArgs(trailingOnly = FALSE)
script_name <- basename(sub(".*=", "", args[grep("--file=", args)]))

if (length(commandArgs(trailingOnly = TRUE)) == 0) {
  cat(paste("Usage:", script_name, "<subcommand> [options]\n"))
  cat("Available subcommands: analyze, convert\n")
  stop("No subcommand provided.", call. = FALSE)
}

sub <- commandArgs(trailingOnly = TRUE)[1]

if (!sub %in% subcommands) {
  stop(paste("Unknown subcommand:", sub, ". Available:", paste(subcommands, collapse = ", ")), call. = FALSE)
}

if (sub == "analyze") {
  # Placeholder for the main analysis logic
  cat("Subcommand 'analyze' is not yet implemented.\n")
  # Here you would parse arguments for the analysis and call the main analysis function.
}

if (sub == "convert") {
  # Define options for the 'convert' subcommand
  option_list <- list(
    make_option("--device", type = "character", help = "The source device name (e.g., 'tobii')."),
    make_option("--input", type = "character", help = "Path to the source data directory."),
    make_option("--out", type = "character", help = "Path to the output directory."),
    make_option("--format", type = "character", default = "bids", help = "The target output format (e.g., 'bids')."),
    make_option("--subject", type = "character", help = "BIDS subject ID (e.g., '01')."),
    make_option("--session", type = "character", help = "BIDS session ID (e.g., '01')."),
    make_option("--task", type = "character", help = "BIDS task name (e.g., 'visualsearch').")
  )
  
  # Note: optparse reads arguments after the script name, so we skip the subcommand
  args_to_parse <- commandArgs(trailingOnly = TRUE)[-1]
  opts <- parse_args(OptionParser(option_list = option_list), args = args_to_parse)
  
  # Collect BIDS info into a list
  bids_info <- list(
    subject_id = opts$subject,
    session_id = opts$session,
    task_name = opts$task
  )
  
  # Call the conversion function from the package
  run_device_conversion(
    device_name = opts$device,
    source_dir = opts$input,
    output_dir = opts$out,
    output_format = opts$format,
    bids_info = bids_info
  )
}