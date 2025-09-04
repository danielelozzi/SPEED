# Internal entry for Windows .bat
main_cli <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  script <- system.file("bin", "speed-analyzer-r", package = "speedAnalyzerR")
  if (nzchar(script) && file.exists(script)) {
    # Re-run the actual script with the same args
    cmd <- paste(shQuote(script), paste(shQuote(args), collapse = " "))
    system(cmd)
  } else {
    cat("speed-analyzer-r script not found in this installation.\n")
  }
  invisible(NULL)
}