# Utilities

#' A helper function to print messages with a package prefix and timestamp.
#' @param ... Items to be printed.
#' @keywords internal
pkg_message <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - ", paste0(..., collapse=""))
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}

# Read CSV if exists, else NULL
read_csv_safe <- function(path) {
  if (file.exists(path)) data.table::fread(path) else NULL
}