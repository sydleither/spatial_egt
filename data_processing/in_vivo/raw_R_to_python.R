library(sf)

extract_data_ellipses <- function(file_name, file) {
  data <- file$ellipses
  data$geometry <- st_coordinates(data$geometry)
  write.csv(data, paste0("data/raw/", file_name, "_ellipses", ".csv"),
            row.names = FALSE)
}

collapse_coords <- function(geometry) {
  c <- st_coordinates(geometry)[, 1:2]
  paste(apply(c, 1, function(row) paste(row, collapse = ",")), collapse = ";")
}

extract_data_regions <- function(file_name, file) {
  data <- file$regions
  data$coords <- sapply(data$geometry, collapse_coords)
  data <- st_drop_geometry(data)
  write.csv(data, paste0("data/raw/", file_name, "_regions", ".csv"),
            row.names = FALSE)
}

file_paths <- list.files(path = "data/raw/R", pattern = ".rds",
                         all.files = TRUE, full.names = TRUE)
for (file_path in file_paths) {
  file <- readRDS(file = file_path)
  file <- file[[1]]
  file_name <- substr(basename(file_path), 1, nchar(basename(file_path)) - 4)
  extract_data_regions(file_name, file)
  extract_data_ellipses(file_name, file)
}