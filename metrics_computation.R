# clear memory and environment
rm(list = ls(all.names = TRUE))

# library installation and load
ensure_package_installed_and_loaded <- function(my_package_name) {
  if (!require(my_package_name, character.only = TRUE)) {
    install.packages(my_package_name, dependencies = TRUE)
    library(package_name, character.only = TRUE)
  }
}
sapply(c("readr", "caret", "cluster"), ensure_package_installed_and_loaded)

adult <- read_csv("tests/tests_files/adult.csv")
car_insurance_claim <- read_csv("tests/tests_files/car_insurance_claim.csv")
diabetes <- read_csv("tests/tests_files/diabetes.csv")
survey <- read_csv("tests/tests_files/survey.csv")

# numerical variables normalization 0-1
normalize <- function(my_input){
  my_numeric_columns <- sapply(my_input, is.numeric)
  
  # check if there are any numeric columns to normalize
  if (sum(my_numeric_columns) == 0) {
    stop("Brak kolumn numerycznych do normalizacji w pliku: ", my_input)
  }
  
  preproc <- preProcess(my_input[, my_numeric_columns], method = c("range"))
  normalized_data <- predict(preproc, my_input)
  
  return(normalized_data)
}