# SUPPLEMENTARY CODE FOR BOE SWP 848: 
# Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 
# This script installs all R packages required to reproduce the results.
install.packages("devtools")
# packages and their versions used
packages <- list(
  c("dplyr", "1.0.7"),
  c("pROC", "1.18.0"),
  c("readxl", "1.3.1"),
  c("xtable", "1.8-4"),
  c("fixest", "0.10.1"),
  c("MLmetrics", "1.1.1"),
  c("PRROC", "1.3.1"),
  c("C50", "0.1.6"),
  c("lmtest", "3.1-3"),
  c("sandwich", "3.0-1"),
  c("openxlsx", "4.1.0.1")
)


lapply(packages, function(package) install.packages(package[1],
                                                    version = package[2],
                                                    repos = "http://cran.us.r-project.org"
                                                    ))
