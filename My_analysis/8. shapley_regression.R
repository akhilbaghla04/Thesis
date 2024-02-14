#### produces Shapley regression (Table VI) ####

source("My_analysis/2. utils_analysis.R")
source("My_analysis/1. global.R")

algo <- "extree"
folder_experiment <- "cross-validation/1"
result_files <- list.files(paste0("results/", folder_experiment))

df_shap <- read.csv(paste0("results/", folder_experiment, "/shapley_append_", algo, ".txt"), sep ="\t")[,-1]
df <- read.csv(paste0("results/", folder_experiment, "/data.txt"), sep ="\t")[,-1]


n_obs <- nrow(df)
n_rep <- nrow(df_shap) / n_obs
iter <- rep(1:n_rep, each = n_obs)



f_names <- setdiff(colnames(df_shap), c("crisis", "iso", "pred", "year", "index"))

# based on this analysis defined a function
tab <- shapley_regression(data = df_shap, iter, y_name = "crisis", features = f_names, avg_fun = mean)
tab <- tab[-1,] # remove intercept
tab$shapley_share <- colMeans(abs(df_shap[,f_names]))
tab$shapley_share <- tab$shapley_share / sum(tab$shapley_share)
print(tab)

# get the "direction" from a logistic regression on the actual features (not Shapely values)
tab$direction <- sign(glm(df[, c("crisis", f_names)], family = binomial(link='logit'))$coefficients[-1])



# Your original data frame
original_tab <- data.frame(
  Variable = c("drate", "cpi_pdiff2", "bmon_gdp_rdiff2", "stock_pdiff2", "cons_pdiff2", "pdebt_gdp_rdiff2", "inv_gdp_rdiff2", "ca_gdp_rdiff2", "tloan_gdp_rdiff2", "tdbtserv_gdp_rdiff2", "global_loan", "global_drate"),
  Estimate = c(0.350140759, 0.346363507, -0.125930235, 0.092226625, 0.194386044, 0.001589028, 0.163821542, 0.096541329, 0.225700612, 0.159108638, 0.390371534, 0.566390066),
  SE = c(0.11890571, 0.11010743, 0.13336946, 0.09292108, 0.05491914, 0.13661695, 0.08284964, 0.07279643, 0.08933500, 0.09651768, 0.07363474, 0.15543577),
  p = c(1.925431e-03, 1.033813e-03, 1.000000e+00, 1.614215e-01, 2.811657e-04, 4.953691e-01, 2.509336e-02, 9.358916e-02, 6.380341e-03, 5.087150e-02, 2.489446e-07, 1.955466e-04),
  shapley_share = c(0.11427699, 0.07623505, 0.03409193, 0.01830194, 0.04076334, 0.04625652, 0.04339621, 0.04298019, 0.10891914, 0.06319236, 0.16916482, 0.24242152),
  direction = c(-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, -1)
)

# Named vector for mapping variable names
var_mapping <- c("drate" = "Yield curve slope",
                 "tloan_gdp_rdiff2" = "Credit",
                 "cpi_pdiff2" = "CPI",
                 "tdbtserv_gdp_rdiff2" = "Debt service ratio",
                 "cons_pdiff2" = "Consumption",
                 "inv_gdp_rdiff2" = "Investment",
                 "pdebt_gdp_rdiff2" = "Public debt",
                 "bmon_gdp_rdiff2" = "Broad money",
                 "stock_pdiff2" = "Stock market",
                 "ca_gdp_rdiff2" = "Current account",
                 "global_drate" = "Global yield curve slope",
                 "global_loan" = "Global credit")

# Rename variables using var_mapping
original_tab$Variable <- var_mapping[original_tab$Variable]
original_tab$p <- round(original_tab$p, 3)
# Subset and rename columns
subset_tab <- subset(original_tab, select = c("Variable", "direction", "shapley_share", "p"))

# Print the modified table
print(subset_tab)



