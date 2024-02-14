# This script produces Table II of the paper
library(dplyr)
source("My_analysis/2. utils_analysis.R")
source("My_analysis/1. global.R")
raw_folder_plot <- "figures/"
crisis_var <- "crisisJST" # In the paper we explored two options: Jorda-SChularickTayolor (crisisJST) and the crisis definition by Baron (crisisBR)

years_exclude <-c(1912:1918, 1931:1945)


df <- df_processed

df$crisis_lead1[is.na(df$crisis_lead1)] <- 0
df$crisis_lead2[is.na(df$crisis_lead2)] <- 0

# our main dependent variable: set to one 1-2 years before a crisis
df$crisis_lead1to2 <- 1 * ((df$crisis_lead1 + df$crisis_lead2) > 0)
df$crisis_lead1to2[df$crisis ==1] <- NA # remove actual crisis year

# add post crisis flag - we remove 1-4 years after a crisis onset from the baseline sample
df$crisis_post <- 1 * ((df$crisis_lag1 + df$crisis_lag2 + df$crisis_lag3 + df$crisis_lag4) > 0)
df$crisis_post[is.na(df$crisis_post)] <- 0


# first model mimics the approach by Greenwood et al. (2022)
form1 <- as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff3 | country")
df_from_raw <- df # save for later analyses
df_reg <- df %>% filter(crisis_post !=1 & !year %in% years_exclude)
df_reg <- df_reg[, c("iso", "country", "year", "crisis_lead1to2", "tloan_gdp_rdiff3")] # only ued variables and remove missings because we want to have correct interpretation of standardised variables
df_reg <- na.omit(df_reg)
df_reg$tloan_gdp_rdiff3 <- scale(df_reg$tloan_gdp_rdiff3) # normalise variable for ease of interpretation
summary(model1 <- fixest::feglm(form1, family = "gaussian", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))



# same as above but Delta = 2
df_reg <- df %>% filter(crisis_post !=1 & !year %in% years_exclude)
df_reg <- df_reg[, c("iso", "country", "year", "crisis_lead1to2", "tloan_gdp_rdiff2")] # only ued variables and remove missings because we want to have correct interpretation of standardised variables
df_reg <- na.omit(df_reg)
df_reg$tloan_gdp_rdiff2 <- scale(df_reg$tloan_gdp_rdiff2) 
form1 <- as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff2 | country")
summary(model1b <- fixest::feglm(form1, family = "gaussian", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))


# model 2 adds our variables with our transformations but large sample (still OLS)
form2 <- as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff2 + global_loan + drate + global_drate | country")
scale_vars <- c("tloan_gdp_rdiff2", "global_loan", "drate", "global_drate")
df_reg <- df %>% filter(crisis_post !=1 & !year %in% years_exclude)
df_reg <- df_reg[, c("iso", "country", "year", "crisis_lead1to2", scale_vars)]
df_reg <- na.omit(df_reg)
df_reg[, scale_vars] <- scale(df_reg[, scale_vars])
summary(model2 <- fixest::feglm(form2, family = "gaussian", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))
summary(model2b <- fixest::feglm(form2, family = "logit", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))
#sum(df_reg$crisis_lead1to2==1)


# test whether model1 changes when having a consistent sample with model 2
summary(fixest::feglm(as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff2 | country"),
                      family = "gaussian",
                      data = df_reg,
                      vcov = "DK",
                      panel.id = c("iso", "year")))


# add additional covariates
scale_vars <- c("tloan_gdp_rdiff2", "global_loan", "drate", "global_drate",
                "cpi_pdiff2",
                "tdbtserv_gdp_rdiff2",
                "cons_pdiff2",
                "inv_gdp_rdiff2",
                "pdebt_gdp_rdiff2",
                "bmon_gdp_rdiff2",
                "stock_pdiff2",
                "ca_gdp_rdiff2"
)
suffix <- paste(scale_vars, collapse = " + ")

form3 <- as.formula(paste0("crisis_lead1to2 ~ ", suffix, "| country"))


df_reg <- df %>% filter(crisis_post !=1 & !year %in% years_exclude)
df_reg <- df_reg[, c("iso", "country", "year", "crisis_lead1to2", scale_vars)]
df_reg <- na.omit(df_reg)
df_reg[, scale_vars] <- scale(df_reg[, scale_vars])

summary(model3 <- fixest::feglm(form3, family = "logit", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))



# we get our baseline number of observations when not using country fixed effects!
form3 <- as.formula(paste0("crisis_lead1to2 ~ ", suffix))
model3b <- fixest::feglm(form3, family = "logit", data = df_reg, vcov = "DK", panel.id = c("iso", "year"))
summary(model3b <- fixest::feglm(form3, family = "logit", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))
#sum(df_reg$crisis_lead1to2==1)


model_list <- list(model1, model1b, model2, model2b, model3, model3b)
table2 <- fixest::etable(model_list, se.below = T, tex = F, dict = features_names_print)
print(table2)
latex_code <- xtable(table2)

# Print the LaTeX code
#print(latex_code, include.rownames = FALSE)

# The earlier regressions qualitatively do not change when using our baseline sample (missing values removed on all covariates) 

summary(fixest::feglm(as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff2 | country"), family = "gaussian", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))
summary(fixest::feglm(as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff2 + global_loan + drate + global_drate | country"), family = "gaussian", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))

summary(fixest::feglm(as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff2"), family = "gaussian", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))
summary(fixest::feglm(as.formula("crisis_lead1to2 ~ tloan_gdp_rdiff2 + global_loan + drate + global_drate"), family = "gaussian", data = df_reg, vcov = "DK", panel.id = c("iso", "year")))




