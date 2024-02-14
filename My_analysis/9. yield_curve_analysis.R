# This code produces the following tables and figures
# - Predictive power of yield curve slope for different horizons (Table VII)
# - recessions and the slope of the yield curve (Table VIII)
# - Analysis of short and long term rates (Table IX)
# - Interaction plot  nominal rates with the slope of the yield curve (Figure X)


library(dplyr)
source("My_analysis/2. utils_analysis.R")
source("My_analysis/1. global.R")
folder_plot <- "figures/"


#### predictive power of the yield curve at different horizons #####


years_exclude <-c(1912:1918, 1931:1945)
df <- df_processed
# if crisis lead > 1, delete years closer to crisis. We do the same in the ML prediction exercises when having a single year landing zone
df$crisis_lag2[df$crisis_lag1 == 1] <- NA
df$crisis_lag3[df$crisis_lag1 == 1 | df$crisis_lag2 == 1] <- NA
df$crisis_lag4[df$crisis_lag1 == 1 | df$crisis_lag2 == 1 | df$crisis_lag3 == 1] <- NA

df_reg <- df
df_reg$drate <- scale(df$drate) 
df_reg$global_drate <- scale(df_reg$global_drate) # standardise predictor


# not consistently significant - what if do sone cleaning?
# remove actual crises
df_reg <- df_reg[df_reg$crisis !=1,]
# remove post crisis (4 years)
df_reg$crisis_post <- 1 * ((df_reg$crisis_lag1 + df_reg$crisis_lag2 + df_reg$crisis_lag3 + df_reg$crisis_lag4) > 0)
df_reg$crisis_post[is.na(df_reg$crisis_post)] <- F
df_reg <- df_reg[df_reg$crisis_post !=1,]

# remove wars
df_reg <- df_reg[!df_reg$year %in% c(1914:1918, 1934:1945),]

h <- 1
df_reg$crisis_lead1[df_reg$year %in% c((1914-h):1918, (1933-h):1945)] <- NA
h <- 2
df_reg$crisis_lead2[df_reg$year %in% c((1914-h):1918, (1933-h):1945)] <- NA
h <- 3
df_reg$crisis_lead3[df_reg$year %in% c((1914-h):1918, (1933-h):1945)] <- NA
h <- 4
df_reg$crisis_lead4[df_reg$year %in% c((1914-h):1918, (1933-h):1945)] <- NA


# both global and domestic slope with all controls

# controls
covariates <- c("tloan_gdp_rdiff2", 
            "global_loan", 
            "cpi_pdiff2",
            "tdbtserv_gdp_rdiff2",
            "cons_pdiff2",
            "inv_gdp_rdiff2",
            "pdebt_gdp_rdiff2",
            "bmon_gdp_rdiff2",
            "stock_pdiff2",
            "ca_gdp_rdiff2"
)
suffix <- paste(covariates, collapse = " + ")

# scale variables



summary(mb_all_1_a <- fixest::feglm(as.formula(paste0("crisis_lead1 ~ drate + global_drate + ", suffix)),
                                    family = binomial(link = "logit"), vcov = "DK", panel.id = c("iso", "year"),
                                    data = df_reg))
summary(mb_all_2_a <- fixest::feglm(as.formula(paste0("crisis_lead2 ~ drate + global_drate + ", suffix)),
                                    family = binomial(link = "logit"), vcov = "DK", panel.id = c("iso", "year"),
                                    data = df_reg))
summary(mb_all_3_a <- fixest::feglm(as.formula(paste0("crisis_lead3 ~ drate + global_drate +",  suffix)),
                                    family = binomial(link = "logit"), vcov = "DK", panel.id = c("iso", "year"),
                                    data = df_reg))

# removing clustered crises
clustered_crises <- c(2007:2008, 1907:1908, 1930:1931)

summary(mb_all_1_b <- fixest::feglm(as.formula(paste0("crisis_lead1 ~ drate + global_drate + ", suffix)),
                                    family = binomial(link = "logit"), vcov = "DK", panel.id = c("iso", "year"),
                                    data = df_reg[!df_reg$year %in% (clustered_crises - 1),]))
summary(mb_all_2_b <- fixest::feglm(as.formula(paste0("crisis_lead2 ~ drate + global_drate + ", suffix)),
                                    family = binomial(link = "logit"), vcov = "DK", panel.id = c("iso", "year"),
                                    data = df_reg[!df_reg$year %in% (clustered_crises - 2),]))
summary(mb_all_3_b <- fixest::feglm(as.formula(paste0("crisis_lead3 ~ drate + global_drate + ", suffix)),
                                    family = binomial(link = "logit"), vcov = "DK", panel.id = c("iso", "year"),
                                    data = df_reg[!df_reg$year %in% (clustered_crises - 3),]))



fixest::etable(list(mb_all_1_a, mb_all_2_a, mb_all_3_a,
                    mb_all_1_b, mb_all_2_b, mb_all_3_b), tex = F)





#### recessions and yield curve effect ####


df_recession <- df_raw %>% group_by(iso) %>% mutate(real_gdp_growth = (rgdppc - lag(rgdppc,1))/lag(rgdppc,1))
df_recession$recession <- ifelse(df_recession$real_gdp_growth < 0, 1, 0)
df_recession$recession[is.na(df_recession$recession)] <- 99 #replace missings
df_recession$first_recession <- 0




df <- df_processed
df$crisis_lead1[is.na(df$crisis_lead1)] <- 0
df$crisis_lead2[is.na(df$crisis_lead2)] <- 0
# our main dependent variable: set to one 1-2 yearsbefore a crisis
df$crisis_lead1to2 <- 1 * ((df$crisis_lead1 + df$crisis_lead2) > 0)
df$crisis_lead1to2[df$crisis ==1] <- NA # remove actual crisis year
df$crisis_post <- 1 * ((df$crisis_lag1 + df$crisis_lag2 + df$crisis_lag3 + df$crisis_lag4) > 0)
df$crisis_post[is.na(df$crisis_post)] <- 0


# remove actual crises
years_exclude <-c(1912:1918, 1931:1945)
df <-  df %>% filter(crisis_post !=1 & !year %in% years_exclude)



df$CandRsame_start <- 0
df$CandR_RbefC <- 0
df$CandR_CbefR <- 0

for(i in 1:nrow(df_recession)){
  
  same_country_lag1 <- (df_recession$iso[i-1] == df_recession$iso[i])
  same_country_lag2 <- (df_recession$iso[i-2] == df_recession$iso[i])
  same_country_lead1 <- (df_recession$iso[i+1] == df_recession$iso[i])
  same_country_lead2 <- (df_recession$iso[i+2] == df_recession$iso[i])
  
  iso <- df_recession$iso[i]
  years_lag <- c(df_recession$year[i] - 1, df_recession$year[i] - 2) 
  ix <- which((df$year %in% years_lag) & (df$iso == iso))
  
  if(df_recession$crisis[i] == 1){ # in all following conditions tested, a crisis must have occurred.
    
    
    if(same_country_lag1 &
       df_recession$recession[i] == 1 & 
       df_recession$recession[i-1] == 0){
      
      # recession onset coincides with crisis onset
      df$CandRsame_start[ix] <- 1
    }
    
    
    # recession preceeds crisis by 1-2 years
    if((same_country_lag1 & df_recession$recession[i-1] == 1) | 
       (same_country_lag2 & df_recession$recession[i-2] == 1)
    ){
      df$CandR_RbefC[ix] <- 1
    }
    
    
    # crisis is followed by a recession
    if((same_country_lead1 & df_recession$recession[i+1] == 1) | 
       (same_country_lead2 & df_recession$recession[i+2] == 1)
    ){
      df$CandR_CbefR[ix] <- 1
    }
    
  }
  
}



# do well cover all crises? 

predictors <- c(
  "drate",
  "global_drate",
  "tloan_gdp_rdiff2", 
  "global_loan",
  "cpi_pdiff2",
  "tdbtserv_gdp_rdiff2",
  "cons_pdiff2",
  "inv_gdp_rdiff2",
  "pdebt_gdp_rdiff2",
  "bmon_gdp_rdiff2",
  "stock_pdiff2",
  "ca_gdp_rdiff2"
)

df <- df[complete.cases(df[,c("crisis_lead1to2", "CandR_RbefC", "CandRsame_start", predictors)]),]


predictors_key <- c("drate", "tloan_gdp_rdiff2", "global_loan_leave", "global_drate_leave" )

# recession before crisis or same year
ix_RbefC <- (df$crisis_lead1to2 == 0 | df$CandR_RbefC | df$CandRsame_start) # preceded and oco-ooccur
dat_RbefC <- df[ix_RbefC,]
dat_RbefC[,predictors] <- scale(dat_RbefC[, predictors])


# recession after crisis or same year onset
ix_otherC <- df$crisis_lead1to2 == 0 | (df$crisis_lead1to2 == 1 & !(df$CandR_RbefC | df$CandRsame_start)) # not preceded and co-occur
dat_otherC <- df[ix_otherC,]
dat_otherC[, predictors] <- scale(dat_otherC[, predictors])


suffix <- paste(predictors, collapse = " + ")

form <- as.formula(paste0("crisis_lead1to2 ~ ", suffix))


# this approach (with clustered standard errors) is preferable:
summary(model_recession <- fixest::feglm(form, family = "logit", data = dat_RbefC, vcov = "DK", panel.id = c("iso", "year")))
summary(model_other <- fixest::feglm(form, family = "logit", data = dat_otherC, vcov = "DK", panel.id = c("iso", "year")))

fixest::etable(list(model_other, model_recession), se.below = T, tex = T, dict = features_names_print)

sum(dat_RbefC$crisis_lead1to2)
sum(dat_otherC$crisis_lead1to2)






#### yield curve interactions with short and long rate #### 


df <- df_processed
df <- df %>% mutate(srate_real = srate - cpi_pdiff1,
              lrate_real = lrate - cpi_pdiff1,
              )


covariates <- c("tloan_gdp_rdiff2", 
            "global_loan", 
            "cpi_pdiff2",
            "tdbtserv_gdp_rdiff2",
            "cons_pdiff2",
            "inv_gdp_rdiff2",
            "pdebt_gdp_rdiff2",
            "bmon_gdp_rdiff2",
            "stock_pdiff2",
            "ca_gdp_rdiff2"
)

scale_vars <- c("srate",
                "lrate",
                "srate_real",
                "lrate_real",
                "drate",
                covariates
)


df$crisis_lead1[is.na(df$crisis_lead1)] <- 0
df$crisis_lead2[is.na(df$crisis_lead2)] <- 0
# our main dependent variable: set to one 1-2 yearsbefore a crisis
df$crisis_lead1to2 <- 1 * ((df$crisis_lead1 + df$crisis_lead2) > 0)
df$crisis_lead1to2[df$crisis ==1] <- NA # remove actual crisis year
df$crisis_post <- 1 * ((df$crisis_lag1 + df$crisis_lag2 + df$crisis_lag3 + df$crisis_lag4) > 0)
df$crisis_post[is.na(df$crisis_post)] <- 0


# remove actual crises
years_exclude <- c(1912:1918, 1931:1945)
df_reg <- df %>% filter(crisis_post !=1 & !year %in% years_exclude)

# remove post crisis (4 years)
df_reg$crisis_post[is.na(df_reg$crisis_post)] <- F


df_rates <- na.omit(df_reg[, c("crisis_lead1to2", scale_vars, c("iso", "year"))])
df_rates_not_scaled <- df_rates # we want to keep actual levels for ease of interpretation of the interaction plot
df_rates[, scale_vars] <- scale(df_rates[, scale_vars])

covariates_term <- paste(covariates, collapse = " + ")

prefixses_nominal <- c("drate",
                       "srate",
                       "lrate",
                       "srate + lrate",
                       "drate * srate",
                       "drate * lrate")

prefixses_real <- c("drate",
                    "srate_real",
                    "lrate_real",
                    "srate_real + lrate_real",
                    "drate * srate_real",
                    "drate * lrate_real")

models_nominal <- lapply(prefixses_nominal, function(x)
  fixest::feglm(as.formula(paste0("crisis_lead1to2 ~ ", x, " + ", covariates_term)), 
                family = "logit", data = df_rates,
                vcov = "DK",
                panel.id = c("iso", "year")))


models_real <- lapply(prefixses_real, function(x)
  fixest::feglm(as.formula(paste0("crisis_lead1to2 ~ ", x, " + ", covariates_term)), 
                family = "logit", data = df_rates,
                vcov = "DK",
                panel.id = c("iso", "year")))





fixest::etable(models_nominal, se.below = T, tex = T, dict = features_names_print, digits = 3)
fixest::etable(models_real, se.below = T, tex = T, dict = features_names_print, digits = 3)


# interaction plot 


# replciate models as a glm to predocue interaction plot
models_nominal_glm <- lapply(prefixses_nominal, function(x)
  fixest::feglm(as.formula(paste0("crisis_lead1to2 ~ ", x, " + ", covariates_term)), 
                family = "logit", data = df_rates,
                vcov = "DK",
                panel.id = c("iso", "year")))




xname <- "drate"

for(yname in c("srate", "lrate")){
  
  if(yname == "srate"){
    model_use <- models_nominal_glm[[5]]
    
  }
  if(yname == "lrate"){
    model_use <- models_nominal_glm[[6]]
  }
  

  
  
  x_scaled <- df_rates[[xname]]
  y_scaled <- df_rates[[yname]]
  x <- df_rates_not_scaled[[xname]]
  y <- df_rates_not_scaled[[yname]]
  dat_reg_inter <- df_rates
  dat_reg_inter[,setdiff(colnames(dat_reg_inter), c("crisis", "year", "iso", xname, yname ))] <- 0
  dat_reg_inter[[xname]] <- sort(dat_reg_inter[[xname]])
  
  dir.create("figures/descriptive/", showWarnings = FALSE)
  cairo_pdf(paste0(folder_plot, "descriptive/interact_regression_", yname, ".pdf"), width = 3.8, height = 3)
  par(mar = c(3,3,1,1))
  plot.new()
  plot.window(xlim = c(min(x), max(x)), ylim = c(0,1))
  axis(1); axis(2)
  title(xlab = features_names_print[xname], ylab = "Predicted probability", line = 2)
  legend("topright", col = c("blue", "black", "red"), lty = 1, 
         legend = paste0(c("+1 SD (", "Mean (", "-1 SD ("), round(c(mean(y) + sd(y), mean(y), mean(y) - sd(y)),2), ")"),
         title = ifelse(yname == "srate", "Nominal short-term rate", "Nominal long-term rate"), bty = "n")
  # legend(x = min(x), y = .7, pch = c(20, 4, 0, 2), legend = c("Great financial crisis", "Great depression", "Crises in 1990s", "Other crises"), bty = "n")
  
  dat_reg_inter[[yname]] <- mean(df_rates[[yname]])
  lines(sort(x), predict(model_use, dat_reg_inter, "response"), type = "l")
  
  dat_reg_inter[[yname]] <- mean(df_rates[[yname]]) + 1
  lines(sort(x), predict(model_use, dat_reg_inter, type = "response"), col = "blue")
  
  dat_reg_inter[[yname]] <- mean(df_rates[[yname]]) - 1
  lines(sort(x), predict(model_use, dat_reg_inter, type = "response"), col = "red")
  dev.off()
}




