# This script creates the following figures:
#  - ROC and Precision Recall curves (Figure II)
#  - Prediction chart (Figure III)
#  - GDP losses (Table IV)
#  - Shapley variable importance (Figure V)
#  - Shapley difference in different periods (Figure XI)
#  - Shapley decomposition by country (Figure VIII)
#  - Learned functional form (Figure IX)

source("My_analysis/2. utils_analysis.R")
source("My_analysis/1. global.R")
library(grDevices)
library(dplyr)

experiment_id <- 511 # 1 refers to baseline specification (see list_experiments.py)

folder_data <- paste0("results/cross-validation/", experiment_id, "/") # where the results are found,  /1/ refers to the baseline experiment
folder_figures <- paste0("figures/cross-validation/", experiment_id, "/") # where the figures are placed 
dir.create(folder_figures)


# load dataset
dataset <- read.csv(paste0(folder_data, "data.txt"), sep = "\t", stringsAsFactors = F)[,-1]
n_observations <- nrow(dataset)
all_years <- min(dataset$year): max(dataset$year)
countries <- unique(dataset$iso)
n_countries <- length(countries)
true_class <- dataset$crisis

# load predictions
predictions <- read.csv(paste0(folder_data, "all_pred.txt"), sep = "\t", stringsAsFactors = F)[,-1]
algos <- setdiff(colnames(predictions), c("year", "iso", "iter", "crisis", "index", "fold"))
n_replications <- nrow(predictions)/n_observations
n_folds <- max(predictions$fold)
predictions$index <- rep(1:n_observations, n_replications)


# compute mean prediction across all iterations
predictions_mean <- predictions[predictions$iter ==0,]
predictions_mean[, algos] <- predictions %>%
  group_by(index) %>%
  select(c(index, !! algos)) %>%
  summarise_all(mean) %>%
  ungroup() %>% select(-index) %>% data.frame()


algos_shapley <- setdiff(algos, "r_c50") # we cannot produce Shapley values for the model trained in R (r_c50)
shapleys <- lapply(algos_shapley, function(algo)
  read.csv(paste0(folder_data, paste0("shapley_append_", algo, ".txt")), sep = "\t", stringsAsFactors = F)[,-1])
features <- setdiff(colnames(shapleys[[1]]), c("year", "iso", "crisis", "pred"))
shapleys_mean <- lapply(algos_shapley, function(algo)
  read.csv(paste0(folder_data, paste0("shapley_mean_", algo, ".txt")), sep = "\t", stringsAsFactors = F)[,-1])

names(shapleys) <- names(shapleys_mean) <-  algos_shapley



algos_show <- algos # the user may want to subset the models she wants to show
algos_show_shapley <- algos_shapley
features_show <- c("drate", "tloan_gdp_rdiff2", "global_drate", "global_loan")# the user may want to subset the features she wants to show
minimum_hitrate <- 0.8 # main model


#### Prediction charts####
model <- "extree"
# try all threshold on the predicted probability and pcik the one that is closest to our minimum hit rate.
all_thresholds <- sort(unique(predictions_mean[,model]))
performance_thresholds <- sapply(all_thresholds, function(x) measurePerf(predictions_mean$crisis,
                                                                         predictions_mean[,model], threshold = x))
threshold_ix <- which.min(abs(performance_thresholds["tp.rate",] - minimum_hitrate))
threshold <- all_thresholds[threshold_ix]

col_hit <- "forestgreen"
col_tn <- "darkseagreen1"
col_fa <- "gray75"
col_miss <- "red"
scaler = .9

cairo_pdf(paste0(folder_figures, "prediction_matrix_",model, ".pdf"), height = 8, width = 12)
cx <- 0.7
par(mar = c(3,7,3,1))
plot.new()
plot.window(xlim = c(min(all_years), max(all_years)+4), ylim = c(1,n_countries+ .5))
axis(1, at = min(all_years):max(all_years), labels = NA, las = 2, cex.axis = 1, tck = -0.01)
axis(1, at = (min(all_years):max(all_years))[seq(1,length((min(all_years):max(all_years))),5)], las = 2, cex.axis = 1)
axis(2, at = .5 + (1:n_countries), labels = country_names_print[countries], las = 2)
par(xpd = T)
legend(x = 1950,y = 19.5, legend = c("Correct crises","Correct non-crises", "Missed crises", "False alarms")[1:2],
       col = c(col_hit, col_tn, col_miss, col_fa)[1:2],
       border = c(col_hit, col_tn, col_miss, col_fa)[1:2],
       fill = c(col_hit, col_tn, col_miss, col_fa)[1:2], 
       bty = "n", y.intersp = 0.8)
legend(x = 1985,y = 19.5, legend = c("Correct crises","Correct non-crises", "Missed crises", "False alarms")[3:4],
       col = c(col_hit, col_tn, col_miss, col_fa)[3:4],
       border = c(col_hit, col_tn, col_miss, col_fa)[3:4],
       fill = c(col_hit, col_tn, col_miss, col_fa)[3:4], 
       bty = "n", y.intersp = 0.8)



par(xpd = F)

counter <- 0
for(country in countries){
  counter <- counter + 1
  abline(h = counter, lty = 3, col = "gray50", lwd = 0.6)
  ix = predictions_mean$iso == country
  pred <- predictions_mean[ix, model]
  crit <- predictions_mean[ix,"crisis"]
  years <- predictions_mean[ix,"year"]
  yearsadd <- setdiff(all_years,predictions_mean[ix,"year"])
  
  years <- c(years, yearsadd)
  oo <- order(years)
  years <- years[oo]
  crit <- c(crit,rep(NA, length(yearsadd)))[oo]
  pred <- c(pred,rep(NA, length(yearsadd)))[oo]
  
  ix_hit = crit == 1 & pred >= threshold; ix_hit[is.na(ix_hit)] <- F
  ix_miss = crit == 1 & pred < threshold; ix_miss[is.na(ix_miss)] <- F
  ix_fa = crit == 0 & pred >= threshold; ix_fa[is.na(ix_fa)] <- F
  ix_tn = crit == 0 & pred < threshold; ix_tn[is.na(ix_tn)] <- F
  n_all <- sum(ix_hit) + sum(ix_miss) + sum(ix_fa) + sum(ix_tn)
  rect(xleft = years[ix_hit], xright = years[ix_hit] + .8, ybottom = rep(counter, sum(ix_hit)), ytop = rep(1 + counter, sum(ix_hit)),col = col_hit, pch = 16, cex = cx + 0.2, border = NA)
  rect(xleft = years[ix_miss], xright = years[ix_miss] + .8, ybottom =  rep(counter, sum(ix_miss)), ytop =  rep(1 + counter, sum(ix_miss)), col = col_miss, bg = col_miss, pch = 24, cex = cx, border = NA)
  rect(xleft = years[ix_fa], xright = years[ix_fa] + .8, ybottom = rep(counter, sum(ix_fa)), ytop = rep(1 + counter, sum(ix_fa)), col = col_fa, bg = col_fa, pch = 25, cex = cx - .05, border = NA)
  rect(xleft = years[ix_tn], xright = years[ix_tn] + .8, ybottom = rep(counter, sum(ix_tn)), ytop = rep(1 + counter, sum(ix_tn)), col = col_tn, bg = col_fa, pch = 25, cex = cx - .05, border= NA)
  
  lines(years + .45, pred + counter, type = "o", pch = 20, cex =.3) # predicted prob  
  
  cols <-  c(col_hit,col_tn, col_fa, col_miss)
  xses <- c(sum(ix_hit), sum(ix_tn), sum(ix_fa), sum(ix_miss))
  cols <- cols[xses!=0]
  xses <- xses[xses!=0]
  
}
dev.off()











#### ROC curves and precision recall curves ####

nrep <- 100
hit_test <- c(.7, .8)
rocpoints_proper <- list()
tp_rates = seq(0.0, 1, by = 0.05)
auc_all <- array(NA, dim = c(nrep, length(algos_show),2))
dimnames(auc_all) <- list(1:nrep, algos_show, c("auroc", "aupr"))
hit_false_alarm_thresholds <- array(NA, dim = c(nrep, length(algos_show), length(hit_test), 3))
dimnames(hit_false_alarm_thresholds) <- list(1:nrep, algos_show, hit_test, c("tpr", "fpr", "threshold"))

for(m in algos_show){
  tt <- predictions[[m]]
  nrep <- min(c(100, length(tt)/n_observations))
  
  out_single_mod <- array(NA, dim = c(nrep, length(tp_rates) + 2,3))
  counter <- 0
  for(i in 1:nrep){
    ix <- (1:n_observations)+(i-1) * n_observations
    auc_all[i,m, ] <- measurePerf(true_class, tt[ix])[c("auc", "aupr")]
    unique_threshes <- sort(unique(tt[ix]))
    unique_threshes <- unique(c(0, unique_threshes, 1))
    pp <- sapply(unique_threshes, function(t) rocPoint(true_class, tt[ix], t))
    ordr <- order(pp[2,])
    pp <- pp[,ordr]
    unique_threshes <- unique_threshes[ordr]
    ix_p <- sapply(tp_rates,function(x)which.min(abs(pp[1,] - x))[1])
    out_single_mod[i,,] <- t(cbind(c(0,0,NA),pp[,ix_p], c(1,1, NA)))
    
    for(h in hit_test){
      ix <- which.min(abs(pp["tp_rate",] - h))
      hit_false_alarm_thresholds[i, m, as.character(h),] <- c(pp["tp_rate", ix], pp["fp_rate", ix], unique_threshes[ix])
    }
  }
  out_single_mod[, 3,]
  rocpoints_proper[[m]] <- apply(out_single_mod,2:3, mean, na.rm = T)
}


auc_mean <- apply(auc_all, 2:3, mean, na.rm = T)
auc_iter <- apply(auc_all, 2:3, function(x) sum(!is.na(x)))
auc_se <- apply(auc_all, 2:3, sd, na.rm = T) / sqrt(auc_iter)

auc_summary <- cbind(auc_mean[,1], auc_se[,1], auc_iter[,1])
aupr_summary <- cbind(auc_mean[,2], auc_se[,2], auc_iter[,2])
dimnames(auc_summary) <- dimnames(aupr_summary) <- list(algos_show, c("mean", "se", "iter"))

hit_false_alarm_thresholds_mean <- apply(hit_false_alarm_thresholds, 2:4, mean, na.rm = T)
hit_false_alarm_thresholds_se <- apply(hit_false_alarm_thresholds, 2:4, sd, na.rm = T) / auc_iter[,1]

# ROC curves

ordr <- order(auc_mean[algos_show, "auroc"], decreasing = T)

cairo_pdf(paste0(folder_figures, "roc.pdf"), height = 5, width = 5)
par(mar = c(3,3,0.5,0.5))
plot.new()
plot.window(xlim = c(0,1), ylim = c(0,1))
axis(1); axis(2)
abline(0,1, lty = 2, col = "gray50")
title(xlab = "False positive rate", ylab = "True positive rate", line = 2)


for(m in algos_show){
  x <- rocpoints_proper[[m]][,2]
  y <- rocpoints_proper[[m]][,1]
  
  ix <- !is.na(x)
  lines(x[ix],y[ix], col = algos_col[m], pch = algos_pch[m], type = "o", lwd = 1.5)
}
legend("bottomright", legend = paste0(algos_names_print[algos_show[ordr]], " (AUC = ", 
                                      format(round(auc_mean[algos_show[ordr], "auroc"],2),  nsmall = 2),   ")"),
       col = algos_col[algos_show][ordr], pch = algos_pch[algos_show][ordr], bty = "n", y.intersp = 0.9, lty = 1, lwd = 2)
dev.off()



# Precision Recall curve

cairo_pdf(paste0(folder_figures, "pre_rec.pdf"), height = 5, width = 5)
par(mar = c(3,3,0.5,0.5))
plot.new()
plot.window(xlim = c(0,1), ylim = c(0,1))
axis(1); axis(2)
#abline(0,1, lty = 2, col = "gray50")
title(xlab = "Recall", ylab = "Precision", line = 2)
for(m in algos_show){
  x <- rocpoints_proper[[m]][,1]
  y <- rocpoints_proper[[m]][,3]
  # kick first three measurements out for stability reasons. 1-2 are NA anyway. And 3: when hit rate (recall) is very low, the precsiion estiamte will be unstable
  x <- x[-c(1:3)]
  y <- y[-c(1:3)]
  
  ix <- !is.na(x)
  lines(x[ix],y[ix], col = algos_col[m], pch = algos_pch[m], type = "o", lwd = 1.5)
  abline(h = mean(true_class), lty = 2, col = "gray50")
}
par(xpd = T)
legend(x=.285, y=1.09, legend = paste0(algos_names_print[algos_show[ordr]], " (AUPR = ", 
                                       format(round(auc_mean[algos_show[ordr], "aupr"],2),  nsmall = 2),   ")"),
       col = algos_col[algos_show][ordr], pch = algos_pch[algos_show][ordr], bty = "n", y.intersp = 0.9, lty = 1, lwd = 2)

par(xpd = F)
dev.off()




#### GDP losses ####

data_merged <- dataset
data_merged$index <- 1:nrow(data_merged)

# get real GDP from the raw data
data_merged <- merge(data_merged[, c("iso", "year", "crisis", "index")], df_raw[, c("year", "iso", "rgdppc")], by = c("year", "iso"), all = T)
data_merged <- data_merged %>% group_by(iso) %>% mutate(gdp_growth_3y_past = perc_change(rgdppc, 3)) # 3 year GDP growth
data_merged <- data_merged %>% group_by(iso) %>% mutate(gdp_growth_3y = leadit(gdp_growth_3y_past, lag = 5)) # lead it by 5* years:
# * 3 years to account for fact that we calculate change in past, 2 years because our crisis variable is 2 years early
data_merged <- data_merged %>% group_by(iso) %>% mutate(crisis_next = leadit(crisis, lag = 1))# help variable to filter 

# we just consider predictions two years before crisis (and no crisis). ie.e if it is one year before crisis we drop the observations
ix_exclude <- data_merged$crisis == 1 & is.na(data_merged$crisis_next) # with is.na(crisis_next) we remove observations one year before actual crisis observation. Because we removed all crisis observations from our data and use the have 1-2 year pre-period.
data_merged <- data_merged[!ix_exclude,]
data_merged <- data_merged[!is.na(data_merged$index),]


# note that crisis= 1, flags two years before crisis
# we iterate through replications  of the cross-validation procedure

gdp_losses <- list()
for(m in algos_show){
  
  preds <- predictions[[m]]
  n_obs <-  nrow(dataset)
  n_rep <- length(preds) / n_obs
  preds <- data.frame(pred = preds, index = rep(1:n_obs, n_rep), iteration = rep(1:n_rep, each = n_obs))
  n_rep <- min(c(n_rep, 100))
  
  hits <- fas <- misses <- correct_reject <- list()
  meta_list <- list(hits, fas, misses, correct_reject)
  
  
  for(i in 1:n_rep){
    dat_use <- data_merged
    dat_use <- merge(dat_use, preds[preds$iteration == i,], by = "index")
    
    pt <- dat_use$pred >=  hit_false_alarm_thresholds[i, m, "0.8", "threshold"] # use thresholding
    ix_hit <- dat_use$crisis == 1 & pt
    ix_hit[is.na(ix_hit)] <- F
    
    ix_fa <- dat_use$crisis == 0 & pt
    ix_fa[is.na(ix_fa)] <- F
    
    ix_miss <- !pt & dat_use$crisis == 1
    ix_miss[is.na(ix_miss)] <- F
    
    ix_correct_rej <- !pt & dat_use$crisis == 0
    ix_correct_rej[is.na(ix_correct_rej)] <- F
    
    # crisis year + 2 years after
    meta_list$hits[[i]] <-  dat_use$gdp_growth_3y[ix_hit]
    meta_list$fas[[i]] <- dat_use$gdp_growth_3y[ix_fa]
    meta_list$misses[[i]] <- dat_use$gdp_growth_3y[ix_miss]
    meta_list$correct_reject[[i]] <- dat_use$gdp_growth_3y[ix_correct_rej]
  }
  gdp_losses[[m]] <- sapply(c("hits", "fas", "misses", "correct_reject"), function(x)
    quantile(unlist(meta_list[[x]]), c(.01, .05, .1, .5, .9,  .95, .99), na.rm = T))
  
}

# Table IV
gdp_losses$extree[c("5%", "10%", "50%"), c("hits", "misses", "fas", "correct_reject")]


#### Shapley variable importance ####

mean_shap_abs <- sapply(shapleys, function(x) colMeans(abs(x[,features]))) # mean absolute values
mean_shap_abs <- apply(mean_shap_abs, 2, function(x) x/sum(x)) # normalise such that mean Shapley vlaues sum to 1 for each model.

cairo_pdf(paste0(folder_figures, "mean_absolute_shapley.pdf"), width = 7, height = 5)
oo <- order(mean_shap_abs[,"extree"], decreasing = T)
par(mar = c(8,4,2,1))
plot.new()
plot.window(xlim = c(1, length(features)), ylim = c(0,.25)) #ylim = c(0, max(pp_allModels)))
axis(2)
#axis(1, at = 1:length(feature_names), labels = features_names_print[oo] , las = 2)
text(1:length(features), -.03, srt = 60, adj= 1, xpd = TRUE, labels = features_names_print[features[oo]], cex=1)
axis(1, at = 1:length(features), labels = NA)
title(ylab = "Mean absolute Shapley values \n(normalized)", line = 2)
legend("topright", legend = algos_names_print[algos_show_shapley],
       col = algos_col[algos_show_shapley], pch = algos_pch[algos_show_shapley],
       lty = algos_lty[algos_show_shapley], bty = "n", y.intersp = 0.8, cex = 1.2)
for(m in algos_show_shapley)
  lines(mean_shap_abs[oo,m], col = makeTransparent(algos_col[m], alpha = .75),
        pch = algos_pch[m], type = ifelse(is.na(algos_lty[m] == 1), "p", "o"), cex = 1.5, lwd = 1.6)
dev.off()

#### Shapley difference in different periods #### 



#### Shapley periods ####
colc = "gray70"
period_all <- list("Complete data (1870 - 2016)", 1500, 2500)
period_pre <- list("Before WW2 (1870 - 1933)", 1870, 1945)
period_post  <- list("After WW2 (1946 - 2016)", 1946, 2016)
period_pre <- list("Before WW2 (1870 - 1933)", 1870, 1945)

period_90  <- list("1990s crises (1985 - 1992)", 1985, 1992)
period_gfc  <- list("Global financial crisis (2004 - 2010)", 2004, 2010)
periods <- list(period_all, period_pre, period_post, period_90, period_gfc) 
names(periods) <- c("all", "pre", "post", "90s", "gfc")



  

oo <- order(mean_shap_abs[,"extree"], decreasing = T) # # order variables
m <- "extree" # choose prediction model for which we plot Shapley values
for(i in 1:length(periods)){
  cairo_pdf(paste0(folder_figures, "shap_period_", names(periods)[i],  ".pdf"), width = 4.5, height = 3)
  par(mar = c(6.6,3,1,0))
  plot.new()
  plot.window(xlim = c(0.5, length(features)), ylim = c(0,.2))
  title(main = periods[[i]][[1]], line = 0)
  abline(h = 0, col = "black")
  axis(2)
  
  ix_time <- shapleys_mean[[m]]$year>= periods[[i]][[2]] & shapleys_mean[[m]]$year<= periods[[i]][[3]] 
  shap_cris <- shapleys_mean[[m]][ix_time & shapleys_mean[[m]]$crisis ==1, features]
  shap_no_cris <- shapleys_mean[[m]][ix_time & shapleys_mean[[m]]$crisis ==0, features]
  shap_dif_period <- (apply(shap_cris, 2, mean) - apply(shap_no_cris, 2, mean))
  
  lines(shap_dif_period[oo], type = "h", lend = 1, lwd = 26, col = colc)
  # superimpose values of whole data set:
  # lines(shap_decade_crisis["all", fplot] - shap_decade_nocrisis["all", fplot], type = "p", lend = 1, lwd = 1, col = "black", pch = 20)
  axis(1, at =1:length(features), labels = NA)
  par(lheight=.7)
  lab <- features_names_print[features[oo]]
  lab <- ifelse(lab == "CPI", "CPI ", lab)
  text(1:length(features), -.02, 
       srt = 60, adj= 1.1, xpd = TRUE,
       labels = lab, cex=1)
  title(ylab = "Shapley difference", line = 2)
  dev.off()
}



#### Learned functonal form ####
for (f in features) {
  cairo_pdf(paste0(folder_figures, "scatter_", f, "_", model, ".pdf"), width = 4, height = 4)
  # png(paste0(plot_folder,"correlation/corr_feature_shap_",m,"_", f, ".png"), width = 400, height = 400)
  par(mar = c(3,3,2,0.5))
  feature_values = dataset[,f]
  shap_values <- shapleys_mean[[model]][,f]
  
  plot.new()
  plot.window(xlim = minmax(feature_values), ylim = minmax(shap_values))
  axis(1); axis(2)
  ix <- true_class == 1
  points(feature_values[!ix], shap_values[!ix], pch = 20, cex = .75,
         col = makeTransparent("gray50", alpha = 0.5))# first non-crises
  points(feature_values[ix], shap_values[ix], pch = 20, cex = .75,
         col = makeTransparent("red",alpha = 0.5))# then crises
  model_degree1 <- lm(shap_values ~ feature_values)
  ol = order(feature_values, decreasing = T)
  lines(feature_values[ol], fitted(model_degree1)[ol], col = "black")
  
  model_degree3 <- lm(shap_values ~ poly(feature_values, 3))
  lines(feature_values[ol], fitted(model_degree3)[ol], col = "blue")
  
  abline(h = 0, lwd = 0.5, col = "gray50")
  title(xlab = "Predictor values", ylab = "Shapley values", line = 1.8)
  title(main = features_names_print[f], line = 1)
  
  legpos <- ifelse(f %in% c("drate", "global_drate", paste0("cpi_pdiff2")), "topright", "topleft")
  
  legend(legpos, bty = "n", text.col = c("black","blue", "black", "black"), pch = c(NA, NA, 20, 20), legend =
           c(as.expression(bquote(R['degree = 1']^2 == .(round(cor(shap_values, feature_values)^2,2)))),
             as.expression(bquote(R['degree = 3']^2  == .(round(cor(shap_values, model_degree3$fitted.values)^2,2)))),
             "Crisis", "Non-crises"), col = c(NA, NA, "red", "gray50")
  )
  dev.off()
}

#### Shapley decomposition by country ####

alpha = 0.75
intercept <- mean(shapleys_mean[[model]][,"pred"])  - mean(rowSums(shapleys_mean[[model]][, features]))

for(country in countries){
  df <- shapleys_mean[[model]]
  ix = df$iso == country
  years_span <- min(df$year):max(df$year)
  lw = 3.7
  ylim <- c(-0.1,.6)
  
  cairo_pdf(paste0(folder_figures, "country_", country, "_" , model, "_shap.pdf"), height = 4, width = 8)
  par(mar = c(4,3,1,0))
  plot.new()
  plot.window(xlim = c(minmax(years_span)), ylim = ylim)
  title(main = country_names_print[country])
  title(ylab = "Predicted probability of crisis")
  
  years_x <- sort(years_span[years_span%%5==0])
  axis(1, las = 2, at = years_x, labels = years_x); axis(2)
  
  if(sum(df$crisis[ix])> 0)
    segments(x0 = df$year[ix][df$crisis[ix]== 1],y0 = ylim[1], y1 = ylim[2],
             col = makeTransparent("red",alpha = 0.2), lwd = 2, lend = 1) # crisis
  abline(h = intercept, lty = 2, lwd = 0.5)
  
  minpos = rep(intercept, sum(ix))
  maxpos = rep(intercept, sum(ix))
  for(f in features_show){
    ycurrent = ifelse(df[ix,f]>0, maxpos,minpos)
    maxpos <- ifelse(df[ix,f]>0, maxpos +df[ix,f], maxpos)
    minpos <- ifelse(df[ix,f]<0, minpos +df[ix,f], minpos)
    segments(x0= df$year[ix], y0 = ycurrent, y1 = ycurrent + df[ix,f],
             col = makeTransparent(features_col[f], alpha=alpha), lwd = lw, lend = 1)
  }
  
  rest_shapley <- rowSums(df[ix, setdiff(features, features_show), drop = F])
  ycurrent = ifelse(rest_shapley>0, maxpos,minpos)
  maxpos <- ifelse(rest_shapley>0, maxpos + rest_shapley, maxpos)
  minpos <- ifelse(rest_shapley<0, minpos + rest_shapley, minpos)
  segments(x0= df$year[ix], y0 = ycurrent, y1 = ycurrent + rest_shapley,
           col = makeTransparent("gray50",alpha=alpha), lwd = lw, lend = 1)
  
  pp <- cbind(df$year[ix],df$pred[ix])
  add <- setdiff(min(years_span):max(years_span), pp[,1])
  pp <- rbind(pp, cbind(add, rep(NA,length(add))))
  pp <- pp[order(pp[,1]),]
  lines(pp[,1], pp[,2], type = "o", col = makeTransparent("black",alpha = 1),
        lwd = 0.9, pch = 20, cex = 1) # pred values
  legend("topleft", legend = c(features_names_print[features_show], "Predicted value"),
         col = c(features_col[features_show], "black"), pch = c(rep(15, length(features_show)), 16), bty = "n", y.intersp = 0.8)
  dev.off()
}






