source("My_analysis/2. utils_analysis.R")
library(dplyr)

experiment_id <- 2 # 1 refers to baseline specification (see list_experiments.py)

folder_data <- paste0("results/cross-validation-interaction/", experiment_id, "/") # where the results are found,  /1/ refers to the baseline experiment
folder_figures <- paste0("figures/cross-validation/", experiment_id, "/") # where the figures are placed 
dir.create(folder_figures)



# load data
dataset <- read.csv(paste0(folder_data,"data.txt"), 
                    sep = "\t",
                    stringsAsFactors = F)[,-1]

all_years <- unique(dataset$year)
countries <- unique(dataset$iso)
n_countries <- length(countries)
true_class <- dataset$crisis

algo <- "extree"
shapleys <- read.csv(paste0(folder_data, paste0("shapley_append_", algo, ".txt")),
                     sep = "\t", stringsAsFactors = F)[,-1]
features <- setdiff(colnames(shapleys), c("year", "iso", "crisis", "pred"))
shapleys_mean <- read.csv(paste0(folder_data, paste0("shapley_mean_", algo, ".txt")),
                          sep = "\t", stringsAsFactors = F)[,-1]

shapleys_inter <- read.csv(paste0(folder_data, paste0("shapley_interaction_", algo, ".txt")),
                           sep = "\t", stringsAsFactors = F, check.names = F)[,-1]
inter_names <- colnames(shapleys_inter)

n_obs <- nrow(dataset)
n_rep <- nrow(shapleys) / n_obs
iter <- rep(1:n_rep, each = n_obs)
shapleys_inter$iter <- iter
shapleys_inter$index <- rep(1:n_obs, n_rep)



# average over replications
key_features <- setdiff(colnames(shapleys_inter),
                        c("index", "base", "others", "year", "iter", "pred", "iso", "crisis",
                          grep("-", colnames(shapleys_inter), value = T)))

features_and_interactions <- setdiff(colnames(shapleys_inter),
                                     c("index", "base", "others", "year", "pred", "iso", "crisis"))


shap_inter_avg <- shapleys_inter %>% group_by(index) %>% summarise_at(features_and_interactions, mean)



#### Shapley regression as in the paper ####

# we train a separate Shapley regression model for each of the key predictions
features <- setdiff(colnames(dataset), c("year", "iter,","iso", "crisis", "crisis_id", "pred"))


output_interactions <- list()
for(a in 1:(length(key_features) - 1)){
  for(b in (1+ a):(length(key_features))){
    
    xname <- key_features[b]
    yname <- key_features[a]
    
    shap_for_reg <- shapleys # use main shapley values
    
    # add interactions
    inter_name <- c(paste0(xname, "-", yname),
                    paste0(yname, "-", xname))
    inter_name <- intersect(inter_name, colnames(shapleys_inter))
    shap_for_reg$inter <- shapleys_inter[[inter_name]]
    
    # remove pairwise interaction from the respective main effects
    shap_for_reg[[xname]] <- shap_for_reg[[xname]]  - shap_for_reg$inter / 2
    shap_for_reg[[yname]] <- shap_for_reg[[yname]]  - shap_for_reg$inter / 2
    
    fnames <- c(features, "inter")
    
    # run actual regression
    
    shap_for_reg[, fnames] <- sapply(shap_for_reg[,fnames], scale)          
    
    
    tab <- shapley_regression(data = shap_for_reg,
                              iter = iter,
                              y_name = "crisis",
                              features = fnames,
                              avg_fun = mean)
    
    output_interactions[[paste0(xname, "_x_", yname)]] <- tab["inter", -1]
  }
}


output_interactions <- simplify2array(output_interactions)["inter",,]





#### plot interaction ####
denom <- 1
pal1 = colorVector(c("white", "red")); pal1 <- pal1(101, expo = denom)
pal2 = colorVector(c("white", "blue")); pal2 <- pal2(101, expo = denom)

ix_crisis <- dataset$crisis == 1

for(a in 1:(length(key_features) - 1)){
  for(b in (1+ a):(length(key_features))){
    
    xname <- key_features[b]
    yname <- key_features[a]
    # observed values
    x <- dataset[[xname]]
    y <- dataset[[yname]]
    
    xlab <- features_names_print[xname]
    ylab <- features_names_print[yname]
    
    inter_name <- c(paste0(xname, "-", yname),
                    paste0(yname, "-", xname))
    inter_name <- intersect(inter_name, colnames(shap_inter_avg))
    z <- shap_inter_avg[[inter_name]]
    
    maxv <- max(abs(z), na.rm = T)
    cols <- ifelse(z >0,
                   pal1[1 + round((abs(z)/maxv) * 100)],
                   pal2[1 + round((abs(z)/maxv) * 100)])
    
    png(paste0(folder_figures, "interact_shap_", algo, "_", xname,"_x_", yname, ".png"), width = 6, height = 5, units = "in", res = 720)
    par(mar = c(3,3,4,6), xpd = F)
    plot.new()
    ylim <- c(min(y), max(y))
    xlim <- c(min(x), max(x))
    plot.window(xlim = xlim, ylim = ylim)
    axis(1);axis(2)
    title(xlab = xlab, ylab = ylab, line = 2)
    cols_other <- "gray50"
    points(x, y, bg = cols, pch = 21, lwd = .2, cex = 1.1, col = "gray90")
    
    abline(h =mean(y), col = ("gray40"))
    abline(v =mean(x), col = ("gray40"))
    
    points(x[ix_crisis], y[ix_crisis], pch = 20, col = "black", cex = .6)
    
    par(xpd = T)
    
    # legend("topleft", legend = c(paste0("Shapley value > ", round(threshold,3)), paste0("Shapley value < ",  -round(threshold,3))), col = c("red", "blue"), pch = c(20, 20), bty = "n", y.intersp = 0.8, cex = 0.95)
    legend("topleft",
           pch = c(20), legend = c("Crises"), bty = "n", y.intersp = 0.8)
    
    
    # Continuous legend
    y0 = 0
    dy = 0.0001
    pal <- c(rev(pal2), pal1)
    dy = (max(y)-min(y))/length(pal)
    # segments(x0 = max(x) + .15 *(max(x)-min(x)) , y0 = min(y), y1 = max(y))
    for(i in 1:length(pal)){
      rect(xleft = max(x) + .05 *(max(x)-min(x)) , xright = max(x) + .1 *(max(x)-min(x)),
           ybottom = min(y) + (i-1)*dy, ytop = min(y) + (i-1)*dy + dy , col = pal[i], border = NA)
    }
    nlevel <- 11
    ys <- seq(min(y), max(y), length.out = nlevel)
    levels <- seq(0, maxv, length.out = (nlevel+1)/2)
    levels <- c(-rev(levels[-1]), levels)
    # text(x = max(x) + .2 *(max(x)-min(x)), y =max(y) + 20*dy, zlab)
    text(x = max(x) + .25 *(max(x)-min(x)), y =ys, format(round(levels, 3), nsmall = 3))
    dev.off()
  }
}




