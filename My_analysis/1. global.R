# We  execute this script before doing  other analyses.
#setting working directory
setwd("~/Desktop/thesis_code")

#### FUNCTIONS ####

colorVector = function (colors, ...) 
{
  ramp <- colorRamp(colors, ...)
  function(n, expo = 2) {
    x <- (ramp(seq.int(0, 1, length.out = n)^(1/expo)))
    if (ncol(x) == 4L) 
      rgb(x[, 1L], x[, 2L], x[, 3L], x[, 4L], maxColorValue = 255)
    else rgb(x[, 1L], x[, 2L], x[, 3L], maxColorValue = 255)
  }
}



perc_change <- function(x, lag = 1){
  xlag <- dplyr::lag(x, n = lag, default = NA)
  return((x - xlag) / xlag)
}

lagit <- function(x, lag = 1){
  return(dplyr::lag(x, n = lag, default = NA))
  
}

leadit <- function(x, lag = 1){
  return(dplyr::lead(x, n = lag, default = NA))
  
}

mean_no_self <- function(x){
  out <- sapply(1:length(x), function(pos) mean(x[-pos], na.rm = T))
  return(out)
}



measurePerf <- function(criterion,predicted,threshold = .5, random = F, weights = c(1,1)){
  # This function computes an array of performance metrics 
  
  # :param numeric vector criterion: observed class labels
  # :param numeric vector predicted: predicted probability of the positive class
  # :param double threshold: values on the "predicted vector" are are assigned to 
  #   the positive class if their value is greater than the threshold 
  # :param boolean random: If true, values with a predicted probability of 0.5
  #are randomly assigned to one of the claases
  #: param numeric vector weights of size 2: First value weights objects in the positive class,
  #   second value weights objects in the negative class.
  ix <- !is.na(predicted)
  criterion <- criterion[ix]
  predicted <- predicted[ix]
  
  predicted.t <- ifelse(predicted > threshold,1,0)
  if(random & threshold == .5)
    predicted.t[predicted==threshold] <- round(stats::runif(sum(predicted==threshold)))
  
  if(length(criterion) >1 && stats::sd(criterion) > 0){
    auc <- calcAUC(predicted,criterion)[1]
  } else auc <- NA
  tp <- as.numeric(sum(predicted.t==1&criterion==1)) * weights[1]
  fp <- as.numeric(sum(predicted.t==1&criterion==0)) * weights[2]
  tn <- as.numeric(sum(predicted.t==0&criterion==0)) * weights[2]
  fn <- as.numeric(sum(predicted.t==0&criterion==1)) * weights[1]
  accuracy <- (tp+tn)/(tp+fp+tn+fn)
  tp.rate <- tp/(tp+fn)
  fp.rate <- fp/(tn+fp)
  balanced <- (tp.rate+(1-fp.rate))/2
  f1 <- 2 * tp/(2 * tp + fp + fn)
  matthews <- (tp * tn - fp * fn)/sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  brier <- mean((criterion - predicted)^2)
  absloss <- mean(abs(criterion - predicted))
  logloss <- MLmetrics::LogLoss(predicted, criterion)
  
  aupr <- PRROC::pr.curve(scores.class0 = predicted, weights.class0 = criterion)$auc.integral
  
  
  if(is.na(matthews))
    matthews <- 0
  performance <- c(auc,accuracy,tp,fp,tn,fn,tp.rate,fp.rate,balanced,f1,matthews, brier, logloss, absloss, aupr)
  names(performance) <- c("auc","accuracy","tp","fp","tn","fn","tp.rate","fp.rate","balanced","f1","matthews", "brier", "logloss", "absloss", "aupr")
  return(performance)
}



calcAUC <- function(x,y){
  # calculate the area under the curve
  return(pROC::roc(y, x, direction = "<", quiet = T)$auc)
  
}


minmax <- function(x)
  return(c(min(x, na.rm = T), max(x, na.rm = T)))

moving_avg <- function(x,n=5){stats::filter(x,rep(1/n,n), sides=2)}



rocPoint <- function(criterion,predicted, threshold){
  # This function computes the true posisitve, false postive rate, and precision.
  # The reuslting numbers can be used to plot ROC and Precision-Recall curves
  
  # :param numeric vector criterion: observed class labels
  # :param numeric vector predicted: predicted probability of the positive class
  # :param double threshold: values on the "predicted vector" are are assigned to 
  #   the positive class if their value is greater than the threshold 
  
  
  predicted[is.na(predicted)] <- runif(sum(is.na(predicted)))
  predicted.t <- ifelse(predicted > threshold,1,0)
  
  tp <- as.numeric(sum(predicted.t==1&criterion==1))
  fp <- as.numeric(sum(predicted.t==1&criterion==0))
  tn <- as.numeric(sum(predicted.t==0&criterion==0))
  fn <- as.numeric(sum(predicted.t==0&criterion==1))
  tp_rate <- tp/(tp+fn)
  fp_rate <- fp/(tn+fp)
  precision <- tp/(tp + fp)
  return(c(tp_rate = tp_rate, fp_rate = fp_rate, precision = precision))
}

clrobustse <- function(fit.model, clusterid) {
  # computes robust standard errors
  
  # https://stackoverflow.com/questions/33927766/logit-binomial-regression-with-clustered-standard-errors
  N.obs <- length(clusterid)            
  N.clust <- length(unique(clusterid))  
  dfc <- N.clust/(N.clust-1)                    
  vcv <- vcov(fit.model)
  estfn <- sandwich::estfun(fit.model)
  uj <- apply(estfn, 2, function(x) tapply(x, clusterid, sum))
  N.VCV <- N.obs * vcv
  ujuj.nobs  <- crossprod(uj)/N.obs
  vcovCL <- dfc*(1/N.obs * (N.VCV %*% ujuj.nobs %*% N.VCV))
  lmtest::coeftest(fit.model, vcov=vcovCL)
}

grep_multi <- function(constraints, corpus){
  # filters those elements from a vector that include all substrings specified as constraints
  # :param character vector constraints: substrings that are ALL required to be in te filtered corpus
  # :param chracter vector corpus: vector of strings that is filtered
  
  if(length(constraints) == 1)
    return(grep(constraints, corpus, value = T, fixed = T) )
  ncorpus <- length(corpus)
  ix <- sapply(constraints, function(x)  grep(x, corpus, value = F, fixed = T)) 
  ix <- sapply(ix, function(x) 1:ncorpus %in% x)  
  return(corpus[apply(ix, 1, all)])
}




winsorize <- function(x, p = .025){
  
  quants <- quantile(x, c(p, 1-p))
  x[x < quants[1]] = quants[1]
  x[x > quants[2]] = quants[2]
  return(x)
  
}

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}
sigmoid_inv <- function(x){
  return(-log(1/x -1))
}

makeTransparent = function(cols, alpha=0.15) {
  # make a color transparent
  if(alpha<0 | alpha>1) stop("alpha must be between 0 and 1")
  alpha = floor(255*alpha)
  newColor = col2rgb(col=unlist(list(cols)), alpha=FALSE)
  .makeTransparent = function(col, alpha) {
    rgb(red=col[1], green=col[2], blue=col[3], alpha=alpha, maxColorValue=255)
  }
  newColor = apply(newColor, 2, .makeTransparent, alpha=alpha)
  newColor[is.na(cols)] <- NA
  return(newColor)
}

resetPar <- function() {
  dev.new()
  op <- par(no.readonly = TRUE)
  dev.off()
  op
}


brier_test <- function(p1,p2, y){
  n <- length(p1)
  pi = mean(y)
  dif = (2/n) * sum((p1 - p2) * pi - (p1-p2)*y)
  var = (4/(n^2)) * sum((p1-p2)^2 * pi * (1-pi))
  z = dif/(var^.5)
  return(2*pnorm(-abs(z)))
}



shapley_regression <- function(data, iter, y_name = "crisis", features, avg_fun = mean){
  
  
  # iter is a vector indicating the iteration of the cross-validation exercise
  data[, features] <- scale(data[,features])
  
  n_obs <- sum(iter == min(iter))
  n_iter <- max(iter)
  
  formula <- as.formula(data[, c(y_name, features)])
  coefs <- lapply(1:n_iter, function(i) fixest::feglm(formula, data = data[iter == i,], family = "logit", vcov = "DK", panel.id = c("iso", "year"))$coeftable)
  
  mod1 <- fixest::feglm(formula, data = data[iter == 1,], family = "logit", vcov = "DK", panel.id = c("iso", "year"))
  dgfree <- fixest::degrees_freedom(mod1, type = "t", vcov = "DK")
  
  # Correction for sample splitting:
  # see Equation 3.13 and 3.14 in Chernozhukov et al. (2018) "Double/debiased machine learning for treatment and structural parameters"
  # https://doi.org/10.1111/ectj.12097
  
  
  coefs <- simplify2array(coefs)
  c_avg <- apply(coefs, 1:2, avg_fun)
  avg_var <- apply((sapply(1:n_iter, function(i) (coefs[, "Std. Error",i] * sqrt(n_obs))^2 + (coefs[, "Estimate",i] - c_avg[,"Estimate"]) ^ 2)), 1, avg_fun)
  
  avg_se <- sqrt(avg_var) / sqrt(n_obs)
  
  avg_p_values <- sapply(c_avg[, "Estimate"] / avg_se, function(i) 
    ifelse(i < 0, 1, pt(i, dgfree, lower.tail = F))) # one-sided
  output <- data.frame(Estimate = c_avg[, "Estimate"], SE = avg_se, p = avg_p_values)
  
  return(output)  
}



# It loads the raw data creates the pre-processed data

library(dplyr)

crises <- read.csv("/Users/akhil/Desktop/thesis_code/data/crises_definitions.csv")
df_raw <- readxl::read_excel("/Users/akhil/Desktop/thesis_code/data/JSTdatasetR6.xlsx", sheet = "Data")
df_raw$crisis <- df_raw$crisisJST
df_raw$crisisJST <- NULL



#### process data ####

df <- df_raw

df$tdbtserv <- df$tloans * df$ltrate / 100

df <- df %>% group_by(country) %>% mutate(
  drate = ltrate - stir,
  tloan_gdp_rdiff3 = tloans / gdp  - lagit(tloans / gdp, lag = 3),
  tloan_gdp_rdiff2 = tloans / gdp  - lagit(tloans / gdp, lag = 2),
  bmon_gdp_rdiff2 = money/gdp - lagit(money/gdp, lag = 2),
  pdebt_gdp_rdiff2 = debtgdp - lagit(debtgdp, lag = 2),
  inv_gdp_rdiff2 = iy - lagit(iy, lag = 2),
  ca_gdp_rdiff2 = ca/gdp - lagit(ca/gdp, lag = 2),
  tdbtserv_gdp_rdiff2 = tdbtserv/gdp - lagit(tdbtserv/gdp, lag = 2),
  
  cpi_pdiff2 = perc_change(cpi, lag = 2),
  #stock_pdiff2 = perc_change(eq_tr, lag = 2), #using equity return instead of stocks data
  stock_pdiff2 = eq_tr,
  cons_pdiff2 = perc_change(rconpc, lag = 2),
  hp_pdiff2 = perc_change(hpnom, lag = 2),
  
  hloan_gdp_rdiff2 = thh / gdp  - lagit(thh / gdp, lag = 2),
  bloan_gdp_rdiff2 = tbus / gdp  - lagit(tbus / gdp, lag = 2)
  
)                     

df$srate <- df$stir
df$lrate <- df$ltrate
df <- df %>% group_by(country) %>% 
  mutate(cpi_pdiff1 = perc_change(cpi, lag = 1) * 100)




df <- df %>% group_by(year) %>% mutate(
  global_loan  = mean_no_self(tloan_gdp_rdiff2),
  global_drate = mean_no_self(drate))

df <- df %>% group_by(country) %>% mutate(crisis_lead1 = leadit(crisis, lag = 1),
                                          crisis_lead2 = leadit(crisis, lag = 2),
                                          crisis_lead3 = leadit(crisis, lag = 3),
                                          crisis_lead4 = leadit(crisis, lag = 4),
                                          
                                          crisis_lag1 = lagit(crisis, lag = 1),
                                          crisis_lag2 = lagit(crisis, lag = 2),
                                          crisis_lag3 = lagit(crisis, lag = 3),
                                          crisis_lag4 = lagit(crisis, lag = 4)
)
df_processed <- df
rm(df)

