# This code produces several of our descriptive results
# - Mean differences of variables before vs. after crisis (Table 1)
# - Figure showing observations (and exclusions)  of the baseline sample (Figure 1)
# - Table showing proportion of missing values (Appendix, Figure A.1)


library(dplyr)
source("My_analysis/2. utils_analysis.R")
source("My_analysis/1. global.R")
folder_plot <- "figures/"
crisis_var <- "crisisJST" #crisisJST, crisisBR
years_exclude <-c(1912:1918, 1931:1945)

crises <- read.csv("data/crises_definitions.csv")


df <- readxl::read_excel("data/JSTdatasetR6.xlsx", sheet = "Data")
df$crisis <- df$crisisJST
df$crisisJST <- NULL


fnames <- c("Yield curve slope" = "drate",
            "Credit" = "tloan_gdp_rdiff2",
            "CPI" = "cpi_pdiff2",
            "Debt service ratio" = "tdbtserv_gdp_rdiff2",
            "Consumption" = "cons_pdiff2",
            "Investment" = "inv_gdp_rdiff2",
            "Public debt" = "pdebt_gdp_rdiff2",
            "Broad money" = "bmon_gdp_rdiff2",
            "Stock market" = "stock_pdiff2",
            "Current account" = "ca_gdp_rdiff2",
            "Global yield curve slope" = "global_drate",
            "Global credit" = "global_loan",
            "Domestic nominal short-term rate" = "stir",
            "Domestic nominal long-term rate" = "ltrate",
            "Houshold loans" = "hloan_gdp_rdiff2",
            "Business loans" = "bloan_gdp_rdiff2",
            "House prices (index)" = "hp_pdiff2"
)

f_names_main <- setdiff(fnames, c("hloan_gdp_rdiff2",
                                  "bloan_gdp_rdiff2",
                                  "hp_pdiff2"))




df <- df_processed


df$crisis_lead1[is.na(df$crisis_lead1)] <- 0
df$crisis_lead2[is.na(df$crisis_lead2)] <- 0

df$crisis_lead1to2 <- 1 * ((df$crisis_lead1 + df$crisis_lead2) > 0)
df$crisis_lead1to2[df$crisis ==1] <- NA # remove actual crisis year

df$crisis_post <- 1 * ((df$crisis_lag1 + df$crisis_lag2 + df$crisis_lag3 + df$crisis_lag4) > 0)
df$crisis_post[is.na(df$crisis_post)] <- 0




# remove actual crisis years, post crises and extreme years
df <- df[!is.na(df$crisis_lead1to2) & !df$year %in% years_exclude & df$crisis_post ==0 , ]




#### Mean differences of variables before vs. after crisis ####

df_main <- df[complete.cases(df[, c("crisis_lead1to2", f_names_main)]),]
df_main <- df_main[order(df_main$year, df_main$iso), ]

mean_tab <- cbind(apply(df_main[df_main$crisis_lead1to2 == 1,fnames], 2, mean, na.rm = T),
                  apply(df_main[df_main$crisis_lead1to2 == 0, fnames], 2, mean, na.rm = T))
transformer <- mean_tab[,1] *0 + 100
transformer[c("drate", "stir", "ltrate", "global_drate")] <- 1
mean_tab <- mean_tab * transformer

sd_tab <- cbind(apply(df_main[df_main$crisis_lead1to2 == 1,fnames], 2, sd, na.rm = T),
                apply(df_main[df_main$crisis_lead1to2 == 0, fnames], 2, sd, na.rm = T))
sd_tab <- sd_tab * transformer


n_tab <- cbind(apply(df_main[df_main$crisis_lead1to2 == 1,fnames], 2, function(x) sum(!is.na(x))),
               apply(df_main[df_main$crisis_lead1to2 == 0, fnames], 2, function(x) sum(!is.na(x)))
)

cbind(n_tab, rowSums(n_tab))



tab_out <- data.frame(mean_tab[,1], sd_tab[,1], mean_tab[,2], sd_tab[,2])
colnames(tab_out) <- c("Mean1", "SD1", "Mean2", "SD2")

rownames(tab_out) <- names(fnames)
tab_out$mean_difference <- tab_out[,1] - tab_out[,3]


round(tab_out,2)


p_values <- round(sapply(fnames, function(fname) 
  t.test(df_main[df_main$crisis_lead1to2 == 1,fname] * transformer[fname],
         df_main[df_main$crisis_lead1to2 == 0,fname] * transformer[fname])$p.value),3)

ses <- round(sapply(fnames, function(fname) 
  t.test(df_main[df_main$crisis_lead1to2 == 1,fname] * transformer[fname],
         df_main[df_main$crisis_lead1to2 == 0,fname] * transformer[fname])$stderr),4)


tab_out$mean_difference <- paste0(format(round(tab_out$mean_difference, 2), nsmall = 2), 
                                  " (",
                                  format(round(ses, 3), nsmall = 3),
                                  
                                  ")")

tab_out$p <- p_values

tab_out$transformation = "\\transgdp"
tab_out$transformation[fnames %in% c("drate", "global_drate_leave", "stir", "ltrate")] <- "level"
tab_out$transformation[fnames %in% c("cpi_pdiff2", "stock_pdiff2", "cons_pdiff2", "hp_pdiff2")] <- "\\transchange"

tab_out <- tab_out[, c(7,1,2,3,4,5,6)]

library(xtable)
print(xtable(tab_out, digits = c(2,2,2,2,2,2,2, 3)), sanitize.text.function = identity)







#### Figure showing observations (and exclusions)  of the baseline sample ####


df <- data.frame(df_processed)


df$crisis_lead1[is.na(df$crisis_lead1)] <- 0
df$crisis_lead2[is.na(df$crisis_lead2)] <- 0

df$crisis_lead1to2 <- 1 * ((df$crisis_lead1 + df$crisis_lead2) > 0)
df$crisis_lead1to2[df$crisis ==1] <- NA # remove actual crisis year

df$crisis_post <- 1 * ((df$crisis_lag1 + df$crisis_lag2 + df$crisis_lag3 + df$crisis_lag4) > 0)
df$crisis_post[is.na(df$crisis_post)] <- 0




lowc <- .3
highc <- .6
countries <- unique(df$iso)
nc <- length(countries)
allyear <- min(df$year):max(df$year)

cairo_pdf(paste0(folder_plot, "country_missings_new.pdf"), height = 6, width = 12)
par(mar = c(3,7,3,1))
plot.new()
plot.window(xlim = c(min(allyear), max(allyear)), ylim = c(1,nc + 1))
axis(1, at = min(allyear):max(allyear), labels = NA, las = 2, cex.axis = 1, tck = -0.01)
axis(1, at = (min(allyear):max(allyear))[seq(1,length((min(allyear):max(allyear))),5)], las = 2, cex.axis = 1, labels = NA)

text(x = (min(allyear):max(allyear))[seq(1,length((min(allyear):max(allyear))),5)], y = -.8,
     srt = 45, labels = (min(allyear):max(allyear))[seq(1,length((min(allyear):max(allyear))),5)], xpd = T, cex = .9)
axis(2, at = .5 +(1:nc), labels = country_names_print[countries], las = 2, tick = T, lwd.ticks = 0)
par(xpd = T)
# standard legend
legend(x = 1850, y = 21, legend = c("Target (1–2 years before crisis)","Non-crises"),
       pt.bg = c( makeTransparent("red", .7), makeTransparent("limegreen",.7)), col = c(makeTransparent("red",.7), makeTransparent("limegreen",.7)),
       pch = 15, bty = "n")

legend(x = 1900, y = 21, legend = c("Excluded target", "Excluded non-crises due to missings"),
       #pt.bg = c(makeTransparent("red",alpha = lowc),  makeTransparent("limegreen",lowc)),
       col = c(makeTransparent("red",alpha = lowc),  makeTransparent("limegreen",lowc)),
       border = c(makeTransparent("red",alpha = lowc),  makeTransparent("limegreen",lowc)),
       fill = c(makeTransparent("red",alpha = lowc),  makeTransparent("limegreen",lowc)),bty = "n",
       text.col = makeTransparent("black", alpha = 0)
)

legend(x = 1900, y = 21, legend = c("Excluded target", "Excluded non-crises due to missings"),
       #pt.bg = c(makeTransparent("red",alpha = lowc),  makeTransparent("limegreen",lowc)),
       #col = c(makeTransparent("red",alpha = lowc),  makeTransparent("limegreen",lowc)),
       border = c(makeTransparent("red",alpha = lowc),  makeTransparent("limegreen",lowc)),
       fill = c("white", "white"),
       density =40, bty = "n")

#error legend
legend(x = 1960, y = 21, legend = c("Actual crises + 4 years post-crisis observations",
                                    "Extraordinary periods"),
       pt.bg = c(makeTransparent("gray50",lowc), makeTransparent("orange4",.45)),
       col = c(makeTransparent("gray50",lowc), makeTransparent("orange4",.45)),
       pch = 15, bty = "n")

counter <- 0
track_pos <- 0
track_neg <- 0

for(country in countries){
  counter <- counter + 1
  abline(h = counter, lty = 3, col = "gray50", lwd = 0.6)
  ix = df$iso == country
  
  feature <- apply(df[ix,f_names_main],1,mean)
  crisis <- df[ix,"crisis"]
  years <- df[ix,"year"]
  names(feature) <- years
  yearsadd <- setdiff(allyear,df[ix,"year"])
  years <- c(years, yearsadd)
  oo <- order(years)
  years <- years[oo]
  feature <- c(feature,rep(NA, length(yearsadd)))[oo]
  crisis <- c(crisis,rep(NA, length(yearsadd)))[oo]
  
  # first missing cue values, as these observations drop out anyway!
  ix_actual_crises <- crisis == 1 
  ix_target <- rowSums(cbind(lead(crisis, 2), lead(crisis, 1)), na.rm = T) > 0
  years_special <- c(1912:1918, 1931:1945)
  ix_special <- (years %in% years_special) 
  
  ix_post_crises <- rowSums(sapply(1:4, function(x) lag(crisis,x)), na.rm = T) > 0
  ix_post_crises <- (ix_post_crises | ix_actual_crises ) & !ix_special 
  
  ix_miss_feature <-  is.na(feature) 
  ix_target_used <- ix_target &!ix_miss_feature &!ix_special & !ix_post_crises & !ix_actual_crises
  ix_target_unused <- ix_target & !ix_target_used
  
  ix_rest <- !ix_miss_feature & !ix_post_crises & !ix_actual_crises & !ix_special & !ix_target
  ix_post_crises_show <- ix_post_crises & !ix_target
  ix_special_show <- ix_special & !ix_target
  ix_miss_feature_show <- ix_miss_feature & !ix_target_unused & !ix_special &!ix_post_crises
  
  ix_any_exclude <- ix_miss_feature | ix_special | ix_post_crises | ix_actual_crises
  # no- crisis - no missings
  if (length(years[ix_rest]) > 0) {
    rect(xleft = years[ix_rest] - 0.4, xright = years[ix_rest] + 0.4, ybottom = counter, ytop = counter + 1, col = makeTransparent("limegreen", highc), border = NA)
  } else {
    print(paste("No elements in years[ix_rest] for", country))
  }
  
  #rect(xleft = years[ix_rest]-.4, xright = years[ix_rest] + .4, ybottom = counter, ytop = counter +1, col = makeTransparent("limegreen",highc), border = NA)
  track_neg <- track_neg + sum(ix_rest)
  track_pos <- track_pos + sum(ix_target_used)
  
  print(country)
  print(c(sum(ix_rest), sum(ix_target_used)))
  # used crises
  if(sum(ix_target_used)>0)
    rect(xleft = years[ix_target_used]-.4, xright = years[ix_target_used] + .4, ybottom = counter, ytop = counter +1, col = makeTransparent("red",highc), border = NA)
  
  # unused crises due to missings
  if(sum(ix_target_unused)>0){
    rect(xleft = years[ix_target_unused]-.4, xright = years[ix_target_unused] + .4, ybottom = counter, ytop = counter +1, col = makeTransparent("red", highc), border = NA)
    
    rect(xleft = years[ix_target_unused]-.4, xright = years[ix_target_unused] + .4,
         ybottom = counter, ytop = counter +1, col = c(makeTransparent("white", highc)),lty = 1, density = 30, border = NA, lwd = 2)
  }
  
  # post crises
  if (!any(is.na(ix_post_crises)) && any(ix_post_crises) && sum(ix_post_crises) > 0)
  #if(sum(ix_post_crises)>0)
    rect(xleft = years[ix_post_crises_show]-.4, xright = years[ix_post_crises_show] + .4, ybottom = counter, ytop = counter +1, col = makeTransparent("gray50", lowc), border = NA)
  
  # special periods
  rect(xleft = years[ix_special_show]-.4, xright = years[ix_special_show] + .4, ybottom = counter, ytop = counter +1, col = makeTransparent("orange4", .45), border = NA)
  
  # missing values
  rect(xleft = years[ix_miss_feature_show]-.4, xright = years[ix_miss_feature_show] + .4, ybottom = counter, ytop = counter +1, col = makeTransparent("limegreen", highc), border = NA)
  
  rect(xleft = years[ix_miss_feature_show]-.4, xright = years[ix_miss_feature_show] + .4,
       ybottom = counter, ytop = counter +1, col = c(makeTransparent("white", highc)),lty = 1, density = 30, border = NA, lwd = 2)
  # any exclude
  # all observations not used
  rect(xleft = years[ix_any_exclude]-.5, xright = years[ix_any_exclude] + .5, ybottom = counter + .45, ytop = counter +.55 , col = makeTransparent("black", .55), border = NA)
}
dev.off()





#### Table showing proportion of missing values (Appendix) ####

df <- df_processed


df$crisis_lead1[is.na(df$crisis_lead1)] <- 0
df$crisis_lead2[is.na(df$crisis_lead2)] <- 0

df$crisis_lead1to2 <- 1 * ((df$crisis_lead1 + df$crisis_lead2) > 0)
df$crisis_lead1to2[df$crisis ==1] <- NA # remove actual crisis year

df$crisis_post <- 1 * ((df$crisis_lag1 + df$crisis_lag2 + df$crisis_lag3 + df$crisis_lag4) > 0)
df$crisis_post[is.na(df$crisis_post)] <- 0


features_raw_used_base <- setdiff(feature_names_raw, c("tmort", "tbus", "thh", "hpnom"))
features_raw_used <- c(features_raw_used_base, c("tbus", "thh", "hpnom","tmort")) # "tmort", 

ix_year <- !(df$year %in% c(1912:1918, 1931:1945))
ix_year_pre <- ix_year & df$year < 1933
ix_year_post <- ix_year & df$year > 1945

mean_missing <- apply(df[ix_year, features_raw_used], 2, function(x) mean(is.na(x)))
mean_pre <- apply(df[ix_year_pre, features_raw_used], 2, function(x) mean(is.na(x)))
mean_post <- apply(df[ix_year_post, features_raw_used], 2, function(x) mean(is.na(x)))

key_featues <- c("gdp", "stir", "ltrate", "tloans")

mean_missing[order(mean_missing)]

out_tab <- data.frame(all = mean_missing, pre = mean_pre, post = mean_post)
rownames(out_tab) <- plyr::mapvalues(rownames(out_tab), vis_feature_names_raw$feature, vis_feature_names_raw$name)
round(out_tab, 2)
xtable::xtable(out_tab)



# plotting pre-processing step to explain missing values

df$crisis_lead1to2 <- 1 * ((df$crisis_lead1 + df$crisis_lead2) > 0)




all_predictors <- c("tloan_gdp_rdiff2", "global_loan", "drate", "global_drate",
                    "cpi_pdiff2",
                    "tdbtserv_gdp_rdiff2",
                    "cons_pdiff2",
                    "inv_gdp_rdiff2",
                    "pdebt_gdp_rdiff2",
                    "bmon_gdp_rdiff2",
                    "stock_pdiff2",
                    "ca_gdp_rdiff2")


ix <- ix1 <- rep(T, nrow(df)) # all
ix <- ix2 <- ix & !is.na(df$tloan_gdp_rdiff2) & !is.na(df$global_loan) # key variables (tloan)
ix <- ix3 <- ix & !is.na(df$drate) & !is.na(df$global_drate) # key variables (tloan)  
ix <- ix4 <- ix & complete.cases(df[, all_predictors])
ix <- ix5 <- ix & !df$year %in% years_exclude # remove extreme periods
ix <- ix6 <- ix & !df$crisis ==1 & df$crisis_post ==0  # remove crisis and post crisis

ix_list <- list(ix1, ix2, ix3, ix4, ix5, ix6)

names <- c("All\nobservations",
           "- missing\ncredit",
           "- missing\nslope",
           "- missing other\npredictors",
           "- 1914-1918,\n1933-1945",
           "- post-crisis"
)
colo <- "gray75"

years <- unique(sort(df$year))
scaler <- .9

pdf(paste0(folder_plot, "missing_sequence.pdf"),  height = scaler * 6, width = scaler* 10)
par(mar = c(4.5,7,1.3,0))
plot.new()
plot.window(xlim = c(min(years), max(years)), ylim = c(0,6))

par(xpd = T)
legend(x = 1970, y = 6.7,
       legend = c("1-2 years before crises", "Non-crises"),
       col = c("red", colo),
       border = c("red", colo),
       fill = c("red", colo),
       bty = "n", y.intersp = 0.8
)

axis(1, at = min(years):max(years), labels = NA, las = 2, cex.axis = 1, tck = -0.01)
show_years <- intersect(years, seq(1875,2020, by = 5))
axis(1, at = show_years, las = 2, cex.axis = 1.1)
axis(2, at = (0:5 + .5), labels = rev(names), las = 2, hadj = 0, line = 3.5, lwd = NA)


i <- 0
for(ix in rev(ix_list)){
  
  tt <- df[ix,] %>% group_by(year) %>% summarise(np = sum(crisis_lead1to2) / 17, nn = sum(1 -crisis_lead1to2) / 17)
  
  rect(xleft = tt$year, xright = tt$year + .8, ybottom = i, ytop = i + .96 * tt$np, col = "red", border = NA)
  rect(xleft = tt$year, xright = tt$year + .8, ybottom = i + .96 * tt$np, ytop = i + .96 *(tt$nn + tt$np), col = colo, border = NA)
  
  i <- i + 1
  table(df$iso[ix])
  table(df$year[ix])
  
}
dev.off()


