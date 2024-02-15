library(dplyr)
source("My_analysis/1. global.R")

df_baron <- haven::read_dta("data/Narrative Crisis List, Panics List, and BVX List.dta")

ix <- df_baron$panic == 1 | (df_baron$bankfailures_widespread == 1 & df_baron$bankeqdecline == 1)
df_baron <- df_baron[ix,]
df_baron$country[df_baron$country == "U.S."] <- "USA"
df_baron$country[df_baron$country == "U.K."] <- "UK"
df_baron <- df_baron[df_baron$country %in% df_jst$country, ]

df_baron <- df_baron[,1:2]
colnames(df_baron)[2] <- "year"
df_baron <- data.frame(df_baron)

# compare manually against paper (p. 60-61) # perect matching!
crises_df <- data.frame(year = df_raw$year,country = df_raw$country,
                        iso = df_raw$iso, crisisJST = df_raw$crisis)

crises_df$crisisBR <- 0

# Baron #
for(i in 1:nrow(df_baron)){
  y = df_baron[i,"year"]
  country = df_baron[i,"country"]
  ix <- crises$year == y & as.character(crises$country) == country
  crises$crisisBR[ix] <- 1
}
write.csv(crises, file = "data/crises_definitions.csv")



