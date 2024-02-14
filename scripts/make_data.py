"""
SUPPLEMENTARY CODE FOR BOE SWP 848: 
Credit Growth, the Yield Curve and Financial Crisis Prediction: Evidence from a Machine Learning Approach 

This script loads the Jorda-Schularick-Taylor Macrohistry database and transforms the variables.
"""
import pandas as pd
from utils import *
import numpy as np
import re
import pandas as pd
from utils import make_ratio, make_shift

def create_data(config):
    """ Create the data set from the raw data from "http://www.macrohistory.net/data/"
    according to the specifications in the Config object"""
        
    horizon = config.data_horizon
    crisis_lag = config.data_crisis_lag

    # the data can be downlaoded directly:
    # df_jst = pd.read_excel('http://www.macrohistory.net/JST/JSTdatasetR3.xlsx', sheet_name="Data") 
    # df = pd.read_excel('data/JSTdatasetR3.xlsx', sheet_name="Data") 
    df = pd.read_excel('data/JSTdatasetR6.xlsx', sheet_name="Data") 

    crises_definitions = pd.read_csv('data/crises_definitions.csv') # different crisis definitions
    crises_definitions["crisis_var"] = pd.Series(crises_definitions["crisis" + config.crisis_definition], dtype='int32')    
    df = df.merge(crises_definitions[["crisis_var", "year", "iso"]], how='left', on=["year", "iso"])


    if config.clustered_crises == "only":
            df["crisis_var"][~np.isin(df["year"].values,(np.array([1907,1908,1930, 1931, 2007, 2008])))] = 0



    # rename variables
    df.rename(columns={
        "crisisJST": "crisis",
        'stir': 'srate',
        'ltrate': 'lrate',
        'iy': 'inv_gdp',
        'debtgdp': 'pdebt_gdp',
        'money': 'bmon',
        'narrowm': 'nmon',
        'tloans': 'tloan',
        'tbus': 'bloan',
        'thh': 'hloan',
        'tmort': 'mort',
        #'stocks': 'stock',
        'eq_tr': 'stock',
        'hpnom': 'hp',
        'rconpc': 'cons'
    }, inplace=True)

    unique_countries = df["iso"].unique()
    unique_years = df["year"].unique()


    horizon = config.data_horizon
    predictors = config.data_feature_names.copy()
    print("Predictors: ", predictors)
    # we do not compute growth rates for the interest rates and the slope for the yield curve
    # And for transformation using the hamilton filter(_ham) or HP filter (_hp)
    # For the other predictors we compute growth rate (percentage change or ratio change)
    # and add the horizon (e.g. 2 year change) to the variable name

    predictors = [p + str(horizon)  if not (p[-1].isdigit()) and (("rdiff" in p) or ("pdiff" in p)) else p for p in predictors ]


    df.loc[:, 'drate'] = df['lrate'] - df['srate']  # slope of the yield curve
#    # df["lrate_real"] = df["lrate"] - 100. * df.groupby('iso')["cpi"].apply(lambda x: lag_pct_change(x, h=1))
    df['lrate_real'] = df['lrate'] - 100. * df.groupby('iso')['cpi'].pct_change(periods=1)

#     #df["srate_real"] = df["srate"] - 100. * df.groupby('iso')["cpi"].apply(lambda x: lag_pct_change(x, h=1))
    df['srate_real'] = df['srate'] - 100. * df.groupby('iso')['cpi'].pct_change(periods=1)


    df['pdebt'] = df['pdebt_gdp'] * df['gdp']
    df['inv'] = df['inv_gdp'] * df['gdp']

    # Debt to service ratios
    df['tdbtserv'] = df['tloan'] * df['lrate'] / 100.0
    df['hdbtserv'] = df['hloan'] * df['lrate'] / 100.0
    df['bdbtserv'] = df['bloan'] * df['lrate'] / 100.0

    df["cons_nom"] = df["cons"] * df["cpi"]

    pre_gdp_ratios = ['bmon', 'nmon', 'tloan', 'bloan', 'hloan', 'mort', 'ca', 'cpi', 'tdbtserv', 'hdbtserv', 'bdbtserv', 'inv', 'pdebt']
    df, gdp_ratios = make_ratio(df, pre_gdp_ratios, denominator='gdp')

    # change in a variable (rdif)_real
    df, _ = make_shift(df, ["lrate", "srate", "drate"] + gdp_ratios, type = "absolute", horizon=horizon)
    df, _ = make_shift(df, ["srate"], type = "absolute", horizon=1) # this is required for a robustness check (#353)



    # percentage change in variable (pdif)
    
    mvs = ["bmon", "pdebt", "inv", "tloan", "gdp", "stock", "cons"]
    for v in mvs:
        df[v + "_real"] = (df[v] / df["cpi"]) * 100 # make variable real

    mvs_real = [m + "_real" for m in mvs]
    
    df, _ = make_shift(df, ['stock', 'cpi', 'cons', "cons_nom", 'gdp', 'hp', "tdbtserv_gdp"] + pre_gdp_ratios + mvs_real, type="percentage", horizon=horizon) # percentage change of nominal variables
    df, _ = make_shift(df, ['gdp'], type="percentage", horizon=1)  # this variable is useful when defining recession
    df, _ = make_shift(df, ['cpi', "gdp"], type="percentage", horizon=1) 


    # log variables
    vars_to_log = [re.sub("(log_|global_|_hp|_ham|_norm)", "",v)  for v in set(predictors) - set(["global_drate", "global_loan"])]
    for v in vars_to_log:
        df["log_" + v] = np.log(df[v])

    df, _ = make_shift(df, mvs_real, type="percentage", horizon=horizon) # percentage change of real variables
    
    
    # compute the filters is slow - w only do it for those variable we actually need
    vars_filters_ham = [re.sub("(_ham|global_|_norm)", "",v)  for v in predictors if "_ham" in v]
    vars_filters_hp =  [re.sub("(_hp|global_|_norm)", "",v)  for v in predictors if "_hp" in v]

    df, _ = make_level_change(df, vars_filters_ham, type="ham") # hamilton filter
    df, _ = make_level_change(df, vars_filters_ham, type="ham" , norm = True) # log for comparability issues of series (see Drehmann 2018)
    
    df, _ = make_level_change(df, vars_filters_hp, type="hp") # hp filter
    df, _ = make_level_change(df, vars_filters_hp, type="hp", norm = True) # log for comparability issues of series

    # --- Computing global variables --- #
    # baseline global credit growth (global_loan)
    for year in df["year"].unique():
        ix = df["year"] == year
        for country in df["iso"].unique():
            # computing the average across all countries but the selected one
            perc_pos = df.loc[ix.values & (df.iso != country).values,
                              "tloan_gdp_rdiff" + str(horizon)].mean()

            if not np.isnan(perc_pos):
                df.loc[ix.values & (df.iso == country).values,
                       "global_loan"] = perc_pos

    # baseline global slope of the yield curve
    for year in df["year"].unique():
        ix = df["year"] == year
        for country in df["iso"].unique():
            # computing the average across all countries but the selected one
            perc_pos = df.loc[ix.values & (df.iso != country).values, "drate"].mean()

            if not np.isnan(perc_pos):
                df.loc[ix.values & (df.iso == country).values, "global_drate"] = perc_pos




    # here we compute other global variables that are used in robustness checks - we average the variables across all countries in a year (excluding the country for which we make the computation)
    global_predictors = set(predictors) - set(["global_loan", "global_drate"])# we have computed our baseline global variables already, here we scan for other global variables we need to compute
    variables_transform_global = [p for p in list(global_predictors) if "global_" in p ]
    variables_transform = [p.replace("global_", "") for p in variables_transform_global]

    for v in variables_transform_global:
        df[v] = np.nan
    if len(variables_transform) > 0:
        for year in unique_years:
            ix = df["year"].values == year
            for country in unique_countries:
                perc_pos = df.loc[ix & (df.iso.values != country), variables_transform].mean()
                df.loc[ix & (df.iso.values == country), variables_transform_global] = perc_pos.values

    # Other countries replacing global slope:
    for country_global in unique_countries: 
        df["global_drate_" + country_global] = np.tile(df["drate"].values[df.iso.values == country_global], len(unique_countries))
        miss_ix = np.isnan(df["global_drate_" + country_global].values)
        df["global_drate_" + country_global].loc[miss_ix] = df["global_drate"].loc[miss_ix]


    # other countries replacing global credit
    for country_global in unique_countries: 
        df["global_loan_" + country_global] = np.tile(df["tloan_gdp_rdiff" + str(horizon)].values[df.iso.values == country_global], len(unique_countries))
        # replace missing values with standard global variable. Just to ensure that the smaples sizes are equivalent
        miss_ix = np.isnan(df["global_loan_" + country_global].values)
        df["global_loan_" + country_global].loc[miss_ix] = df["global_loan"].loc[miss_ix]

    




    # --- creating the 'landing zone' on the crisis outcome --- #
    years = df.year.values
    isos = df['iso'].values
    min_yr = np.min(years)

    # CRISIS variable

    crisis_in = df.crisis_var.values == 1
    # Flag pre-crisis years
    crisis = crisis_in * 0
    for i, (yr, cr) in enumerate(zip(years, crisis_in)):
        if cr:
            for l in np.arange(1, crisis_lag+1):
                if yr > (min_yr + l-1):
                    crisis[i-l] = 1
            if config.data_include_crisis_year:
                   crisis[i] = 1  # crisis year

    # Define crisis ID
    # the same crisis gets identical id. is used for cross-validation.
    # This function generalizes to any length of crises
    crisis_id = np.zeros(len(df))
    count = int(1)
    for i in np.arange(2, len(df)):
        if crisis[i] == 1:
            if not ((crisis[i - 1] == 1) & (isos[i] == isos[i - 1])):
                count += 1
            crisis_id[i] = count
    # All other observations get unique identifier
    crisis_id[crisis_id == 0] = np.random.choice(sum(crisis_id == 0),
                                                 size=sum(crisis_id == 0), replace=False) + 2 + int(max(crisis_id))
    # flag post crisis
    i_keep = np.ones(len(df), dtype=int)
    for i, (yr, cr, iso) in enumerate(zip(years, crisis_in, df.iso)):
        if cr:
            if not config.data_include_crisis_year:
                i_keep[i] = 0

            if config.data_crisis_single_lag: #  If FALSE: when crisis is j, then all years 1,2,3,..,j is set to positive label). This is our standard approach. . If False only single year gets positive crisis label
                for j in np.arange(1, crisis_lag): # range one before actual crisis lag
                    if i-j > 0: 
                        if df.iso[i] == df.iso[i-j]:
                            i_keep[i-j] = 0

            # post crisis
            for j in range(1, config.data_post_crisis + 1):
                k = i + j
                if (iso == df.iso[k]) & (k < len(df)):
                    i_keep[k] = 0



    if len(set(predictors).difference(set(df.columns.values))) > 0:
        raise ValueError('Features ' + ', '.join(set(predictors).difference(set(df.columns.values))) + "\n" +
                         "could not be found in the data!")


    # create the data set
    features = df.loc[:, predictors]
    data = features
    data['crisis'] = crisis.astype(int)
    data['crisis_id'] = crisis_id.astype(int)
    data['year'] = years.astype(int)
    data['iso'] = isos # name of countries

    # exclude special periods
    exclude_ix = np.isin(data.year.values, list(range(1914 - crisis_lag,1919)) + list(range(1933 - crisis_lag, 1946)))
    exclude_ix = exclude_ix | (i_keep == 0)
    
    if config.clustered_crises == "remove":

        exclude_clusters = np.isin(data.year.values, list(range(1907 - crisis_lag, 1908 + 1)) + list(range(1930 - crisis_lag, 1931 + 1)) + list(range(2007 - crisis_lag, 2008 + 1)))                                                         
        exclude_ix = exclude_ix | exclude_clusters
    
    if config.data_period  == 'pre-ww2':
        exclude_ix = exclude_ix | np.array(data["year"] > 1939)

    if config.data_period == 'post-ww2':
        exclude_ix = exclude_ix | np.array(data["year"] < 1946)


    data = data.loc[~exclude_ix, :]


    X = data.copy()
    X = X.drop(columns=['crisis', 'crisis_id', 'year', 'iso'])
    feature_names = list(X)

    # exclude instances with missing cue values
    data.dropna(inplace=True)

    # Update index
    data.reset_index(drop=True, inplace=True)
    data.to_csv("r_data/current_data.csv")

    return data
