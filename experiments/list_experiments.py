experiments = {
        
    1: {
        "description": "baseline experiment",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },
    
    
    2: {
        "description": "baseline experiment with logreg and extree only",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },
    
    3: {
        "description": "baseline experiment",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },
    
    405: {
        "description": "use percentage change variables on real variables",
        "data_feature_names": ["drate", "global_drate", "cpi_pdiff", 
                                     "stock_real_pdiff", "cons_pdiff", # indices (consumption is already real)
                                     "ca_gdp_rdiff", # contains negative values, therefore no percentage change transformation
                                    "bmon_real_pdiff", "pdebt_real_pdiff", "inv_real_pdiff" ,  "tloan_real_pdiff" ,
                                     "tdbtserv_gdp_pdiff", # is defined as GDP ratio thus we keep it as such!
                                      "global_tloan_real_pdiff", "gdp_real_pdiff"]


    },
     406: {
        "description": "use percentage changes on nominal variables, add GDP as another",    
        "data_feature_names": ["drate", "cpi_pdiff" , "bmon_pdiff",
                                     "stock_pdiff", "pdebt_pdiff",
                                     "cons_nom_pdiff", # consumption is normally a real series, here we use a nominal series so that all series are nominal 
                                     "inv_pdiff", "tloan_pdiff",
                                    "ca_gdp_rdiff", # contains negative values, therefore no percentage change transformation
                                     "tdbtserv_gdp_pdiff",  # is defined as GDP ratio, thus we need to use it as such
                                     "global_tloan_pdiff",
                                    "global_drate", "gdp_pdiff"]
     },

    500: {
        "description": "HP filter on GDP ratios",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_hp", "stock_pdiff", "cons_pdiff" ,
                                     "pdebt_gdp_hp", "inv_gdp_hp", "ca_gdp_hp", "tloan_gdp_hp",
                                     "tdbtserv_gdp_hp", "global_tloan_gdp_hp", "global_drate"]

    },
    #  505: {
    #         "description": "HP filter on nominal variables",
    #         "data_feature_names":  ["drate", "global_drate", 
    #                                     "cpi_pdiff", "stock_pdiff", "cons_pdiff",
    #                                     "tdbtserv_gdp_hp", # is defined as GDP ration, thus we keep it as such
    #                                     "log_bmon_hp_norm", "log_pdebt_hp_norm", "log_inv_hp_norm", "log_tloan_hp_norm", # use log change on those variables to make comparison possible (Drehman 2019)
    #                                     "ca_gdp_hp", # current account has many negative values, thus we cannot take the log
    #                                     "global_log_tloan_hp_norm", "log_gdp_hp_norm"]
    # },
    # 504: {
    #         "description": "HP filter on nominal variables",
    #         "data_feature_names":  ["drate", "global_drate", 
    #                                     "cpi_pdiff", "stock_pdiff", "cons_pdiff",
    #                                     "tdbtserv_gdp_hp", # is defined as GDP ration, thus we keep it as such
    #                                     "log_bmon_real_hp_norm", "log_pdebt_real_hp_norm", "log_inv_real_hp_norm", "log_tloan_real_hp_norm", # use log change on those variables to make comparison possible (Drehman 2019)
    #                                     "ca_gdp_hp", # current account has many negative values, thus we cannot take the log
    #                                     "global_log_tloan_real_hp_norm", "log_gdp_real_hp_norm"]
    # },

    # 510: {
    #     "description": "Hamilton filter on GDP ratios",
    #     "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_ham", "stock_pdiff", "cons_pdiff",
    #                                  "pdebt_gdp_ham", "inv_gdp_ham", "ca_gdp_ham", "tloan_gdp_ham",
    #                                  "tdbtserv_gdp_ham", "global_tloan_gdp_ham", "global_drate"]

    # },
    515: {
        "description": "Hamilton filter on nominal variables",
        "data_feature_names": ["drate", "global_drate", 
                                        "cpi_pdiff", "stock_pdiff", "cons_pdiff",
                                        "tdbtserv_gdp_ham", # is defined as GDP ration, thus we keep it as such
                                        "log_bmon_ham_norm", "log_pdebt_ham_norm", "log_inv_ham_norm", "log_tloan_ham_norm", # use log change on those variables to make comparison possible (Drehman 2019)
                                        "ca_gdp_ham", # current account has many negative values, thus we cannot take the log
                                        "global_log_tloan_ham_norm", "log_gdp_ham_norm"]
    },
    514: {
        "description": "Hamilton filter on real  variables",
        "data_feature_names": ["drate", "global_drate", 
                                        "cpi_pdiff", "stock_pdiff", "cons_pdiff",
                                        "tdbtserv_gdp_ham", # is defined as GDP ration, thus we keep it as such
                                        "log_bmon_real_ham_norm", "log_pdebt_real_ham_norm", "log_inv_real_ham_norm", "log_tloan_real_ham_norm", # use log change on those variables to make comparison possible (Drehman 2019)
                                        "ca_gdp_ham", # current account has many negative values, thus we cannot take the log
                                        "global_log_tloan_real_ham_norm", "log_gdp_real_ham_norm"]
    },
    108: {
    "description":  "Transformation horizon (i.e. percentage change) of 1 year",
    "data_horizon": 1
    },

    109: {
    "description":  "Transformation horizon (i.e. percentage change) of 3 years",
    "data_horizon": 3
    },

    110: {
    "description":  "Transformation horizon (i.e. percentage change) of 5 years",
    "data_horizon": 4
    },

    111: {
    "description":  "Transformation horizon (i.e. percentage change) of 5 years",
    "data_horizon": 5
    },

    750: {
            "description": "Forecast horizon: 1 year",
            "data_crisis_lag": 1
        },
    752: {"description": "Forecast horizon: 1-3 years",
          "data_crisis_lag": 3
        },
    753: {"description": "Forecast horizon: 1-4 years",
          "data_crisis_lag": 4
        },
    754: {
        "description": "Forecast horizon: 2 year (single year)",
        "data_crisis_lag": 2,
        "data_crisis_single_lag": True
    },
    755: {  
        "description": "Forecast horizon: 3 year (single year)",
        "data_crisis_lag": 3,
        "data_crisis_single_lag": True
        },
    756: {  
        "description": "Forecast horizon: 4 year (single year)",
        "data_crisis_lag": 4,
        "data_crisis_single_lag": True
        },

    700: {
        "description": "Crisis definition by Baron (2021, QJE)",
        "crisis_definition": "BR",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
    },

    102: {
        "description": "Replace slope of the yield curve with nominal short and long rate",
         "data_feature_names": ["srate", "lrate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]


    },
    103: {
        "description": "Replace slope of the yield curve with real short and long rate",
        "data_feature_names": ["srate_real", "lrate_real", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
    },
    353: {
      "description": "Add 1-year change in short-term rate",
        "data_feature_names": ["srate_rdiff1", "drate",  "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
    },
    352: {
      "description": "Add 2-year change in short-term rate",
      "data_feature_names": ["srate_rdiff", "drate",  "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
    },
    1007: {
        "description": "Total loans replaced by household and business loans (including global) but we keep using total loans for debt-service ratio.",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                        "cons_pdiff",
                                        "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                        "hloan_gdp_rdiff", "bloan_gdp_rdiff" ,
                                        "tdbtserv_gdp_rdiff", 
                                        "global_hloan_gdp_rdiff", "global_bloan_gdp_rdiff",
                                        "global_drate"]

    },
    107: {
        "description": "Add house prices",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate",
                                     "hp_pdiff"]

    },
    600: {
        "description": "Remove stock prices because we have many missing values on this variable",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", 
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
    },
    720: {
        "description": "Do not remove post-crisis period. These observations are treated as non-crisis events",
        "data_post_crisis": 0
    },
    12: {
        "description": "remove clustered crises",
        "clustered_crises": "remove"
    },
    13: {"description": "remove non-clustered crises",
        "clustered_crises": "only" 
    },

    7112: {
        "description": "remove clustered crises",
        "clustered_crises": "remove",
        "data_crisis_lag": 1,
        "data_crisis_single_lag": True
    },
    7113: {"description": "remove non-clustered crises",
        "clustered_crises": "only" ,
        "data_crisis_lag": 1,
        "data_crisis_single_lag": True
    },

    7212: {
        "description": "remove clustered crises",
        "clustered_crises": "remove",
        "data_crisis_lag": 2,
        "data_crisis_single_lag": True
    },
    7213: {"description": "remove non-clustered crises",
        "clustered_crises": "only" ,
        "data_crisis_lag": 2,
        "data_crisis_single_lag": True
    },

    7312: {
        "description": "remove clustered crises",
        "clustered_crises": "remove",
        "data_crisis_lag": 3,
        "data_crisis_single_lag": True
    },
    7313: {"description": "remove non-clustered crises",
        "clustered_crises": "only" ,
        "data_crisis_lag": 3,
        "data_crisis_single_lag": True
    },

    501: {
        "description": "Checking importance of variable global_drate (yield curve)",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan"]
        
        },

    502: {
        "description": "Checking importance of variable drate (yield curve)",
        "data_feature_names": ["cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },

    503: {
        "description": "removed global_loan",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_drate"]
        
        },

    504: {
        "description": "removed domestic loan ",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                    "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },

    505: {
        "description": "removed investment ",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },

    506: {
        "description": " stock price removed",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },

    507: {
        "description": "cpi removed",
        "data_feature_names": ["drate", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },

    508: {
        "description": "consumption removed",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },
    
    509: {
        "description": "broad money removed",
        "data_feature_names": ["drate", "cpi_pdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },

    510: {
        "description": "public debt removed",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },
    
    511: {
        "description": "current account removed",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff",
                                     "tloan_gdp_rdiff", "tdbtserv_gdp_rdiff", "global_loan", "global_drate"]
        
        },

    512: {
        "description": "debt service ratio removed",
        "data_feature_names": ["drate", "cpi_pdiff", "bmon_gdp_rdiff", "stock_pdiff",
                                     "cons_pdiff", "pdebt_gdp_rdiff", "inv_gdp_rdiff", "ca_gdp_rdiff",
                                     "tloan_gdp_rdiff", "global_loan", "global_drate"]
        
        }  

}
    