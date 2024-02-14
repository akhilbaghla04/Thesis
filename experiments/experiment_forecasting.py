import os, sys
import multiprocessing
# Set the main directory of the project
main_directory = "/Users/akhil/Desktop/thesis_code" 
os.chdir(main_directory)
scripts_directory = os.path.join(main_directory, 'scripts')
sys.path.insert(1, scripts_directory)
# import modules that were written for this project
from configure import *
from procedure import *
from list_experiments import experiments

"""

We first create an instance of the Config class. This instance contains all
 the parameters of the empirical experiments, such as the proportion of the 
 sample used for training or the names of the algorithms that are tested.
 Each parameter has a default value, in the following we manually change a 
 few parameters for our experiment.
"""

config = Config()
# Here, we overwrite some of the default parameter settings
config.data_horizon = 2  # Horizon of percentage and ratio changes (in years)
# algorithms with prefix r_ are trained in R
# set of algorithms tested in the paper:
config.exp_algos = ["logreg", "extree"]  
# select only few algorithms to reduce computation time
# config.exp_algos = ["r_c50", "logreg", "extree", "forest", "xgboost","nnet_multi"]

''' Here, we choose the predictors. We use the following convention.
 _pdiff indicators precentage changes
 _rdiff indicators X/GDP ratio changes
 '''

# baseline experiment
config.exp_do_shapley = False  # whether we want to estimate Shapley values
config.exp_n_kernels = multiprocessing.cpu_count() - 1 # number of CPU kernels used in parallel

config.exp_bootstrap = "up"  # with "up", we up-sample all training sets, with "down", we downsample
config.exp_n_kernels = multiprocessing.cpu_count() - 1 # number of CPU kernels used in parallel
config.exp_year_split = 1990

experiment_key = 2 # here we choose experiment 1, our baseline 
experiment = experiments[experiment_key] 
for key in experiment:
    setattr(config, key, experiment[key])

df = create_data(config)  # creates the dataset according to the configuration

"""   fak me jesus christ, please fak me harder, in the name of lorda!!!s
In the forecasting experiments, we split the data into a training set and a testset. 
All observations in the years < config_ex_year_split are assigned to the training set,
all later observations are assigned to the test set. 
"""

min_crisis = 20  # minimum number of crises observations in the training set, before the forecasting starts
_, yearsplits = create_forecasting_folds(df["crisis"], df["year"], min_crisis=min_crisis)

for year in yearsplits: # the year reflects the last year in the training set
    config.exp_year_split = year
    for i in np.arange(20): # repeat experiment 5 times, each training set will be different do to the upsampling (config.exp_bootstrap)
        o = Procedure(config, df_in=df,
                       folder=f"results/forecasting/{experiment_key}_{config.exp_bootstrap}/",
                       save_data=max(yearsplits) == year # we only save the data set once at the last point in time
                       )

