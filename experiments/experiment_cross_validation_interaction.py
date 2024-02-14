# new experiment: key difference: use the new data generation approach V4
import os, sys
import multiprocessing

# Set the main directory of the project
# Set the main directory of the project
main_directory = "/Users/akhil/Desktop/thesis_code" 
os.chdir(main_directory)
scripts_directory = os.path.join(main_directory, 'scripts')
sys.path.insert(1, scripts_directory)
# import modules that were written for this project
from configure import *
from procedure import *
from list_experiments import experiments
'''
We first create an instance of the Config class. This instance contains all the 
parameters of the empirical experiments, such as the proportion of the sample used 
for training or the names of the algorithms that are tested. Each parameter 
has a default value (see scripts/configure.py) - in the following we manually 
change a few parameters for our experiment.
'''

config = Config()

# Here, we overwrite some of the default parameter settings
config.data_horizon = 2  # Horizon of percentage and ratio changes (in years)
# algorithms with prefix r_ are trained in R
# set of algorithms tested in the paper:
# config.exp_algos = ["r_c50", "logreg", "extree", "forest", "svm_multi", "nnet_multi"]  
# select only few algorithms to reduce computation time
config.exp_algos = ["extree", "logreg"]

''' Here, we choose the predictors. We use the following convention.
 _pdiff indicators precentage changes
 _rdiff indicators X/GDP ratio changes
 '''

# baseline experiment

config.exp_do_shapley = True  # whether we want to estimate Shapley values
config.exp_shapley_interaction = True
config.exp_n_kernels = multiprocessing.cpu_count() - 1 # number of CPU kernels used in parallel


config.exp_taylor_feature_select = ["drate",
                                    "tloan_gdp_rdiff", 
                                    "global_loan",
                                    "global_drate"]

''' All other experiments are specified in the "experiments" dictionary.
 The experiments are indexed by a number. The dictionary specifies for each experiment, which parameters are changed 
 with respect to the baseline. The "description" parameters describes the experiments.
 With the following lines of code we overwrite the respective baseline
 parameters. 
'''

# here we choose experiment 1, our baseline 
experiment_key = 1
experiment = experiments[experiment_key] # here we choose experiment 1, our baseline 
for key in experiment:
    setattr(config, key, experiment[key])

    
df = create_data(config) # creates the dataset according to the configuration
for i in np.arange(22): # repeat the cross-validation experiment 10 times
    o = Procedure(config, df_in=df, folder=f"results/cross-validation-interaction/{experiment_key}/")




