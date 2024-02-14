"""
Code to estiamte the Shapley Taylor index. We use it to compute Shapley value interactions.
"""

import numpy  as np
import pandas as pd
from   itertools import chain, combinations
import math
import shap
import random
import multiprocessing as mp
import pathos.pools as pp

# %%

def logit(p):
    return np.log(p)- np.log(1 -p)

def Shapley_taylor_wrapper(model, train_x, test_x, k = 3, 
                        features_select = None,
                        background_size = None,
                        probabilities = True, # if True, we compute Shapley values directly on the predicted probabilities.
                                               # This is sensible for tree models but not for linear models, for which estimate the Shapley vlaues on the linear output
                        sample_coalitions = 10000):  
    
    '''
    Wrapper function, which is called from the _compute_shap method of the PredictionModel class
    '''
    if features_select is None:
        features_select = list(train_x.columns)

    if background_size < train_x.shape[0]:
        background = pd.DataFrame(shap.kmeans(train_x, background_size).data, columns = train_x.columns)
    else:
        background = train_x

    output = Shapley_Taylor_index(model,
                            F = features_select,
                            df_x = test_x,
                            df_b = background,
                            k = k,
                            is_skl = True,
                            verbose = True,
                            sample_coalitions = sample_coalitions,
                            classify = True,
                            probabilities = probabilities,
                            parallel = False)
    return(output)


def cond_exp(model,S,F,df_x,df_b,other_F=None,is_skl=True, classify = True, probabilities = True):
    '''Expectations of model for input df_x over background df_b
       conditioned on set of coalition variables S not in F.

       Returns vector of len(c).                               '''
    
    
    if ('others' in S) or ('others' in F):
        if 'others' in S: # conditioning features
            S  = list(S) + list(other_F)
            S.remove('others')
        if 'others' in F: # all features
            F  = list(F) + list(other_F)
            F.remove('others')
    not_S = list(set(F)-set(S)) # variables to be integrated out
    # df_x, df_b = df_x[F], df_b[F] # assure comparability
    df_x.index = np.arange(len(df_x)) # standardise indices
    df_b.index = np.arange(len(df_b))
    df_nx = pd.concat([df_x]*len(df_b),ignore_index=True)      # copy for each observation in background
    # insert input repeated index for grouping
    df_nx.loc[:,'rep_idx'] = pd.concat([pd.DataFrame(df_x.index)]*len(df_b),ignore_index=True)[0].values
    df_nb = df_b.loc[df_b.index.repeat(len(df_x))]  # repeat rows for each observation in input data
    df_nx.loc[:,not_S] = df_nb[not_S].values                     # insert background data conditioned on S
    
    cols_use_ordered = list(df_x.columns)
    if is_skl==True:
        if classify:
            preds = model.predict_proba(df_nx[cols_use_ordered])[:,1]
            if not probabilities:
                preds = logit(preds)
            df_nx.loc[:,'pred'] = preds
        else:
            df_nx.loc[:,'pred'] = model.predict(df_nx[cols_use_ordered])
    else:
        df_nx.loc[:,'pred'] = model(df_nx[cols_use_ordered]) # any custom function

    # mean of model averaged over background for each input observation
    return df_nx.groupby('rep_idx')['pred'].mean().values


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    l = list(iterable)
    # note we return an iterator rather than a list
    return list(chain.from_iterable(combinations(l,n) for n in range(len(l)+1)))

def set_derivative(model,S,T,F,df_x,df_b,other_F=None,is_skl=True, classify = False, probabilities = True):
    '''Feature set derivative of order len(S) of model relative to features in T.'''
    d_vals = np.zeros(len(df_x)) # initialisation

    for w in powerset(S):
        d_vals += (-1)**(len(w)-len(S)) * cond_exp(model,T+list(w),F,df_x,df_b,other_F=other_F,is_skl=is_skl, classify = classify, probabilities = probabilities)

    return d_vals

def Shapley_Taylor_index(model, F, df_x, df_b, k=None, is_skl=True, verbose=False, sample_coalitions=2**20, classify = True, probabilities = True, parallel = True):
    '''Model Shapley decomposition with interactions up to order k.'''
    
    if parallel:
        # approach 1: # import from main fails on windows! https://github.com/uqfoundation/pathos/issues/118, might work on Linux cluster!
        pool = pp.ProcessPool(processes=14)
        
    def taylor_index(S):
        print(".",  end = "")
        S = list(S)
        '''Order-k Taylor Index of model with respect to feature coalition S.'''
        # this function adjust the weights according to the subsampling. Otherwise interaction effects are underestimated as their weights do not sum to 1
 
        if len(S)<k: # set derivative relative to the empty set
            I = set_derivative(model, S, [], F, df_x, df_b, other_F=other_F, is_skl=is_skl, classify = classify, probabilities = probabilities)
        else: # |S|==k
            p_set = powerset(set(F)-set(S))
            n_set = len(p_set)

            if n_set > sample_coalitions:
                p_set = random.sample(p_set, sample_coalitions)
            n_set_use = len(p_set)
            I = np.zeros(len(df_x)) # initialisation
            # loop over feature coalitions not containing S
            weights = list()
            for t in p_set:
                weights.append((k/len(F))*((math.factorial(len(t))*math.factorial(len(F)-len(t)-1))/math.factorial(len(F)-1)))
            weight_factor = 1.0/np.array(weights).sum()
            for t in range(n_set_use):
                I += weights[t] * weight_factor * set_derivative(model,S, list(p_set[t]), F, df_x, df_b, other_F=other_F, is_skl=is_skl, classify = classify, probabilities = probabilities)
        return I

    # sort features by main and others
    other_F = []
    for f in df_x.columns:
        if f not in F:
            other_F.append(f)
    if len(other_F)==0:
        other_F = None
    else:
        F = list(F)+['others']

    # max order of explanation: number of features
    if (k==None) or (k>len(F)): k=len(F)

    # baseline value F(0)
    
    if is_skl==True:
        if classify:
            preds = model.predict_proba(df_b)[:, 1]
            if not probabilities:
                preds = logit(preds)
            base = preds.mean()
    else:
        base = np.mean(model(df_b))

    # initialise output dataframe
    df_sv = pd.DataFrame(np.ones(len(df_x))*base,columns=['base'])

    if verbose==True:
        N, n = 0, 0
        for i in range(k):
            N += len(list(combinations(F,i+1)))

    # loop up to order k
    for i in range(k):
        print("k: " + str(i))
        s_set = list(combinations(F,i+1))
        print(len(s_set))
        if parallel:
            out = pool.map(taylor_index, s_set)
        else:
            out = list(map(taylor_index, s_set))
        df_sv = pd.concat([df_sv, pd.DataFrame(out).T], axis=1)

    df_sv.columns = Shapley_Taylor_namer(F,k)
    return df_sv



def Shapley_Taylor_namer(F, k=None):
    '''This function returns the names of the interactions estimated by the Shapley Taylor index method'''

    names_out = list(["base"])

    # loop up to order k
    for i in range(k):
        s_set = list(combinations(F,i+1))
        # loop over all order-k terms

        for j,s in enumerate(s_set):
            S = list(s)
            # column name
            if len(S)==1:
                col_name = S[0]
            else:
                col_name = '-'.join([str(f) for f in S])

            names_out.append(col_name)
    return names_out


