# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:48:08 2024

@author: lizzi
"""

import numpy as np 
import pandas as pd
import pickle 
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product


def hyperparam_scan(model,data,hyperparams,hyperparams_list,name,accuracy_score=True,roc=False,roc_auc=False,cross_val=False):
    
    """
    Implements a manual hyperparameter scan for a given ML model
    model: ML model to use, with no parameters implmented (e.g. tree.DecisionTreeClassifier)
    hyperparams: list of hyperparams to test - [[h1a,h1b,h1c...],[h2a,h2b,h2c...],[h3a,h3b,h3c....],...]
    hyperparams: names of the hyperparams 
    name: name to save for the best model
    can choose which metrics to implement
    
    """
    X_train,X_test,y_train,y_test=data
    combinations=product(*hyperparams_list)
    columns=hyperparams+['accuracy_score']
    results_df=pd.DataFrame(columns=columns)
    

    best=0

    for params in combinations:
        test_model=model(*params)
        
        test_model.fit(X_train,y_train)
        pred=test_model.predict(X_test)
        acc=accuracy_score(y_test,pred)
        if acc>best:
            best=acc
            with open(name,'wb') as f:
                pickle.dump(test_model,f)
                
        model_df=pd.DataFrame(np.array([[*params]+[acc]]))
        results_df=pd.concat(results_df,model_df)
    
    return results_df

#%%


