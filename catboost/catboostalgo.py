from catboost import CatBoostClassifier
import cv_engine as cve
import pandas as pd

import numpy as np

cb_param_space = {
        "learning_rate": ("float", 0.01, 1.0, True), # (type, low, high, log_scale)
        "depth": ("int", 2, 10),                     # (type, low, high)
        "l2_leaf_reg": ("float", 0.0, 10.0),
        "iterations": ("int", 20, 5000, True),
        #"max_features": ("float", 0.0, 1.0)
    }
params = {
            "random_state":42,
        }

cve.run(model_class=CatBoostClassifier, model_params=params, param_space=cb_param_space, tune_model_bool=True, n_jobs=-1, n_trials=10)
