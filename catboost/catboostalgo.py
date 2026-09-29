from sklearn.ensemble import HistGradientBoostingClassifier
import cv_engine as cve
import pandas as pd

import numpy as np

hist_param_space = {
        "learning_rate": ("float", 0.01, 1.0, True), # (type, low, high, log_scale)
        "max_depth": ("int", 2, 10),                     # (type, low, high)
        "l2_regularization": ("float", 0.0, 10.0),
        "max_iter": ("int", 20, 5000, True),
        "max_features": ("float", 0.0, 1.0)
    }
params = {
            "random_state":42,
        }

cve.run(HistGradientBoostingClassifier, params, hist_param_space, True, n_jobs=-1, n_trials=10)
