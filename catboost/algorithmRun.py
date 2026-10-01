from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier 
from xgboost import XGBClassifier
import cv_engine as cve
estimators = [
    [
        XGBClassifier,
        {
            "max_depth": 4,
            "learning_rate": 0.1, 
            "n_estimators": 1000, 
            "subsample": 0.8,
            'random_seed': 42, 
        }
    ],
    [
        RandomForestClassifier,
        {
            "n_estimators": 453,
            "max_features": 3,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            'random_seed': 42, 
        }
    ],
    [
        HistGradientBoostingClassifier,
        {
            "learning_rate": 0.0815,
            "max_depth": 9,
            "l2_regularization": 2.1466,
            "max_iter": 2678,
            "max_features": 0.7605,
            'random_seed': 42, 
        }
    ],
    [
        CatBoostClassifier,
        {
            'verbose': 0, 
            'random_seed': 42, 
            'learning_rate': 0.1456, 
            'depth': 5, 
            'l2_leaf_reg': 2.7181, 
            'iterations': 1060
        }
    ]
]

results = {}

for estimator in estimators:
    r = cve.run(
        model_class=estimator[0],
        model_params=estimator[1],
        n_jobs=-1,
    )
    results[estimator[0].__name__] = r

import pandas as pd

rslt = pd.DataFrame(data=results, index=estimators)
    