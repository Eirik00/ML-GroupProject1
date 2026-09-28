from catboost import CatBoostClassifier
from cv_engine import run_full_pipeline

def main():
    param_grid = {
        "learning_rate": [0.01, 0.1, 0.6],#list(map(lambda x:(x+5)/100, list(range(0,100,5)))),#
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5, 10, 20],
        "rsm": [0.7, 0.85, 1.0],
        "subsample": [0.7, 0.85, 1.0],
        "bootstrap_type":['Bernoulli', 'MVS']
    }

    static_params = {
        "iterations": 2500,
        "random_seed": 42,
        "verbose": 0,
        "early_stopping_rounds": 50,
    }

    return run_full_pipeline(
        model_class=CatBoostClassifier,
        param_grid=param_grid,
        static_params=static_params
    )

if __name__ == "__main__":
    results = main()
