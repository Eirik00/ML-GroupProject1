from catboost import CatBoostClassifier
from cv_engine import run_full_pipeline
def main():
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.2],
        "depth": [4, 6, 8],
        "subsample": [0.6, 0.8, 1.0]
    }

    static_params = {
        "iterations": 1500,
        "random_seed": 42,
        "verbose": 0
    }

    results = run_full_pipeline(
        model_class=CatBoostClassifier,
        param_grid=param_grid,
        static_params=static_params
    )


if __name__ == "__main__":
    main()
