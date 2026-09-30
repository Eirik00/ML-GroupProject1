import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any, Tuple, Optional, Type, Callable
from scipy.io import arff
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, 
    recall_score, f1_score, 
    roc_auc_score, fbeta_score)
from sklearn.feature_selection import mutual_info_classif
import time

def lap_data():
    """Import and preprocess dataset."""
    data, meta = arff.loadarff('datasets/Training Dataset.arff')
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.decode('utf-8')

    df = df.astype(int)
    df['Result'] = (df['Result'] == -1).astype(int)

    t_set = df['Result']
    f_set = df.drop('Result', axis=1)

    n_phish = int((t_set == 1).sum())
    n_legit = int((t_set == 0).sum())

    print("==========================================")
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape} | {len(df)} rows, {len(df.columns)-1} features")
    print(f"Class 1 = phishing: {n_phish} | Class 0 = legitimate: {n_legit}")
    print(f"Majority baseline accuracy: {max(n_phish, n_legit) / len(df):.4f}")
    print("==========================================")

    X_train, X_test, y_train, y_test = train_test_split(
        f_set, t_set,
        test_size=0.1,
        random_state=42,
        stratify=t_set
    )
    return X_train, X_test, y_train, y_test

def tune_model(
    model_class: Type[Any],
    param_space: Dict[str, Tuple[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    static_params: Optional[Dict[str, Any]] = None,
    n_trials: int = 50,
    scoring_func: Callable = f1_score,
    scoring_kwargs: Optional[Dict[str, Any]] = None,
    tune_threshold: bool = False,
    n_jobs: int = 1,
) -> Tuple[Any, Dict[str, Any], optuna.Study]:
    """
    Modular Optuna hyperparameter tuning function compatible with Sklearn,
    XGBoost, LightGBM, and CatBoost models.

    Parameters:
    -----------
    param_space : Dict where key is hyperparameter name and value is a tuple:
        - ("int", low, high) or ("int", low, high, log_bool)
        - ("float", low, high) or ("float", low, high, log_bool)
        - ("categorical", [list_of_options])
    tune_threshold : bool
        If True, optimizes the classification threshold alongside hyperparameters.
    """
    static_params = static_params or {}
    print("\n--- STARTING HYPERPARAMETER TUNING ---")

    static_params = static_params or {}
    scoring_kwargs = scoring_kwargs or {}
    print(f"\n=================\n{model_class.__name__}\n===============")

    trainF, valF, trainT, valT = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    model = ()

    def objective(trial: optuna.Trial)->float:
        suggested_params={}
        for param_name, spec in param_space.items():
            param_type = spec[0]

            if param_type == "int":
                low, high = spec[1], spec[2]
                log = spec[3] if len(spec) > 3 else False
                suggested_params[param_name] = trial.suggest_int(param_name, low, high, log=log)

            elif param_type == "float":
                low, high = spec[1], spec[2]
                log = spec[3] if len(spec) > 3 else False
                suggested_params[param_name] = trial.suggest_float(param_name, low, high, log=log)

            elif param_type == "categorical":
                choices = spec[1]
                suggested_params[param_name] = trial.suggest_categorical(param_name, choices)

        current_params = {**static_params, **suggested_params}
        threshold = trial.suggest_float("threshold", 0.01, 0.50) if tune_threshold else 0.5

        model = model_class(**current_params)
        model.fit(trainF, trainT)

        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(valF)[:, 1]
            y_pred = (y_probs >= threshold).astype(int)
        else:
            y_pred = model.prediction(valF)

        score = scoring_func(valT, y_pred, **scoring_kwargs)
        return score

    #optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    best_params = {**static_params, **study.best_params}
    best_threshold = best_params.pop("threshold", None)

    print(f"\n--- OPTUNA TUNING COMPLETE ({model_class.__name__}) ---")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params:{best_params}")
    print(f"Best threshold:{best_threshold}")
    print("----------------------------------------------------------")

    return best_params, study

    

def feature_selection(X_train, y_train): # Source = https://karyailham.com.my/index.php/arca/article/view/1119/1257
    score = mutual_info_classif(X_train, y_train, discrete_features=True, random_state=42, n_jobs=2)
    result = pd.DataFrame({"X_train": X_train.columns, "mutual_info": score}).sort_values(["mutual_info", "X_train"], ascending=[False, True]).reset_index(drop=True)
    return list(result["X_train"])

def evaluate_robustness(
    model_class: Any,
    best_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tests model sensitivity/robustness against:
      1. Reduced Training Data Sizes (% of rows)
      2. Reduced Feature Sets (Top N columns based on feature importance)
    """
    print("\n==========================================")
    print("RUNNING ROBUSTNESS & SENSITIVITY TESTS")
    print("==========================================")

    model = model_class(**best_params)

    # --- EXPERIMENT A: LESS TRAINING DATA ---
    print("\n[Test A] Evaluating sensitivity to reduced training data...")
    data_size_results = {}
    data_fractions = [0.5, 0.25, 0.05]
    
    for frac in data_fractions:
        if frac == 1.0:
            X_tr_sub, y_tr_sub = X_train, y_train
        else:
            _, X_tr_sub, _, y_tr_sub = train_test_split(
                X_train, y_train, train_size=frac, random_state=42, stratify=y_train
            )

        model.fit(X_tr_sub, y_tr_sub)
        scores = model.predict_proba(X_test)[:, 1]
        pred = (scores >= 0.5).astype(int)
        i=int(frac*100)
        data_size_results[i] = {"accuracy": accuracy_score(y_test, pred),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, pred),
                    "f1": f1_score(y_test, pred, zero_division=0)}

    df_data_size = pd.DataFrame.from_dict(data_size_results, orient="index")

    # --- EXPERIMENT B: FEWER FEATURES ---
    print("\n[Test B] Evaluating missing values...")
    model.fit(X_train, y_train)

    missing_res = {}
    missing_fracts = [0.0, 0.1, 0.3, 0.5, 0.8]
    for frac in missing_fracts:
        X_corrupt = X_test.copy()
        np.random.seed(42)
        mask = np.random.rand(*X_corrupt.shape)<frac
        X_corrupt[mask] = np.nan

        scores = model.predict_proba(X_corrupt)[:, 1]
        pred = (scores >= 0.5).astype(int)
        i=int(frac*100)
        missing_res[i] = {"accuracy": accuracy_score(y_test, pred),
                    "recall": recall_score(y_test, pred, zero_division=0),
                    "precision": precision_score(y_test, pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, pred),
                    "f1": f1_score(y_test, pred, zero_division=0)}

    df_missing = pd.DataFrame.from_dict(missing_res, orient="index")

    return df_data_size, df_missing

def run(
    model_class: Type[Any],
    model_params: Dict[str, Any],
    param_space: Dict[str, Tuple[str, Any]] = None,
    tune_model_bool: bool = False,
    n_jobs: int=1,
    n_trials: int=20,
)-> pd.DataFrame:
    """
    All-in-one high level function to load data, tune hyperparameters,
    evaluate holdout performance, and run robustness experiments.
    """
    X_train, X_test, y_train, y_test = lap_data()

    metrics = {
        "accuracy": "accuracy",
        "recall": "precision",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
    }

    dataFolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

    if (tune_model_bool and param_space is not None):
        model_params, _ = tune_model(
            model_class=model_class,
            param_space=param_space,
            X_train=X_train,
            y_train=y_train,
            static_params=model_params,
            n_trials=n_trials,
            scoring_func=roc_auc_score,
            tune_threshold=False,
            n_jobs=n_jobs,
        )
    else:
        print("NEITHER TUNING ENABLED NOR PARAM_SPACE GIVEN")
        return        


    model = model_class(**model_params)
    total_res = {}
    strtTm = time.time()
    print(f"\n Model[{model_class.__name__}] starting training \n============================================")
    cv_results = cross_validate(model, X_train, 
                                y_train, cv=dataFolds, scoring=metrics, 
                                n_jobs=1, return_train_score=False)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    pred = (scores >= 0.5).astype(int)
    test_run = {"accuracy": accuracy_score(y_test, pred),
                "recall": recall_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, pred),
                "f1": f1_score(y_test, pred, zero_division=0)}
    flme = lambda a, b:float(np.mean([np.mean(a),np.mean(b)]))

    # Step 3: Robustness & Sensitivity Tests
    df_data_size, df_missing = evaluate_robustness(
        model_class, model_params, X_train, y_train, X_test, y_test
    )
    endTm = time.time() - strtTm
    print(f"\n Model[{model_class.__name__}] trained after {endTm}s\n============================================")
    
    total_res = {
        "Accuracy": flme(cv_results['test_accuracy'],test_run['accuracy']),
        "Recall": flme(cv_results['test_recall'],test_run["recall"]),
        "Precision": flme(cv_results['test_precision'],test_run["precision"]),
        "ROC-AUC": flme(cv_results['test_roc_auc'],test_run["roc_auc"]),
        "CV-F1": float(np.mean(cv_results['test_f1'])),
        "Test-F1": float(np.mean(test_run['f1'])),
    }

    df_results = pd.DataFrame([total_res])

    print("\n----------------- Results ----------------")
    print(df_results.round(3).to_string(index=False))
    print("\n------------------------------------------")
    print("\n------- Missing Data Sensitivity ---------")
    print(df_data_size.round(3).to_string())
    print("\n------------------------------------------")
    print("\n------- Missing values Sensitivity -------")
    print(df_missing.round(3).to_string())
    
    total_res = {**total_res,
        "data_size_experiments": df_data_size,
        "missing_values_experiments": df_missing,
    }


    endTm = time.time() - strtTm
    print(f"\n Experiments finished after {endTm}s\n============================================")

    return total_res
