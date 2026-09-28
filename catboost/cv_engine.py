import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.io import arff
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def lap_data():
    """Import and preprocess dataset."""
    data, meta = arff.loadarff('datasets/Training Dataset.arff')
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.decode('utf-8')

    df = df.astype(int)
    df['Result'] = df['Result'].replace(-1, 0) # Convert -1 to 0 for binary classification

    t_set = df['Result']
    f_set = df.drop('Result', axis=1)
    
    print("==========================================")
    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape} | {len(df)} rows, {len(df.columns)-1} features")
    print("==========================================")

    X_train, X_test, y_train, y_test = train_test_split(
        f_set, t_set,
        test_size=0.1,
        random_state=42,
        stratify=t_set
    )
    return X_train, X_test, y_train, y_test

def evaluate_test_set(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Helper function to calculate test set metrics."""
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0)
    }

def tune_model(
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    static_params: Dict[str, Any] = None,
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Modular tuning function. Uses native grid search for CatBoost
    and standard GridSearchCV for sklearn/XGBoost models.
    """
    static_params = static_params or {}
    print("\n--- STARTING HYPERPARAMETER TUNING ---")
    base_model = model_class(**static_params)
    if "CatBoost" in model_class.__name__:
        grid_res = base_model.grid_search(
            param_grid, 
            X=X_train, 
            y=y_train, 
            cv=5, 
            verbose=True, 
            plot=False, 
            stratified=True)
        best_params = {**static_params, **grid_res['params']}
        best_model = model_class(**best_params)
        best_model.fit(X_train, y_train, verbose=False)
        print(f"Best CatBoost Params: {grid_res['params']}")
        return best_model, best_params, grid_res
    else:
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=3
        )
        grid_search.fit(X_train, y_train)
        
        best_params = {**static_params, **grid_search.best_params_}
        print(f"Best {model_class.__name__} Params: {grid_search.best_params_}")
        return grid_search.best_estimator_, best_params, grid_search

def evaluate_robustness(
    model_class: Any,
    best_params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    data_fractions: List[float] = [1.0, 0.5, 0.25],
    feature_counts: List[int] = [None, 15, 5]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tests model sensitivity/robustness against:
      1. Reduced Training Data Sizes (% of rows)
      2. Reduced Feature Sets (Top N columns based on feature importance)
    """
    print("\n==========================================")
    print("RUNNING ROBUSTNESS & SENSITIVITY TESTS")
    print("==========================================")

    # --- EXPERIMENT A: LESS TRAINING DATA ---
    print("\n[Test A] Evaluating sensitivity to reduced training data...")
    data_size_results = []
    
    for frac in data_fractions:
        if frac == 1.0:
            X_tr_sub, y_tr_sub = X_train, y_train
        else:
            X_tr_sub, _, y_tr_sub, _ = train_test_split(
                X_train, y_train, train_size=frac, random_state=42, stratify=y_train
            )

        model = model_class(**best_params)
        model.fit(X_tr_sub, y_tr_sub)
        metrics = evaluate_test_set(model, X_test, y_test)
        
        data_size_results.append({
            "Train_Data_Pct": f"{int(frac*100)}%",
            "Train_Rows": len(X_tr_sub),
            **metrics
        })

    df_data_size = pd.DataFrame(data_size_results)

    # --- EXPERIMENT B: FEWER FEATURES ---
    print("\n[Test B] Evaluating sensitivity to missing attributes/features...")
    
    # Fit baseline model on full features to extract feature importances
    base_model = model_class(**best_params)
    base_model.fit(X_train, y_train)
    
    if hasattr(base_model, "feature_importances_"):
        importances = base_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = X_train.columns[sorted_indices].tolist()
    else:
        sorted_features = X_train.columns.tolist()

    feature_results = []
    for num_feats in feature_counts:
        selected_cols = sorted_features[:num_feats] if num_feats else X_train.columns.tolist()
        feat_label = f"Top {num_feats}" if num_feats else "All Features"

        X_tr_feat = X_train[selected_cols]
        X_te_feat = X_test[selected_cols]

        model = model_class(**best_params)
        model.fit(X_tr_feat, y_train)
        metrics = evaluate_test_set(model, X_te_feat, y_test)

        feature_results.append({
            "Feature_Subset": feat_label,
            "Num_Features": len(selected_cols),
            **metrics
        })

    df_features = pd.DataFrame(feature_results)

    return df_data_size, df_features

def run_full_pipeline(
    model_class: Any,
    param_grid: Dict[str, List[Any]],
    static_params: Dict[str, Any] = None
):
    """
    All-in-one high level function to load data, tune hyperparameters,
    evaluate holdout performance, and run robustness experiments.
    """
    X_train, X_test, y_train, y_test = lap_data()

    # Step 1: Tune Model
    best_model, best_params, gs_model = tune_model(model_class, param_grid, static_params, X_train, y_train)

    # Step 2: Baseline Holdout Evaluation
    baseline_metrics = evaluate_test_set(best_model, X_test, y_test)
    print("\n--- BASELINE HOLDOUT TEST METRICS ---")
    for metric, score in baseline_metrics.items():
        print(f"Test {metric.capitalize()}: {score:.4f}")

    # Step 3: Robustness & Sensitivity Tests
    df_data_size, df_features = evaluate_robustness(
        model_class, best_params, X_train, y_train, X_test, y_test
    )

    print("\n--- DATA REDUCTION RESULTS ---")
    print(df_data_size.to_string(index=False))

    print("\n--- FEATURE SHORTENING RESULTS ---")
    print(df_features.to_string(index=False))

    return {
        "best_params": best_params,
        "baseline_metrics": baseline_metrics,
        "data_size_experiments": df_data_size,
        "feature_experiments": df_features,
        "gs_model": gs_model,
    }