from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report

from datasetloader import load_dataset

def fitmodel(cbModel):
    x_training, y_training, x_validation, y_validation, x_testing, y_testing = load_dataset()

    train_data = Pool(data=x_training,   label=y_training)
    valid_data = Pool(data=x_validation, label=y_validation)
    test_data  = Pool(data=x_testing,    label=y_testing)

    model = cbModel

    model.fit(
        train_data,
        eval_set=valid_data,
        early_stopping_rounds=50,
        use_best_model=True,
    )

    test_preds = model.predict(test_data)
    test_probs = model.predict_proba(test_data)

    print(classification_report(y_testing, test_preds),"--------------------------\n",accuracy_score(y_testing, test_preds))
    evals_result = model.get_evals_result()

    # Access specific metrics
    train_loss = evals_result["learn"]["Logloss"]
    val_loss = evals_result["validation"]["Logloss"]

    print(f"Training Loss: {train_loss[-1]}")
    print(f"Validation Loss: {val_loss[-1]}")

def autoTuner(oIteration, vpTarget):
    while True:


        pass;

if __name__ == "__main__":
    cbModel = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=5,
        eval_metric='Logloss', # research what this does, potential other metrics to use
        l2_leaf_reg=4,
        random_seed=42,
        verbose=True)

    fitmodel(cbModel)
