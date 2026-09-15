import pandas as pd
from scipy.io import arff
from split import split_data

def load_dataset():
    # Import dataset
    data, meta = arff.loadarff('datasets/Training Dataset.arff')
    df = pd.DataFrame(data)

    # Converts from objects to str | fixes b'-1' -> -1
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.decode('utf-8')

    df = df.astype(int) # Convert to Int

    df['Result'] = df['Result'].replace(-1, 0) # Convert -1 to 0 for binary classification

    t_set = df['Result']                # Target set
    f_set = df.drop('Result', axis=1)   # Feature set
    print("==========================\nDataset loaded.")
    print(f"Dataset shape: {df.shape} | {len(df)} rows and {len(df.columns)} columns")

    # Split data into training, validation, and testing sets
    featTrainingSet, targetTrainingSet, featValidationSet, targetValidationSet, featTestingSet, targetTestingSet = split_data(t_set, f_set, 0.15, 0.1)

    print(f"""
    Data split completed.
    -------------------------
    Training set size: {len(featTrainingSet)} | {round(len(featTrainingSet)/len(df)*100, 2)}%
    Validation set size: {len(featValidationSet)} | {round(len(featValidationSet)/len(df)*100, 2)}%
    Testing set size: {len(featTestingSet)} | {round(len(featTestingSet)/len(df)*100, 2)}%
    ==========================""")
    return featTrainingSet, targetTrainingSet, featValidationSet, targetValidationSet, featTestingSet, targetTestingSet