import pandas as pd

from pathlib import Path
from tensorflow import keras
from sklearn.metrics import accuracy_score

predictions_path = Path('predictions')
predictions_path.mkdir(exist_ok=True)

root_path = Path('ReproducibilityPackage/Table2')

dataset_label = {'PD': 'Outcome', 'HP': 'AboveMedianPrice'}
name_matches = {'BM': 'bankcustomer', 'GC': 'german'}


for model_path in (root_path / 'Models').iterdir():
    if model_path.suffix != '.h5':
        continue

    model_prefix = model_path.stem[:2]
    model_name = model_path.stem
    model_predictions_path = predictions_path / model_prefix
    model_predictions_path.mkdir(exist_ok=True)

    # paths
    d_test = root_path / f'Data/{model_prefix}_unseen.csv'
    print(f"Predicting for {model_prefix} - Model")

    if model_prefix in dataset_label:
        X_predict = pd.read_csv(d_test, sep=',')
        label = dataset_label[model_prefix]
        ActualOutcome = X_predict[label].copy()
        X_predict.drop(columns=[label], inplace=True)
    else:
        path_actual_outcome = root_path / f'Other/{name_matches[model_prefix]}_Actual_df.csv'
        X_predict = pd.read_csv(d_test, sep=',')
        ActualOutcome = pd.read_csv(path_actual_outcome, sep=',')

    # predict
    model = keras.models.load_model(model_path)
    model_predictions = (model.predict(X_predict.to_numpy()) > 0.5).astype(int)
    pd.DataFrame(model_predictions, columns=['y']).to_csv(model_predictions_path / f'{model_name}.csv', index=False)

    # accuracy
    acc_gt = accuracy_score(ActualOutcome, model_predictions)

    print(f"Acc. for {model_name} - Model: {round(acc_gt*100, 2)}")
