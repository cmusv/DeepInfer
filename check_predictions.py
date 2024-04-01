import pandas as pd

from pathlib import Path
from tensorflow import keras
from sklearn.metrics import accuracy_score

name_matches = {'BM': 'bankcustomer', 'GC': 'german'}

root_path_table_2 = 'ReproducibilityPackage/Table2'
root_path_table_3 = 'ReproducibilityPackage/Table3'

root_path = Path(root_path_table_3)

if str(root_path) == root_path_table_3:
    name_matches.update({'HP': '', 'PD': ''})

# Why all BM saved predictions yield random chance accuracy?
for model_path in (root_path / 'Models').iterdir():
    model_prefix = model_path.stem[:2]
    model_name = model_path.stem

    if model_prefix not in name_matches:
        continue

    # paths
    d_test = root_path / f'Data/{model_prefix}_unseen.csv'

    if name_matches[model_prefix] is '':
        path_actual_outcome = root_path / f'Other/{model_name}_actual.csv'
        path_prediction = root_path / f'Other/{model_name}_prediction.csv'
    else:
        path_actual_outcome = root_path / f'Other/{name_matches[model_prefix]}_Actual_df.csv'
        path_prediction = root_path / f'Other/{name_matches[model_prefix]}_predictions{model_name}.csv'

    # data
    X_predict = pd.read_csv(d_test, sep=',')
    ActualOutcome = pd.read_csv(path_actual_outcome, sep=',')

    if model_prefix == 'HP':
        del X_predict['AboveMedianPrice']
    elif model_prefix == 'PD':
        del X_predict['Outcome']

    saved_predictions = pd.read_csv(path_prediction, sep=',')

    # predict
    model = keras.models.load_model(model_path)
    model_predictions = (model.predict(X_predict.to_numpy()) > 0.5).astype(int)

    # accuracy
    acc_gt = accuracy_score(ActualOutcome, model_predictions)
    acc_pred = accuracy_score(ActualOutcome, saved_predictions)

    print(f"Acc. for {model_name} - Model vs Saved predictions: {round(acc_gt*100, 2)} - {round(acc_pred*100, 2)}")
