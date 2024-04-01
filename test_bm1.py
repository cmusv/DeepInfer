import pandas as pd
from tensorflow import keras
from sklearn.metrics import accuracy_score

root_path_table_2 = 'ReproducibilityPackage/Table2'
root_path_table_3 = 'ReproducibilityPackage/Table3'

path_actual_outcome_table_2 = root_path_table_2 + '/' + 'Other/bankcustomer_Actual_df.csv'
path_actual_outcome_table_3 = root_path_table_3 + '/' + 'Other/bankcustomer_Actual_df.csv'

path_prediction_table_2 = root_path_table_2 + '/' + 'Other/bankcustomer_predictionsBM1.csv'
path_prediction_table_3 = root_path_table_3 + '/' + 'Other/bankcustomer_predictionsBM1.csv'

d_test_table_2 = root_path_table_2 + '/' + 'Data/BM_unseen.csv'
d_test_table_3 = root_path_table_3 + '/' + 'Data/BM_unseen.csv'

X_predict_table_2 = pd.read_csv(d_test_table_2, sep=',')
X_predict_table_3 = pd.read_csv(d_test_table_3, sep=',')
ActualOutcome_table_2 = pd.read_csv(path_actual_outcome_table_2, sep=',')
ActualOutcome_table_3 = pd.read_csv(path_actual_outcome_table_3, sep=',')

model_path_table_2 = root_path_table_2 + '/' + 'Models/BM1.h5'
model_path_table_3 = root_path_table_3 + '/' + 'Models/BM1.h5'
fairify_model_path = 'BM-1.h5'

model_table_2 = keras.models.load_model(model_path_table_2)
print("Model Table 2")
model_table_2.summary()
model_table_3 = keras.models.load_model(model_path_table_3)
print("Model Table 3")
model_table_3.summary()
fairify_model = keras.models.load_model(fairify_model_path)
print("Fairify Model")
fairify_model.summary()

prediction_table_2 = pd.read_csv(path_prediction_table_2, sep=',')
prediction_table_3 = pd.read_csv(path_prediction_table_3, sep=',')

predictionvalue_table_2 = (model_table_2.predict(X_predict_table_2.to_numpy()) > 0.5).astype(int)
model_2_prediction_value_table_3 = (model_table_2.predict(X_predict_table_3.to_numpy()) > 0.5).astype(int)

predictionvalue_table_3 = (model_table_3.predict(X_predict_table_3.to_numpy()) > 0.5).astype(int)
model_3_prediction_value_table_2 = (model_table_3.predict(X_predict_table_2.to_numpy()) > 0.5).astype(int)

fairify_predictions = (fairify_model.predict(X_predict_table_2.to_numpy()) > 0.5).astype(int)
# BM1
# - Source: Kaggle
# - Layers: 4
# - Neurons: 97 89.20

# BM dataset - 16 attributes

# compute accuracy with actual outcome

acc_gt_table_2 = accuracy_score(ActualOutcome_table_2, predictionvalue_table_2)
acc_gt_table_3 = accuracy_score(ActualOutcome_table_3, predictionvalue_table_3)
acc_gt_table_2_model_3 = accuracy_score(ActualOutcome_table_2, model_3_prediction_value_table_2)
acc_gt_table_3_model_2 = accuracy_score(ActualOutcome_table_3, model_2_prediction_value_table_3)
acc_gt_table_2_fairify = accuracy_score(ActualOutcome_table_2, fairify_predictions)

print(f"Accuracy (Actual Outcome Saved in Table 2) - (Predictions by Model in Table 2): {acc_gt_table_2}")
print(f"Accuracy (Actual Outcome Saved in Table 3) - (Predictions by Model in Table 2): {acc_gt_table_3_model_2}")
print(f"Accuracy (Actual Outcome Saved in Table 3) - (Predictions by Model in Table 3): {acc_gt_table_3}")
print(f"Accuracy (Actual Outcome Saved in Table 2) - (Predictions by Model in Table 3): {acc_gt_table_2_model_3}")
print(f"Accuracy (Actual Outcome Saved in Table 2) - (Predictions by Fairify Model): {acc_gt_table_2_fairify}")
print("\n")
# Actual Outcome does not differ between tables 2 and 3
# The models in table 2 and 3 differ, as the difference in accuracy is significant

acc_pred_table_2 = accuracy_score(ActualOutcome_table_2, prediction_table_2)
acc_pred_table_3 = accuracy_score(ActualOutcome_table_3, prediction_table_3)

print(f"Accuracy (Actual Outcome Saved in Table 2) - (Saved Predictions in Table 2): {acc_pred_table_2}")
print(f"Accuracy (Actual Outcome Saved in Table 3) - (Saved Predictions in Table 3): {acc_pred_table_3}")
print("\n")
# The predictions stored in table 2 are the actual predictions made by the model in table 2
# The predictions stored in table 3 differ from the ones in table 2 and are much worse than the expected predictions

acc_model_table_2_pred_table_2 = accuracy_score(predictionvalue_table_2, prediction_table_2)
acc_model_table_3_pred_table_3 = accuracy_score(predictionvalue_table_3, prediction_table_3)
acc_fairify_pred_table_2 = accuracy_score(fairify_predictions, prediction_table_2)

print(f"Accuracy (Predictions by Model Table in 2) - (Saved Predictions in Table 2): {acc_model_table_2_pred_table_2}")
print(f"Accuracy (Predictions by Model Table in 3) - (Saved Predictions in Table 3): {acc_model_table_3_pred_table_3}")
print(f"Accuracy (Predictions by Fairify Model) - (Saved Predictions in Table 2): {acc_fairify_pred_table_2}")
print("\n")
# The predictions stored in table 2 were generated using the model in table 2
# The predictions stored in table 3 were not generated using the model in table 3

acc_model_table_3_pred_table_2 = accuracy_score(predictionvalue_table_3, prediction_table_2)
acc_model_table_2_pred_table_3 = accuracy_score(predictionvalue_table_2, prediction_table_3)

print(f"Accuracy (Predictions by Model in Table 3) - (Saved Predictions in Table 2): {acc_model_table_3_pred_table_2}")
print(f"Accuracy (Predictions by Model in Table 2) - (Saved Predictions in Table 3): {acc_model_table_2_pred_table_3}")
print("\n")

# Saved predictions in table 3 are more similar with the predictions of the model in table 2 than the one in table 3

# Model in table 3 yields much better results than model in table 2.
# However, the saved predictions in table 3 are more similar to the predictions of the model in table 2.

# Why there are different models for the tables?
# What model has been used to generate the predictions in table 3?
# What model do you use to get the Implications?
