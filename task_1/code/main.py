# %%
# import sys
# sys.path.append("./")
import pandas as pd
import numpy as np

# %%
import sklearn as sk
from joblib import load
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from task_1 import execute_task_1, prepare_train_1
from task_2 import execute_task_2, prepare_train_2

if __name__ == "__main__":
    np.random.seed(0)
    data = pd.read_csv("./hackathon_code/data/agoda_cancellation_train.csv")
    # model_1 = make_pipeline(sklearn.preprocessing.StandardScaler(),
    #                         sklearn.ensemble.AdaBoostClassifier(sk.tree.DecisionTreeClassifier(max_depth=1),
    #                                                             n_estimators=50))
    # model_2 = make_pipeline(sklearn.preprocessing.StandardScaler(),
    #                         sklearn.ensemble.HistGradientBoostingRegressor())
    # prepare_train_1(data, model_1)
    # prepare_train_2(data, model_2)
    model_1 = load("model_1.joblib")
    model_2 = load("model_2.joblib")

    test_1 = pd.read_csv("./hackathon_code/data/Agoda_Test_1.csv")
    booking_id_1 = test_1["h_booking_id"]
    prediction_1 = execute_task_1(model_1, test_1)
    result_1 = pd.DataFrame({'ID': booking_id_1, 'cancellation': prediction_1})
    result_1.to_csv("agoda_cancellation_prediction.csv", index=False)
    test_2 = pd.read_csv("./hackathon_code/data/Agoda_Test_2.csv")
    booking_id_2 = test_2["h_booking_id"]
    test_2 = execute_task_2(model_2, test_2)
    prediction_cancel_2 = execute_task_1(model_1, test_2)
    prediction_2 = test_2["original_selling_amount"].where(prediction_cancel_2 == 1, other=-1)
    result_2 = pd.DataFrame({'ID': booking_id_2, 'predicted_selling_amount': prediction_2})
    result_2.to_csv("agoda_cost_of_cancellation.csv", index=False)

# %%
