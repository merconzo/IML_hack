# %%
import pandas as pd
import numpy as np
from joblib import load
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from task_1.code.hackathon_code.task_1 import execute_task_1, prepare_train_1
from task_1.code.hackathon_code.task_2 import execute_task_2, prepare_train_2
from task_1.code.hackathon_code.explore_data import explore

if __name__ == "__main__":
    np.random.seed(0)
    ###  part one  ###
    data = pd.read_csv("./hackathon_code/data/agoda_cancellation_train.csv")
    # # the following code is for training the data:
    # # (uncomment for not training)
    # model_1 = make_pipeline(sk.preprocessing.StandardScaler(),
    #                         sk.ensemble.AdaBoostClassifier(
    #                             sk.tree.DecisionTreeClassifier(max_depth=2),
    #                                                             n_estimators=50))
    # model_2 = make_pipeline(sk.preprocessing.StandardScaler(),
    #                         sk.ensemble.HistGradientBoostingRegressor())
    # prepare_train_1(data, model_1)
    # prepare_train_2(data, model_2)
    # # done training!

    # load trained model
    model_1 = load("hackathon_code/model_1.joblib")
    model_2 = load("hackathon_code/model_2.joblib")

    ###  part two  ###
    # task 1
    try:
        test_1 = pd.read_csv("./hackathon_code/data/Agoda_Test_1.csv")
        booking_id_1 = test_1["h_booking_id"]
        prediction_1 = execute_task_1(model_1, test_1)
        result_1 = pd.DataFrame({'ID': booking_id_1, 'cancellation': prediction_1})
        result_1.to_csv("../predictions/agoda_cancellation_prediction.csv",
                        index=False)
    except:
        pass

    # task 2
    try:
        test_2 = pd.read_csv("./hackathon_code/data/Agoda_Test_2.csv")
        booking_id_2 = test_2["h_booking_id"]
        test_2 = execute_task_2(model_2, test_2)
        prediction_cancel_2 = execute_task_1(model_1, test_2)
        prediction_2 = test_2["original_selling_amount"].where(prediction_cancel_2 == 1, other=-1)
        result_2 = pd.DataFrame({'ID': booking_id_2, 'predicted_selling_amount': prediction_2})
        result_2.to_csv("../predictions/agoda_cost_of_cancellation.csv",
                        index=False)
    except:
        pass

    ###  part three  ###
    # explore some data :) you can uncomment & call the following function:
    # explore(data)

# %%
