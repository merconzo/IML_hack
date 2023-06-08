# %%
# import sys
# sys.path.append("./")
import pandas as pd
import numpy as np
import sklearn as sk
from typing import Optional
import re

# %%
from joblib import load

from task_1 import execute_task_1

if __name__ == "__main__":
    np.random.seed(0)
    data = pd.read_csv("./hackathon_code/data/agoda_cancellation_train.csv")
    model_1 = load("model_1.joblib")
    # model_2 = load("model_2.joblib")
    # TODO: model two

    test_1 = pd.read_csv("./hackathon_code/data/Agoda_Test_1.csv")
    test_2 = pd.read_csv("./hackathon_code/data/Agoda_Test_2.csv")
    booking_id = test_1["h_booking_id"]
    prediction_1 = execute_task_1(data, test_1)
    # prediction_2 = execute_task_2(data, test_2)
    result_1 = pd.DataFrame({'ID': booking_id, 'cancellation': prediction_1})
    # result_2 = pd.DataFrame({'ID': booking_id, 'predicted_selling_amount': prediction_2})
    result_1.to_csv("agoda_cancellation_prediction.csv", index=False)
    # result_2.to_csv("agoda_cost_of_cancellation.csv", index=False)
    print(result_1)
# %%
