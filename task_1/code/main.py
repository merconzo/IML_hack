# %%
import sys 
sys.path.append("./")
import pandas as pd
import numpy as np
import sklearn as sk
from typing import Optional
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# %%
def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    X = X.drop(["h_booking_id"])
    X["booking_day_of_year"] = pd.to_datetime(X["booking_datetime"]).dt.dayofyear


# %%
if __name__ == "__main__":
    data = pd.read_csv("./hackathon_code/data/agoda_cancellation_train.csv")
# %%
