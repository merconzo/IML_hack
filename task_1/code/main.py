# %%
import sys

sys.path.append("./")
import pandas as pd
import numpy as np
import sklearn as sk
from typing import Optional
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from hackathon_code.explore_data import random_forest_exploring


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
    Post-processed design matrix and response vector (prices) - either as a
    single
    DataFrame or a Tuple[DataFrame, Series]
    """
    try:
        X = X.drop(["h_booking_id"])
    except:
        pass
    X.replace("UNKNOWN", None, inplace=True)
    X.is_user_logged_in = X.is_user_logged_in.astype(int)
    X.is_first_booking = X.is_first_booking.astype(int)

    # dates
    X["booking_day_of_year"] = pd.to_datetime(
        X["booking_datetime"]).dt.dayofyear

    return X, y


# %%
if __name__ == "__main__":
    np.random.seed(0)
    data = pd.read_csv("./hackathon_code/data/agoda_cancellation_train.csv")
    cols = data.columns.values
    train, test = sk.model_selection.train_test_split(data, test_size=0.2)
    X_train, y_train = train.drop(["cancellation_datetime"], axis=1), train[
        "cancellation_datetime"]
    X_test, y_test = test.drop(["cancellation_datetime"], axis=1), test[
        "cancellation_datetime"]
    # %%
    X_train, y_train = preprocess_data(X_train, y_train)

    # %%
    random_forest_exploring(X_train, y_train)
# %%
