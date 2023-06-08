# %%
# import sys
# sys.path.append("./")
import pandas as pd
import numpy as np
import sklearn as sk
from typing import Optional
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from hackathon_code.explore_data import random_forest_exploring

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# typehints
df = pd.DataFrame
col = pd.Series
op_col = Optional[col]

# constants
Y_COL = "cancellation_datetime"


# %%
def preprocess_data(X: df, y: op_col = None):
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
    Post-processed design matrix and response vector (prices) - either as a single DataFrame or a Tuple[DataFrame, Series]
    """
    if y is not None:  # train
        y.rename(Y_COL, inplace=True)
        X = pd.concat([X, y], axis=1)
    try:
        X.drop(["h_booking_id"], axis=1, inplace=True)
    except:
        pass
    X.is_user_logged_in = X.is_user_logged_in.astype(int)
    X.is_first_booking = X.is_first_booking.astype(int)

    # rooms
    X["no_total_guests"] = X.no_of_adults + X.no_of_children
    X["guests_to_rooms_ratio"] = X["no_total_guests"] / X["no_of_room"]
    X = X[
        (1 <= X["guests_to_rooms_ratio"]) & (X["guests_to_rooms_ratio"] < 17)]
    
    # booking times
    X["booking_year"] = pd.to_datetime(X["booking_datetime"]).dt.year
    X["booking_month"] = pd.to_datetime(X["booking_datetime"]).dt.month
    X["booking_day_of_week"] = pd.to_datetime(
        X["booking_datetime"]).dt.day_of_week
    X["booking_day_of_year"] = pd.to_datetime(
        X["booking_datetime"]).dt.dayofyear
    X["booking_hour"] = pd.to_datetime(
        X["booking_datetime"]).dt.hour
    X[["checkin_date", "checkout_date"]] = X[["checkin_date", "checkout_date"]].apply(pd.to_datetime)
    X["checkin_dayofyear"] = pd.to_datetime(X["checkin_date"]).dt.dayofyear
    X["checkout_dayofyear"] = pd.to_datetime(X["checkout_date"]).dt.dayofyear
    X["days_book_to_checkin"] = (pd.to_datetime(X_train.checkin_date) - pd.to_datetime(X_train.booking_datetime)).dt.days

    return X.drop(Y_COL, axis=1), X[Y_COL] if y is not None else X


# %%
if __name__ == "__main__":
    np.random.seed(0)
    data = pd.read_csv("./hackathon_code/data/agoda_cancellation_train.csv")
    cols = data.columns.values
    train, test = sk.model_selection.train_test_split(data, test_size=0.2)
    X_train, y_train = train.drop([Y_COL], axis=1), train[
        Y_COL]
    X_test, y_test = test.drop([Y_COL], axis=1), test[
        Y_COL]

    X_train, y_train = preprocess_data(X_train, y_train)

    # %% plots
    # px.scatter(X_train, x="no_of_adults", y="no_of_room").show()

    # go.Figure([
    #     go.Scatter(x=X_train.no_of_adults,
    #                y=X_train.no_of_room,
    #                mode='markers',
    #                marker=dict(color='purple'),
    #                name=r'$Loss$')]).update_layout(
    #     title=f"adults over rooms",
    #     xaxis=dict(title=f"no_of_adults", showgrid=True),
    #     yaxis=dict(title=f"no_of_room", showgrid=True)
    #     ).show()

# %%
