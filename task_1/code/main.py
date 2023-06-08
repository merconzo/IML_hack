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


def make_dummies(X: df, column_name: str, ratio: int):
    value_counts = X[column_name].value_counts()
    to_replace = value_counts[value_counts < ratio].index
    X[column_name] = X[column_name].replace(to_replace, 'UNPOPULAR')
    popular_list = X[column_name].unique().tolist()
    X = pd.get_dummies(X, prefix=column_name, columns=[column_name], dtype=int)
    return X, popular_list

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
    Post-processed design matrix and response vector (prices) - either as a
    single DataFrame or a Tuple[DataFrame, Series]
    """
    if y is not None:  # train
        y.rename(Y_COL, inplace=True)
        X = pd.concat([X, y], axis=1)
    try:
        X.drop(["h_booking_id"], axis=1, inplace=True)
    except:
        pass

    # nans replacements
    X.replace("UNKNOWN", np.nan, inplace=True)
    X["charge_option"] = X["charge_option"].apply(
        lambda x: np.where(x == 'Pay Now', 1, 0))

    requests = ["request_earlycheckin", "request_airport", "request_twinbeds",
                "request_largebed",
                "request_highfloor", "request_latecheckin",
                "request_nonesmoke"]

    for request in requests:
        X["is_available_to_" + request] = X[request].notnull().astype(int)
    X.loc[:, requests] = X[requests].fillna(0)
    codes = ["hotel_brand_code", "hotel_chain_code", "hotel_country_code",
             "origin_country_code",
             "original_payment_method", "customer_nationality",
             "cancellation_policy_code", "accommadation_type_name",
             "guest_nationality_country_name"]
    # for code in codes:
    #     X["has_" + code] = X[code].notnull().astype(int)
    X.loc[:, codes] = X[codes].fillna("UNKNOWN")

    # dummies
    X = pd.get_dummies(X, prefix='accommadation_type_name', columns=['accommadation_type_name'], dtype=int)
    X, popular_brand_codes = make_dummies(X, 'hotel_brand_code', 100)
    X, popular_hotel_id = make_dummies(X, 'hotel_id', 30)
    X, popular_hotel_country_code = make_dummies(X, 'hotel_country_code', 30)
    # print(len(popular_hotel_country_code))
    X, popular_h_customer_id = make_dummies(X, 'h_customer_id', 20)
    # print(len(popular_h_customer_id))
    X, popular_customer_nationality = make_dummies(X, 'customer_nationality', 30)
    # print(len(popular_customer_nationality))
    X, popular_guest_nationality_country_name = make_dummies(X, 'guest_nationality_country_name', 30)
    X, popular_origin_country_code = make_dummies(X, 'origin_country_code', 30)
    X, popular_language = make_dummies(X, 'language', 30)
    X, popular_original_payment_method = make_dummies(X, 'original_payment_method', 30)
    X, popular_original_payment_type = make_dummies(X, 'original_payment_type', 30)
    X, popular_original_payment_currency = make_dummies(X, 'original_payment_currency', 30)



    # means TODO: smarter means?
    means_cols = ["hotel_star_rating", "no_of_adults", "no_of_children", "no_of_extra_bed", "no_of_room", "original_selling_amount"]
    mean_dict = X[means_cols].mean().to_dict()
    mean_dict["hotel_star_rating"] = round(mean_dict["hotel_star_rating"] * 2) / 2
    mean_dict["no_of_adults"] = int(mean_dict["no_of_adults"])
    mean_dict["no_of_children"] = int(mean_dict["no_of_children"])
    mean_dict["no_of_extra_bed"] = int(mean_dict["no_of_extra_bed"])
    mean_dict["no_of_room"] = int(mean_dict["no_of_room"])
    for key, val in mean_dict.items():
        X[key] = X[key].apply(lambda x: x if x >= 0 else val)

    # bool cols
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
    X[["checkin_date", "checkout_date"]] = X[
        ["checkin_date", "checkout_date"]].apply(pd.to_datetime)
    X["checkin_dayofyear"] = pd.to_datetime(X["checkin_date"]).dt.dayofyear
    X["checkout_dayofyear"] = pd.to_datetime(X["checkout_date"]).dt.dayofyear
    X["days_book_to_checkin"] = (
            pd.to_datetime(X_train.checkin_date) - pd.to_datetime(
        X_train.booking_datetime)).dt.days
    X.loc[X.days_book_to_checkin < 0, "days_book_to_checkin"] = 0
    X["satying_duration"] = (pd.to_datetime(X_train.checkout_date) -
                             pd.to_datetime(X_train.checkin_date)).dt.days
    X["hotel_age_days"] = (
            pd.to_datetime(X_train.checkin_date) - pd.to_datetime(
        X_train.hotel_live_date)).dt.days

    # prices
    X["total_price_per_night"] = X.original_selling_amount / X.satying_duration
    X["room_price_per_night"] = X.total_price_per_night / X.no_of_room
    X["total_price_for_adult"] = X.original_selling_amount / X.no_of_adults
    X["total_price_for_adult_per_night"] = X.total_price_for_adult / \
                                           X.satying_duration
    dates_cols = ["booking_datetime", "checkin_date", "checkout_date",
                  "hotel_live_date"]
    X.drop(dates_cols, axis=1, inplace=True)

    # costumer 
    X["no_orders_history"] = X_train.h_customer_id.map(
        X_train.h_customer_id.value_counts())

    # dummies_cols  TODO: delete after treatment
    dummies_cols = ["hotel_id", "hotel_country_code",
                    "accommadation_type_name", "charge_option",
                    "h_customer_id", "customer_nationality",
                    "guest_nationality_country_name", "origin_country_code",
                    "language", "original_payment_method",
                    "original_payment_type", "original_payment_currency",
                    "cancellation_policy_code", "hotel_area_code",
                    "hotel_brand_code", "hotel_chain_code",
                    "hotel_city_code"]
    X.drop(dummies_cols, axis=1, inplace=True)

    # DONE!
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
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.expand_frame_repr', False)  # Disable line breaks

    # Print the DataFrame
    with pd.option_context('display.max_colwidth', None):
        # bla = _train["hotel_id"].value_counts()
        # print(len(bla[bla > 30]))
        # print(len(set(X_train["guest_nationality_country_name"])))
        # print(X_train[:100])
        all = pd.concat([X_train, y_train], axis=1)
        null_rows = all[X_train.isnull().any(axis=1)]
        print(all[:1])
        print(all.size)
# %%
