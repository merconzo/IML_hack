import copy
import pandas as pd
import numpy as np
from typing import Optional
from joblib import dump
import re

# typehints
df = pd.DataFrame
col = pd.Series
op_col = Optional[col]

# constants
Y_COL = "original_selling_amount"
CANCEL_COL = "cancellation_policy_code"
PRED_COL = "predicted_selling_amount"
OUT_FILE_NAME = "../../predictions/agoda_cost_of_cancellation.csv"
NO_CANCEL = (-1)


class OurModel2:
    def __init__(self, model):
        self.means = None
        self.popular_list = None
        self.columns = None
        self.model = model


def make_dummies(X: df, column_name: str, ratio: int):
    value_counts = X[column_name].value_counts()
    to_replace = value_counts[value_counts < ratio].index
    X[column_name] = X[column_name].replace(to_replace, 'UNPOPULAR')
    popular_list = X[column_name].unique().tolist()
    X = pd.get_dummies(X, prefix=column_name, columns=[column_name], dtype=int)
    return X, popular_list


def get_cancel_days(row):
    cancel_codes = row[CANCEL_COL].split('_')
    if '100P' in cancel_codes:
        cancel_codes.remove('100P')
    days_book_to_checkin = row.days_book_to_checkin
    if not cancel_codes:
        return np.nan, np.nan, 1
    code_pattern = r'\d+D\d+[PN]'
    after_lst = []
    days = []
    possible_p = []
    for code in cancel_codes:
        if not re.match(code_pattern, code):
            continue
        parts = code.split('D')
        days_cancel_before_checkin, charge_num = int(parts[0]), parts[1]
        is_p = True if code[-1] == 'P' else False
        is_after_deadline = days_book_to_checkin <= days_cancel_before_checkin
        after_lst.append(is_after_deadline)
        days.append(days_cancel_before_checkin)
        possible_p.append(is_p)

    max_days = max(days) if days else np.nan
    min_days = min(days) if days else np.nan
    all_after = 1 if all(after_lst) else 0
    return max_days, min_days, all_after


# %%
def preprocess_data(X: df, y: op_col = None, popular_list=None, means=None):
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
    X.loc[:, codes] = X[codes].fillna("UNKNOWN")

    # means
    means_cols = ["hotel_star_rating", "no_of_adults", "no_of_children",
                  "no_of_extra_bed", "no_of_room"]
    if y is not None:
        means = X[means_cols].mean().to_dict()
        means["hotel_star_rating"] = round(means["hotel_star_rating"] * 2) / 2
        means["no_of_adults"] = int(means["no_of_adults"])
        means["no_of_children"] = int(means["no_of_children"])
        means["no_of_extra_bed"] = int(means["no_of_extra_bed"])
        means["no_of_room"] = int(means["no_of_room"])

    for key, val in means.items():
        X[key] = X[key].apply(lambda x: x if x >= 0 else val)

    # bool cols
    X.is_user_logged_in = X.is_user_logged_in.astype(int)
    X.is_first_booking = X.is_first_booking.astype(int)

    # rooms
    X["no_total_guests"] = X.no_of_adults + X.no_of_children
    X["guests_to_rooms_ratio"] = X["no_total_guests"] / X["no_of_room"]

    if y is not None:
        X = X[(1 <= X["guests_to_rooms_ratio"]) & (
                X["guests_to_rooms_ratio"] < 17)]

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
            pd.to_datetime(X.checkin_date) - pd.to_datetime(
        X.booking_datetime)).dt.days
    X.loc[X.days_book_to_checkin < 0, "days_book_to_checkin"] = 0
    X["staying_duration"] = (pd.to_datetime(X.checkout_date) - pd.to_datetime(
        X.checkin_date)).dt.days
    X["hotel_age_days"] = (pd.to_datetime(X.checkin_date) -
                           pd.to_datetime(X.hotel_live_date)).dt.days
    dates_cols = ["booking_datetime", "checkin_date", "checkout_date",
                  "hotel_live_date"]
    X.drop(dates_cols, axis=1, inplace=True)

    # costumer
    X["no_orders_history"] = X.h_customer_id.map(
        X.h_customer_id.value_counts())
    X[['max_days_to_cancel', 'min_days_to_cancel', 'is_after_deadline']
    ] = X.apply(get_cancel_days, axis=1, result_type='expand')
    X.drop(["cancellation_policy_code"], axis=1, inplace=True)
    X.loc[:, "max_days_to_cancel"] = X["max_days_to_cancel"].fillna(0)
    X.loc[:, "min_days_to_cancel"] = X["min_days_to_cancel"].fillna(0)

    # dummies
    dummis = ["accommadation_type_name", "hotel_brand_code",
              "hotel_chain_code", "hotel_city_code", "hotel_area_code",
              "hotel_id", "hotel_country_code", "h_customer_id",
              "customer_nationality",
              "guest_nationality_country_name",
              "origin_country_code", "language", "original_payment_method",
              "original_payment_type",
              "original_payment_currency"]
    if y is not None:
        popular_list["accommadation_type_name"] = X[
            "accommadation_type_name"].unique()
        X = pd.get_dummies(X, prefix='accommadation_type_name',
                           columns=['accommadation_type_name'], dtype=int)
        X, popular_list["hotel_brand_code"] = make_dummies(X,
                                                           'hotel_brand_code', 20)
        X, popular_list["hotel_chain_code"] = make_dummies(X,
                                                           'hotel_chain_code', 20)
        X, popular_list["hotel_city_code"] = make_dummies(X, 'hotel_city_code', 20)
        X, popular_list["hotel_area_code"] = make_dummies(X, 'hotel_area_code', 20)
        X, popular_list["hotel_id"] = make_dummies(X, 'hotel_id', 10)
        X, popular_list["hotel_country_code"] = make_dummies(X,
                                                             'hotel_country_code', 10)
        X, popular_list["h_customer_id"] = make_dummies(X, 'h_customer_id', 5)
        X, popular_list["customer_nationality"] = make_dummies(X,
                                                               'customer_nationality', 10)
        X, popular_list["guest_nationality_country_name"] = make_dummies(X,
                                                                         'guest_nationality_country_name',
                                                                         10)
        X, popular_list["origin_country_code"] = make_dummies(X,
                                                              'origin_country_code', 10)
        X, popular_list["language"] = make_dummies(X, 'language', 10)
        X, popular_list["original_payment_method"] = make_dummies(X,
                                                                  'original_payment_method',
                                                                  10)
        X, popular_list["original_payment_type"] = make_dummies(X,
                                                                'original_payment_type', 10)
        X, popular_list["original_payment_currency"] = make_dummies(X,
                                                                    'original_payment_currency', 10)
    else:
        for dummy in dummis:
            X[dummy] = X[dummy].where(X[dummy].isin(popular_list[dummy]),
                                      other=pd.NA)
            X = pd.get_dummies(X, prefix=dummy, columns=[dummy], dtype=int)

    # DONE!
    if y is not None:
        return X.drop(Y_COL, axis=1), X[Y_COL], means, popular_list
    else:
        return X, None, None, None

def execute_task_2(our_model, test):
    processed_test, _, _, _ = preprocess_data(copy.deepcopy(test), means=our_model.means,
                                              popular_list=our_model.popular_list)
    processed_test = processed_test.reindex(columns=our_model.columns, fill_value=0)
    predicted_value = our_model.model.predict(processed_test)
    test['original_selling_amount'] = predicted_value
    return test


def prepare_train_2(data, model):
    our_model = OurModel2(model)
    X_train, y_train = data.drop([Y_COL, "cancellation_datetime"], axis=1), data[Y_COL]
    X_train, y_train, our_model.means, our_model.popular_list = preprocess_data(X_train, y_train, dict(), dict())
    our_model.columns = X_train.columns
    our_model.model.fit(X_train, y_train)
    dump(our_model, "model_2.joblib")
