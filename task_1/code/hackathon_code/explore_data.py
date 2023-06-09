import pandas as pd
import numpy as np
import plotly.express as px

REQUESTS = ["request_nonesmoke", "request_latecheckin",
            "request_highfloor", "request_largebed", "request_twinbeds",
            "request_airport", "request_earlycheckin"]


def explore_days_between_booking_and_cancelation(df):
    canceled_indices = df[df["cancellation_datetime"].apply(lambda x: type(x) == str)]
    booking_until_cancelation = pd.to_datetime(
        canceled_indices["cancellation_datetime"]).dt.day_of_year - pd.to_datetime(
        canceled_indices["booking_datetime"]).dt.day_of_year
    booking_until_cancelation = booking_until_cancelation.apply(lambda x: x if x > 0 else 0)
    unique_vals = booking_until_cancelation.unique()
    unique_vals.sort()
    # counting the values of booking_until_cancelation
    counts = [np.sum(booking_until_cancelation == val) for val in unique_vals]
    # plotting the counts
    fig = px.bar(pd.DataFrame({"Days since booking": unique_vals, "Number of Cancelations": counts}),
                 x="Days since booking", y="Number of Cancelations", color="Number of Cancelations"
                 , title="Days Between Booking and Cancelation")
    fig.show()


def explore_days_between_cancelation_and_checkin(df):
    canceled_indices = df[df["cancellation_datetime"].apply(lambda x: type(x) == str)]
    days_till_checkin = pd.to_datetime(canceled_indices["checkin_date"]).dt.day_of_year - pd.to_datetime(
        canceled_indices["cancellation_datetime"]).dt.day_of_year
    days_till_checkin = days_till_checkin.apply(lambda x: x if x > 0 else 0)
    unique_vals = days_till_checkin.unique()
    unique_vals.sort()
    # counting the values of booking_until_cancelation
    counts = [np.sum(days_till_checkin == val) for val in unique_vals]
    # plotting the counts
    fig = px.bar(pd.DataFrame({"Days since cancelation till checkin": unique_vals, "Number of Cancelations": counts}),
                 x="Days since cancelation till checkin", y="Number of Cancelations", color="Number of Cancelations"
                 , title="Days Between Cancelation and Checkin")
    fig.show()


# def interpolate_distance_between_booking_and_checkin(df):
#     canceled_indices = df[df["cancellation_datetime"].apply(lambda x: type(x) == str)]
#     days_till_checkin = pd.to_datetime(canceled_indices["checkin_date"]).dt.day_of_year - pd.to_datetime(canceled_indices["cancellation_datetime"]).dt.day_of_year
#     days_till_checkin = days_till_checkin.apply(lambda x: x if x > 0 else 0)
#     booking_until_cancelation = pd.to_datetime(canceled_indices["cancellation_datetime"]).dt.day_of_year - pd.to_datetime(canceled_indices["booking_datetime"]).dt.day_of_year
#     booking_until_cancelation = booking_until_cancelation.apply(lambda x: x if x > 0 else 0)
#     #interpolating the distance between booking and checkin
#     distance_between_booking_and_checkin = booking_until_cancelation + days_till_checkin
#     unique_vals = distance_between_booking_and_checkin.unique()
#     unique_vals.sort()
#     #counting the values of booking_until_cancelation
#     counts = [np.sum(distance_between_booking_and_checkin == val) for val in unique_vals]
#     #plotting the counts
#     fig = px.bar(pd.DataFrame({"Distance between booking and checkin": unique_vals, "Number of Cancelations": counts}),
#                     x="Distance between booking and checkin", y="Number of Cancelations", color="Number of Cancelations"
#                 , title="Distance Between Booking and Checkin")
#     fig.show()

def explore_relation_of_booking_and_checkin_difference(df):
    day_between_booking_and_checkin = pd.to_datetime(df["checkin_date"]).dt.weekofyear - pd.to_datetime(
        df["booking_datetime"]).dt.weekofyear
    day_between_booking_and_checkin = day_between_booking_and_checkin.apply(lambda x: x if x >= 0 else x + 52)
    # getting the mean of cancelation per this data
    binary_y = df["cancellation_datetime"].apply(lambda x: 1 if type(x) == str else 0)
    # getting the cancelation rate per day between booking and checkin
    number_of_reservations = [np.sum(day_between_booking_and_checkin == val) for val in
                              day_between_booking_and_checkin.unique()]
    cancelation_rate = [np.mean(binary_y[day_between_booking_and_checkin == val]) for val in
                        day_between_booking_and_checkin.unique()]
    # plotting the cancelation rate

    fig = px.bar(pd.DataFrame({"Weeks between booking and checkin": day_between_booking_and_checkin.unique(),
                               "Number Of Reservations": number_of_reservations
                                  , "Cancelation Rate": cancelation_rate}),
                 x="Weeks between booking and checkin", y="Number Of Reservations", color="Cancelation Rate"
                 , title="Cancelation Rate per Weeks Between Booking and Checkin")
    # making y on log scale
    fig.update_layout(yaxis_type="log")
    fig.show()


def explore_parameter_relation(df, param, proportion=False, log_y=False):
    data = pd.DataFrame()
    if param == "original_selling_amount":
        data["Sell Price, log scale"] = df[param].apply(lambda x: np.log(x) if type(x) != str and x > 0 else 0)
        param = "Sell Price, log scale"
    elif param == "cancellation_policy_code":
        data[param] = df[param].apply(lambda x: x.split("D")[0]).apply(
            lambda x: "Can't cancel without a fine" if x == "365" else "Some time allowed for cancellation")
    elif param == "hour of booking":
        data[param] = pd.to_datetime(df["booking_datetime"]).dt.hour
    elif param == "duration of stay":
        data[param] = (pd.to_datetime(df["checkout_date"]).dt.day_of_year - pd.to_datetime(
            df["checkin_date"]).dt.day_of_year).apply(lambda x: min(x, 10))
    elif param == "year hotel went online":
        data[param] = pd.to_datetime(df["hotel_live_date"]).dt.year
        data.sort_values(by=param, inplace=True)
        data[param] = data[param].astype(str)
    elif param == "number of requests":
        data[param] = df[REQUESTS].sum(axis=1)
    else:
        data[param] = df[param]
    data["cancelled"] = df["cancellation_datetime"].apply(lambda x: 1 if type(x) == str else 0)
    if not proportion:
        fig = px.histogram(data, x=param, color="cancelled", title="Relation of " + param + " to Cancelation Rate")
    else:
        agg_df = pd.DataFrame()
        agg_df[param] = data[param].unique()
        agg_df["Cancellation Rate"] = [np.mean(data[data[param] == val]["cancelled"]) for val in agg_df[param]]
        agg_df["number of orders"] = [np.sum(data[param] == val) for val in agg_df[param]]
        appendix = ""
        if log_y:
            appendix = ", log scale"
            agg_df["number of orders" + appendix] = agg_df["number of orders"].apply(lambda x: np.log(x))
        fig = px.bar(agg_df, x=param, y="number of orders" + appendix,
                     title="Relation of " + param + " to Cancelation Rate", color="Cancellation Rate")

    fig.show()


def explore(data):
    explore_days_between_booking_and_cancelation(data)
    explore_days_between_cancelation_and_checkin(data)
    explore_relation_of_booking_and_checkin_difference(data)
    explore_parameter_relation(data, "cancellation_policy_code")
    # data_exploring.explore_parameter_relation(train, "original_selling_amount")
    # data_exploring.explore_parameter_relation(train, "hour of booking")
    # data_exploring.explore_parameter_relation(train, "is_first_booking")
    # data_exploring.explore_parameter_relation(train, "is_user_logged_in")
    # data_exploring.explore_parameter_relation(train, "guest_is_not_the_customer")
    # data_exploring.explore_parameter_relation(train, "original_payment_type")
    # data_exploring.explore_parameter_relation(train, "hotel_star_rating")
    # data_exploring.explore_parameter_relation(train, "no_of_room")
    # data_exploring.explore_parameter_relation(train, "no_of_adults")
    # data_exploring.explore_parameter_relation(train, "duration of stay", proportion=True, log_y=True)
    # data_exploring.explore_parameter_relation(train, "year hotel went online", proportion=True)
    # data_exploring.explore_parameter_relation(train, "number of requests", proportion=True)
