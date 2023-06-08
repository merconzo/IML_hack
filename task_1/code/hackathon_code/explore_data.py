import pandas as pd
import numpy as np
import plotly.express as px


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
    cancelation_rate = [np.mean(binary_y[day_between_booking_and_checkin == val]) for val in
                        day_between_booking_and_checkin.unique()]
    # plotting the cancelation rate
    fig = px.bar(pd.DataFrame({"Weeks between booking and checkin": day_between_booking_and_checkin.unique(),
                               "Cancelation Rate": cancelation_rate}),
                 x="Weeks between booking and checkin", y="Cancelation Rate", color="Cancelation Rate"
                 , title="Cancelation Rate per Weeks Between Booking and Checkin")
    fig.show()
