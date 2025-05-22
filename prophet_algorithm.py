'''Script to generate time series prediction using Facebook's Prophet Package'''
from datetime import datetime
import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd

def data_preprocess(start_time, raw_data, data_frequency):
    '''Takes starting time of the series in yyyy-mm-dd format, the raw
        datafile and data frequency - daily, weekly or monthly
    '''
    datetime_object = datetime.strptime(start_time, '%Y-%m-%d')
    frequency_encoding_dict = {'daily':'D', 'weekly':'W', 'monthly':'M'}
    date_lst = (pd.date_range(start=datetime_object, periods=len(raw_data), \
        freq=frequency_encoding_dict[data_frequency][0]))
    df_ts = pd.DataFrame({'ds':date_lst.values, 'y':raw_data['y']})
    return df_ts

def prophet_ts(raw_df, date_string, model_type, data_type, model_specs_lst):
    '''accepts the raw data file, date string for the first time stamp in yyyy-mm-dd format.
    model_type - linear or logistic, data_type - daily, weekly, or monthly, and confidence interval
    which is a value between 0 and 1
    '''
    pred_periods = model_specs_lst[0]
    confidence_int = model_specs_lst[1]
    model_dict = {'daily': Prophet(growth=model_type, interval_width=confidence_int, \
    daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True), \
    'weekly':Prophet(growth=model_type, interval_width=confidence_int, weekly_seasonality=True), \
    'monthly':Prophet(growth=model_type, interval_width=confidence_int)}
    frequency_encoding_dict = {'daily':'D', 'weekly':'W', 'monthly':'M'}
    df_ts = data_preprocess(date_string, raw_df, data_type)
    try:
        math_model = model_dict[data_type]
        if model_type == 'logistic':
            floor_value = input("What is the minimum value for the metric being predicted? ")
            ceiling_value = input("What is the maximum value for the metric being predicted? ")
            df_ts['floor'] = int(floor_value)
            df_ts['cap'] = int(ceiling_value)
        math_model.fit(df_ts)
        future = math_model.make_future_dataframe(periods=pred_periods, \
        freq=frequency_encoding_dict[data_type])
        future['floor'] = int(floor_value) if model_type == "logistic" else None
        future['cap'] = int(ceiling_value) if model_type == "logistic" else None
        forecast = math_model.predict(future)
        prediction_df = forecast.iloc[-pred_periods:, 0:]\
            [['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        prediction_df.columns = ['Date', 'Expected Value', \
            'Lower Confidence Interval', 'Upper Confidence Interval']
        plt.figure(figsize=(15, 10), dpi=80)
        plt.plot(prediction_df['Date'], prediction_df['Expected Value'])
        plt.plot(prediction_df['Date'], prediction_df['Lower Confidence Interval'])
        plt.plot(prediction_df['Date'], prediction_df['Upper Confidence Interval'])
        plt.title("Prophet Algorithm Results")
        plt.xlabel("Date")
        plt.ylabel("Predictions And Confidence Intervals")
        plt.savefig("Prophet_Results.png")
        plt.close()
        prediction_df.to_csv("Prophet_TS.csv")
        return prediction_df
    except ValueError:
        print("The floor or ceiling values have to be a number or integer. \
It's possible that you entered a non-integer ")
        return None
#print(prophet_ts("AirPassengers.csv", "1949-01-01", "linear", "weekly", 0.95 ))
