'''A script to generate weekly and monthly predictions using ARIMA/SARIMA models'''
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt

def arima_sarima_model(df_formatted, prediction_periods, seasonality, data_type, confidence_int):
    '''Takes a formatted dataframe - df_formatted - using the data_preprocess package,
    number of prediction periods, whether data has seasonality or not and data_type - daily,
    weekly, monthly and type of confidence interval which is a value between 0 and 1'''
    seasonality_cycle = {'weekly': 52, 'monthly': 12}
    frequency_encoding_dict = {'daily':'D', 'weekly':'W', 'monthly':'M'}
    if seasonality:
        model_choice = pm.auto_arima(df_formatted['y'], start_p=1, start_q=1, test='adf', \
                                    max_p=3, max_q=3, m=seasonality_cycle[data_type], start_P=0,\
                                        seasonal=True, d=None, D=1, trace=False, \
                                        error_action='ignore', suppress_warnings=True, \
stepwise=True)
    else:
        model_choice = pm.auto_arima(df_formatted['y'], start_p=1, start_q=1, test='adf', \
                            max_p=3, max_q=3, d=None)

    model_choice.plot_diagnostics(figsize=(15, 12))
    plt.savefig("ARIMA-SARIMA-Model-Diagnostics.png")
    plt.close()
    n_period = prediction_periods
    fitted_values, confidence_interval = model_choice.predict(n_periods=n_period,\
                                        alpha=1-confidence_int, return_conf_int=True)
    date_lst = (pd.date_range(start=df_formatted['ds'].values[-1], periods=n_period, \
        freq=frequency_encoding_dict[data_type]))
    prediction_dataframe = pd.DataFrame({'Date': date_lst,\
'Expected Value':fitted_values,\
'Lower Confidence Interval': [round(values[0], 0) for values in confidence_interval],\
'Upper Confidence Interval': [round(values[1], 0) for values in confidence_interval]})
    prediction_dataframe.to_csv("ARIMA-SARIMA_Model.csv")
    print("The predictions have been saved to a file name ARIMA/SARIMA_Model.csv. \
The plots and diagnostics have also been saved")
    plt.figure(figsize=(15, 10), dpi=80)
    plt.plot(prediction_dataframe['Date'], prediction_dataframe['Expected Value'])
    plt.plot(prediction_dataframe['Date'], prediction_dataframe['Lower Confidence Interval'])
    plt.plot(prediction_dataframe['Date'], prediction_dataframe['Upper Confidence Interval'])
    plt.xlabel("Date")
    plt.ylabel("Predictions and Confidence Intervals")
    plt.title("ARIMA/SARIMA algorithm results")
    plt.savefig("ARIMA_SARIMA_MODEL_RESULTS.png")
    return prediction_dataframe
