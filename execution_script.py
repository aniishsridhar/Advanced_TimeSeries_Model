'''A script that takes user input from the command line, and also asks the user some questions
about the data and gives predictions using both the Prophet and ARIMA/SARIMA algorithm if the data
is weekly or monthly and only the Prophet algorithm for daily data'''
import sys
import pandas as pd
from prophet_algorithm import prophet_ts, data_preprocess
from arima_sarima_algorithm import arima_sarima_model

FILE_NAME_URL = sys.argv[1]
NUM_PERIODS = sys.argv[2]
try:
    CONFIDENCE_INTERVAL = float(sys.argv[3]) if (len(sys.argv) >= 4 and float(sys.argv[3]) > 0\
    and float(sys.argv[3]) < 1) else 0.95
except ValueError:
    CONFIDENCE_INTERVAL = 0.95
try:
    print(CONFIDENCE_INTERVAL)
    data_input = pd.read_csv(FILE_NAME_URL)
    START_DATE = input("Please enter the starting time period for \
this dataset in yyyy-mm-dd format: ")
    PREDICTION_TYPE = None
    GROWTH_FUNCTION = None
    ARIMA_SEASONALITY = None
    while PREDICTION_TYPE not in ['weekly', 'monthly', 'daily'] or \
GROWTH_FUNCTION not in ['linear', 'logistic', 'flat']:
        PREDICTION_TYPE = input("What type of time series prediction are you looking - daily, \
weekly, monthly? ")
        GROWTH_FUNCTION = input("For the Prophet model, is the the trend linear, \
logistic or flat? ")

    if PREDICTION_TYPE in ['weekly', 'monthly']:
        ARIMA_SEASONALITY = input("For the ARIMA/SARIMA model, should we add seasonality? Type 'y' \
for yes and 'n' for no: ")
        ARIMA_MODEL_TUNING = bool(ARIMA_SEASONALITY == 'y')
    data_formatted = data_preprocess(START_DATE, data_input, PREDICTION_TYPE)
    if PREDICTION_TYPE in ['weekly', 'monthly']:
        print(prophet_ts(data_input, START_DATE, GROWTH_FUNCTION, PREDICTION_TYPE, \
            [int(NUM_PERIODS), CONFIDENCE_INTERVAL]))
        print(arima_sarima_model(data_formatted, int(NUM_PERIODS), \
            ARIMA_MODEL_TUNING, PREDICTION_TYPE, CONFIDENCE_INTERVAL))
    else:
        print(prophet_ts(data_input, START_DATE, GROWTH_FUNCTION, PREDICTION_TYPE, \
            [int(NUM_PERIODS), CONFIDENCE_INTERVAL]))
except FileNotFoundError:
    print("The source file is not in the directory")
except ValueError:
    print("Please check whether the start date is legitimate and in yyyy-mm-dd format. \
        If that is correct, please check to ensure that the confidence interval is between \
            0 and 1")
except TypeError:
    print("Please check whether the number of predictions requested is a number")
except KeyError:
    print("Please check the format of the data file - \
It should have one column with header name y")
