
# Goals
The goal of this project is to generate forecasts from a time series dataset. The specific goals of the project are as follows:         
•	Develop a time series prediction functionality using Facebook’s Prophet prediction algorithm, and auto-fit ARIMA and SARIMA models in Python. The ARIMA and SARIMA models, as mentioned in the proposal document, are extensions, the primary goal is the Prophet model                                   
•	Output results of the models - raw predictions and confidence interval – into a .csv file               
•	Output a line chart of the raw predictions along with the confidence intervals, and also diagnostics if the ARIMA and SARIMA models were used           




# Design

The prophet_algorithm.py contains 2 functions: data_preprocess and prophet_ts  

### Function data_preprocess()
This function will preprocess the input time series data. It takes three arguments   
start_time - First date/day of the time series in yyyy-mm-dd format             
raw_data - path to a .csv file in the working directory which contains one column of time series observations with name 'y'       
data_frequency - Whether the time series was measured daily, monthly or weekly

### The Prophet time series algorithm designed by Facebook
The script prophet_algorithm.py contains my implementation of this algorithm. The script contains a function called prophet_ts() and takes the arguments below 

raw_df - The path to the source file which contains one column, 'y', of time series observations  
date_string - The start date for the time series being measured in 'yyyy-mm-dd' format 
model_type - Prophet accepts 'linear', 'logistic' or 'flat' for the trend function  
data_type - This argument is to check if data is 'weekly', 'monthly' or 'daily' 
model_specs_lst - This is a list that takes two values - first argument is for the number of predictions 
to be generated and the second for the type of confidence interval

### ARIMA/SARIMA algorithm
The arima_sarima_algorithm.py contains the arima_sarima_model function. The function takes five arguments 

df_formatted : A formatted dataframe which has been passed through the data_preprocess function and contains two columns- 'ds' and 'y' where 'ds' stands for datestamps 

prediction_periods : The number of predictions into the future that need to be generated  

seasonality : This is a boolean - If true, the model will use a Seasonal ARIMA or a SARIMA model.  Else, it will only do an ARIMA model

data_type : The type of data - is it 'daily', 'weekly' or 'monthly'  

confidence_int : The type of confidence interval to be generated 

### The script execution.py that implements both algorithm

The execution.py script will accept user input through the command line and generate predictions using the Prophet and/or
ARIMA/SARIMA model. If the data is daily, only predictions from Prophet will be generated. The ARIMA and SARIMA models
work well when the data is weekly or monthly; For for daily data, these models do not have the flexibility to capture 
large amounts of signal. Hence, when the user needs daily predictions, only the Prophet algorithm is set in action to generate predictions 

# Limitations
•	The algorithms use a generic tuning for their parameters; there are no options for the user to customize some of the parameters from the command line 

•	Time series models can benefit from the addition of external regressors. Holidays impact traffic, especially in the case of the sample data set above which is about airline passenger traffic. However, the tunings for the algorithms in the current execution.py script does not allow for these options.  

•	The code design could have arranged the try-except blocks more systematically and also incorporated errors related to the algorithms as well. The script is also asking the user for input beyond the command line parameters, it will be interesting to see how much it can be minimized. 

•	A machine learning approach involving cross validation with training, validation and test sets would have been interesting. 

•	The ARIMA/SARIMA algorithm is effectively for relatively long range predictions on weekly or monthly data. They can generate daily predictions, but only to a certain minimum horizon, like 2-3 days from my personal experience. If the data has multiple signals like daily, weekly, monthly, and quarterly seasonality, then these models are not flexible enough to model growth. Hence, for daily predictions, the script will not generate ARIMA/SARIMA predictions


# Usage

### External Libraries

Before running the scripts, the user needs to install the neccessary packages from the [requirments.txt](requirements.txt) file.

After the installation, the script can be run. Before running, the user should also check whether any existing libraries in their environment need an update. 

### Executing the execution_script.py file
After these steps, to run the script, the user needs to input two mandatory arguments through the command line. They are as below :  
python file name - execution_script.py  

file_path : raw data file with one column, 'y', with the time series observsation arranged row-wise   

number_predictions : number of future predictions. This is a positive integer generally.

[confidence_interval] - This is optional - will default to 0.95 if not entered or if non float or value outside of 0 or 1 is entered  accidentally

### Sample Command line Code and expected behavior
Sample 1 : python execution_script.py AirPassengers2.csv 11 0.95 

Sample 2 without confidence interval : python 
 execution_script.py AirPassengers2.csv 11

Sample 3 with string confidence interval : python 
execution_script.py AirPassengers2.csv 11 abc - Here confidence interval will default to 0.95

Sample 4 : with infeasible confidence interval : python execution_script.py AirPassengers2.csv 11 1.2 - Here confidence interval will default to 0.95

# Demo

I have uploaded a word document called Project_Snapshots.docx which also contains a walk through of how the user will interact with the script. I was hoping to upload this to the readme.MD file itself, but felt a word document will be better. Thank you 

