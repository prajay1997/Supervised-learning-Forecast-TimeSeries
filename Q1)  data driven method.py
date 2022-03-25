import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Datasets\Airlines Data.xlsx")

data.Passengers.plot()   # time series plot 

# Splitting the data into Train and Test data
# Recent 4 time period values are Test data
Train = data.head(84)
Test = data.tail(12)


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

# Moving Average for the time series
mv_pred = data["Passengers"].rolling(12).mean()
mv_pred.tail(12)
MAPE(mv_pred.tail(12), Test.Passengers)

# Plot with Moving Averages
data.Passengers.plot(label = "org")
for i in range(2, 15, 2):
    data["Passengers"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)


# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average

decompose_ts_add = seasonal_decompose(data.Passengers, model = "additive", period = 12)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(data.Passengers, model = "multiplicative", period = 12)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(data.Passengers, lags = 12 )
tsa_plots.plot_pacf(data.Passengers, lags = 12)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Passengers) 

# Holt method 
hw_model = Holt(Train["Passengers"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Passengers) 


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Passengers"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Passengers) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Passengers) 

# Final Model on 100% Data

hwe_model_mul_add = ExponentialSmoothing(data["Passengers"], seasonal = "mul", trend = "add", seasonal_periods = 12).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q1\Airlines Datanew.xlsx")
new_data.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace= True)

newdata_pred = hwe_model_mul_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred.tail(60)
