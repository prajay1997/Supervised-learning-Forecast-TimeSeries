################## Forecasting - Time Series Q2) #########################
########################## Model Based Approach ############################

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q4\solarpower_cumuldaybyday2.csv")
data.info()
data.describe()

# Pre processing
import numpy as np

data['t'] = np.arange(1,2559)
data['t_square'] =  data['t']*data['t']
data['log_cum_power'] = np.log(data['cum_power'])
data.columns

data['date'] = pd.to_datetime(data['date'])
data.info()

data['day'] = data['date'].dt.day_name()


data1 = pd.get_dummies(data, columns=['day'])
data1.columns
data1 = data1.rename(columns={ 'day_Sunday':'sun','day_Monday':'mon','day_Tuesday':'tue','day_Wednesday':'wed','day_Thursday':'thur','day_Friday':'fri', 'day_Saturday':'sat'})

# rearranging the column

data1 = data1.iloc[:,[0,1,2,3,4,8,6,10,11,9,5,7]]
data1.columns

# Visualization - Timeplot
data1.cum_power.plot()

# data partition
Train = data1.head(2193)
Test  = data1.tail(365)

###################### Linear ####################################

import statsmodels.formula.api as smf

linear_model = smf.ols('cum_power ~ t', data = Train).fit()
pred_linear = linear_model.predict(Test['t'])
resd_lin = Test['cum_power'] - pred_linear
rmse_linear = np.sqrt(np.mean(resd_lin*resd_lin))
rmse_linear 

##################### Exponential ###################################

exp_model = smf.ols('log_cum_power ~ t',data = Train).fit()
pred_exp = exp_model.predict(Test['t'])
resd_exp = Test['cum_power'] - np.exp(pred_exp)
rmse_exp = np.sqrt(np.mean(resd_exp*resd_exp))
rmse_exp

################# Quadratic ######################################

quad_model = smf.ols('cum_power ~ t + t_square',data = Train).fit()
pred_quad = quad_model.predict(Test[['t', 't_square']])
resd_quad = Test['cum_power'] - pred_quad
rmse_quad = np.sqrt(np.mean(resd_quad*resd_quad))
rmse_quad

#################### Additive Seasonality #########################

add_sea = smf.ols('cum_power ~ sun + mon + tue + wed + thur + fri + sat',data = Train).fit()
pred_add_sea = add_sea.predict(Test[[ 'sun', 'mon', 'tue', 'wed', 'thur', 'fri', 'sat']])
resd_add_sea = Test['cum_power'] - pred_add_sea
rmse_add_sea = np.sqrt(np.mean(resd_add_sea*resd_add_sea))
rmse_add_sea

################## Multiplicative Seasonality #########################################

mul_sea = smf.ols('log_cum_power ~  sun + mon + tue + wed + thur + fri + sat',data = Train).fit()
pred_mul_sea = mul_sea.predict(Test)
resd_mul_sea = Test['cum_power'] - np.exp(pred_mul_sea)
rmse_mul_sea = np.sqrt(np.mean(resd_mul_sea*resd_mul_sea))
rmse_mul_sea

################### Additive Seasonality Quadratic Trend #########################

add_sea_quad = smf.ols('cum_power ~ t + t_square + sun + mon + tue + wed + thur + fri + sat',data = Train).fit()
pred_add_sea_quad = add_sea_quad.predict(Test)
resd_add_sea_quad = Test['cum_power'] - pred_add_sea_quad
rmse_add_sea_quad = np.sqrt(np.mean(resd_add_sea_quad*resd_add_sea_quad))
rmse_add_sea_quad


################## Multiplicative Seasonality Linear Trend  ###########

mul_add_sea = smf.ols('log_cum_power ~ t + sun + mon + tue + wed + thur + fri + sat',data = Train).fit()
pred_mul_add_sea = mul_add_sea.predict(Test)
resd_mul_add_sea = Test['cum_power'] -  np.exp(pred_mul_add_sea)
rmse_mul_add_sea = np.sqrt(np.mean(resd_mul_add_sea*resd_mul_add_sea ))
rmse_mul_add_sea


################## Testing #######################################

data2 = {"model":pd.Series(['rmse_linear','rmse_exp','rmse_quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']), "RMSE_values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
table_rmse = pd.DataFrame(data2)
table_rmse

# rmse_linear has the least value among the models prepared so far Predicting new values 

predict_data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q4\solar_predict_new.xlsx")
predict_data = predict_data.iloc[:,1:]

model_final = smf.ols('cum_power ~ t ',data = data1).fit()

predict_new = model_final.predict(predict_data)
predict_new

predict_data["Predicted_cum_power"] = pd.Series(predict_new)

# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV

full_resd = data1['cum_power'] - model_final.predict(data1)
# ACF plot on residuals 

import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_resd, lags = 7 )

# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.


# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_resd, lags=7)


# AR model 
from statsmodels.tsa.ar_model import AutoReg

model_ar = AutoReg(full_resd, lags = [1])
model_fit = model_ar.fit()

print('Coefficients: %s' % model_fit.params)

pred_res = model_fit.predict(start = len(full_resd), end = len(full_resd) + len(predict_data)-1, dynamic = False)
pred_res.reset_index(drop = True, inplace = True)

# The Final Predictions using ASQT and AR(1) Model

final_pred = predict_new + pred_res
final_pred

final_pred.to_excel("plasic_model_based.xlsx" ,encoding="utf-8")
import os
os.getcwd()

##########################  Data Driven Technique ############################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q4\solarpower_cumuldaybyday2.csv")

data.cum_power.plot()   # time series plot 

# Splitting the data into Train and Test data
# Recent 365 days time period values are Test data
Train = data.head(2193)
Test = data.tail(365)


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

# Moving Average for the time series
mv_pred = data["cum_power"].rolling(7).mean()
mv_pred.tail(7)
MAPE(mv_pred.tail(7), Test.cum_power)

# Plot with Moving Averages

data.cum_power.plot(label = "org")
for i in range(0, 35, 7):
    data["cum_power"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)

# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average

decompose_ts_add = seasonal_decompose(data.cum_power, model = "additive", period = 7)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(data.cum_power, model = "multiplicative", period = 7)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(data.cum_power, lags = 7)
tsa_plots.plot_pacf(data.cum_power, lags = 7)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["cum_power"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.cum_power) 

# Holt method 
hw_model = Holt(Train["cum_power"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.cum_power) 


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["cum_power"], seasonal = "add", trend = "add", seasonal_periods = 7).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.cum_power) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 7).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.cum_power) 

# Final Model on 100% Data

hwe_model_mul_add = ExponentialSmoothing(data["cum_power"], seasonal = "mul", trend = "add", seasonal_periods = 7).fit()

# Load the new data which includes the entry for future 365 values
new_data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q4\solarpower_data_driven.csv")


newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred
newdata_pred.tail(365)

####################################################################################

########################## ARIMA MOdel Q2) ####################

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

data= pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q4\solarpower_cumuldaybyday2.csv")
# Data Partition
Train = data.head(2193)
Test = data.tail(365)

tsa_plots.plot_acf(data.cum_power, lags = 7)
tsa_plots.plot_pacf(data.cum_power,lags = 7)

# ARIMA with AR = 1, MA = 7
model1 = ARIMA(Train.cum_power, order = (1,1,7))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 365 days
start_index = len(Train)
end_index = start_index + 364
forecast_test = res1.predict(start=start_index, end=end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.cum_power, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.cum_power)
pyplot.plot(forecast_test, color='red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm

ar_model = pm.auto_arima(Train.cum_power, start_p=0, start_q=0,
                      max_p=7, max_q=7, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, trace=True,
                      error_action='warn', stepwise=True)

# Best Parameters ARIMA
# ARIMA with AR=2, I = 1, MA = 1
model = ARIMA(Train.cum_power, order = (2,1,1))
res = model.fit()
print(res.summary())

# Forecast for next 365 days
start_index = len(Train)
end_index = start_index + 364
forecast_best = res1.predict(start=start_index, end=end_index)

print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.cum_power, forecast_best))
print('Test RMSE: %.3f' % rmse_best)

# plot forecasts against actual outcomes
pyplot.plot(Test.cum_power)
pyplot.plot(forecast_best, color='red')
pyplot.show()


# Forecast for future 365 days
start_index = len(data)
end_index = start_index + 364
forecast = res1.predict(start=start_index, end=end_index)

print(forecast)

pyplot.plot(forecast, color='red')
pyplot.show()

#############################################################

comparision = {"Method":pd.Series(['model_based','data_driven','ARIMA']), "Forecasted_value":pd.Series([final_pred,newdata_pred.tail(12),forecast])}

comparision1 = pd.DataFrame(comparision)
comparision1

comparision1.to_excel("solar_comp.xlsx", encoding  = 'utf-8')
