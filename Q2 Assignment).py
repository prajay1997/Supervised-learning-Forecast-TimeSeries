################## Forecasting - Time Series Q2) #########################
########################## Model Based Approach ############################

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q2)\CocaCola_Sales_Rawdata.xlsx")
data.info()
data.describe()


# Pre processing
import numpy as np

data['t'] = np.arange(1,43)
data['t_square'] =  data['t']*data['t']
data['log_Sales'] = np.log(data['Sales'])
data.columns


p = data["Quarter"][0]
p[0:2]

data['quarter'] = 0

for i in range(42):
    p = data["Quarter"][i]
    data['quarter'][i]= p[0:2]

data1 = pd.get_dummies(data, columns=['quarter'])
data1.columns
data1 = data1.rename(columns = {'quarter_Q1':'Q1','quarter_Q2':'Q2','quarter_Q3':'Q3','quarter_Q4':'Q4'})

# Visualization - Timeplot
data1.Sales.plot()

# data partition
Train = data1.head(38)
Test  = data1.tail(4)

###################### Linear ####################################

import statsmodels.formula.api as smf

linear_model = smf.ols('Sales ~ t', data = Train).fit()
pred_linear = linear_model.predict(Test['t'])
resd_lin = Test['Sales'] - pred_linear
rmse_linear = np.sqrt(np.mean(resd_lin*resd_lin))
rmse_linear 

##################### Exponential ###################################

exp_model = smf.ols('log_Sales~t',data = Train).fit()
pred_exp = exp_model.predict(Test['t'])
resd_exp = Test['Sales'] - np.exp(pred_exp)
rmse_exp = np.sqrt(np.mean(resd_exp*resd_exp))
rmse_exp

################# Quadratic ######################################

quad_model = smf.ols('Sales~t + t_square',data = Train).fit()
pred_quad = quad_model.predict(Test[['t', 't_square']])
resd_quad = Test['Sales'] - pred_quad
rmse_quad = np.sqrt(np.mean(resd_quad*resd_quad))
rmse_quad

#################### Additive Seasonality #########################

add_sea = smf.ols('Sales ~ Q1 + Q2 + Q3 + Q4',data = Train).fit()
pred_add_sea = add_sea.predict(Test[['Q1','Q2','Q3','Q4']])
resd_add_sea = Test['Sales'] - pred_add_sea
rmse_add_sea = np.sqrt(np.mean(resd_add_sea*resd_add_sea))
rmse_add_sea

################## Multiplicative Seasonality #########################################

mul_sea = smf.ols('log_Sales ~  Q1 + Q2 + Q3 + Q4',data = Train).fit()
pred_mul_sea = mul_sea.predict(Test)
resd_mul_sea = Test['Sales'] - np.exp(pred_mul_sea)
rmse_mul_sea = np.sqrt(np.mean(resd_mul_sea*resd_mul_sea))
rmse_mul_sea

################### Additive Seasonality Quadratic Trend #########################

add_sea_quad = smf.ols('Sales ~ t + t_square + Q1 + Q2 + Q3 + Q4',data = Train).fit()
pred_add_sea_quad = add_sea_quad.predict(Test)
resd_add_sea_quad = Test['Sales'] - pred_add_sea_quad
rmse_add_sea_quad = np.sqrt(np.mean(resd_add_sea_quad*resd_add_sea_quad))
rmse_add_sea_quad


################## Multiplicative Seasonality Linear Trend  ###########

mul_add_sea = smf.ols('log_Sales ~ t + Q1 + Q2 + Q3 + Q4',data = Train).fit()
pred_mul_add_sea = mul_add_sea.predict(Test)
resd_mul_add_sea = Test['Sales'] -  np.exp(pred_mul_add_sea)
rmse_mul_add_sea = np.sqrt(np.mean(resd_mul_add_sea*resd_mul_add_sea ))
rmse_mul_add_sea


################## Testing #######################################

data2 = {"model":pd.Series(['rmse_linear','rmse_exp','rmse_quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']), "RMSE_values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
table_rmse = pd.DataFrame(data2)
table_rmse

# rmse_mul_add_sea has the least value among the models prepared so far Predicting new values 

predict_data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q2)\cocacola_new Q2.xlsx")
predict_data.columns

predict_data.drop(['Unnamed: 0','Unnamed: 9'], axis = 1, inplace = True)

model_final = smf.ols('log_Sales ~ t + Q1 + Q2 + Q3 + Q4',data = data1).fit()

predict_new = model_final.predict(predict_data)
predict_new = np.exp(predict_new)
predict_new

predict_data["Predicted_Sales"] = pd.Series(predict_new)

# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV

b = model_final.predict(data1)
c= np.exp(b)
full_resd = data1['Sales'] - c
# ACF plot on residuals 

import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_resd, lags =4)

# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.


# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_resd, lags=4)


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

final_pred.to_excel("cocacola_model_based.xlsx" ,encoding="utf-8")
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


data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\CocaCola_Sales_Rawdata.xlsx")

data.Sales.plot()   # time series plot 

# Splitting the data into Train and Test data
# Recent 4 time period values are Test data
Train = data.head(38)
Test = data.tail(4)


# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

# Moving Average for the time series
mv_pred = data["Sales"].rolling(4).mean()
mv_pred.tail(4)
MAPE(mv_pred.tail(4), Test.Sales)

# Plot with Moving Averages

data.Sales.plot(label = "org")
for i in range(2, 9, 2):
    data["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)

# Time series decomposition is the process of separating data into its core components.
# Time series decomposition plot using Moving Average

decompose_ts_add = seasonal_decompose(data.Sales, model = "additive", period =4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(data.Sales, model = "multiplicative", period = 12)
decompose_ts_mul.plot()

# ACF and PACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(data.Sales, lags = 4 )
tsa_plots.plot_pacf(data.Sales, lags = 4)
# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.

# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 

# Final Model on 100% Data

hwe_model_add_add = ExponentialSmoothing(data["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 8 values
new_data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q2)\CocaCola_new data driven.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred
newdata_pred.tail(8)

####################################################################################

########################## ARIMA MOdel Q2) ####################

import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

data= pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Q2)\CocaCola_Sales_Rawdata.xlsx")
# Data Partition
Train = data.head(38)
Test = data.tail(4)

tsa_plots.plot_acf(data.Sales, lags = 4)
tsa_plots.plot_pacf(data.Sales,lags = 4)

# ARIMA with AR = 1, I =1  MA = 4
model1 = ARIMA(Train.Sales, order = (1,1,4))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 4 Quarter
start_index = len(Train)
end_index = start_index + 3
forecast_test = res1.predict(start=start_index, end=end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Sales, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Sales)
pyplot.plot(forecast_test, color='red')
pyplot.show()


# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm

ar_model = pm.auto_arima(Train.Sales, start_p=0, start_q=0,
                      max_p=4, max_q=4, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, trace=True,
                      error_action='warn', stepwise=True)


# Best Parameters ARIMA
# ARIMA with AR=0, I = 1, MA = 2
model = ARIMA(Train.Sales, order = (0,1,2))
res = model.fit()
print(res.summary())

# Forecast for next 4 quarter
start_index = len(Train)
end_index = start_index + 3
forecast_best = res1.predict(start=start_index, end=end_index)

print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Sales, forecast_best))
print('Test RMSE: %.3f' % rmse_best)

# plot forecasts against actual outcomes
pyplot.plot(Test.Sales)
pyplot.plot(forecast_best, color='red')
pyplot.show()


# Forecast for future 8 quarter
start_index = len(data)
end_index = start_index + 7
forecast = res1.predict(start=start_index, end=end_index)

print(forecast)

pyplot.plot(forecast, color='red')
pyplot.show()

#############################################################

comparision = {"Method":pd.Series(['model_based','data_driven','ARIMA']), "Forecasted_value":pd.Series([final_pred,newdata_pred.tail(8),forecast])}

comparision1 = pd.DataFrame(comparision)
comparision1

comparision1.to_excel("Q2 cococola_comp.xlsx", encoding  = 'utf-8')
