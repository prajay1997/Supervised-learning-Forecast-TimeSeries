######################## Forecast Time Series Q1) #########################
 ####################### Regression model ##########################
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Datasets\Airlines Data.xlsx")
data.info()
data.describe()


# Pre processing
import numpy as np

data["Month"] = pd.to_datetime(data['Month'])
data['month'] = data["Month"].dt.month

data['t'] = np.arange(1,97)
data['t_square'] =  data['t']*data['t']
data['log_passengers'] = np.log(data['Passengers'])
data.columns


month_dummies = pd.DataFrame(pd.get_dummies(data['month']))

data1 = pd.concat([data, month_dummies], axis =1)
data1.drop(['Month'], axis =1 , inplace = True)
data1 = data1.rename(columns = {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'july',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})


# Visualization - Timeplot
data1.Passengers.plot()

# data partition
Train = data1.head(84)
Test  = data1.tail(12)

# To change the index of test data
Test.set_index(np.arange(1,13), inplace = True)

###################### Linear ####################################

import statsmodels.formula.api as smf

linear_model = smf.ols('Passengers ~ t', data = Train).fit()
pred_linear = linear_model.predict(Test['t'])
resd_lin = Test['Passengers'] - pred_linear
rmse_linear = np.sqrt(np.mean(resd_lin*resd_lin))
rmse_linear 

##################### Exponential ###################################

exp_model = smf.ols('log_passengers~t',data = Train).fit()
pred_exp = exp_model.predict(Test['t'])
resd_exp = Test['Passengers'] - pred_exp
rmse_exp = np.sqrt(np.mean(resd_exp*resd_exp))
rmse_exp

################# Quadratic ######################################

quad_model = smf.ols('Passengers~t + t_square',data = Train).fit()
pred_quad = quad_model.predict(Test[['t', 't_square']])
resd_quad = Test['Passengers'] - pred_quad
rmse_quad = np.sqrt(np.mean(resd_quad*resd_quad))
rmse_quad

#################### Additive Seasonality #########################

add_sea = smf.ols('Passengers ~ jan + feb + mar + apr + may + jun + july + aug + sep + oct + nov + dec',data = Train).fit()
pred_add_sea = add_sea.predict(Test[['jan', 'feb','mar', 'apr', 'may', 'jun', 'july', 'aug', 'sep', 'oct', 'nov', 'dec']])
resd_add_sea = Test['Passengers'] - pred_add_sea
rmse_add_sea = np.sqrt(np.mean(resd_add_sea*resd_add_sea))
rmse_add_sea

################## Multiplicative Seasonality #########################################

mul_sea = smf.ols('log_passengers ~ jan + feb + mar + apr + may + jun + july + aug + sep + oct + nov + dec',data = Train).fit()
pred_mul_sea = mul_sea.predict(Test)
resd_mul_sea = Test['Passengers'] - pred_mul_sea
rmse_mul_sea = np.sqrt(np.mean(resd_mul_sea*resd_mul_sea))
rmse_mul_sea

################### Additive Seasonality Quadratic Trend #########################

add_sea_quad = smf.ols('Passengers ~ t + t_square + jan + feb + mar + apr + may + jun + july + aug + sep + oct + nov + dec',data = Train).fit()
pred_add_sea_quad = add_sea_quad.predict(Test)
resd_add_sea_quad = Test['Passengers'] - pred_add_sea_quad
rmse_add_sea_quad = np.sqrt(np.mean(resd_add_sea_quad*resd_add_sea_quad))
rmse_add_sea_quad


################## Multiplicative Seasonality Linear Trend  ###########

mul_add_sea = smf.ols('log_passengers ~ t + jan + feb + mar + apr + may + jun + july + aug + sep + oct + nov + dec',data = Train).fit()
pred_mul_add_sea = mul_add_sea.predict(Test)
resd_mul_add_sea = Test['Passengers'] -  pred_mul_add_sea
rmse_mul_add_sea = np.sqrt(np.mean(resd_mul_add_sea*resd_mul_add_sea ))
rmse_mul_add_sea


################## Testing #######################################

data2 = {"model":pd.Series(['rmse_linear','rmse_exp','rmse_quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']), "RMSE_values":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
table_rmse = pd.DataFrame(data2)
table_rmse

# 'rmse_add_sea_quad' has the least value among the models prepared so far Predicting new values 

predict_data = pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Datasets\predictnew Q1).xlsx")
predict_data.columns

predict_data.drop(['Unnamed: 15','Unnamed: 16'], axis = 1, inplace = True)

model_final = smf.ols('Passengers ~ t + t_square + jan + feb + mar + apr + may + jun + july + aug + sep + oct + nov + dec',data = data1).fit()

predict_new = model_final.predict(predict_data)
predict_new

predict_data["Predicted_passengers"] = pd.Series(predict_new)

# Autoregression Model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV

full_resd = data1['Passengers'] - model_final.predict(data1)
# ACF plot on residuals 

import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_resd, lags =12)

# ACF is an (complete) auto-correlation function gives values 
# of auto-correlation of any time series with its lagged values.


# PACF is a partial auto-correlation function. 
# It finds correlations of present with lags of the residuals of the time series 
tsa_plots.plot_pacf(full_resd, lags=12)


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

#############################################################################################
##################################### Data Driven Method ###############################

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

############################################################################################################

#################################### ARIMA method #####################################
import pandas as pd
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot

data= pd.read_excel(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Forecast Time Series\Datasets\Airlines Data.xlsx")
# Data Partition
Train = data.head(84)
Test = data.tail(12)

tsa_plots.plot_acf(data.Passengers, lags = 12)
tsa_plots.plot_pacf(data.Passengers,lags = 12)

# ARIMA with AR = 1, MA = 12
model1 = ARIMA(Train.Passengers, order = (1,1,12))
res1 = model1.fit()
print(res1.summary())

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_test = res1.predict(start=start_index, end=end_index)

print(forecast_test)

# Evaluate forecasts
rmse_test = sqrt(mean_squared_error(Test.Passengers, forecast_test))
print('Test RMSE: %.3f' % rmse_test)

# plot forecasts against actual outcomes
pyplot.plot(Test.Passengers)
pyplot.plot(forecast_test, color='red')
pyplot.show()

# Auto-ARIMA - Automatically discover the optimal order for an ARIMA model.
# pip install pmdarima --user
import pmdarima as pm

ar_model = pm.auto_arima(Train.Passengers, start_p=0, start_q=0,
                      max_p=12, max_q=12, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, trace=True,
                      error_action='warn', stepwise=True)


# Best Parameters ARIMA
# ARIMA with AR=2, I = 1, MA = 1
model = ARIMA(Train.Passengers, order = (2,1,1))
res = model.fit()
print(res.summary())

# Forecast for next 12 months
start_index = len(Train)
end_index = start_index + 11
forecast_best = res1.predict(start=start_index, end=end_index)

print(forecast_best)

# Evaluate forecasts
rmse_best = sqrt(mean_squared_error(Test.Passengers, forecast_best))
print('Test RMSE: %.3f' % rmse_best)

# plot forecasts against actual outcomes
pyplot.plot(Test.Passengers)
pyplot.plot(forecast_best, color='red')
pyplot.show()


# Forecast for future 60 months
start_index = len(data)
end_index = start_index + 59
forecast = res1.predict(start=start_index, end=end_index)

print(forecast)
