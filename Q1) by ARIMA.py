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
