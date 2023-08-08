# Using SARIMA Time Series Model to forecast future passenger traffic at Long Thanh International Airport (LTIA) Phase 1 from 2023-2035

Long Thanh International Airport is a national major and important air gateway of Vietnam. Phase 1 is expected to be completed in 2026, allowing the airport to handle 25 million passengers a year. LTIA aims to become one of the hub airports in Southeast Asia, with a total capacity of 100 million passengers annual and 5 million tons cargo annual. Long Thanh International Airport construction shall overcome the current overload situation of Tan Son Nhat International Airport (SGN).

The SARIMA (Seasonal Autoregressive Integrated Moving Average) model stands as a powerful tool in the realm of time series forecasting, particularly suited for complex data patterns characterized by seasonality, trend, and irregular fluctuations. It extends the foundational ARIMA model by incorporating seasonal components, making it especially apt for forecasting airport passenger traffic data. As airports witness dynamic changes in passenger numbers throughout the year due to holidays, travel seasons, and events, SARIMA's ability to capture both short-term and long-term patterns proves invaluable.

The SARIMA model combines autoregressive (AR) and moving average (MA) components with differencing and seasonal differencing, thus enabling accurate representation and prediction of temporal dependencies in the data. The autoregressive part accounts for past values' influence on the current value, while the moving average component considers the impact of past prediction errors. These components are augmented by differencing to stabilize and remove trends, which is especially crucial for non-stationary data like passenger traffic.

The introduction of seasonality involves an additional set of AR, MA, and differencing terms, allowing the model to account for recurring patterns over specific time intervals (e.g., daily, weekly, or monthly). This is especially pertinent for airport passenger traffic, which often exhibits consistent high and low seasons throughout the year due to factors such as holidays, vacation periods, and local events. By capturing both the seasonal variations and underlying trends, the SARIMA model ensures more accurate and reliable predictions.

# Modules
    import warnings
    import itertools
    import pandas as pd
    import numpy as np
    import sklearn
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_predict
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
    from pmdarima.arima import auto_arima
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Dataset
Monthly passenger data at SGN airport from 2012 to 2022 is used as input to train the model and validate model predicted traffic to actual values. Then, the model will be used to forecast out-of-sample traffic data.

    excel_file = "SGN_LTIA_forecastpax.xlsx"
    df = pd.read_excel(excel_file, index_col=0)
    df = df.asfreq('MS')
    df.rename(columns= {'Total_pax': 'Monthly Pax'},inplace=True)
    print(df.head(132))   

| Date | Monthly pax |
| -------- | --------- |        
| 2012-01-01 | 1,614,296 |
| 2012-02-01 | 1,420,632 |
| 2012-03-01 | 1,484,730 |
| 2012-04-01 | 1,380,141 |
| 2012-05-01 | 1,321,895 |
| ... | ... |
| 2022-08-01 | 3,453,091 |
| 2022-09-01 | 2,988,763 |
| 2022-10-01 | 3,049,318 |
| 2022-11-01 | 3,001,299 |
| 2022-12-01 | 3,258,710 |

[132 rows x 1 columns]

# Seasonality/ Trend
![Figure_1.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Figure_1.png)

    from statsmodels.tsa.seasonal import seasonal_decompose
    decompose_data = seasonal_decompose(df['Monthly Pax'], model='additive')
    decompose_data.plot()

# Split train-test set manually
    train_size = int(len(df['Monthly Pax']) * 0.55)
    train_data, test_data = df['Monthly Pax'][:train_size + 1], df['Monthly Pax'][train_size:]
    print("Train data")
    print(train_data)
    print("Test data")
    print(test_data)
    print("")
    
# Plot train-test using matplotlib.pyplot
![train data.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/train%20data.png)

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Monthly Pax')
    plt.plot(train_data, 'green', label='Train data')
    plt.plot(test_data, 'blue', label='Test data')
    plt.legend()

# Augmented Dickey-Fuller test aims to reject the null hypothesis that the given time-series data is non-stationary
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(df['Monthly Pax'].dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    for key, value in result[4].items():
        print("\t{}: {} - The data is {} stationary with {}% confidence".format(key, value, "not" if value < result[0] else "",100 - int(key[:-1])))
    print("")

#H0: It is a stationary\
#H1: It is a non-stationary

    if result[1] > 0.05:
      print("Reject null hypothesis, take this series as non-stationary")
    else:
      print("Accept null hypothesis, take this series as stationary")

    print("")
    result = adfuller(df['Monthly Pax'].diff().dropna())
    print('p-value 1st Order Differencing: %f' % result[1])
    result = adfuller(df['Monthly Pax'].diff().diff().dropna())
    print('p-value 2nd Order Differencing: %f' % result[1])
    print('Critical Values:')
    print("")

ADF Statistic: -2.371819\
p-value: 0.149849\
\
1%: -3.482920063655088 - The data is not stationary with 99% confidence\
5%: -2.884580323367261 - The data is not stationary with 95% confidence\
10%: -2.5790575441750883 - The data is not stationary with 90% confidence\
\
Reject null hypothesis, take this series as non-stationary\
\
p-value 1st Order Differencing: 0.000026\
p-value 2nd Order Differencing: 0.000000

# Finding the value of the d parameter (I = Integrated)
![Figure_3.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Figure_3.png)
# Original Series
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(df['Monthly Pax']);
    ax1.set_title('Original Series');
    ax1.axes.xaxis.set_visible(False)
# 1st Differencing
    ax2.plot(df['Monthly Pax'].diff());
    ax2.set_title('1st Order Differencing');
    ax2.axes.xaxis.set_visible(False)
# 2nd Differencing
    ax3.plot(df['Monthly Pax'].diff().diff());
    ax3.set_title('2nd Order Differencing');
    ax3.axes.xaxis.set_visible(False)

# Differencing Auto-correlation
![Figure_4.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Figure_4.png)

    from statsmodels.graphics.tsaplots import plot_acf
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    plot_acf(df['Monthly Pax'], ax=ax1)
    plot_acf(df['Monthly Pax'].diff().dropna(), ax=ax2)
    plot_acf(df['Monthly Pax'].diff().diff().dropna(), ax=ax3)

# PACF (p) and ACF (q) Plot
![Figure_5.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Figure_5.png)

    from statsmodels.graphics.tsaplots import plot_pacf
    fig, (ax1, ax2) = plt.subplots(2)
    plot_acf(df['Monthly Pax'].diff().dropna(),lags=24, ax=ax1)
    plot_pacf(df['Monthly Pax'].diff().dropna(),lags=24, ax=ax2)

# Grid search (or hyperparameter optimization) for model selection
# Define the p, d and q parameters to take any value between 0 and 2
    p = range(0, 2)
    d = range(0, 2)
    q = range(0, 2)

# Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        for param in pdq:
        for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            mod_fit = mod.fit(disp=0)
            forecast_test = mod_fit.forecast(len(test_data))
            mape = mean_absolute_percentage_error(test_data, forecast_test)
            print('SARIMA{}x{}12 - AIC:{} - MAPE:{}'.format(param, param_seasonal, mod_fit.aic,mape))
            except:
                continue
    print("")

# Auto ARIMA
    autoarima = auto_arima(df, start_p=0, d=1, start_q=0,
                       max_p=5, max_d=2, max_q=5,
                       start_P=0, D=1, start_Q=0,
                       max_P=5, max_D=2, max_Q=5, m=12,
                       seasonal=True,
                       error_action='warn',
                       trace=True,
                       supress_warnings=True,
                       stepwise=True,
                       random_state=20,
                       n_fits=50)
    print("Auto Arima")
    print(autoarima.summary())
    print("")

Performing stepwise search to minimize aic
 ARIMA(0,1,0)(0,1,0)[12]             : AIC=3489.431, Time=0.03 sec\
 ARIMA(1,1,0)(1,1,0)[12]             : AIC=3447.216, Time=0.09 sec\
 ARIMA(0,1,1)(0,1,1)[12]             : AIC=3436.490, Time=0.09 sec\
 ARIMA(0,1,1)(0,1,0)[12]             : AIC=3485.024, Time=0.03 sec\
 ARIMA(0,1,1)(1,1,1)[12]             : AIC=3435.357, Time=0.13 sec\
 ARIMA(0,1,1)(1,1,0)[12]             : AIC=3448.392, Time=0.08 sec\
 ARIMA(0,1,1)(2,1,1)[12]             : AIC=3436.202, Time=0.35 sec\
 ARIMA(0,1,1)(1,1,2)[12]             : AIC=3436.672, Time=0.50 sec\
 ARIMA(0,1,1)(0,1,2)[12]             : AIC=3434.694, Time=0.30 sec\
 ARIMA(0,1,1)(0,1,3)[12]             : AIC=3436.658, Time=0.49 sec\
 ARIMA(0,1,1)(1,1,3)[12]             : AIC=3438.494, Time=0.93 sec\
 ARIMA(0,1,0)(0,1,2)[12]             : AIC=3496.638, Time=0.17 sec\
 ARIMA(1,1,1)(0,1,2)[12]             : AIC=3435.863, Time=0.42 sec\
 ARIMA(0,1,2)(0,1,2)[12]             : AIC=3436.528, Time=0.34 sec\
 ARIMA(1,1,0)(0,1,2)[12]             : AIC=3432.601, Time=0.25 sec\
 ARIMA(1,1,0)(0,1,1)[12]             : AIC=3434.795, Time=0.07 sec\
 ARIMA(1,1,0)(1,1,2)[12]             : AIC=3434.585, Time=0.44 sec\
 ARIMA(1,1,0)(0,1,3)[12]             : AIC=3434.574, Time=0.41 sec\
 ARIMA(1,1,0)(1,1,1)[12]             : AIC=3433.302, Time=0.11 sec\
 ARIMA(1,1,0)(1,1,3)[12]             : AIC=3436.411, Time=0.86 sec\
 ARIMA(2,1,0)(0,1,2)[12]             : AIC=3426.541, Time=0.32 sec\
 ARIMA(2,1,0)(0,1,1)[12]             : AIC=3427.725, Time=0.11 sec\
 ARIMA(2,1,0)(1,1,2)[12]             : AIC=3428.414, Time=0.54 sec\
 ARIMA(2,1,0)(0,1,3)[12]             : AIC=3428.310, Time=0.53 sec\
 ARIMA(2,1,0)(1,1,1)[12]             : AIC=3427.276, Time=0.15 sec\
 ARIMA(2,1,0)(1,1,3)[12]             : AIC=3430.036, Time=0.83 sec\
 ARIMA(3,1,0)(0,1,2)[12]             : AIC=3424.209, Time=0.38 sec\
 ARIMA(3,1,0)(0,1,1)[12]             : AIC=3424.779, Time=0.13 sec\
 ...

# Building SARIMA(p,d,q)(P,D,Q,L) = (2,1,2)(0,1,1,12)
![Figure_6.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Figure_6.png)  

    model = SARIMAX(df,order=(2, 1, 2),
                seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

    model_fit = model.fit(disp=0)
    model_fit.plot_diagnostics(figsize=(15, 12))
    print(model_fit.summary())
    print("")

# SARIMAX Results                                       

Dep. Variable:                          Monthly Pax   
No. Observations:                  132  
Model:             SARIMAX(2, 1, 2)x(0, 1, [1], 12)   
Date:                              Fri, 14 Jul 2023    
Time:                                      10:59:17   
Sample:                                  01-01-2012 - 12-01-2022                         
Covariance Type:                                opg                                         

Log Likelihood:               -1492.409  
AIC:                           2996.819  
BIC:                           3012.685  
HQIC:                          3003.247
    
Ljung-Box (L1) (Q):                   0.22   
Jarque-Bera (JB):                    71.83  
Prob(Q):                              0.64  
Prob(JB):                             0.00  
Heteroskedasticity (H):               9.59  
Skew:                                -0.67  
Prob(H) (two-sided):                  0.00  
Kurtosis:                             6.84  

# Validate model accuracy
    forecast_test = model_fit.forecast(len(test_data))
    mae = mean_absolute_error(test_data, forecast_test)
    mape = mean_absolute_percentage_error(test_data, forecast_test)
    rmse = np.sqrt(mean_squared_error(test_data, forecast_test))

    print(f'mae - Mean Absolute Error: {mae}')
    print(f'mape - Mean Absolute Percentage Error: {mape}')
    print(f'rmse - Root-mean-square deviation: {rmse}')
    print("")

mae - Mean Absolute Error: 1756864.940118484\
mape - Mean Absolute Percentage Error: 13.598499090494714\
rmse - Root-mean-square deviation: 2049002.3282535903
    
![Figure_7.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Figure_7.png)

    #y_pred = model_fit.get_prediction(start=test_data.index[0], end=test_data.index[-1], dynamic=False)
    y_pred = model_fit.get_prediction(start=pd.to_datetime('2018-01-01'), end=pd.to_datetime('2022-12-01'), dynamic=False)
    pred_ci = y_pred.conf_int()
    ax = df['2012':].plot(label='Actual')
    y_pred.predicted_mean.plot(ax=ax, label='One-step ahead', alpha=.5)
    ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Date')
    ax.set_ylabel('Pax')
    plt.title('Testing HCM area Pax traffic forecast')
    plt.legend()
    print(y_pred.predicted_mean)
    print("")

# Fure forecast (2023-2035)
![Figure_8.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Figure_8.png)

    future = model_fit.get_forecast(steps=156, signal_only=False)
    future_ci = future.conf_int()
    ax = df['2012':].plot(label='Actual', figsize=(9, 7))
    future.predicted_mean.plot(ax=ax, label='Forecasted value')
    ax.fill_between(future_ci.index,
                future_ci.iloc[:, 0],
                future_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Pax')
    plt.title('5-year HCM area Pax traffic forecast')
    plt.legend()
    plt.show()
    df2 = future.predicted_mean
    print('Forecast monthly pax')
    print(df2.tail(60))

| Date | Forecast Monthly pax |
| -------- | --------- |
| 2023-01-01 | 3,508,168 |
| 2023-02-01 | 3,437,096 |
| 2023-03-01 | 3,374,886 |
| 2023-04-01 | 3,360,680 |
| 2023-05-01 | 3,347,965 |
| 2023-06-01 | 3,401,006 |
| 2023-07-01 | 3,571,990 |
| 2023-08-01 | 3,264,586 |
| 2023-09-01 | 3,148,431 |
| 2023-10-01 | 3,282,201 |
| 2023-11-01 | 3,302,302 |
| 2023-12-01 | 3,467,837 |
| ... | ... |
| 2035-01-01 | 7,430,841 |
| 2035-02-01 | 7,406,244 |
| 2035-03-01 | 7,278,502 |
| 2035-04-01 | 7,185,733 |
| 2035-05-01 | 7,192,210 |
| 2035-06-01 | 7,328,995 |
| 2035-07-01 | 7,526,894 |
| 2035-08-01 | 7,155,600 |
| 2035-09-01 | 6,980,052 |
| 2035-10-01 | 7,141,748 |
| 2035-11-01 | 7,232,346 |
| 2035-12-01 | 7,409,532 |

Freq: MS, Name: predicted_mean, dtype: float64\

Process finished with exit code 0

# Total pax forecast (Updated half-year data as of Jun 2023)
The following table shows the comparison of the forecasted data versus the actual data. The first 6-month pax traffic data was provided by SGN airport to validate the accuracy of the model. The Mean Absolute Percentage Error (MAPE) is 6.02%. It seems the results of SARIMA(2,1,2)(0,1,1,12) are quite close to the actual traffic data. The last 6-month traffic data will be compared at the end of this year.
| Date | Forecast |	Actual | Absolute % Error|
| -------- | --------- | --------- | ----- |
| 2023-01-01 | 3,508,168 | 3,816,658 | 8.08% |
| 2023-02-01 | 3,437,096 | 3,185,644 | 7.89% |
| 2023-03-01 | 3,374,886 | 3,262,966 | 3.43% |
| 2023-04-01 | 3,360,680 | 3,127,879 | 7.44% |
| 2023-05-01 | 3,347,965 | 3,295,803 | 1.58% |
| 2023-06-01 | 3,401,006 | 3,684,908 | 7.70% |

# Conclusion
Passenger traffic in Ho Chi Minh City area will be expected to recover in late 2023 and 2024 at the latest, and continue to grow steadily at a compound annual growth rate of 6.76% over the five-year period from 2023 to 2027.
As airports play a pivotal role in global travel and commerce, the insights provided by the SARIMA model can aid stakeholders in making informed decisions, optimizing operations, and planning for contingencies. By harnessing the power of SARIMA, accurate predictions and proactive strategies become attainable, ensuring the seamless management of airport resources and passenger experiences.

# References
- Box, G.E.P., and G.M. Jenkins (1976). Time Señes Analysis: Forecasting and Control, Revised Edition, Holden Day, San Francisco.
- Chaitip, R, Chaiboonsri and R. Mukhjang (2008). Time Series Models for Forecasting International Visitor Arrivals to Thailand, International Conference on Applied Economics, 2008, 159-163.
- Tsui, Kan & Balli, Hatice & Gilbey, Andrew & Gow, Hamish. (2014). Forecasting of Hong Kong airport's passenger throughput. Tourism Management. 42. 62–76. 10.1016/j.tourman.2013.10.008.


