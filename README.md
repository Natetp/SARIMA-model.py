# SARIMA-model.py
Using SARIMA Time Series Model to forecast future passenger traffic at Long Thanh International Airport (LTIA) from 2023-2035

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

    excel_file = "SGN_LTIA_forecastpax.xlsx"
    df = pd.read_excel(excel_file, index_col=0)
    df = df.asfreq('MS')
    df.rename(columns= {'Total_pax': 'Monthly Pax'},inplace=True)
    print(df.head(132))
           
Date           Monthly Pax           
2012-01-01        1614296\
2012-02-01        1420632\
2012-03-01        1484730\
2012-04-01        1380141\
2012-05-01        1321895\
...                 ...\
2022-08-01        3453091\
2022-09-01        2988763\
2022-10-01        3049318\
2022-11-01        3001299\
2022-12-01        3258710\
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
      print("Reject null hypothesis, take this series as non-stationery")
    else:
      print("Accept null hypothesis, take this series as stationary")

    print("")
    result = adfuller(df['Monthly Pax'].diff().dropna())
    print('p-value 1st Order Differencing: %f' % result[1])
    result = adfuller(df['Monthly Pax'].diff().diff().dropna())
    print('p-value 2nd Order Differencing: %f' % result[1])
    print('Critical Values:')
    print("")

ADF Statistic: -2.371819
p-value: 0.149849\
1%: -3.482920063655088 - The data is not stationary with 99% confidence\
5%: -2.884580323367261 - The data is not stationary with 95% confidence\
10%: -2.5790575441750883 - The data is not stationary with 90% confidence\
\
Reject null hypothesis, take this series as non-stationery\
\
p-value 1st Order Differencing: 0.000026\
p-value 2nd Order Differencing: 0.000000

# Finding the value of the d parameter (I = Integrated) = 1
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
    forecast_test = model_fit.forecast(len(df))
    mae = mean_absolute_error(df, forecast_test)
    mape = mean_absolute_percentage_error(df, forecast_test)
    rmse = np.sqrt(mean_squared_error(df, forecast_test))
    
mae - Mean Absolute Error: 2721280.2643994256\
mape - Mean Absolute Percentage Error: 6.278220989175696\
rmse - Root-mean-square deviation: 2974174.424456362

    print(f'mae - Mean Absolute Error: {mae}')
    print(f'mape - Mean Absolute Percentage Error: {mape}')
    print(f'rmse - Root-mean-square deviation: {rmse}')
    print("")
    
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

# Total pax forecast
![Totalpax_graph.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Totalpax_graph.png)

# International pax forecast
![Intl_graph.png](https://github.com/Natetp/SARIMA-model.py/blob/main/Pax%20Graph/Intl_graph.png)



