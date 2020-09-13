"""
Project:    CSIS4260
            Fall 2019
Instructor: Nikhil Bhardwaj
Student:    Jose Luiz Mattos Gomes 
            #300291877
File:       ProjectJLMG_CSIS4260_Defs.py

"""

## Import libraries
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from math import sqrt

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.api as sm
import itertools


# functions

# Verify is value is NaN
def is_nan(x):
    return (x is np.nan or x != x)


# Draw chart
def plotChart(train_set, test_set, y_hat, title, yLabel):
    plt.figure(figsize=(12,8))
    plt.plot(train_set.index, train_set['Sales'], label='train_set')
    plt.plot(test_set.index,test_set['Sales'], label='test_set')
    plt.plot(y_hat.index,y_hat[yLabel], label=title)
    plt.legend(loc='best')
    plt.title(title)
    plt.show()


# split data into training (80%) and test (20%)
def splitData(df):

    # copy dataframe
    df_copy = df.copy()

    # get data size
    dsIndex = int(len(df)*0.8)

    # train dataset 80%
    train_set = df_copy[0:dsIndex]
    if len(train_set) == 0:
        train_set = df.copy()
    # print('Training size (80%): '+ str(len(train_set)))

    # test dataset 20%
    test_set = df_copy.drop(train_set.index)
    if len(test_set) == 0:
        test_set = df.copy()
    # print('Test size (20%): ' + str(len(test_set)))

    # return train and test dataframes
    return (train_set, test_set)


# Naive
def doNaive(train_set, test_set, predict_set):
    print('>Naive')
  
    # copy test df
    y_hat = test_set.copy()

    # get last value
    last_value = train_set.iloc[-1, 0]
    y_hat['naive'] = last_value
    
    # do prediction
    predict_set['FutureValue'] = last_value

    # calculate error
    rms = sqrt(mean_squared_error(test_set.Sales, y_hat.naive))

    # plot chart
    #plotChart(train_set, test_set, y_hat, 'Naive Forecast', 'naive')

    # return dataframes: error and prediction
    return(rms, predict_set)


# Simple Mean
def doSimpleMean(train_set, test_set, predict_set):
    print('>Simple Mean')

    # copy test df
    y_hat_avg = test_set.copy()

    # calculate the mean
    y_hat_avg['Sales'] = train_set['Sales'].mean()

    # do prediction
    predict_set['FutureValue'] = train_set['Sales'].mean()
    
    # calculate error
    rms = sqrt(mean_squared_error(test_set.Sales, y_hat_avg.Sales))

    # plot chart
    #plotChart(train_set, test_set, y_hat_avg, 'Average Forecast', 'Sales')

    # return dataframes: error and prediction
    return (rms, predict_set)
 


# Moving Average N periods
def doMovingAvgN(train_set, test_set, predict_set, n):
    print('>Moving Average ' + str(n) )

    # set window size
    n_window = n
    if len(train_set)< n:
        n_window = len(train_set)

    # copy last window of test df
    df_aux = train_set.copy()[-n_window:]

    # calculate the mean
    v_mean = df_aux['Sales'].mean()

    # copy teste df
    y_hat_avg = test_set.copy()

    # find first position of y_hat_avg
    v_aux = 0
    label = y_hat_avg.index.values[v_aux]

    # set value = mean
    y_hat_avg.at[label, 'Sales']=v_mean

    # update pointer
    v_aux += 1

    # do calculation for the remaining values of train df 
    while v_aux < n_window and v_aux < len(y_hat_avg):
        df_aux = train_set.copy()[v_aux - n_window :]
        df_aux = df_aux.append(y_hat_avg.copy()[:v_aux])
        v_mean = df_aux['Sales'].mean()
        label = y_hat_avg.index.values[v_aux]
        y_hat_avg.at[label, 'Sales']=v_mean
        v_aux += 1

    # do calculation for the remaining values of test df
    while v_aux < len(y_hat_avg):
        df_aux = y_hat_avg.copy()[v_aux - n_window : v_aux]
        v_mean = df_aux['Sales'].mean()
        label = y_hat_avg.index.values[v_aux]
        y_hat_avg.at[label, 'Sales']=v_mean
        v_aux += 1

    # calculate error
    rms = sqrt(mean_squared_error(test_set.Sales, y_hat_avg.Sales))

    # do calculations for predict dataframe
    n_window = n
    if len(test_set)< n:
        n_window = len(test_set)

    #copy test_set to df_aux
    df_aux = test_set.copy()[-n_window:]

    #calculate mean
    v_mean = df_aux['Sales'].mean()

    #reset counter
    v_aux = 0

    #copy prdict_set to pred_df
    pred_df = predict_set.copy()

    #drop Store and Dept from index
    pred_df.index = pred_df.index.droplevel((0,1))

    #rename FutureValue to Sales and set zero to Sales on pred_df
    pred_df.rename(columns={'FutureValue':'Sales'}, inplace=True)
    pred_df['Sales'] = 0

    #find v_aux pos in index in pred_df
    label = pred_df.index.values[v_aux]

    #set Sales = calcualted mean on the position found
    pred_df.at[label, 'Sales']=v_mean
    v_aux += 1
    
    # do calculation for the remaining values of test df
    while v_aux < n_window and v_aux < len(pred_df):
        df_aux = test_set.copy()[v_aux - n_window :]
        df_aux = df_aux.append(pred_df.copy()[:v_aux])
        v_mean = df_aux['Sales'].mean()
        label = pred_df.index.values[v_aux]
        pred_df.at[label, 'Sales']=v_mean
        v_aux += 1

    # do calculation for the remaining values of pred_df df
    while v_aux < len(pred_df):
        df_aux = pred_df.copy()[v_aux - n_window : v_aux]
        v_mean = df_aux['Sales'].mean()
        label = pred_df.index.values[v_aux]
        pred_df.at[label, 'Sales']=v_mean
        v_aux += 1


    # set index
    pred_df['Store'] = predict_set.index.get_level_values('Store')
    pred_df.set_index('Store', append=True, inplace=True)
    
    pred_df['Dept'] = predict_set.index.get_level_values('Dept')
    pred_df.set_index('Dept', append=True, inplace=True)
    
    pred_df = pred_df.reorder_levels(['Store', 'Dept', 'Date'])

    # update final predict dataframe
    predict_set['FutureValue']= pred_df['Sales']

    # plot chart
    #plotChart(train_set, test_set, y_hat_avg, 'Moving Average ' + str(n), 'Sales')

    # return dataframes: error and prediction
    return (rms, predict_set)    


# Holts Linear
def doHoltsLinear(train_set, test_set, predict_set):
    print('>Holts Linear')

    try:

        # copy test dataframe dates
        y_hat_avg = pd.DataFrame(index=test_set.index.copy())

        # fit model
        fit1 = Holt(np.asarray(train_set['Sales'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)

        # predict test dataframe
        y_hat_avg['Sales'] = fit1.forecast(len(test_set))

        # calculate error
        rms = sqrt(mean_squared_error(test_set.Sales, y_hat_avg.Sales))

        # create final predict dataframe
        predict_set['FutureValue'] = fit1.forecast(len(predict_set))
        
        # plot chart        
        #plotChart(train_set, test_set, y_hat_avg, 'Holt_linear', 'Sales')

    except:
        rms = 999999999

    # return dataframes: error and prediction
    return (rms, predict_set) 


# Holt Winter / Exponential Smoothing
def doHoltWinter(train_set, test_set, predict_set):
    print('>Holt Winters')

    try:
    
        # define window for sazonality
        n_window = 12
        if len(train_set)< 12:
            n_window = len(train_set)        

        # copy test dataframe dates
        y_hat_avg = pd.DataFrame(index=test_set.index.copy())

        # fit model
        fit1 = ExponentialSmoothing(np.asarray(train_set['Sales']) ,seasonal_periods=n_window ,trend='add', seasonal='add',).fit()

        # predict test dataframe
        y_hat_avg['Sales'] = fit1.forecast(len(test_set))

        # calculate error
        rms = sqrt(mean_squared_error(test_set.Sales, y_hat_avg.Sales))

        # create final predict dataframe
        predict_set['FutureValue'] = fit1.forecast(len(predict_set))
        
        # plot chart
        #plotChart(train_set, test_set, y_hat_avg, 'Holt_Winter', 'Sales')
        
    except:
        rms = 999999999

    # return dataframes: error and prediction
    return (rms, predict_set)


# ARIMA
def doArima(train_set, test_set, predict_set):
    print('>Arima')

    try:

        # copy test dataframe dates
        y_hat_avg = pd.DataFrame(index=test_set.index.copy())

        # copy predict dataframe dates
        predict_idx = pd.DataFrame(index=predict_set.index.levels[2].copy())

        # fit model
        fit1 = sm.tsa.statespace.SARIMAX(train_set.Sales,order=(0, 0, 1),seasonal_order=(0, 1, 1, 12)).fit(disp=0)

        # predict test dataframe
        y_hat_avg['Sales'] = fit1.predict(start=test_set.index.min(), end=test_set.index.max(), dynamic=True)

        # calculate error
        rms = sqrt(mean_squared_error(test_set.Sales, y_hat_avg.Sales))

        # create final predict dataframe
        predict_set['FutureValue'] = fit1.predict(start=predict_idx.index.min(), end=predict_idx.index.max(), dynamic=True)

        # plot chart
        #plotChart(train_set, test_set, y_hat_avg, 'Arima', 'Sales')

    except:
        rms = 999999999

    # return dataframes: error and prediction
    return (rms, predict_set) 


# Multiple Linear Regression
def doLinearRegression(df_X, df_Y, df_P, v_dept):
    print('>Multiple Linear Regression')
    
    try:
  
        # split data into training (80%) and test (20%)
        (train_set_X, test_set_X) = splitData(df_X)
        (train_set_Y, test_set_Y) = splitData(df_Y)
        
        # set model
        lin_reg_mod = linear_model.LinearRegression()
        lin_reg_mod.fit(train_set_X, train_set_Y)

        # do prediction
        dd = lin_reg_mod.predict(test_set_X)
    
        # create dataFrafe objects
        df_temp = pd.DataFrame({'Actual': test_set_Y, 'Sales': dd})
        y_hat = df_temp.drop(columns=['Actual'])
        df_P['FutureValue'] = lin_reg_mod.predict(df_P)
        
        # calculate error
        rms = sqrt(mean_squared_error(test_set_Y, y_hat.Sales))
        
        # plot chart
        #     y_test = test_set_Y.to_frame()
        #     y_train = train_set_Y.to_frame()
        #     plotChart(y_train, y_test, y_hat, 'MLR Forecast', 'Sales')

    except:
        rms = 999999999

    # return dataframes: error and prediction
    return(rms, df_P[['FutureValue']])
    


# Update Final Prediction
def updateFinalPrediction(df_pred, df_temp):

    # remove any negative value
    df_temp['FutureValue'] = df_temp['FutureValue'].apply(lambda x: 0 if x < 0 else x)
 
    # merge dataframes
    df_pred = pd.merge(df_pred, df_temp, on=list(['Store','Dept', 'Date']), how='outer', suffixes=('_f','_t'))
    df_pred['FutureValue'] = df_pred['FutureValue_f'].where(df_pred['FutureValue_t'].isnull(),df_pred['FutureValue_t'])
    df_pred.drop(['FutureValue_f','FutureValue_t'], axis=1, inplace=True)

    # return dataframe updated
    return(df_pred)
