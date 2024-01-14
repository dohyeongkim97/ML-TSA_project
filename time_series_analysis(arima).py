import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from sklearn.metrics import mutual_info_score
import statsmodels.tsa.seasonal 
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime
from datetime import datetime

import statsmodels.tsa.api as tsa
df = pd.read_csv("df2.csv")



import warnings
warnings.filterwarnings('ignore')

import itertools

indexes = []
df['period'] = pd.to_datetime(df['period'])
df = df.set_index('period')
indexes = []

for i in range(len(df.index)):
    if (i+1) % 7 == 0:
        indexes.append(df.index[i])
        print(i)
        
df_idx = df.loc[indexes, :]


test_result_catlist = []
test_result_paramlist = []
test_result_seasonalist = []
test_result_aic = []

for i in df_idx.columns:
    print(i)
    p = range(0, 2)
    d = range(0, 2)
    q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    
    seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
    
    param_list=[]
    param_seasonal_list = []
    results_aic_list = []
    
    for param in pdq:
        print(param)
        for param_seasonal in seasonal_pdq:
            try:
                mode = tsa.statespace.SARIMAX(df_idx[i][:300], order=param,
                                             seasonal_order = param_seasonal,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False,)
                results = mode.fit()
                param_list.append(param)
                param_seasonal_list.append(param_seasonal)
                results_aic_list.append(results.aic)
                print(param_seasonal)
                
            except:
                continue
                
    arima_dataset = pd.DataFrame({'Parameter': param_list, 'Seasonal': param_seasonal_list, 'AIC': results_aic_list})
    
    test_result_catlist.append(i)
    test_result_paramlist.append(arima_dataset.loc[arima_dataset['AIC'].idxmin(), 'Parameter'])
    test_result_seasonalist.append(arima_dataset.loc[arima_dataset['AIC'].idxmin(), 'Seasonal'])
    test_result_aic.append(arima_dataset['AIC'].min())
    
param_dataset = pd.DataFrame({'catlist': test_result_catlist, 'paramlist': test_result_paramlist,
                             'seasonal': test_result_seasonalist, 'aic': test_result_aic})

test_result_catlist_week = []
test_result_paramlist_week = []
test_result_seasonalist_week = []
test_result_aic_week = []

for i in df_idx.columns:
    print(i)
    p = range(0, 2)
    d = range(0, 2)
    q = range(0, 2)
    pdq_week = list(itertools.product(p, d, q))
    
    seasonal_pdq_week = [(x[0], x[1], x[2], 26) for x in list(itertools.product(p, d, q))]
    
    param_list_week=[]
    param_seasonal_list_week = []
    results_aic_list_week = []
    
    for param_week in pdq_week:
        print(param_week)
        for param_seasonal_week in seasonal_pdq_week:
            try:
                mode_week = tsa.statespace.SARIMAX(df_idx[i][:300], order=param_week,
                                             seasonal_order = param_seasonal_week,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False,)
                results_week = mode_week.fit()
                param_list_week.append(param_week)
                param_seasonal_list_week.append(param_seasonal_week)
                results_aic_list_week.append(results_week.aic)
                print(param_seasonal_week)
                
            except:
                continue
                
    arima_dataset_week = pd.DataFrame({'Parameter': param_list_week, 'Seasonal': param_seasonal_list_week, 'AIC': results_aic_list_week})
    
    test_result_catlist_week.append(i)
    test_result_paramlist_week.append(arima_dataset.loc[arima_dataset_week['AIC'].idxmin(), 'Parameter'])
    test_result_seasonalist_week.append(arima_dataset.loc[arima_dataset_week['AIC'].idxmin(), 'Seasonal'])
    test_result_aic_week.append(arima_dataset_week['AIC'].min())
    
param_dataset_week = pd.DataFrame({'catlist': test_result_catlist_week, 'paramlist': test_result_paramlist_week,
                             'seasonal': test_result_seasonalist_week, 'aic': test_result_aic_week})

test_result_catlist_week = []
test_result_paramlist_week = []
test_result_seasonalist_week = []
test_result_aic_week = []

for i in df_idx.columns:
    print(i)
    p = range(0, 2)
    d = range(0, 2)
    q = range(0, 2)
    pdq_week = list(itertools.product(p, d, q))
    
    seasonal_pdq_week = [(x[0], x[1], x[2], 26) for x in list(itertools.product(p, d, q))]
    
    param_list_week=[]
    param_seasonal_list_week = []
    results_aic_list_week = []
    
    for param_week in pdq_week:
        print(param_week)
        for param_seasonal_week in seasonal_pdq_week:
            try:
                mode_week = tsa.statespace.SARIMAX(df_idx[i][:300], order=param_week,
                                             seasonal_order = param_seasonal_week,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False,)
                results_week = mode_week.fit()
                param_list_week.append(param_week)
                param_seasonal_list_week.append(param_seasonal_week)
                results_aic_list_week.append(results_week.aic)
                print(param_seasonal_week)
                
            except:
                continue
                
    arima_dataset_week = pd.DataFrame({'Parameter': param_list_week, 'Seasonal': param_seasonal_list_week, 'AIC': results_aic_list_week})
    
    test_result_catlist_week.append(i)
    test_result_paramlist_week.append(arima_dataset.loc[arima_dataset_week['AIC'].idxmin(), 'Parameter'])
    test_result_seasonalist_week.append(arima_dataset.loc[arima_dataset_week['AIC'].idxmin(), 'Seasonal'])
    test_result_aic_week.append(arima_dataset_week['AIC'].min())
    
param_dataset_week = pd.DataFrame({'catlist': test_result_catlist_week, 'paramlist': test_result_paramlist_week,
                             'seasonal': test_result_seasonalist_week, 'aic': test_result_aic_week})

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error

cat_name = []
annual_forecast = []
halfyear_forecast = []
weight = []
annual_mean = []
halfyear_mean = []


for cat in range(len(param_dataset)):
    print(cat)
    pred_annual = []
    pred_halfyear = []
    weight_of_pred = []
    mean_annual = []
    mean_halfyear = []
    
    scorelist = []
    
    best_mode = tsa.statespace.SARIMAX(df_idx[param_dataset.loc[cat, 'catlist']][:300],
                                      order = param_dataset.loc[cat, 'paramlist'],
                                       seasonal_order = param_dataset.loc[cat, 'seasonal'],
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
    
    best_result = best_mode.fit()
    forecast = best_result.forecast(30)
    
    best_mode_half = tsa.statespace.SARIMAX(df_idx[param_dataset_week.loc[cat, 'catlist']][:300],
                                           order = param_dataset.loc[cat, 'paramlist'],
                                            seasonal_order = param_dataset.loc[cat, 'seasonal'],
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
    best_result_half = best_mode_half.fit()
    forecast_half = best_result_half.forecast(30)
    
    target = df_idx[param_dataset.loc[cat, 'catlist']][300:]
    
    for i in range(0, 11):
        print(cat, 'annual_weight:', i)
        for j in range(0, 11):
            print(cat, 'halfyear_weight:', j)
            for k in range(0, 11):
                for l in range(0, 11):
                    for m in range(0, 11):
                        train_data = sdc.fit_transform(np.array(forecast * i/10 + forecast_half * j/10).reshape(-1, 1))*k + np.mean(forecast * l/10 + forecast_half*m/10)
                        
                        pred_annual.append(i)
                        pred_halfyear.append(j)
                        weight_of_pred.append(k)
                        mean_annual.append(l)
                        mean_halfyear.append(m)
                        
                        score = mae(train_data, target)
                        scorelist.append(score)
                        
                        
    parameter_dataset = pd.DataFrame({'annual_weight': pred_annual, 'halfyear_weight': pred_halfyear, 'weight_of_pred': weight_of_pred,
                         'mean_annual': mean_annual, 'mean_halfyear': mean_halfyear, 'scorelist': scorelist})
    
    cat_name.append(param_dataset.loc[cat, 'catlist'])
    annual_forecast.append(parameter_dataset.loc[parameter_dataset['scorelist'].idxmin(), 'annual_weight'])
    halfyear_forecast.append(parameter_dataset.loc[parameter_dataset['scorelist'].idxmin(), 'halfyear_weight'])
    weight.append(parameter_dataset.loc[parameter_dataset['scorelist'].idxmin(), 'weight_of_pred'])
    annual_mean.append(parameter_dataset.loc[parameter_dataset['scorelist'].idxmin(), 'mean_annual'])
    halfyear_mean.append(parameter_dataset.loc[parameter_dataset['scorelist'].idxmin(), 'mean_halfyear'])