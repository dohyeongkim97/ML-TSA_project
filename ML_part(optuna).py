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

for i in range(len(train_df)):
    if (i+1)%7 == 0:
        continue
    else:
        train_df.drop(index = i, inplace=True)
        
train_df = train_df.reset_index(drop=True)
train_data = train_df[:300]
target_data = train_df[300:]
train_features = df[['고용률', '물가', '평균기온(°C)', '생활물가총지수']]
train_features = train_features.rolling(window=7).mean()


for i in range(len(train_features)):
    if (i+1)%7 == 0:
        continue
    else:
        train_features.drop(index = i, inplace=True)
        
train_features = train_features.reset_index(drop=True)
train_feat = train_features[:300]
target_feat = train_features[300:]

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost
from xgboost import XGBRegressor

import lightgbm
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor

import pycaret
from pycaret import classification

import optuna
from optuna import Trial

import random
from random import randint

def random_number(start, end, term=1):
    multiples_of_term = [num for num in range(start, end) if num % term == 0]
    selected = random.choice(multiples_of_term)
    
    return selected

target_feat = target_feat.reset_index(drop=True)
target_data = target_data.reset_index(drop=True)

def objective_rf(trial):
    n_estimator_param = trial.suggest_categorical('n_estimators', [100, 200, 500])
    max_depth_param = trial.suggest_int('max_depth', 5, 40, 5)

    model = RandomForestRegressor(n_estimators = n_estimator_param,
                             max_depth = max_depth_param,
                             random_state=42)

    model.fit(xtr, ytr)
    preds = model.predict(xte)
    score = np.sqrt(mean_squared_error(preds, yte))

    return score

optimized_rf = pd.DataFrame()

for i in range(len(train_data.columns)):
    ytr = pd.DataFrame(train_data[train_data.columns[i]])
    xtr = pd.DataFrame(train_feat)
    yte = pd.DataFrame(target_data[target_data.columns[i]])
    xte = pd.DataFrame(target_feat)
    
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective_rf, n_trials=30)
    
    optimized_rf.loc[train_data.columns[i], 'rf_n_estimator'] = study.best_trial.params['n_estimators']
    optimized_rf.loc[train_data.columns[i], 'rf_max_depth'] = study.best_trial.params['max_depth']
    optimized_rf.loc[train_data.columns[i], 'best_score'] = study.best_value
    
    optimized_mlp = pd.DataFrame()

def objective_mlp(trial):
    
    hidden_layer_size = trial.suggest_categorical('hidden_layer_sizes', [(100,),(50,100,),(25,50,100,50,),(300,),(150,300,),
                                                                        (75,150,300,150,),(400,),(200,400,),
                                                                        (100,200,400,200,),(700,),(350,700,),(175,350,700,350,)])
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
    learning_rate_init = trial.suggest_categorical('learning_rate_init', [0.01, 0.001, 0.0001])
    tol = trial.suggest_categorical('tol', [1e-3, 1e-4, 1e-5])
    alpha = trial.suggest_categorical('alpha', [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    model = MLPRegressor(hidden_layer_sizes = hidden_layer_size,
                        activation = activation,
                        learning_rate_init = learning_rate_init,
                        tol = tol,
                        alpha = alpha)
    
    model.fit(xtr, ytr)
    preds = model.predict(xte)
    score = np.sqrt(mean_squared_error(preds, yte))

    return score

for i in range(len(train_data.columns)):
    ytr = pd.DataFrame(train_data[train_data.columns[i]])
    xtr = pd.DataFrame(train_feat)
    yte = pd.DataFrame(target_data[target_data.columns[i]])
    xte = pd.DataFrame(target_feat)
    
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective_mlp, n_trials=120)
    
#     optimized_rf.loc[train_data.columns[i], 'n_estimator'] = study.best_trial.params['n_estimators']

    optimized_mlp.loc[train_data.columns[i], 'hidden_layer_sizes'] = str(study.best_trial.params['hidden_layer_sizes'])
    optimized_mlp.loc[train_data.columns[i], 'activation'] = study.best_trial.params['activation']
    optimized_mlp.loc[train_data.columns[i], 'learning_rate_init'] = study.best_trial.params['learning_rate_init']
    optimized_mlp.loc[train_data.columns[i], 'tol'] = study.best_trial.params['tol']
    optimized_mlp.loc[train_data.columns[i], 'alpha'] = study.best_trial.params['alpha']
    
    optimized_mlp.loc[train_data.columns[i], 'best_score'] = study.best_value
    
    optimized_xgb = pd.DataFrame()

def objective_xgb(trial):
    
    n_estimator = trial.suggest_categorical('n_estimators', [100, 200, 500])
    max_depth = trial.suggest_int('max_depth', 5, 40, 5)
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.02, 0.05, 0.1, 0.15])
    reg_alpha = trial.suggest_categorical('reg_alpha', [0, 0.2, 0.4, 0.6, 0.8, 1])
    gamma = trial.suggest_categorical('gamma', [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    model = XGBRegressor(n_estimators = n_estimator,
                        max_depth = max_depth,
                        learning_rate = learning_rate,
                        reg_alpha = reg_alpha,
                        gamma = gamma)
    
    model.fit(xtr, ytr)
    preds = model.predict(xte)
    score = np.sqrt(mean_squared_error(preds, yte))

    return score

for i in range(len(train_data.columns)):
    ytr = pd.DataFrame(train_data[train_data.columns[i]])
    xtr = pd.DataFrame(train_feat)
    yte = pd.DataFrame(target_data[target_data.columns[i]])
    xte = pd.DataFrame(target_feat)
    
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective_xgb, n_trials=150)
    
#     optimized_rf.loc[train_data.columns[i], 'n_estimator'] = study.best_trial.params['n_estimators']

    optimized_xgb.loc[train_data.columns[i], 'n_estimators'] = str(study.best_trial.params['n_estimators'])
    optimized_xgb.loc[train_data.columns[i], 'max_depth'] = study.best_trial.params['max_depth']
    optimized_xgb.loc[train_data.columns[i], 'learning_rate'] = study.best_trial.params['learning_rate']
    optimized_xgb.loc[train_data.columns[i], 'reg_alpha'] = study.best_trial.params['reg_alpha']
    optimized_xgb.loc[train_data.columns[i], 'gamma'] = study.best_trial.params['gamma']
    
    optimized_xgb.loc[train_data.columns[i], 'best_score'] = study.best_value
    
    optimized_lgbm = pd.DataFrame()

def objective_lgbm(trial):
    
    n_estimator = trial.suggest_categorical('n_estimators', [100, 200, 500])
    max_depth = trial.suggest_int('max_depth', 5, 40, 5)
    learning_rate = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.02, 0.05, 0.1, 0.15])
    reg_alpha = trial.suggest_categorical('reg_alpha', [0, 0.2, 0.4, 0.6, 0.8, 1])
#     gamma = trial.suggest_categorical('gamma', [0, 0.2, 0.4, 0.6, 0.8, 1])
    
    model = LGBMRegressor(n_estimators = n_estimator,
                        max_depth = max_depth,
                        learning_rate = learning_rate,
#                         reg_alpha = reg_alpha,
                        gamma = gamma)
    
    model.fit(xtr, ytr)
    preds = model.predict(xte)
    score = np.sqrt(mean_squared_error(preds, yte))

    return score

for i in range(len(train_data.columns)):
    ytr = pd.DataFrame(train_data[train_data.columns[i]])
    xtr = pd.DataFrame(train_feat)
    yte = pd.DataFrame(target_data[target_data.columns[i]])
    xte = pd.DataFrame(target_feat)
    
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective_xgb, n_trials=120)
    
#     optimized_rf.loc[train_data.columns[i], 'n_estimator'] = study.best_trial.params['n_estimators']

    optimized_lgbm.loc[train_data.columns[i], 'n_estimators'] = str(study.best_trial.params['n_estimators'])
    optimized_lgbm.loc[train_data.columns[i], 'max_depth'] = study.best_trial.params['max_depth']
    optimized_lgbm.loc[train_data.columns[i], 'learning_rate'] = study.best_trial.params['learning_rate']
    optimized_lgbm.loc[train_data.columns[i], 'reg_alpha'] = study.best_trial.params['reg_alpha']
#     optimized_lgbm.loc[train_data.columns[i], 'gamma'] = study.best_trial.params['gamma']
    
    optimized_lgbm.loc[train_data.columns[i], 'best_score'] = study.best_value
    
optimized_lgbm.columns = ['n_estimators_lgbm', 'max_depth_lgbm', 'learning_rate_lgbm', 'reg_alpha_lgbm', 'best_score_lgbm']
optimized_xgb.columns = ['n_estimators_xgb', 'max_depth_xgb', 'learning_rate_xgb', 'reg_alpha_xgb', 'gamma_xgb', 'best_score_xgb']
optimized_mlp.columns = ['hidden_layer_sizes_mlp', 'activation_mlp', 'learning_rate_init_mlp', 'tol_mlp', 'alpha_mlp', 'best_score_mlp']
optimized_rf.columns = ['rf_n_estimator', 'rf_max_depth', 'best_score_rf']
optimized_df = pd.concat([optimized_rf, optimized_mlp, optimized_xgb, optimized_lgbm], axis=1)

best_scores = optimized_df[['best_score_rf', 'best_score_mlp', 'best_score_xgb', 'best_score_lgbm']]
best_scores.T['남성골프웨어'].idxmin()
best_scores.T.idxmin()

optimized_df.to_csv("ML_df.csv")
optimized_mlp.to_csv("MLP.csv")