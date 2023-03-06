import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import cohen_kappa_score, mean_squared_error
import xgboost as xgb 
import lightgbm as lgbm
from HROCH import PHCRegressor
import optuna
from optuna.samplers import TPESampler
import datetime
import pickle
import json
import os

# Suppressing warnings
import warnings
warnings.simplefilter("ignore")

# To be configured
target = "Strength"
target_type = "regression"

# Loading the data
train = pd.read_csv("data/processed/train/train.csv")

# Setting the target and feature variables
features = [col for col in train.columns if col != target]
n_classes = len(train[target].unique())

def objective_lgbm(trial):
    params_lgbm = {}
    if target_type == "classification":
        params_lgbm = {
            'objective' : "multiclass",
            'metric' :'multi_logloss',
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.0, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'num_iterations': 10000
        }
    elif target_type == "regression":
        params_lgbm = {
            'objective': 'regression',
            'metric': 'rmse',
            'feature_pre_filter': False,
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.0, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'num_iterations': 10000,
            'early_stopping_round': 100
        }
    else:
        raise Exception("target_type is not recognized")
        
    cv = KFold(5, shuffle=True, random_state=42)

    fold_scores = []
    for i, (train_idx, val_idx) in enumerate(cv.split(train[features], train[target])):
        X_train, y_train = train.loc[train_idx, features],train.loc[train_idx, target]
        X_val, y_val = train.loc[val_idx, features],train.loc[val_idx, target]

        model = None
        if target_type == "classification":
            model = lgbm.LGBMClassifier(**params_lgbm)
        elif target_type == "regression":
            model = lgbm.LGBMRegressor(**params_lgbm)
        else:
            raise Exception("target_type is not recognized")
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50,
                  verbose=500)

        pred_val = model.predict(X_val)

        score = np.sqrt(mean_squared_error(y_val, pred_val))

        fold_scores.append(score)
    return np.mean(fold_scores)

study = optuna.create_study(direction='minimize', sampler = TPESampler())
study.optimize(func=objective_lgbm, n_trials=1)

model = None
if target_type == "classification":
    model = lgbm.LGBMClassifier(**study.best_params)
elif target_type == "regression":
    model = lgbm.LGBMRegressor(**study.best_params)
else:
    raise Exception("target_type is not recognized")

model.fit(train.loc[:, features],
                 train.loc[:, target])

date_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
file_path = "models/lgbm/" + date_time_str
os.mkdir(file_path)

with open(file_path + '/model.bin', 'wb') as f:
    pickle.dump(model, f)
    
with open(file_path + "/params.json", 'w') as f:
    json.dump(study.best_params, f)