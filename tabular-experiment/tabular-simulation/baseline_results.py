import pandas as pd
import numpy as np
import numpy.random as npr
import random
from generate_data import *

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def get_baseline_results(train, test, n_cat, seed):
    
    np.random.seed(seed)
    random.seed(seed)
    
    D = train.shape[1] - 1
    train_feat_oh = pd.get_dummies(train, columns = ['x'+str(i) for i in range(D)], drop_first=True)
    test_feat_oh = pd.get_dummies(test, columns = ['x'+str(i) for i in range(D)], drop_first=True)
    
    
    ## Logistic Regression
    lr = LogisticRegression(random_state = seed, max_iter = 2000).fit(train_feat_oh.iloc[:,1:], train_feat_oh['y'])
    mse_train_lr = ((lr.predict(train_feat_oh.iloc[:,1:]) - train['y'])**2).mean()
    mse_test_lr = ((lr.predict(test_feat_oh.iloc[:,1:]) - test['y'])**2).mean()
    acc_train_lr = (lr.predict(train_feat_oh.iloc[:,1:]) == train['y']).mean()
    acc_test_lr = (lr.predict(test_feat_oh.iloc[:,1:]) == test['y']).mean()
    
    
    ## Random Forest CV
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    clf = RandomForestClassifier(n_jobs = -1, random_state = seed)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'n_estimators': [50, 100, 200],
        'max_depth': [1, 3, None]
    }
    rf = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        scoring = 'accuracy',
        n_jobs = -1,
        cv = cv)
    rf.fit(train_feat_oh.iloc[:,1:], train_feat_oh['y'])
    mse_train_rf = ((rf.predict(train_feat_oh.iloc[:,1:]) - train['y'])**2).mean()
    mse_test_rf = ((rf.predict(test_feat_oh.iloc[:,1:]) - test['y'])**2).mean()
    acc_train_rf = (rf.predict(train_feat_oh.iloc[:,1:]) == train['y']).mean()
    acc_test_rf = (rf.predict(test_feat_oh.iloc[:,1:]) == test['y']).mean()

    
    ## Gradient Boosting CV
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    clf = GradientBoostingClassifier(random_state = seed)
    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [50, 100, 200],
        'max_depth': [1, 3, 5]
    }
    gb = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        scoring = 'accuracy',
        n_jobs = -1,
        cv = cv)
    gb.fit(train_feat_oh.iloc[:,1:], train_feat_oh['y'])
    mse_train_gb = ((gb.predict(train_feat_oh.iloc[:,1:]) - train['y'])**2).mean()
    mse_test_gb = ((gb.predict(test_feat_oh.iloc[:,1:]) - test['y'])**2).mean()
    acc_train_gb = (gb.predict(train_feat_oh.iloc[:,1:]) == train['y']).mean()
    acc_test_gb = (gb.predict(test_feat_oh.iloc[:,1:]) == test['y']).mean()

    ## MLP Classifier CV
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    clf = MLPClassifier(random_state = seed, max_iter = 2000, learning_rate_init = 0.01)
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    }
    mlp = GridSearchCV(
        estimator = clf,
        param_grid = param_grid,
        scoring = 'accuracy',
        n_jobs = -1,
        cv = cv)
    mlp.fit(train_feat_oh.iloc[:,1:], train_feat_oh['y'])
    mse_train_mlp = ((mlp.predict(train_feat_oh.iloc[:,1:]) - train['y'])**2).mean()
    mse_test_mlp = ((mlp.predict(test_feat_oh.iloc[:,1:]) - test['y'])**2).mean()
    acc_train_mlp = (mlp.predict(train_feat_oh.iloc[:,1:]) == train['y']).mean()
    acc_test_mlp = (mlp.predict(test_feat_oh.iloc[:,1:]) == test['y']).mean()
    
    return {'mse_train': [mse_train_lr, mse_train_rf, mse_train_gb, mse_train_mlp],
           'acc_train': [acc_train_lr, acc_train_rf, acc_train_gb, acc_train_mlp],
           'mse_test': [mse_test_lr, mse_test_rf, mse_test_gb, mse_test_mlp],
           'acc_test': [acc_test_lr, acc_test_rf, acc_test_gb, acc_test_mlp]}