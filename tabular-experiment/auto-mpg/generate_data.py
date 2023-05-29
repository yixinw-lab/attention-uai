import pandas as pd
import numpy as np
import numpy.random as npr

def generate_data(N, D, num_corefea, corr, x_noise, y_noise, n_cat, seed, shift = True):
    
    ##### Parameters:
    ## N = number of observations in training (test) set
    ## D = dimension of covariates
    ## num_corefea = number of covariates that actually generate the data
    ## corr = correlation parameter
    ## x_noise = amount of noise in the covariates
    ## y_noise = amount of noise in the response
    ## n_cat = number of categories
    ## seed = random seed
    ## shift = whether there is a covariate shift between training and test
    
    np.random.seed(seed)
   
    mean = np.zeros(D)
    true_cause = np.arange(num_corefea).astype(int)
    train_cov = np.ones((D, D)) * corr + np.eye(D) * (1 - corr)
    train_x_true = npr.multivariate_normal(mean, train_cov, size=N)

    # create both positive and negatively correlated covariates
    train_x_true = train_x_true * np.concatenate([-1 * np.ones(D//2), np.ones(D - D//2)]) 

    # simulate different correlation patterns for testing
    if shift:
        test_cov = np.ones((D, D)) * (1 - corr) + np.eye(D) * corr
    else:
        test_cov = train_cov

    test_x_true = npr.multivariate_normal(mean, test_cov, size=N)
    test_x_true = test_x_true * np.concatenate([-1 * np.ones(D//2), np.ones(D - D//2)]) 

    # add observation noise to the x
    # spurious correlation more often occurs when the signal to noise ratio is lower
    x_noise = np.array(list(np.ones(num_corefea)*0.4) + list(np.ones(D-num_corefea)*0.3)) * x_noise

    train_x = train_x_true + x_noise * npr.normal(size=[N,D])
    test_x = test_x_true + x_noise * npr.normal(size=[N,D])

    # generate outcome
    # toy model y = x + noise
    truecoeff = npr.uniform(size=num_corefea) * 10
    train_y = train_x_true[:,true_cause].dot(truecoeff) + y_noise * npr.normal(size=N)
    test_y = test_x_true[:,true_cause].dot(truecoeff) + y_noise * npr.normal(size=N)

    cov_train = np.column_stack([train_x])
    cov_test = np.column_stack([test_x])

    train = pd.DataFrame(cov_train)
    train.columns = ['x' + str(i) for i in range(D)]
    train['y'] = train_y

    test = pd.DataFrame(cov_test)
    test.columns = ['x' + str(i) for i in range(D)]
    test['y'] = test_y

    data = pd.concat([train, test]).reset_index(drop=True)

    for i in range(D):
        data['x' + str(i)] = pd.qcut(data['x' + str(i)], n_cat, labels=False)

    data['y'] = pd.qcut(data['y'], n_cat, labels=False)

    train = data.iloc[0:N].reset_index(drop=True)
    test = data.iloc[N:2*N].reset_index(drop=True)
    
    return train, test