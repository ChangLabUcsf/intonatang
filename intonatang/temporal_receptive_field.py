"""Code for fitting temporal receptive models using Ridge (L2 regularized) regression.

Given two ndarrays, stim and resp, there are two ways to fit trfs in terms of the cross validation strategy.
Both CV strategies split the data into three mutually exclusive sets, training, validation (ridge), and test.
The regression weights are fit to the training data. The best ridge parameter is found by testing on the ridge
set. The final model performance (correlation between actual and predicted response) is calculated from the
test set. 

The two strategies are:

1. Simple KFold: run_cv_temporal_ridge_regression_model
    The total samples of the data are split into a (K-1)/K train set, 1/2K ridge set, and 1/2K test set
    K times.

2. User-defined: run_cv_temporal_ridge_regression_model_fold
    Use this function when you want to specify your own train, ridge, and test sets (e.g. I use this to
    make sure training sets have TIMIT sentences with low-to-high pitch variability so that I don't end up
    with a training set that is only low pitch variability or only high pitch variability)
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import sklearn.model_selection as model_selection

def get_alphas(start=2, stop=7, num=10):
    """Returns alphas from num^start to num^stop in log space.
    """
    return np.logspace(start, stop, num)

def get_delays(delay_seconds=0.4, fs=100):
    """Returns 1d array of delays for a given window (in s). Default sampling frequency (fs) is 100Hz.
    """
    return np.arange(np.floor(delay_seconds * fs), dtype=int)

def get_dstim(stim, delays=get_delays(), add_edges=True):
    """Returns stimulus features with given delays.

    Args:
        stim: (n_samples, n_features)
        delays: list of delays to use, values in delays have units of indices for stim.
        add_edges: adds 3 additional delays to both sides of the delays list to account for edge effects in temporal
            receptive fields.

    Returns:
        dstim (ndarray): (n_samples, n_features x n_delays (including edge delays if added))
    """
    n_samples, n_features = stim.shape
    if add_edges:
        step = delays[1] - delays[0]
        delays_beg = [delays[0]-3*step, delays[0]-2*step, delays[0]-step]
        delays_end = [delays[-1]+step, delays[-1]+2*step, delays[-1]+3*step]
        delays = np.concatenate([delays_beg, delays, delays_end])
    dstim = []
    for i, d in enumerate(delays):
        dstim_slice = np.zeros((n_samples, n_features))
        if d<0:
            dstim_slice[:d, :] = stim[-d:, :]
        elif d>0:
            dstim_slice[d:, :] = stim[:-d, :]
        else:
            dstim_slice = stim.copy()
        dstim.append(dstim_slice)

    dstim = np.hstack(dstim)
    return dstim

def run_cv_temporal_ridge_regression_model(stim, resp, delays=get_delays(), alphas=get_alphas(), n_folds=5):
    """Given stim and resp, fit temporal receptive fields using ridge regression and KFold cross validation.

    Args:
        stim: (n_samples, n_features)
        resp: (n_samples, n_chans)
        delays: (n_delays)
        alphas: (n_alphas)
        n_folds (int): number of folds to use for KFold cross validation. The 1/K fraction of data usually used
            for the test set is split in half for the ridge parameter validation set and the test set.

    Returns:
        (tuple)
            * **test_corr_folds** (*ndarray*): Correlation between predicted and actual responses on
                test set using wts computed for alpha with best performance on validation set. 
                Shape of test_corr_folds is (n_folds, n_chans)
            * **wts_folds** (*ndarray*): Computed regression weights. Shape of wts_folds is 
                (n_folds, n_features, n_chans)
    """
    dstim = get_dstim(stim, delays)

    n_features = dstim.shape[1]
    n_chans = resp.shape[1]

    test_corr_folds = np.zeros((n_folds, n_chans))
    wts_folds = np.zeros((n_folds, n_features, n_chans))
    best_alphas = np.zeros((n_folds, n_chans))

    kf = model_selection.KFold(n_splits=n_folds)

    for i, (train, test) in enumerate(kf.split(dstims)):
        print('Running fold ' + str(i) + ".", end=" ")

        train_stim = dstim[train, :]
        train_resp = resp[train, :]

        #Use half of the test set returned by KFold for validation and half for test.
        ridge_stim = dstim[test[:round(len(test)/2)], :]
        ridge_resp = resp[test[:round(len(test)/2)], :]
        test_stim = dstim[test[round(len(test)/2):], :]
        test_resp = resp[test[round(len(test)/2):], :]

        wts_alphas, ridge_corrs = run_ridge_regression(train_stim, train_resp, ridge_stim, ridge_resp, alphas)
        best_alphas[i, :] = ridge_corrs.argmax(0) #returns array with length nchans. 

        #For each chan, see which alpha did the best on the validation and choose the wts for that alpha
        best_wts = [wts_alphas[best_alphas[i, chan], :, chan] for chan in range(n_chans)]
        test_pred = [np.dot(test_stim, best_wts[chan]) for chan in range(n_chans)]
        test_corr = np.array([np.corrcoef(test_pred[chan], test_resp[:, chan])[0,1] for chan in range(resp.shape[1])])
        test_corr[np.isnan(test_corr)] = 0

        test_corr_folds[i, :] = test_corr
        wts_folds[i, :, :] = np.array(best_wts).T

    return test_corr_folds, wts_folds

def run_cv_temporal_ridge_regression_model_fold(stims, resps, delays=get_delays(), alphas=get_alphas()):
    """Fit trf models with user-given split of data into training, validation, and test.

    Args:
        stims (list): list of stim data split into training, validation, and test
            i.e. [train_stim, ridge_stim, test_stim] where train_stim is (n_training_samples x n_features)
        resps (list): list of resp data split into training, validation, and test.
            The number of samples in each set should match that for the stims.

    Returns:
        (tuple)
            * **test_corr** (*ndarray*): Correlation between predicted and actual responses on
                test set using wts computed for alpha with best performance on validation set. 
                Shape of test_corr is (n_chans)
            * **wts** (*ndarray*): Computed regression weights using best alpha from testing on 
                validation set. Shape of wts is (n_chans, n_features)

    """
    dstims = [get_dstim(stim, delays) for stim in stims]
    n_chans = resps[0].shape[1]

    wts_alphas, ridge_corrs_alphas = run_ridge_regression(dstims[0], resps[0], dstims[1], resps[1], alphas)
    best_alphas = ridge_corrs_alphas.argmax(0) #returns array with length nchans.
    best_wts = [wts_alphas[best_alphas[chan], :, chan] for chan in range(n_chans)]
    test_pred = [np.dot(dstims[2], best_wts[chan]) for chan in range(n_chans)]
    test_corr = np.array([np.corrcoef(test_pred[chan], resps[2][:, chan])[0,1] for chan in range(n_chans)])
    test_corr[np.isnan(test_corr)] = 0

    wts = np.array(best_wts)
    return test_corr, wts

def run_ridge_regression(train_stim, train_resp, ridge_stim, ridge_resp, alphas):
    """Runs ridge (L2 regularized) regression for ridge parameters in alphas and returns wts fit
    on training data and correlation between actual and predicted on validation data for each alpha.

    Args:
        train_stim: (n_training_samples x n_features)
        train_resp: (n_training_samples x n_chans)
        ridge_stim: (n_validation_samples x n_features)
        ridge_resp: (n_validation_samples x n_chans)
        alphas: 1d array with ridge parameters to use

    Returns:
        (tuple):
            * **wts** (*ndarray*): Computed regression weights. Shape of wts is 
                (n_alphas, n_features, n_chans)
            * **ridge_corrs** (*ndarray*): Correlation between predicted and actual responses on
                ridge validation set. Shape of ridge_corrs is (n_alphas, n_chans)

    For multiple regression with stim X and resp y and wts B:

    1. XB = y
    2. X'XB = X'y
    3. B = (X'X)^-1 X'y

    Add L2 (Ridge) regularization:

    4. B = (X'X + aI)^-1 X'y

    Because covariance X'X is a real symmetric matrix, we can decompose it to QLQ', where
    Q is an orthogonal matrix with the eigenvectors and L is a diagonal matrix with the eigenvalues
    of X'X. Furthermore, (QLQ')^-1 = QL^-1Q'

    5. B = (QLQ' + aI)^-1 X'y
    6. B = Q (L + aI)^-1 Q'X'y

    Variables in code below:

    * `covmat` is X'X
    * `l` contains the diagonal entries of L
    * `Q` is Q
    * `Usr` is Q'X'y
    * `D_inv` is (L + aI)^-1

    The wts (B) can be calculated by the matrix multiplication of [Q, D_inv, Usr]
    """
    n_features = train_stim.shape[1] #stim shape is time x features
    n_chans = train_resp.shape[1] #resp shape is time x channels
    n_alphas = alphas.shape[0]

    wts = np.zeros((n_alphas, n_features, n_chans))
    ridge_corrs = np.zeros((n_alphas, n_chans))

    dtype = np.single
    covmat = np.array(np.dot(train_stim.astype(dtype).T, train_stim.astype(dtype)))
    l, Q = np.linalg.eigh(covmat)
    Usr = np.dot(Q.T, np.dot(train_stim.T, train_resp))

    for alpha_i, alpha in enumerate(alphas):
        D_inv = np.diag(1/(l+alpha)).astype(dtype)
        wt = np.array(reduce(np.dot, [Q, D_inv, Usr]).astype(dtype))
        pred = np.dot(ridge_stim, wt)
        ridge_corr = np.zeros((n_chans))
        for i in range(ridge_resp.shape[1]):
            ridge_corr[i] = np.corrcoef(ridge_resp[:, i], pred[:, i])[0, 1]
        ridge_corr[np.isnan(ridge_corr)] = 0

        ridge_corrs[alpha_i, :] = ridge_corr
        wts[alpha_i, :, :] = wt

    return wts, ridge_corrs

def get_all_pred(wts, dstim):
    all_pred = np.array([np.dot(dstim, wts[chan]) for chan in range(wts.shape[0])])
    return all_pred

def reshape_wts_to_2d(wts, delays_used=get_delays(), delay_edges_added=True):
    """Expand the 1d array of wts to the 2d shape of n_delays x n_features.

    Args:
        wts: (n_chans, n_features x n_delays)

    Returns:
        wts_2d: (n_chans, n_delays, n_features)
    """
    n_chans = wts.shape[0]
    n_delays = len(delays_used) + 6 if delay_edges_added else len(delays_used)
    n_features = wts.shape[1]/n_delays
    print(n_features)
    if delay_edges_added:
        wts_2d = wts.reshape(n_chans, n_delays, n_features)[:, 3:-3, :]
    else:
        wts_2d = wts.reshape(n_chans, n_delays, n_features)
    return wts_2d

__all__ = ['get_alphas', 'get_delays', 'run_cv_temporal_ridge_regression_model_fold', 'get_dstim', 
           'run_cv_temporal_ridge_regression_model', 'get_all_pred', 'run_ridge_regression']
