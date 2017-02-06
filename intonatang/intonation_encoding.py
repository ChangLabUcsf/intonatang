from __future__ import division, print_function, absolute_import

import os
results_path = os.path.join(os.path.dirname(__file__), 'results')

import numpy as np
import statsmodels.formula.api as sm
import scipy.io as sio
from scipy.stats import f

default_which_chans = np.arange(256)

def single_electrode_encoding(Y_mat, xs, which_chans=default_which_chans, use_adj_r2=True, return_weights=False):
    """Y_mat: n_chans x n_timepoints x n_trials. 
    """
    n_chans = Y_mat.shape[0]
    n_timepoints = Y_mat.shape[1]
    r2s_adj = np.zeros((Y_mat.shape[0], Y_mat.shape[1], len(xs)))
    p_values = np.zeros((Y_mat.shape[0], Y_mat.shape[1], len(xs)))
    betas = []
    beta_p_values = []
    
    for i, x in enumerate(xs):
        bs = np.zeros((n_chans, n_timepoints, x.shape[1]))
        b_pvalues = np.zeros((n_chans, n_timepoints, x.shape[1]))
        for chan in range(n_chans):
            if chan in which_chans:
                for t in range(Y_mat.shape[1]):
                    result = sm.OLS(Y_mat[chan, t, :], x).fit()
                    if use_adj_r2:
                        r2s_adj[chan, t, i] = result.rsquared_adj
                    else:
                        r2s_adj[chan, t, i] = result.rsquared
                    p_values[chan, t, i] = result.f_pvalue
                    bs[chan, t] = result.params
                    b_pvalues[chan, t] = result.pvalues
        betas.append(bs)
        beta_p_values.append(b_pvalues)
        
    if return_weights:
        return r2s_adj, p_values, betas, beta_p_values
    else:
        return r2s_adj, p_values

def single_electrode_encoding_all_weights(Y_mat, sns, sts, speakers, which_chans=default_which_chans, control_stim=False):
    """Returns weights for encoding when using all groups of predictors
    """
    if control_stim:
        x = dummy_code_control(sns, sts, speakers)
    else:
        x = dummy_code(sns, sts, speakers)
    r2 = np.zeros((Y_mat.shape[0], Y_mat.shape[1]))
    f_values = np.zeros((Y_mat.shape[0], Y_mat.shape[1]))
    f_p_values = np.zeros((Y_mat.shape[0], Y_mat.shape[1]))
    betas = np.zeros((Y_mat.shape[0], Y_mat.shape[1], x.shape[1]))
    betas_p_values = np.zeros((Y_mat.shape[0], Y_mat.shape[1], x.shape[1]))
    
    for chan in range(Y_mat.shape[0]):
        if chan in which_chans:

            for i in range(Y_mat.shape[1]):
                result = sm.OLS(Y_mat[chan, i, :], x).fit()
                r2[chan, i] = result.rsquared_adj
                betas[chan, i,:] = result.params
                betas_p_values[chan, i,:] = result.pvalues
                f_values[chan, i] = result.fvalue
                f_p_values[chan, i] = result.f_pvalue

    return f_values, f_p_values, betas, betas_p_values, r2

def save_encoding_results_all_weights(subject_number, f, fp, b, bp, r2, control_stim=False):
    control_string = "_control" if control_stim else ""
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_encoding_full_model' + control_string + '.mat')
    sio.savemat(filename, {'f': f, 'fp': fp, 'b': b, 'bp': bp, 'r2':r2})

def load_encoding_results_all_weights(subject_number, control_stim=False):
    control_string = "_control" if control_stim else ""
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_encoding_full_model' + control_string + '.mat')
    data = sio.loadmat(filename)
    return data['f'], data['fp'], data['b'], data['bp'], data['r2']

def single_electrode_encoding_varpart(Y_mat, sns, sts, sps, which_chans=default_which_chans, use_adj_r2=True, control_stim=False):
    """Returns unique variance of each group of predictors, must use xs from get_xs_dummy_code_varpart
    
    Calculates difference in explained variance between full model and model excluding one group of predictors.
    The groups of predictors are (sn, st, sp, snxst, snxsp, stxsp, snxstxsp), where x indicates interaction 
    terms. The p_values returned take into account the number of variables in each group.

    Args:
        Y_mat (ndarray): dimensions are n_chans x n_timepoints x n_trials
        sns (list): list of sentence conditions (n_trials length)
        sts (list): list of intonation conditions (n_trials length)
        sps (list): list of speaker conditions (n_trials length)
        which_chans (list): list of channels to do encoding analysis on
        use_adj_r2: use adjusted r2 instead of r2
        control_stim: whether analysis is on non-speech control task (changes coding of categorical variables)

    Returns:
        (tuple):
            * **r2s_varpart** (*ndarray*): dimensions n_chans x n_timepoints x 7 (groups of predictors)
            * **p_values** (*ndarray*): dimensions n_chans x n_timepoints x 7
            * **f_stats** (*ndarray*): dimensions n_chans x n_timepoints x 7
    """

    xs = get_xs_dummy_code_varpart(sns, sts, sps, control_stim=control_stim)
    r2s_varpart = np.zeros((Y_mat.shape[0], Y_mat.shape[1], 7))
    p_values = np.zeros((Y_mat.shape[0], Y_mat.shape[1], 7))
    f_stats = np.zeros((Y_mat.shape[0], Y_mat.shape[1], 7))

    r2s_wo = np.zeros((Y_mat.shape[0], Y_mat.shape[1], 8))
    
    #First calculate the r2s for the full model
    for chan in range(256):
        if chan in which_chans:
            for t in range(Y_mat.shape[1]):
                result = sm.OLS(Y_mat[chan, t, :], xs[-1]).fit()
                if use_adj_r2:
                    r2s_wo[chan, t, 7] = result.rsquared_adj
                else:
                    r2s_wo[chan, t, 7] = result.rsquared
                    
    #Then calcuate r2 differences and assess significance with the F statistic
    xs = xs[:-1]
    N = Y_mat.shape[2]
    if control_stim:
        k = 40 # 5 sn x 4 st x 2 speakers
    else:
        k = 48 #hard-coded to work with 4 sn, 4 st, 3 sp
    for i, x in enumerate(xs):
        fstat = np.zeros((Y_mat.shape[0], Y_mat.shape[1]))
        for chan in range(256):
            if chan in which_chans:
                for t in range(Y_mat.shape[1]):
                    result = sm.OLS(Y_mat[chan, t, :], x).fit()
                    if use_adj_r2:
                        r2s_wo[chan, t, i] = result.rsquared_adj
                        r2s_varpart[chan, t, i] = r2s_wo[chan, t, 7] - r2s_wo[chan, t, i]
                    else:
                        r2s_wo[chan, t, i] = result.rsquared
                        r2s_varpart[chan, t, i] = r2s_wo[chan, t, 7] - r2s_wo[chan, t, i]
        m = k - x.shape[1]
        fstat = (r2s_varpart[:,:,i]/m)/((1 - r2s_wo[:,:,7])/(N - k - 1))
        f_stats[:,:,i] = fstat
        p_values[:, :, i] = f.sf(fstat, m, N-k-1)
    return r2s_varpart, p_values, f_stats

def save_encoding_results(subject_number, r2s, p_values, f_stats, varpart=True, control_stim=False):
    varpart_string = "_varpart" if varpart else ""
    control_string = "_control" if control_stim else ""
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_encoding' + varpart_string + control_string + '.mat')
    sio.savemat(filename, {'r2s': r2s, 'p_values': p_values, 'f_stats': f_stats})

def load_encoding_results(subject_number, varpart=True, control_stim=False):
    varpart_string = "_varpart" if varpart else ""
    control_string = "_control" if control_stim else ""
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_encoding' + varpart_string + control_string + '.mat')
    data = sio.loadmat(filename)
    r2s = data['r2s']
    p_values = data['p_values']
    f_stats = data['f_stats']
    return r2s, p_values, f_stats

def encoding_varpart_permutation_test(Y_mat, sns, sts, sps, n_perms=250, which_chans=default_which_chans, use_adj_r2=True, control_stim=False):
    """Runs a permutation test on variance partitioning analysis by shuffling trials. Uses single_electrode_encoding_varpart.
    """
    r2s_varpart_perm = np.zeros((Y_mat.shape[0], Y_mat.shape[1], 7, n_perms))
    p_values_perm = np.zeros((Y_mat.shape[0], Y_mat.shape[1], 7, n_perms))
    f_stats_perm = np.zeros((Y_mat.shape[0], Y_mat.shape[1], 7, n_perms))

    N = Y_mat.shape[2]
    for i in range(n_perms):
        indexes = np.random.permutation(N)
        Y_mat_perm = Y_mat[:,:,indexes]
        r2s_varpart, p_values, f_stats = single_electrode_encoding_varpart(Y_mat_perm, sns, sts, sps, which_chans=which_chans, use_adj_r2=use_adj_r2, control_stim=control_stim)
        r2s_varpart_perm[:,:,:,i] = r2s_varpart
        p_values_perm[:,:,:,i] = p_values
        f_stats_perm[:,:,:,i] = f_stats
    return r2s_varpart_perm, p_values_perm, f_stats_perm

def get_xs_dummy_code_varpart(sns, sts, speakers, control_stim=False):
    """Returns list of xs for variance partitioning analysis.
    
    Each item in xs contains predictors variables that exclude one group of predictor variables, 
    except the last item in xs which contains all predictor variables (full model with main effects,
    all pairwise, and three-way interaction)

    xs = [x_wo_sn, x_wo_st, x_wo_sp, x_wo_sn_st, x_wo_sn_sp, x_wo_st_sp, x_wo_sn_st_sp, x_all]
    """
    if control_stim:
        dummy_code_func = dummy_code_varpart_control
    else:
        dummy_code_func = dummy_code_varpart
    x_wo_sn = dummy_code_func(sns, sts, speakers, 'sn')
    x_wo_st = dummy_code_func(sns, sts, speakers, 'st')
    x_wo_sp = dummy_code_func(sns, sts, speakers, 'sp')
    x_wo_sn_st = dummy_code_func(sns, sts, speakers, 'sn st')
    x_wo_sn_sp = dummy_code_func(sns, sts, speakers, 'sn sp')
    x_wo_st_sp = dummy_code_func(sns, sts, speakers, 'st sp')
    x_wo_sn_st_sp = dummy_code_func(sns, sts, speakers, 'sn st sp')
    if control_stim:
        x_all = dummy_code_control(sns, sts, speakers)
    else:
        x_all = dummy_code(sns, sts, speakers)
    xs = [x_wo_sn, x_wo_st, x_wo_sp, x_wo_sn_st, x_wo_sn_sp, x_wo_st_sp, x_wo_sn_st_sp, x_all]    
    return xs

def dummy_code_varpart(sns, sts, speakers, to_exclude=None):
    """Returns subsets of coded variables for variance partitioning analysis

    to_exclude can be one of the seven following strings:
    ["sn", "st", "sp", "sn st", "sn sp", "st sp", "sn st sp"]

    "sn", "st", and "sp" each exclude the variables coding for the main effect of sentence,
    intonation, and speaker, respectively. 

    "sn st", "sn sp", "st sp", and "sn st sp" each exclude the variables coding for each pairwise
    and the three-way interaction. 

    This function can't be used to exclude more than one group of predictor variables.

    Args:
        to_exclude (str): which predictor group to exclude
    """
    x = dummy_code(sns, sts, speakers)

    if to_exclude == 'sn':
        indexes = np.concatenate([[0], np.arange(4,48)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'st':
        indexes = np.concatenate([[0,1,2,3], np.arange(7,48)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'sp':
        indexes = np.concatenate([np.arange(0,7), np.arange(9,48)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'sn st':
        indexes = np.concatenate([np.arange(0,9), np.arange(18,48)], axis=0)
        x = x[:,indexes]   
    elif to_exclude == 'sn sp':
        indexes = np.concatenate([np.arange(0,18), np.arange(24,48)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'st sp':
        indexes = np.concatenate([np.arange(0,24), np.arange(30,48)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'sn st sp':
        x = x[:, np.arange(0,30)]

    return x 

def dummy_code_varpart_control(sns, sts, speakers, to_exclude=None):
    """Returns subsets of coded variables for variance partitioning analysis for non-speech control

    See docstring/documentation for dummy_code_varpart for more information.

    Args:
        to_exclude (str): which predictor group to exclude
    """
    x = dummy_code_control(sns, sts, speakers)

    if to_exclude == 'sn':
        indexes = np.concatenate([[0], np.arange(5,40)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'st':
        indexes = np.concatenate([[0,1,2,3,4], np.arange(8,40)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'sp':
        indexes = np.concatenate([np.arange(0,8), np.arange(9,40)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'sn st':
        indexes = np.concatenate([np.arange(0,9), np.arange(21,40)], axis=0)
        x = x[:,indexes]   
    elif to_exclude == 'sn sp':
        indexes = np.concatenate([np.arange(0,21), np.arange(25,40)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'st sp':
        indexes = np.concatenate([np.arange(0,25), np.arange(28,40)], axis=0)
        x = x[:,indexes]
    elif to_exclude == 'sn st sp':
        x = x[:, np.arange(0,28)]

    return x 

def dummy_code(sns, sts, speakers, kind='simple'):
    """Used to code categorical variables of sentence, intonation, and speaker.

    Simple coding means each condition is compared to a reference level with the intercept 
    being the grand mean. Here, the reference levels are the Neutral condition (st == 1) for
    intonation, sentence 1 (sn == 1) for sentence, and speaker 3 (sp == 3) for speaker.

    The total number of independent variables is 48:
        intercept             1
        sentence (sn)   4-1 = 3
        intonation (st) 4-1 = 3
        speaker (sp)    3-1 = 2
        sn x st         3*3 = 9
        sn x sp         3*2 = 6
        st x sp         3*2 = 6
        sn x st x sp  3*3*2 = 18

    Args:
        sns (list): list of sentence conditions (1, 2, 3, 4)
        sts (list): list of intonation conditions (1, 2, 3, 4)
        sps (list): list of speaker conditions (1, 2, 3)

    Returns:
        (ndarray): 
            * x with dimensions  (n_trials x 48)
    """
    x = np.zeros((len(sns), 48))

    if kind == 'simple':
        for i, (sn, st, speaker) in enumerate(zip(sns, sts, speakers)):
            x[i,0] = 1

            if sn == 1:
                x[i,1:3+1] = [-0.25, -0.25, -0.25]
            elif sn == 2:
                x[i,1:3+1] = [0.75, -0.25, -0.25]
            elif sn == 3:
                x[i,1:3+1] = [-0.25, 0.75, -0.25]
            elif sn == 4:
                x[i,1:3+1] = [-0.25, -0.25, 0.75]

            if st == 1:
                x[i,4:6+1] = [-0.25, -0.25, -0.25]
            elif st == 2:
                x[i,4:6+1] =  [0.75, -0.25, -0.25]
            elif st == 3:
                x[i,4:6+1] =  [-0.25, 0.75, -0.25]
            elif st == 4:
                x[i,4:6+1] =  [-0.25, -0.25, 0.75]

            if speaker == 1:
                x[i, 7:8+1] = [0.6667, -0.3333]
            elif speaker == 2:
                x[i, 7:8+1] = [-0.3333, 0.6667]
            elif speaker == 3:
                x[i, 7:8+1] = [-0.3333, -0.3333]

            x[i,9:11+1] = x[i,1]*x[i,4:6+1]
            x[i,12:14+1] = x[i,2]*x[i,4:6+1]
            x[i,15:17+1] = x[i,3]*x[i,4:6+1]

            x[i,18:20+1] = x[i,1:3+1]*x[i,7]
            x[i,21:23+1] = x[i,1:3+1]*x[i,8]
            
            x[i,24:26+1] = x[i,4:6+1]*x[i,7]
            x[i,27:29+1] = x[i,4:6+1]*x[i,8]

            x[i,30:38+1] = x[i,7]*x[i,9:17+1]
            x[i,39:47+1] = x[i,8]*x[i,9:17+1]

    return x

def dummy_code_control(sns, sts, speakers, kind='simple'):
    """Used to code categorical variables in non-speech control task

    The total number of independent variables here is 40:
        intercept             1
        sentence (sn)   5-1 = 4
        intonation (st) 4-1 = 3
        speaker (sp)    2-1 = 1
        sn x st         4*3 = 12
        sn x sp         4*1 = 4
        st x sp         3*1 = 3
        sn x st x sp  4*3*2 = 12

    Args:
        sns (list): list of sentence conditions (1, 2, 3, 4, 5)
        sts (list): list of intonation conditions (1, 2, 3, 4)
        sps (list): list of speaker conditions (1, 2)

    Returns:
        (ndarray):
            * x with dimensions (n_trials x 40)
    """
    x = np.zeros((len(sns), 40))

    if kind == 'simple':
        for i, (sn, st, speaker) in enumerate(zip(sns, sts, speakers)):
            x[i,0] = 1

            if sn == 1:
                x[i,1:5] = [0.8, -0.2, -0.2, -0.2]
            elif sn == 2:
                x[i,1:5] = [-0.2, 0.8, -0.2, -0.2]
            elif sn == 3:
                x[i,1:5] = [-0.2, -0.2, 0.8, -0.2]
            elif sn == 4:
                x[i,1:5] = [-0.2, -0.2, -0.2, 0.8]
            elif sn == 5:
                x[i,1:5] = [-0.2, -0.2, -0.2, -0.2]

            if st == 1:
                x[i,5:8] = [-0.25, -0.25, -0.25]
            elif st == 2:
                x[i,5:8] =  [0.75, -0.25, -0.25]
            elif st == 3:
                x[i,5:8] =  [-0.25, 0.75, -0.25]
            elif st == 4:
                x[i,5:8] =  [-0.25, -0.25, 0.75]

            if speaker == 1:
                x[i, 8] = -0.5
            elif speaker == 2:
                x[i, 8] = 0.5

            x[i,9:12] = x[i,1]*x[i,5:8]
            x[i,12:15] = x[i,2]*x[i,5:8]
            x[i,15:18] = x[i,3]*x[i,5:8]
            x[i,18:21] = x[i,4]*x[i,5:8]

            x[i,21:25] = x[i,1:5]*x[i,8]

            x[i,25:28] = x[i,5:8]*x[i,8]

            x[i,28:40] = x[i,8]*x[i,9:21]

    return x