from __future__ import division, print_function, absolute_import

import os
results_path = os.path.join(os.path.dirname(__file__), 'results')
timit_data_path = os.path.join(os.path.dirname(__file__), 'data', 'timit')
processed_timit_data_path = os.path.join(os.path.dirname(__file__), 'processed_timit_data')

import numpy as np
import scipy.io as sio
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pandas as pd
import random

from . import timit
from .intonation_stims import get_pitch_and_intensity
from .temporal_receptive_field import *

def generate_all_results(regenerate_shuffled_timit_data=False):
    subject_numbers = [113, 118, 122, 123, 125, 129, 131]

    if regenerate_shuffled_timit_data:
        timit_pitch = timit.get_timit_pitch()
        for i in range(25):
            randomize_timit_pitch_contours(timit_pitch, save_as=i)

    for subject_number in subject_numbers:
        run_ptrf_analysis_permutation_test(subject_number)
        run_ptrf_analysis_pipeline_for_subject_number(subject_number)

def run_ptrf_analysis_pipeline_for_subject_number(subject_number, pitch_scaling="log"):
    """Pitch temporal receptive field analysis pipeline.

    The ptrf pipeline consists of: 
        1. Loading TIMIT data for each subject. 
        2. For each of the 25 stratified folds for cross validation:
            - Discretize the pitch features (parameterized from f0 values in Hz in timit.save_timit_pitch) into bins.
            - Run ridge regression between stimulus features (pitch bins, and other features like intensity to statistically control for) and neural activity on each electrode.
        3. Save R2 values and weights (which are the temporal receptive fields)
    """
    timit_pitch = timit.get_timit_pitch_phonetic()

    out = timit.load_h5py_out(subject_number)
    for trial in out:
        n_chans = out[trial]['ecog'].shape[0]
        break

    test_corr_all = np.zeros((n_chans, 25))
    test_corr_abs_bin = np.zeros((n_chans, 25))
    test_corr_rel_bin = np.zeros((n_chans, 25))

    wts_all = np.zeros((n_chans, 1058, 25))
    wts_abs = np.zeros((n_chans, 598, 25))
    wts_rel = np.zeros((n_chans, 598, 25))

    abs_bin_edges, rel_bin_edges = get_bin_edges_abs_rel(timit_pitch, pitch_scaling=pitch_scaling)

    for i in range(25):
        pitch_intensity, neural_activity, last_indexes = get_neural_activity_and_pitch_phonetic_for_fold(out, timit_pitch, i, pitch_scaling=pitch_scaling)
        stims_all, resps_all = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, feat="all")
        stims_abs_bin, resps_abs_bin = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, feat="abs_bin")
        stims_rel_bin, resps_rel_bin = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, feat="rel_bin")
        test_corr_all[:,i], wts_all[:, :, i] = run_cv_temporal_ridge_regression_model_fold(stims_all, resps_all)
        test_corr_abs_bin[:,i], wts_abs[:, :, i] = run_cv_temporal_ridge_regression_model_fold(stims_abs_bin, resps_abs_bin)
        test_corr_rel_bin[:,i], wts_rel[:, :, i] = run_cv_temporal_ridge_regression_model_fold(stims_rel_bin, resps_rel_bin)

    r2_abs_folds = test_corr_all ** 2 - test_corr_rel_bin ** 2
    r2_rel_folds = test_corr_all ** 2 - test_corr_abs_bin ** 2
    r2_abs = np.mean(r2_abs_folds, axis=1)
    r2_rel = np.mean(r2_rel_folds, axis=1)

    save_cv_model_fold(subject_number, test_corr_all, test_corr_abs_bin, test_corr_rel_bin, r2_abs, r2_rel, wts_all, wts_abs, wts_rel, pitch_scaling=pitch_scaling)

def run_ptrf_analysis_pipeline_for_subject_number_testing_rel_versus_change(subject_number, pitch_scaling="log"):
    timit_pitch = timit.get_timit_pitch_phonetic()

    out = timit.load_h5py_out(subject_number)
    for trial in out:
        n_chans = out[trial]['ecog'].shape[0]
        break

    test_corr_all = np.zeros((n_chans, 25))
    test_corr_rel = np.zeros((n_chans, 25))
    test_corr_change = np.zeros((n_chans, 25))

    wts_all = np.zeros((n_chans, 1518, 25)) #1058 = 46 * 23
    wts_rel = np.zeros((n_chans, 1058, 25)) #598 = 46*13
    wts_change = np.zeros((n_chans, 1058, 25))

    abs_bin_edges, rel_bin_edges = get_bin_edges_abs_rel(timit_pitch, pitch_scaling=pitch_scaling)
    abs_change_bin_edges = get_bin_edges_abs_pitch_change(timit_pitch, pitch_scaling=pitch_scaling)

    for i in range(25):
        pitch_intensity, neural_activity, last_indexes = get_neural_activity_and_pitch_phonetic_for_fold(out, timit_pitch, i, pitch_scaling=pitch_scaling)
        stims_all, resps_all = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, abs_change_bin_edges=abs_change_bin_edges, feat="all_with_change")
        stims_rel, resps_rel = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, abs_change_bin_edges=abs_change_bin_edges, feat="abs_rel")
        stims_change, resps_change = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, abs_change_bin_edges=abs_change_bin_edges, feat="abs_change")
        test_corr_all[:,i], wts_all[:, :, i] = run_cv_temporal_ridge_regression_model_fold(stims_all, resps_all)
        test_corr_rel[:,i], wts_rel[:, :, i] = run_cv_temporal_ridge_regression_model_fold(stims_rel, resps_rel)
        test_corr_change[:,i], wts_change[:, :, i] = run_cv_temporal_ridge_regression_model_fold(stims_change, resps_change)

    r2_rel_folds = test_corr_all ** 2 - test_corr_change ** 2
    r2_change_folds = test_corr_all ** 2 - test_corr_rel ** 2
    r2_rel = np.mean(r2_rel_folds, axis=1)
    r2_change = np.mean(r2_change_folds, axis=1)

    save_cv_model_fold(subject_number, test_corr_all, test_corr_rel, test_corr_change, r2_rel, r2_change, wts_all, wts_rel, wts_change, pitch_scaling=pitch_scaling, note="_rel_versus_pitch_change")

def get_subject_permutation_test_data(subject_number, n_perms=200, pitch_scaling="log"):
    out = timit.load_h5py_out(subject_number)
    for trial in out:
        n_chans = out[trial]['ecog'].shape[0]
        break

    try:
        r2_all_perms, r2_abs_perms, r2_rel_perms = load_cv_shuffle_fold(subject_number, pitch_scaling=pitch_scaling)
        if r2_all_perms.shape[1] != n_perms:
            r2_all_perms = np.zeros((n_chans, n_perms))
            r2_abs_perms = np.zeros((n_chans, n_perms))
            r2_rel_perms = np.zeros((n_chans, n_perms))
    except:
        r2_all_perms = np.zeros((n_chans, n_perms))
        r2_abs_perms = np.zeros((n_chans, n_perms))
        r2_rel_perms = np.zeros((n_chans, n_perms))

    return r2_all_perms, r2_abs_perms, r2_rel_perms

def run_ptrf_analysis_permutation_test(subject_number, n_perms=200, pitch_scaling="log", which_perms=None):
    print("Running ptrf permutation for EC" + str(subject_number))
    print("permutations:")
    print(which_perms)
    timit_pitch = timit.get_timit_pitch_phonetic()
    abs_bin_edges, rel_bin_edges = get_bin_edges_abs_rel(timit_pitch, pitch_scaling=pitch_scaling)

    out = timit.load_h5py_out(subject_number)
    for trial in out:
        n_chans = out[trial]['ecog'].shape[0]
        break

    if which_perms is None:
        r2_all_perms_run = np.zeros((n_chans, n_perms))
        r2_abs_perms_run = np.zeros((n_chans, n_perms))
        r2_rel_perms_run = np.zeros((n_chans, n_perms))
    else:
        r2_all_perms_run = np.zeros((n_chans, len(which_perms)))
        r2_abs_perms_run = np.zeros((n_chans, len(which_perms)))
        r2_rel_perms_run = np.zeros((n_chans, len(which_perms)))

    for perm_i, perm in enumerate(range(n_perms)) if which_perms is None else enumerate(which_perms):
        perm = int(perm)
        print("Perm: " + str(perm))
        timit_pitch_shuffled = load_timit_shuffled(perm)

        test_corr_all = np.zeros((n_chans, 25))
        test_corr_abs_bin = np.zeros((n_chans, 25))
        test_corr_rel_bin = np.zeros((n_chans, 25))

        for i in range(25):
            print(i)
            pitch_intensity, neural_activity, last_indexes = get_neural_activity_and_pitch_phonetic_for_fold(out, timit_pitch_shuffled, i, pitch_scaling=pitch_scaling)
            stims_all, resps_all = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, feat="all")
            stims_abs_bin, resps_abs_bin = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, feat="abs_bin")
            stims_rel_bin, resps_rel_bin = get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, feat="rel_bin")
            test_corr_all[:,i], wts = run_cv_temporal_ridge_regression_model_fold(stims_all, resps_all)
            test_corr_abs_bin[:,i], wts = run_cv_temporal_ridge_regression_model_fold(stims_abs_bin, resps_abs_bin)
            test_corr_rel_bin[:,i], wts = run_cv_temporal_ridge_regression_model_fold(stims_rel_bin, resps_rel_bin)
        
        r2_abs_folds = test_corr_all ** 2 - test_corr_rel_bin ** 2
        r2_rel_folds = test_corr_all ** 2 - test_corr_abs_bin ** 2
        r2_abs_perms_run[:, perm_i] = np.mean(r2_abs_folds, axis=1)
        r2_rel_perms_run[:, perm_i] = np.mean(r2_rel_folds, axis=1)
        r2_all_perms_run[:, perm_i] = np.mean(test_corr_all**2, axis=1)

    r2_all_perms, r2_abs_perms, r2_rel_perms = get_subject_permutation_test_data(subject_number, n_perms=n_perms, pitch_scaling=pitch_scaling)

    if which_perms is None:
        r2_all_perms = r2_all_perms_run
        r2_abs_perms = r2_abs_perms_run
        r2_rel_perms = r2_rel_perms_run
    else:
        r2_all_perms[:, which_perms] = r2_all_perms_run
        r2_abs_perms[:, which_perms] = r2_abs_perms_run
        r2_rel_perms[:, which_perms] = r2_rel_perms_run

    save_cv_shuffle_fold(subject_number, r2_all_perms, r2_abs_perms, r2_rel_perms)

def get_abs_and_rel_sig(subject_number):
    r_all, r_abs, r_rel, abs_r2, rel_r2, wts_all, wts_abs, wts_rel = load_cv_model_fold(subject_number)
    ptrf_permutation_data = sio.loadmat(os.path.join(results_path, 'EC' + str(subject_number) + '_shuffle200_25fold_ptrf_results_10bins.mat'))
    r2_abs_perms = ptrf_permutation_data['r2_abs']
    r2_rel_perms = ptrf_permutation_data['r2_rel']
    abs_sig = np.ones((256)) * -1
    rel_sig = np.ones((256)) * -1
    for chan in np.arange(256):
        if rel_r2[0, chan] > np.percentile(r2_rel_perms[chan], [95])[0]:
            rel_sig[chan] = 1
        else:
            rel_sig[chan] = 0
        if abs_r2[0, chan] > np.percentile(r2_abs_perms[chan], [95])[0]:
            abs_sig[chan] = 1
        else:
            abs_sig[chan] = 0
    return abs_sig, rel_sig

def get_neural_activity_and_pitch_phonetic_for_fold(out_h5py, timit_pitch, fold, pitch_scaling="log"):
    out = out_h5py
    n_sentences = len(out)
    last_index_train = np.floor(0.8 * n_sentences)
    last_index_ridge = np.floor(0.9 * n_sentences)
    last_index_test = n_sentences
    nt, nchans = get_nt_nchans_for_out(out_h5py, timit_pitch)
    pitch_intensity = np.zeros((nt, 5))
    neural_activity = np.empty((nchans, nt))
    neural_activity.fill(np.nan)

    timit_strat = load_timit_strat(fold)
    timit_index = 0
    last_indexes = []
    index = 0

    for timit_name in timit_strat.values:
        if timit_name in out:
            timit_index = timit_index + 1
            if timit_index == last_index_train:
                last_indexes.append(index)
            elif timit_index == last_index_ridge:
                last_indexes.append(index)
            n_pitch = timit_pitch.loc[timit_name].pitch.shape[0]
            ecog = out[timit_name]['ecog']
            for i in range(ecog.shape[2]):
                if pitch_scaling == "log":
                    pitch_intensity[index:index+n_pitch,0] = timit_pitch.loc[timit_name]['abs_pitch']
                    pitch_intensity[index:index+n_pitch,1] = timit_pitch.loc[timit_name]['rel_pitch_global']
                    pitch_intensity[index:index+n_pitch,4] = timit_pitch.loc[timit_name]['abs_pitch_change']
                elif pitch_scaling == "erb":
                    pitch_intensity[index:index+n_pitch,0] = timit_pitch.loc[timit_name]['abs_pitch_erb']
                    pitch_intensity[index:index+n_pitch,1] = timit_pitch.loc[timit_name]['rel_pitch_global_erb']
                    pitch_intensity[index:index+n_pitch,4] = timit_pitch.loc[timit_name]['abs_pitch_erb_change']
                pitch_intensity[index:index+n_pitch,2] = timit_pitch.loc[timit_name]['zscore_intensity']
                time_indexes = timit_pitch.loc[timit_name].pitch.index.values + 1
                pitch_intensity[index:index+n_pitch,3] = time_indexes/100.0
                
                ecog_trial = ecog[:, :, i]
                try:
                    neural_activity[:, index:index+n_pitch] = ecog_trial[:, time_indexes + 50]
                except:
                    neural_activity[:, index:index+n_pitch] = np.nan
                    print('error at offset ' + str(offset) + ' ' + timit_name)
                index = index + n_pitch
    return pitch_intensity, neural_activity, last_indexes

def load_timit_strat(fold=0):
    """Loads an ordering of TIMIT sentences so that cross-validation of models will be done with stratification.

    Each of the 25 stratified orderings was generated using code that is available in a Notebook.
    """
    assert fold < 25
    filename = os.path.join(timit_data_path, 'timit_strat_25folds.h5')
    timit_strat = pd.read_hdf(filename, 'timit_strat' + str(fold))
    return timit_strat

def load_timit_shuffled(fold=0):
    """Loads a shuffled version of timit_pitch which was created and saved by randomize_timit_pitch_contours
    """
    filename = os.path.join(processed_timit_data_path, 'timit_pitch_shuffle_' + str(fold) + '.h5')
    timit_pitch_shuffled = pd.read_hdf(filename, 'timit_pitch_shuffle_' + str(fold))
    return timit_pitch_shuffled

def get_nt_nchans_for_out(out, timit_pitch):
    """Returns number of time points and channels for one subject

    out is returned from timit.load_h5py_out(subject_number) and contains TIMIT data for one subject.

    This function goes through each TIMIT trial to determine how many timepoints of data are there.
    """
    nt = 0
    for trial in out:
        nchans = out[trial]['ecog'].shape[0]
        timit_name = out[trial].attrs['timit_name'][0]
        for i in range(out[trial]['ecog'].shape[2]):
            nt = nt + timit_pitch.loc[timit_name].pitch.shape[0]
    return nt, nchans

def get_stim_and_resp_from_pitch_intensity_neural_activity_fold(pitch_intensity, neural_activity, last_indexes, abs_bin_edges, rel_bin_edges, abs_change_bin_edges=None, nbins=10, feat="all"):
    """Returns matrices for independent variables (stimulus features) and dependent variables (neural activity on all channels) for training, hyperparamater optimization, and testing.

    This function further processes the output of get_neural_activity_and_pitch_phonetic_for_fold. Starting with the nt x n_continuous_features matrix of pitch_intensity and 
    the n_chans x nt matrix of neural activity, this function produces the matrices representing the binary matrix of binned pitch features as the stim and transposes the neural 
    activity to get the resp. These are then split into three matrices along the time dimension.
    """
    resp = neural_activity.T
    pi_train = pitch_intensity[0:last_indexes[0], :]
    na_train = resp[0:last_indexes[0], :]
    pi_ridge = pitch_intensity[last_indexes[0]: last_indexes[1], :]
    na_ridge = resp[last_indexes[0]:last_indexes[1], :]
    pi_test = pitch_intensity[last_indexes[1]:, :]
    na_test = resp[last_indexes[1]:, :]
    pis = [pi_train, pi_ridge, pi_test]
    nas = [na_train, na_ridge, na_test]

    full_pitch_intensity = np.copy(pitch_intensity)
    full_intensity = full_pitch_intensity[:, 2]
    full_pitch_intensity = full_pitch_intensity[~np.isnan(full_intensity), :]
    stims = []
    resps = []
    for pitch_intensity, neural_activity in zip(pis, nas):
        intensity = pitch_intensity[:, 2]
        pitch_intensity = pitch_intensity[~np.isnan(intensity), :]
        neural_activity = neural_activity[~np.isnan(intensity), :]
        abs_pitch = pitch_intensity[:, 0]
        rel_pitch = pitch_intensity[:, 1]
        abs_pitch_change = pitch_intensity[:, 4]

        if nbins==10:
            stim_pitch_abs = get_pitch_matrix(abs_pitch, abs_bin_edges)
            stim_pitch_rel = get_pitch_matrix(rel_pitch, rel_bin_edges)
            if abs_change_bin_edges is not None:
                stim_pitch_abs_change = get_pitch_matrix(abs_pitch_change, abs_change_bin_edges)

        pitch_binary = np.any(stim_pitch_rel, axis=1).astype(np.int)[:, np.newaxis]
        stim_int = transform_intensity(pitch_intensity[:, 2])
        stim_onset = transform_time_indexes(pitch_intensity[:, 3])
        bias_ones = np.ones((stim_int.shape[0], 1))
        if feat == "abs_bin":
            stim = np.hstack([stim_pitch_abs, pitch_binary, stim_int, bias_ones])
        elif feat == "rel_bin":
            stim = np.hstack([stim_pitch_rel, pitch_binary, stim_int, bias_ones])
        elif feat == "all":
            stim = np.hstack([stim_pitch_abs, stim_pitch_rel, pitch_binary, stim_int, bias_ones])
        elif feat == "abs_rel":
            stim = np.hstack([stim_pitch_abs, stim_pitch_rel, pitch_binary, stim_int, bias_ones])
        elif feat == "abs_change":
            stim = np.hstack([stim_pitch_abs, stim_pitch_abs_change, pitch_binary, stim_int, bias_ones])
        elif feat == "all_with_change":
            stim = np.hstack([stim_pitch_abs, stim_pitch_rel, stim_pitch_abs_change, pitch_binary, stim_int, bias_ones])
        stims.append(stim)
        resps.append(neural_activity)

    return stims, resps

def transform_intensity(intensity):
    assert len(intensity.shape) == 1
    stim_int = intensity[:, np.newaxis]
    return (stim_int/2.0) + 0.75

def transform_time_indexes(time_indexes):
    assert len(time_indexes.shape) == 1
    stim_onset = (time_indexes == 0.01).astype(np.float)[:, np.newaxis]
    return stim_onset

def get_bin_edges_percent_range(a, bins=10, percent=95):
    assert percent > 1 
    assert percent < 100
    tail_percentage = (100 - percent)/2
    a_range = np.percentile(a, [tail_percentage, 100-tail_percentage])
    counts, bin_edges = np.histogram(a, bins=bins, range=a_range)
    return bin_edges

def get_bin_edges_abs_rel(timit_pitch, bins=10, percent=95, pitch_scaling="log"):
    """Returns abs_bin_edges and rel_bin_edges"""
    if pitch_scaling == "log":
        abs_pitch = timit_pitch['abs_pitch']
        rel_pitch = timit_pitch['rel_pitch_global']
    elif pitch_scaling == "erb":
        abs_pitch = timit_pitch['abs_pitch_erb']
        rel_pitch = timit_pitch['rel_pitch_global_erb']

    abs_bin_edges = get_bin_edges_percent_range(abs_pitch[~np.isnan(abs_pitch)], bins=bins, percent=percent)
    rel_bin_edges = get_bin_edges_percent_range(rel_pitch[~np.isnan(rel_pitch)], bins=bins, percent=percent)
    return abs_bin_edges, rel_bin_edges

def get_bin_edges_abs_pitch_change(timit_pitch, bins=10, percent=95, pitch_scaling="log"):
    if pitch_scaling == "log":
        abs_pitch_change = timit_pitch['abs_pitch_change']
    elif pitch_scaling == "erb":
        abs_pitch_change = timit_pitch['abs_pitch_erb_change']

    abs_change_bin_edges = get_bin_edges_percent_range(abs_pitch_change[~np.isnan(abs_pitch_change)], bins=bins, percent=percent)
    return abs_change_bin_edges

def get_pitch_matrix(pitch, bin_edges):
    pitch[pitch < bin_edges[0]] = bin_edges[0] + 0.0001
    pitch[pitch > bin_edges[-1]] = bin_edges[-1] - 0.0001
    bin_indexes = np.digitize(pitch, bin_edges) - 1
    stim_pitch = np.zeros((len(pitch), 10))
    for i, b in enumerate(bin_indexes):
        if b < 10:
            stim_pitch[i, b] = 1
    return stim_pitch

def save_cv_model_fold(subject_number, test_corr_all, test_corr_abs_bin, test_corr_rel_bin, r2_abs, r2_rel, wts_all, wts_abs, wts_rel, pitch_scaling="log", note=""):
    filename = 'EC' + str(subject_number) + '_25fold_ptrf_results_10bins' + note
    if pitch_scaling != "log":
        filename = filename + "_" + pitch_scaling
    filename = os.path.join(results_path, filename + ".mat")
    sio.savemat(filename, {'r_all': test_corr_all, 'r_abs_bin': test_corr_abs_bin,
          'r_rel_bin': test_corr_rel_bin, 'r2_abs': r2_abs, 'r2_rel': r2_rel, 'wts_all': wts_all, 'wts_abs': wts_abs, 'wts_rel': wts_rel})

def load_cv_model_fold(subject_number, pitch_scaling="log", note=""):
    filename = 'EC' + str(subject_number) + '_25fold_ptrf_results_10bins' + note
    if pitch_scaling != "log":
        filename = filename + "_" + pitch_scaling
    filename = os.path.join(results_path, filename + ".mat")
    data = sio.loadmat(filename)
    return data['r_all'], data['r_abs_bin'], data['r_rel_bin'], data['r2_abs'], data['r2_rel'], data['wts_all'], data['wts_abs'], data['wts_rel']

def save_cv_shuffle_fold(subject_number, r2_all, r2_abs, r2_rel, pitch_scaling="log"):
    filename = 'EC' + str(subject_number) + '_shuffle200_25fold_ptrf_results_10bins.mat'
    if pitch_scaling != "log":
        filename = filename + "_" + pitch_scaling
    filename = os.path.join(results_path, filename)
    sio.savemat(filename, {'r2_all': r2_all, 'r2_abs': r2_abs, 'r2_rel': r2_rel})

def load_cv_shuffle_fold(subject_number, pitch_scaling="log"):
    filename = 'EC' + str(subject_number) + '_shuffle200_25fold_ptrf_results_10bins.mat'
    if pitch_scaling != "log":
        filename = filename + "_" + pitch_scaling
    filename = os.path.join(results_path, filename)
    data = sio.loadmat(filename)
    return data['r2_all'], data['r2_abs'], data['r2_rel']

def get_intonation_tokens_stim():
    pitch_intensity_intonation = get_pitch_and_intensity()
    timit_pitch = timit.get_timit_pitch_phonetic()
    abs_bin_edges, rel_bin_edges = get_bin_edges_abs_rel(timit_pitch)

    token_stim = {}
    for token, pi in pitch_intensity_intonation.iteritems():
        token_stim[token] = {}
        token_stim[token]['abs_pitch'] = timit.zscore_abs_pitch(pi['pitch'])
        token_stim[token]['intensity'] = timit.zscore_intensity(pi['intensity'])
        token_stim[token]['rel_pitch'] = nan_zscore(np.log(pi['pitch']))
        token_stim[token]['stim_pitch_abs'] = get_pitch_matrix(token_stim[token]['abs_pitch'], abs_bin_edges)
        token_stim[token]['stim_pitch_rel'] = get_pitch_matrix(token_stim[token]['rel_pitch'], rel_bin_edges)
        token_stim[token]['pitch_binary'] = np.any(token_stim[token]['stim_pitch_rel'], axis=1).astype(np.int)[:, np.newaxis]
        token_stim[token]['stim_int'] = transform_intensity(token_stim[token]['intensity'])
        token_stim[token]['bias_ones'] = np.ones((token_stim[token]['stim_int'].shape[0], 1))

    delays = get_delays()
    padding_all = np.zeros((75, 23))
    padding_all[:,22] = 1
    padding_other = np.zeros((75, 13))
    padding_other[:,12] = 1

    token_stim_mat = {}
    for token, s in token_stim.iteritems():
        token_stim_mat[token] = {}
        token_stim_mat[token]['abs_bin'] = np.hstack([s['stim_pitch_abs'], s['pitch_binary'], s['stim_int'], s['bias_ones']])
        token_stim_mat[token]['rel_bin'] = np.hstack([s['stim_pitch_rel'], s['pitch_binary'], s['stim_int'], s['bias_ones']])
        token_stim_mat[token]['all'] = np.hstack([s['stim_pitch_abs'], s['stim_pitch_rel'], s['pitch_binary'], s['stim_int'], s['bias_ones']])
        token_stim_mat[token]['all'][np.isnan(token_stim_mat[token]['all'])] =0
        token_stim_mat[token]['abs_bin'][np.isnan(token_stim_mat[token]['abs_bin'])] =0 
        token_stim_mat[token]['rel_bin'][np.isnan(token_stim_mat[token]['rel_bin'])] =0 
        token_stim_mat[token]['all_padded'] = np.concatenate([padding_all, token_stim_mat[token]['all'], padding_all], axis=0)
        token_stim_mat[token]['abs_padded'] = np.concatenate([padding_other, token_stim_mat[token]['abs_bin'], padding_other], axis=0)
        token_stim_mat[token]['rel_padded'] = np.concatenate([padding_other, token_stim_mat[token]['rel_bin'], padding_other], axis=0)

        token_stim_mat[token]['dstims_all'] = get_dstim(token_stim_mat[token]['all_padded'], delays)
        token_stim_mat[token]['dstims_abs'] = get_dstim(token_stim_mat[token]['abs_padded'], delays)
        token_stim_mat[token]['dstims_rel'] = get_dstim(token_stim_mat[token]['rel_padded'], delays)

    return token_stim_mat

def predict_response_to_intonation_stims(subject_number, chan):
    token_stim_mat = get_intonation_tokens_stim()
    r_all, r_abs, r_rel, abs_r2, rel_r2, wts_all, wts_abs, wts_rel = load_cv_model_fold(subject_number)
    wts_all = np.nanmean(wts_all, axis=2)
    wts_abs = np.nanmean(wts_abs, axis=2)
    wts_rel = np.nanmean(wts_rel, axis=2)

    tokens_male_st1 = ['sn1_st1_sp1', 'sn2_st1_sp1', 'sn3_st1_sp1', 'sn4_st1_sp1']
    tokens_male_st2 = ['sn1_st2_sp1', 'sn2_st2_sp1', 'sn3_st2_sp1', 'sn4_st2_sp1']
    tokens_male_st3 = ['sn1_st3_sp1', 'sn2_st3_sp1', 'sn3_st3_sp1', 'sn4_st3_sp1']
    tokens_male_st4 = ['sn1_st4_sp1', 'sn2_st4_sp1', 'sn3_st4_sp1', 'sn4_st4_sp1']

    tokens_female_st1 = ['sn1_st1_sp2', 'sn2_st1_sp2', 'sn3_st1_sp2', 'sn4_st1_sp2', 'sn1_st1_sp3', 'sn2_st1_sp3', 'sn3_st1_sp3', 'sn4_st1_sp3']
    tokens_female_st2 = ['sn1_st2_sp2', 'sn2_st2_sp2', 'sn3_st2_sp2', 'sn4_st2_sp2', 'sn1_st2_sp3', 'sn2_st2_sp3', 'sn3_st2_sp3', 'sn4_st2_sp3']
    tokens_female_st3 = ['sn1_st3_sp2', 'sn2_st3_sp2', 'sn3_st3_sp2', 'sn4_st3_sp2', 'sn1_st3_sp3', 'sn2_st3_sp3', 'sn3_st3_sp3', 'sn4_st3_sp3']
    tokens_female_st4 = ['sn1_st4_sp2', 'sn2_st4_sp2', 'sn3_st4_sp2', 'sn4_st4_sp2', 'sn1_st4_sp3', 'sn2_st4_sp3', 'sn3_st4_sp3', 'sn4_st4_sp3']

    tokens = [tokens_male_st1, tokens_male_st2, tokens_male_st3, tokens_male_st4, tokens_female_st1, tokens_female_st2, tokens_female_st3, tokens_female_st4]

    all_preds_tokens = [[get_all_pred(wts_all, token_stim_mat[token]['dstims_all']) for token in tokens_list] for tokens_list in tokens]
    abs_pred_tokens = [[get_all_pred(wts_abs, token_stim_mat[token]['dstims_abs']) for token in tokens_list] for tokens_list in tokens]
    rel_pred_tokens = [[get_all_pred(wts_rel, token_stim_mat[token]['dstims_rel']) for token in tokens_list] for tokens_list in tokens]

    all_preds1 = [np.mean(np.array(a), axis=0)[:,50:350] for a in all_preds_tokens]
    abs_pred1 = [np.mean(np.array(a), axis=0)[:,50:350] for a in abs_pred_tokens]
    rel_pred1 = [np.mean(np.array(a), axis=0)[:,50:350] for a in rel_pred_tokens]

    all_preds_ste1 = [np.std(np.array(a), axis=0)[:,50:350]/np.sqrt(len(a)) for a in all_preds_tokens]
    abs_pred_ste1 = [np.std(np.array(a), axis=0)[:,50:350]/np.sqrt(len(a)) for a in abs_pred_tokens]
    rel_pred_ste1 = [np.std(np.array(a), axis=0)[:,50:350]/np.sqrt(len(a)) for a in rel_pred_tokens]

    tokens_male = tokens_male_st1 + tokens_male_st2 + tokens_male_st3 + tokens_male_st4
    tokens_female = tokens_female_st1 + tokens_female_st2 + tokens_female_st3 + tokens_female_st4

    tokens = [tokens_male, tokens_female]

    all_preds_tokens = [[get_all_pred(wts_all, token_stim_mat[token]['dstims_all']) for token in tokens_list] for tokens_list in tokens]
    abs_pred_tokens = [[get_all_pred(wts_abs, token_stim_mat[token]['dstims_abs']) for token in tokens_list] for tokens_list in tokens]
    rel_pred_tokens = [[get_all_pred(wts_rel, token_stim_mat[token]['dstims_rel']) for token in tokens_list] for tokens_list in tokens]

    all_preds2 = [np.mean(np.array(a), axis=0)[:,50:350] for a in all_preds_tokens]
    abs_pred2 = [np.mean(np.array(a), axis=0)[:,50:350] for a in abs_pred_tokens]
    rel_pred2 = [np.mean(np.array(a), axis=0)[:,50:350] for a in rel_pred_tokens]

    all_preds_ste2 = [np.std(np.array(a), axis=0)[:,50:350]/np.sqrt(len(a)) for a in all_preds_tokens]
    abs_pred_ste2 = [np.std(np.array(a), axis=0)[:,50:350]/np.sqrt(len(a)) for a in abs_pred_tokens]
    rel_pred_ste2 = [np.std(np.array(a), axis=0)[:,50:350]/np.sqrt(len(a)) for a in rel_pred_tokens]

    return all_preds1, all_preds_ste1, abs_pred1, abs_pred_ste1, rel_pred1, rel_pred_ste1, all_preds2, all_preds_ste2, abs_pred2, abs_pred_ste2, rel_pred2, rel_pred_ste2

def nan_zscore(a):
    a[~np.isnan(a)] = zscore(a[~np.isnan(a)])
    return a

def randomize_timit_pitch_contours(timit_pitch, save_as=None):
    timit_names = timit_pitch.index.get_level_values(0).unique().values
    lengths = []
    for sentence_name in timit_names:
        sentence = timit_pitch.loc[sentence_name]
        lengths.append(len(sentence))
    lengths = np.array(lengths)
    counts, bins = np.histogram(lengths, 5) #bins == [98, 130.2, 162.4, 194.6, 226.8, 259]
    bins[-1] = bins[-1] + 1
    indexes = np.arange(len(timit_names))
    bin_indexes = []
    shuffled_bin_indexes = []
    for i in range(1, 6):
        bin_index = indexes[np.digitize(lengths, bins) == i]
        bin_indexes.append(np.copy(bin_index))
        shuffled_bin_index = np.copy(bin_index)
        random.shuffle(shuffled_bin_index)
        shuffled_bin_indexes.append(shuffled_bin_index)
    shuffled_indexes = np.zeros((len(timit_names)), dtype=int)
    for i in range(5):
        shuffled_indexes[bin_indexes[i]] = shuffled_bin_indexes[i]

    relevant_columns = ['abs_pitch', 'rel_pitch_global', 'abs_pitch_erb', 'rel_pitch_global_erb',
                         'abs_pitch_change', 'abs_pitch_erb_change', 'zscore_intensity']

    pitch_intensity_tables = []
    for sentence_name, shuffle_i in zip(timit_names, shuffled_indexes):
        pitch_intensity = timit_pitch.loc[sentence_name]
        current_length = len(pitch_intensity)
        new_pitch_intensity = timit_pitch.loc[timit_names[shuffle_i]]
        new_length = len(new_pitch_intensity)
        if current_length - new_length < 0:
            pitch_intensity = new_pitch_intensity.iloc[0:current_length - new_length]
        else:
            pitch_intensity.loc[0:new_length, relevant_columns] = new_pitch_intensity.loc[0:new_length, relevant_columns]
            if current_length - new_length > 0:
                lowest_intensity = np.nanmin(new_pitch_intensity['zscore_intensity'])
                pitch_intensity.loc[new_length:current_length, 'abs_pitch'] = np.NaN
                pitch_intensity.loc[new_length:current_length, 'abs_pitch_erb'] = np.NaN
                pitch_intensity.loc[new_length:current_length, 'rel_pitch_global'] = np.NaN
                pitch_intensity.loc[new_length:current_length, 'rel_pitch_global_erb'] = np.NaN
                pitch_intensity.loc[new_length:current_length, 'zscore_intensity'] = lowest_intensity
        pitch_intensity_tables.append(pitch_intensity)

    timit_pitch = pd.concat(pitch_intensity_tables, keys=timit_names)
    #print(np.mean(timit_pitch['log_hz']))
    #print(np.std(timit_pitch['log_hz']))

    if save_as is not None:
        filename = os.path.join(processed_timit_data_path, 'timit_pitch_shuffle_' + str(save_as) + '.h5')
        timit_pitch.to_hdf(filename, 'timit_pitch_shuffle_' + str(save_as))

    return timit_pitch

def plot_prediction_overlay(test_corr, resp, all_pred, pitch_intensity, chans=[], start_time=5000):
    chans = np.array(chans)
    if chans.shape[0] == 0:
        sorted_chans = test_corr.argsort()
        chans = sorted_chans[-3:]
    chans = np.array(chans)
    corrs = np.array(test_corr)[chans]
    n_chans = chans.shape[0]
    fig = plt.figure(figsize=(10, 2*n_chans + 2))
    for i, (chan, r) in enumerate(zip(chans, corrs)):
        ax = fig.add_subplot(n_chans+1, 1, i+1)
        ax.plot(all_pred[chan, start_time:start_time+500], 'k')
        ax.plot(resp[start_time:start_time+500, chan], 'r')
        ax.set_title("Channel %d, r=%2.2f"%(chan, r))
    ax = fig.add_subplot(n_chans+1, 1, n_chans+1)
    ax.plot(pitch_intensity[3,start_time:start_time+500], 'm')
    ax.set_title('Stimulus')
    fig.tight_layout()
    return fig

def get_channel_order():
    channel_order = []
    for i in np.arange(16):
        x = np.arange(256-i,16-i-1,-16)
        channel_order.append(x)
    return np.hstack(channel_order)

def plot_trf(wts, chan, wts_shape=(46,23), wts1=(0, 10), wts2=(10, 20), min_max=(0, 20), wts1_label=None, wts2_label=None):
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)
    min_value = np.min(wts[chan].reshape(*wts_shape)[3:43, min_max[0]:min_max[1]])
    max_value = np.max(wts[chan].reshape(*wts_shape)[3:43, min_max[0]:min_max[1]])
    abs_value = np.max(np.abs([min_value, max_value]))
    min_value = -1 * abs_value
    max_value = abs_value
    im1 = axs[0].imshow(np.fliplr(np.flipud(wts[chan].reshape(*wts_shape)[3:43, wts1[0]:wts1[1]].T)), cmap=plt.get_cmap('RdBu_r'), aspect="auto")
    im3 = axs[1].imshow(np.fliplr(np.flipud(wts[chan].reshape(*wts_shape)[3:43, wts2[0]:wts2[1]].T)), cmap=plt.get_cmap('PuOr_r'), aspect="auto")
    for im in [im1, im3]:
        im.set_clim((-1 * abs_value, abs_value))
    min_tick_value = np.trunc(np.ceil(min_value * 100))/100
    max_tick_value = np.trunc(np.floor(max_value * 100))/100
    fig.colorbar(im1, ax=axs[0], ticks=[min_tick_value, 0, max_tick_value], aspect=10)
    fig.colorbar(im3, ax=axs[1], ticks=[min_tick_value, 0, max_tick_value], aspect=10)
    if wts1_label is None:
        im1.axes.set(yticks=(0, 3, 6, 9), yticklabels=[250, 200, 150, 90], ylabel="Absolute pitch (Hz)")
    if wts2_label is None:
        im3.axes.set(xticks=[0, 39], xticklabels=[400, 0], xlabel="Delay (ms)", 
            yticks=(0,3,6,9), yticklabels=[1.7, 0.6, -0.5, -1.7], ylabel="Relative pitch (z-score)")
    fig.tight_layout()
    return fig

def plot_trf_rel_versus_change(wts, chan):
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)
    min_value = np.min(wts[chan].reshape(46,33)[3:43, 10:30])
    max_value = np.max(wts[chan].reshape(46,33)[3:43, 10:30])
    abs_value = np.max(np.abs([min_value, max_value]))
    min_value = -1 * abs_value
    max_value = abs_value
    im1 = axs[0].imshow(np.fliplr(np.flipud(wts[chan].reshape(46, 33)[3:43,20:30].T)), cmap=plt.get_cmap('RdBu_r'), aspect="auto")
    im3 = axs[1].imshow(np.fliplr(np.flipud(wts[chan].reshape(46, 33)[3:43,10:20].T)), cmap=plt.get_cmap('PuOr_r'), aspect="auto")
    for im in [im1, im3]:
        im.set_clim((-1 * abs_value, abs_value))
    min_tick_value = np.trunc(np.ceil(min_value * 100))/100
    max_tick_value = np.trunc(np.floor(max_value * 100))/100
    fig.colorbar(im1, ax=axs[0], ticks=[min_tick_value, 0, max_tick_value], aspect=10)
    fig.colorbar(im3, ax=axs[1], ticks=[min_tick_value, 0, max_tick_value], aspect=10)
    #im1.axes.set(yticks=(0, 3, 6, 9), yticklabels=[250, 200, 150, 90], ylabel="Absolute pitch change (Hz')")
    #im3.axes.set(xticks=[0, 39], xticklabels=[400, 0], xlabel="Delay (ms)", 
    #    yticks=(0,3,6,9), yticklabels=[1.7, 0.6, -0.5, -1.7], ylabel="Relative pitch (z-score)")
    fig.tight_layout()
    return fig

def plot_trfs(wts, test_corr, delays, vlim=None, with_edges=True):
    if with_edges:
        trfs = [wts[chan, :].reshape(len(delays)+6, -1)[3:-3,:-2].T for chan in range(wts.shape[0])]
    else:
        trfs = [wts[chan, :].reshape(len(delays), -1).T for chan in range(wts.shape[0])]
    titles = [test_corr[chan] for chan in range(wts.shape[0])]
    if vlim is None:
        val = np.max(np.abs([np.min(trfs), np.max(trfs)]))
        vlim = [-1*val, val]
    figs = []
    figs.append(plot_grid(trfs[:256], titles[:256], vlim=vlim))
    if(len(trfs) == 288):
        figs.append(plot_heschls(trfs[256:], titles[256:], vlim=vlim))
    return figs

def plot_grid(data, titles, vlim=(-0.2, 0.2)):
    channel_order = get_channel_order()
    fig = plt.figure(figsize=(40, 40))

    for i, (data_chan, title) in enumerate(zip(data, titles)):
        ax = fig.add_subplot(16, 16, channel_order[i])
        plt.imshow(data_chan, vmin=vlim[0], vmax=vlim[1], cmap=plt.get_cmap("RdBu_r"), aspect="auto", interpolation="none")
        ax.set_title('%d, r=%2.2f'%(i, title))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig

def plot_heschls(data, titles, vlim=(-0.2, 0.2)):
    channel_order = np.arange(32) + 256 + 1
    fig = plt.figure(figsize=(20, 10))

    for i, (data_chan, title) in enumerate(zip(data, titles)):
        ax = fig.add_subplot(4, 8, channel_order[i] - 256)
        plt.imshow(data_chan, vmin=vlim[0], vmax=vlim[1], cmap=plt.get_cmap("RdBu_r"), aspect="auto", interpolation="none")
        ax.set_title('%d, r=%2.2f'%(i + 256, title))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig
