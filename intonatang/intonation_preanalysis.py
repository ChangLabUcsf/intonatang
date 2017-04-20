from __future__ import division, print_function, absolute_import

import os
processed_data_path = os.path.join(os.path.dirname(__file__), 'processed_neural_data')
subject_data_path = os.path.join(os.path.dirname(__file__), 'data', 'subject_data')

import numpy as np
from scipy.stats import zscore
import scipy.io as sio

from .intonation_subject_data import get_blocks_for_subject_number, get_stims_for_subject_number
from .intonation_subject_data import get_sentence_numbers_sentence_types_speakers_for_stims_list


def get_times_hg_for_subject_number(subject_number, only_good_trials=False, control_stim=False, missing_f0_stim=False, use_log_hg=False):
    """Used to process .mat data files, called by save_Y_mat_sns_sts_sps_for_subject_number

    For each subject, all block data (hg, times, bcs, and badTimeSegments) is loaded and neural data is processed 
    (bad time segments are removed, i.e. set as NaN, and each channel is z-scored across time). 

    Args:
        subject_number (int): xx in ECxx
        only_good_trials (bool): return only good_trials
        control_stim (bool): set as True to process non-speech control data
        use_log_hg (bool): return log hg instead of hg (log is taken for each high-gamma band and then averaged during preprocessing)

    Returns:
        (tuple):
            * **times** (*list of ndarrays*): list of stimulus onsets (seconds) for each block. Each item in times has dimensions 1 x number of trials
            * **good_trials** (*list of lists*): list of good trials for each block
            * **hgs_toreturn** (*list of ndarrays*): list of processed hg data for each block
            * **gcs** (*list*): good channels (channels that were good in every block)
    """
    assert (control_stim and missing_f0_stim) is False

    blocks = get_blocks_for_subject_number(subject_number, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    hg = []
    times = []
    bcs = []
    bad_times = []

    #.mat data files contain hg, log_hg, times, bcs, and badTimeSegments.
    #This first loop loads these variables from the .mat file and saves them in lists for further processing.
    for block in blocks:
        data = sio.loadmat(get_full_data_path_for_subject_number_and_block(subject_number, block))
        if use_log_hg:
            hg.append(data['EC' + str(subject_number) + '_B' + str(block) + '_log_hg_100Hz'])
        else:
            hg.append(data['EC' + str(subject_number) + '_B' + str(block) + '_hg_100Hz'])
        times.append(data['times'])
        bcs.append(data['bcs'])
        bad_times.append(data['badTimeSegments'])
        hg[-1][np.isnan(hg[-1])] = -1000 #to make any NaN values not NaN

    #Using python sets to get list of channels that were marked "bad" in any block
    all_bcs = set()
    for bcs_block in bcs:
        all_bcs.update(set(bcs_block[0].tolist()))
    gcs = list(set(np.arange(256)) - all_bcs) #good channels were not "bad" in any block

    #The following loop sets NaN values in the neural data for each bad time segment and then
    # saves a list of bad indexes (indexing with respect to the neural data)
    bad_indexes = []
    for bad_times_block, hg_block in zip(bad_times, hg):
        for bad_time in bad_times_block:
            hg_block[:, int(np.round(bad_time[0]*100)): int(np.round(bad_time[1]*100))] = np.NaN
        bad_indexes.append(np.isnan(hg_block))

    #To get a list of good trials for each block, we define bad trials as ones where stimulus on times overlap with
    # bad time segments, and take trials which aren't bad trials.
    good_trials = []
    for times_block, bad_indexes_block, hg_block in zip(times, bad_indexes, hg):
        stim_times = [[t * 100, (t+2.2)*100] for t in times_block]
        stim_times = np.array(stim_times)[0]
        bad_trials_block = []
        for i, stim_time in enumerate(stim_times.T):
            if stim_time[0] == 1: #This was a hack to remove a trial that wasn't recorded. The flag value had to be 1 (instead of 0 or a neg. number) because Y_mat is calculated before bad trials are removed.
                bad_trials_block.append(i)
            else:
                indexes = np.zeros((hg_block.shape[1]))
                indexes[int(stim_time[0]): int(stim_time[1])] = 1
                if np.logical_and(indexes==1, bad_indexes_block==True).any():
                    bad_trials_block.append(i)
        if control_stim:
            good_trials.append(list(set(np.arange(120)) - set(bad_trials_block)))
        elif missing_f0_stim:
            good_trials.append(list(set(np.arange(144)) - set(bad_trials_block)))
        else:
            good_trials.append(list(set(np.arange(96)) - set(bad_trials_block)))
    if only_good_trials:
        return good_trials

    #The list of hg data for each block is z-scored across time. Bad times are ignored and left as NaNs.
    hgs_toreturn = []
    for hg_block in hg:
        hg_zscore = []
        for hg_chan in hg_block:
            z = np.copy(hg_chan)
            z[~np.isnan(z)] = zscore(z[~np.isnan(z)])
            hg_zscore.append(z)
        hgs_toreturn.append(np.array(hg_zscore))

    return times, good_trials, hgs_toreturn, gcs

def get_bcs(subject_number):
    """Returns list of all bad channels for each subject

    Bad channels are channels that were bad in at least one block. 

    Args:
        subject_number (int): xx in ECxx
    
    Returns:
        (list of ints):
            * **bcs**: list of bad channels
    """
    blocks = get_blocks_for_subject_number(subject_number)
    blocks_control = get_blocks_for_subject_number(subject_number, control_stim=True)
    blocks_missing_f0 = get_blocks_for_subject_number(subject_number, missing_f0_stim=True)
    bcs = []
    for block in blocks + blocks_control + blocks_missing_f0:
        data = sio.loadmat(get_full_data_path_for_subject_number_and_block(subject_number, block))
        bcs.append(data['bcs'])
    all_bcs = set()
    for bcs_block in bcs:
        all_bcs.update(set(bcs_block[0].tolist()))
    bcs = list(all_bcs)
    bcs.sort()
    return bcs

def get_gcs(subject_number):
    """Returns list of all good channels for each subject

    Good channels are channels that were good in all blocks (said another way, they are channels which 
    were never bad).

    Args:
        subject_number (int): xx in ECxx
    
    Returns:
        (list of ints): 
            * **gcs**: list of good channels
    """
    bcs = get_bcs(subject_number)
    gcs = list(set(np.arange(256)) - set(bcs))
    gcs.sort()
    return gcs

def get_full_data_path_for_subject_number_and_block(subject_number, block, data_path_different=None):
    """Returns path to the .mat data for each subject and block combination.

    adds "ECxx/ECxx_Bxxx/ECxx_Bxxx.mat" where xx is the subject_number and xxx is the block to the data_path.
    data_path is defined at the top of this file as a global constant.
    A different data path can also be passed in through the parameter data_path_different 

    Args:
        subject_number (int): xx in ECxx
        block (int): block number
        data_path_different (str): optional if path to data is not the globally defined data_path

    Returns:
        (str):
            * **full_data_path**: data_path (or data_path_different when passed in) + "ECxx/ECxx_Bxxx/ECxx_Bxxx.mat"
    """
    if data_path_different is not None:
        return os.path.join(data_path_different, 'EC' + str(subject_number), 'EC' + str(subject_number) + '_B' + str(block), 'EC' + str(subject_number) + '_B' + str(block) + '.mat')
    else:
        return os.path.join(subject_data_path, 'EC' + str(subject_number), 'EC' + str(subject_number) + '_B' + str(block), 'EC' + str(subject_number) + '_B' + str(block) + '.mat')

def get_all_good_trials(good_trials, control_stim=False, missing_f0_stim=False):
    """From list of good trials for each block, return one list of all good trials.

    The numbers in all_good_trials are used to index into data which is concatenated from all blocks. 
    For example, for two blocks with 96 trials in which the first two trials of each block are good, 
    all_good_trials would be [0, 1, 96, 97]

    Args: 
        good_trials (list of lists): list of good trials for each block, returned from get_times_hg_for_subject_number
        control_stim (bool): whether using non-speech control stimuli
    
    Returns:
        (list):
            * **all_good_trials**: one list of all good trials
    """
    assert (control_stim and missing_f0_stim) is False

    if control_stim:
        trials_per_block = 120
    elif missing_f0_stim:
        trials_per_block = 144
    else:
        trials_per_block = 96
    all_good_trials = []
    for i, g_block in enumerate(good_trials):
        for g in g_block:
            all_good_trials.append(g + i*trials_per_block)
    return all_good_trials

def get_sentence_numbers_sentence_types_speakers_for_subject_number(subject_number, good_trials=None, control_stim=False, missing_f0_stim=False):
    """Returns lists of sentence, intonation, and speaker conditions with bad trials removed for a given subject

    Each list of conditions contains integers from 1 to 4, 1 to 4, and 1 to 3, respectively.

    Args:
        subject_number (int): xx in ECxx
        good_trials (list of lists): list of good trials in each block returned from get_times_hg_for_subject_number
        control_stim (bool): set to True for non-speech control_stim

    Returns:
        (tuple):
            * **sns** (*list*): sentence conditions (sentence numbers, 1 to 4)
            * **sts** (*list*): intonation conditions (sentence types, 1 to 4)
            * **sps** (*list*): speaker conditions (speakers, 1 to 3)
    """
    assert (control_stim and missing_f0_stim) is False

    if good_trials is None:
        good_trials = get_times_hg_for_subject_number(subject_number, only_good_trials=True, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    all_good_trials = get_all_good_trials(good_trials, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    stims_list = get_stims_for_subject_number(subject_number, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    sns, sts, sps = get_sentence_numbers_sentence_types_speakers_for_stims_list(stims_list, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    return sns[all_good_trials], sts[all_good_trials], sps[all_good_trials]

def save_Y_mat_sns_sts_sps_for_subject_number(subject_number, control_stim=False, missing_f0_stim=False, return_raw_data=False, zscore_to_silence=True):
    """Pre-analysis processing pipeline. Creates matrix of hg activity and lists of stimulus information.

    This function saves a .mat file called ECXXX_Y_mat.mat containing the variables: 
    Y_mat, sentence_numbers, sentence_types, speakers, and Y_mat_plotter

    Y_mat is time-averaged data for encoding analysis. 
    Y_mat_plotter is full time series data for visualization and use with the Plotter.labels

    To load the data that is saved, use load_Y_mat_sns_sts_sps_for_subject_number

    Args:
        subject_number (int): xx in ECxx
        control_stim (bool): set to True for non-speech control_stim
        return_raw_data (bool): set to True to return (times, good_trials, hg, gcs) like get_times_hg_for_subject_number
        zscore_to_silence (bool): normalize neural data to pre-stimulus baseline rather than entire block
    """
    assert (control_stim and missing_f0_stim) is False

    times, good_trials, hg, gcs = get_times_hg_for_subject_number(subject_number, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    all_good_trials = get_all_good_trials(good_trials, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    sns, sts, sps = get_sentence_numbers_sentence_types_speakers_for_subject_number(subject_number, good_trials, control_stim=control_stim, missing_f0_stim=missing_f0_stim)
    Y_mat_plotter = get_concatenated_data(times, [hg_block[:256] for hg_block in hg], zscore=False, back=25, forward=275, zscore_to_silence=zscore_to_silence)
    Y_mat_plotter = Y_mat_plotter[:, :, all_good_trials]
    Y_mat, centers = get_time_averaged_data(times, [hg_block[:256] for hg_block in hg], zscore=False, zscore_to_silence=zscore_to_silence)
    Y_mat = Y_mat[:, :, all_good_trials]

    base_string = "EC" + str(subject_number) + "_Y_mat"
    if zscore_to_silence:
        base_string = base_string + "_zscore_to_silence"
    else:
        base_string = base_string + "_zscore_to_block"
    if control_stim:
        base_string = base_string + "_control"
    if missing_f0_stim:
        base_string = base_string + "_missing_f0"

    data =  {'Y_mat': Y_mat, 'Y_mat_plotter': Y_mat_plotter, 'sentence_numbers': sns, 'sentence_types': sts, 'speakers': sps}
    filename = os.path.join(processed_data_path, base_string + ".mat")
    sio.savemat(filename, data)

    if return_raw_data:
        return times, good_trials, hg, gcs


def load_Y_mat_sns_sts_sps_for_subject_number(subject_number, control_stim=False, missing_f0_stim=False, zscore_to_silence=True):
    """Loads data for analysis from file saved by save_Y_mat_sns_sts_sps_for_subject_number

    Returns:
        (tuple):
            * **Y_mat**: time averaged neural data for encoding analysis
            * **sentence_numbers**: list of sentence numbers (1, 2, 3, or 4)
            * **sentence_types**: list of sentence types or intonation conditions (1, 2, 3, 4)
            * **speaker**: list of speakers (1, 2, 3)
            * **Y_mat_plotter**: neural data for visualization (not time averaged).
    """
    assert (control_stim and missing_f0_stim) is False

    base_string = "EC" + str(subject_number) + "_Y_mat"
    if zscore_to_silence:
        base_string = base_string + "_zscore_to_silence"
    else:
        base_string = base_string + "_zscore_to_block"
    if control_stim:
        base_string = base_string + "_control"
    if missing_f0_stim:
        base_string = base_string + "_missing_f0"

    filename = os.path.join(processed_data_path, base_string + ".mat")
    data = sio.loadmat(filename)

    assert data['Y_mat'].shape[1] == 101
    return data['Y_mat'], data['sentence_numbers'][0], data['sentence_types'][0], data['speakers'][0], data['Y_mat_plotter']

def get_stg(subject_number, path='Imaging/elecs/', filename='TDT_elecs_all.mat'):
    """Returns list of electrode numbers that are on STG.

    Loads the anatomy files from the data_path + 'EC' + subject_number + path + filename.
    """
    data = sio.loadmat(data_path + '/EC' + str(subject_number) + '/' + path + filename)['anatomy']
    stg= np.arange(256)[np.logical_and(data[:,3] == 'superiortemporal',data[:,2] == 'grid')]
    return stg

def get_timelocked_activity(times, hg, zscore=True, hz=100, back=0, forward=250, zscore_to_silence=True):
    """Returns a n_chans x n_timepoints x n_trials matrix of high-gamma activity. 

    Args:
        times: times[0] contains trial start times in seconds.
        hg: full time-series of high-gamma for multiple electrodes (n_chans x nt)
        back: number of time samples to take preceding trial start
        forward: number of time samples to take following trial start.
        zscore_to_silence: boolean, whether z-scoring should be done to a prestimulus baseline
    """
    if zscore:
        for i in np.arange(hg.shape[0]):
            hg[i, ~np.isnan(hg[i])] = zscore(hg[i, ~np.isnan(hg[i])])

    Y_mat = np.zeros((hg.shape[0], int(back+forward), np.shape(times)[1]), dtype=float)

    if zscore_to_silence:
        baseline = []
        for i, seconds in enumerate(times[0]):
            index = int(np.round(seconds*hz))
            baseline.append(hg[:, index-30:index])

        baseline = np.concatenate(baseline, axis=1)

        baseline_mean = np.nanmean(baseline, axis=1)
        baseline_std = np.nanstd(baseline, axis=1)

        print("baseline length: " + str(len(baseline[199])))

        print("baseline_mean: " + str(baseline_mean[199]))
        print("baseline_std: " + str(baseline_std[199]))

    for i, seconds in enumerate(times[0]):
        index = int(np.round(seconds * hz))
        if zscore_to_silence:
            try:
                Y_mat[:,:,i] = hg[:, int(index-back):int(index+forward)]
            except:
                print('Error creating Y_mat at trial i: ' + str(i) + ' index: ' + str(index))
            for t in range(Y_mat.shape[1]):
                Y_mat[:, t, i] = (Y_mat[:, t, i] - baseline_mean)/baseline_std
        else:
            try:
                Y_mat[:,:,i] = hg[:, index-back:index+forward]
            except:
                print('Error creating Y_mat at trial i: ' + str(i) + ' index: ' + str(index))
            if np.sum(np.isnan(Y_mat[:,:,i])) > 0:
                print(i)
                print(np.sum(np.isnan(Y_mat[:,:,i])))

    return Y_mat

def get_concatenated_data(times_list, hg_list, hz=100, back=0, forward=250, zscore=True, zscore_to_silence=True):
    """Use to get time locked activity for multiple blocks. 

    Args:
        times_list: list of individual times variables where times[0] are start times_list
        hg_list: list of hg for blocks in same order as times_list
    """
    Y_cat = np.concatenate([get_timelocked_activity(times, hg, zscore=zscore, hz=hz, back=back, forward=forward, zscore_to_silence=zscore_to_silence) for (times, hg) in zip(times_list, hg_list)], 2)
    return Y_cat
    

def get_time_averaged_data(times_list, hg_list, window=6, hz=100, back=15, forward=285, zscore=True, zscore_to_silence=True):
    """Averages data in a moving window (each step is half a window) to smooth high-gamma for encoding analyses.

    Args:
        window: number of samples in each window. Use an even number, so steps are an integer number of samples.
        back: number of samples back in time, center of the first window
        forward: center of the last window. 
    """
    Y_cat = get_concatenated_data(times_list, hg_list, hz=hz, zscore=zscore, back=back+window/2, forward=forward+window/2, zscore_to_silence=zscore_to_silence)
    centers = get_centers(back=back, forward=forward, window=6)
    Y_mat = np.zeros((Y_cat.shape[0], centers.shape[0], Y_cat.shape[2]))
    
    for i, c in enumerate(centers):
        Y_mat[:, i, :] = np.nanmean(Y_cat[:, int(c+back):int(c+back+window), :], axis=1)
    return Y_mat, centers

def get_centers(back=15, forward=285, window=6):
    """Returns a numpy array of center indexes (no parameters needed for default)

    The returned list of `centers` starts at `-1 * back` and steps every half `window`. `centers` will
    end at `forward` if `forward` can be reached from `-1 * back` in half `window` steps.

    The default arguments (back=15, forward=285, and window=6) returns a length 101 ndarray.
    """
    centers = np.arange(-1*back, forward+window/2, window/2)
    return centers