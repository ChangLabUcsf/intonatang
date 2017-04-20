from __future__ import print_function, division, absolute_import

import os
results_path = os.path.join(os.path.dirname(__file__), 'results')

import numpy as np
import scipy.io as sio

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tqdm import tqdm

from .intonation_preanalysis import load_Y_mat_sns_sts_sps_for_subject_number

def test_invariance_control(subject_number, solver="lsqr", shrinkage=1, n_perms=1000):
    """Run the LDA invariance analysis on nonspeech control data.

    This function contains the pipeline for the nonspeech invariance analysis. Here, we use LDA to fit a model on the 
    neural activity time series to predict intonation condition using speech data. We then test this model on both held 
    out speech data and the nonspeech control data. The accuracies for the model on the held out speech data and nonspeech
    data are then returned and can be saved with ``save_control_test_accs``.

    Args:
        subject_number: xxx in ECxxx
        solver: The type of solver to use for LDA. Can be "svd" or "lsqr". Only "lsqr" supports shrinkage/regularization.
        shrinkage: The shrinkage parameter between 0 and 1. Default of 1 is diagonal LDA. 
        n_perms: The number of permutations for held out speech data to run.

    Returns:
        (tuple):
            * **accs** (*ndarray*): shape is (n_chans x n_perms x 3). The last dimension contains accuracy values for held out 
                speech data, shuffled speech data, and shuffled nonspeech data
            * **accs_test** (*ndarray*): shape is (n_chans). Contains accuracy value for nonspeech data. 
    """
    Y_mat, sns, sts, sps, Y_mat_plotter = load_Y_mat_sns_sts_sps_for_subject_number(subject_number)
    Y_mat_c, sns_c, sts_c, sps_c, Y_mat_plotter_c = load_Y_mat_sns_sts_sps_for_subject_number(subject_number, control_stim=True)

    # find time-points with NaNs across all trials to be excluded.
    Y_all = np.concatenate([Y_mat, Y_mat_c], axis=2)
    bad_time_indexes = np.isnan(np.sum(Y_all, axis=2))

    print(Y_mat.shape)
    n_chans, n_timepoints, n_trials = Y_mat.shape

    if solver == "svd":
        lda = LinearDiscriminantAnalysis()
    elif solver == "lsqr":
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    accs = np.zeros((n_chans, n_perms, 3))
    accs_test = np.zeros((n_chans))

    n_train_100 = len(sts)
    n_test_100 = len(sts_c)
    n_train_80 = int(np.round(n_train_100*0.8))
    n_train_20 = n_train_100 - n_train_80

    ofs1 = sts
    ofs2 = sts_c

    # calculate performance accuracy of speech-fit LDA model on nonspeech data for each chan.
    for chan in np.arange(n_chans):
        Y_train = Y_mat[chan][~bad_time_indexes[chan]].T
        Y_test = Y_mat_c[chan][~bad_time_indexes[chan]].T

        if Y_train.shape[1] < 1 or Y_test.shape[1] < 1:
            accs_test[chan] = np.NaN
        else:
            lda.fit(Y_train, ofs1)
            accs_test[chan] = lda.score(Y_test, ofs2)

    # calculate distribution of performance accuracies on held on speech data, shuffled speech data, and shuffled nonspeech data.
    for p in tqdm(np.arange(n_perms)):
        rand_perm_train = np.random.permutation(n_train_100)
        rand_perm_test = np.random.permutation(n_test_100)

        shuffle_train = np.random.permutation(n_train_100)
        shuffle_test = np.random.permutation(n_test_100)

        for chan in np.arange(n_chans):
            Y_train = Y_mat[chan][~bad_time_indexes[chan]].T
            Y_test = Y_mat_c[chan][~bad_time_indexes[chan]].T

            if Y_train.shape[1] < 1 or Y_test.shape[1] < 1:
                accs[chan, p, :] = np.NaN
            else:
                # fit the model on a random 80% of the speech data.
                lda.fit(Y_train[rand_perm_train][np.arange(n_train_80)], ofs1[rand_perm_train][np.arange(n_train_80)])

                # use the remaining 20% to bootstrap a set with n_test_100 trials.
                Y_speech_test = Y_train[rand_perm_train][np.arange(n_train_80, n_train_100)]
                Y_speech_shuffle = Y_train[shuffle_train][np.arange(n_train_80, n_train_100)]
                ofs1_test = ofs1[rand_perm_train][np.arange(n_train_80, n_train_100)]
                sample_inds = np.random.randint(0, Y_speech_test.shape[0], size=(n_test_100))
                Y_speech_test = Y_speech_test[sample_inds]
                Y_speech_shuffle = Y_speech_shuffle[sample_inds]
                ofs1_test = ofs1_test[sample_inds]

                # save the performance accuracies
                accs[chan, p, 0] = lda.score(Y_speech_test, ofs1_test)
                accs[chan, p, 1] = lda.score(Y_speech_shuffle, ofs1_test)
                accs[chan, p, 2] = lda.score(Y_test[shuffle_test], ofs2)

    return accs, accs_test

def save_control_test_accs(subject_number, accs, accs_test, chans=None, diagonal=True, missing_f0=False, zscore_to_silence=True):
    """Used to save the nonspeech control invariance analysis results. 

    After running ``test_invariance_control``, save ``accs`` and ``accs_test`` for a subject.

    Args:
        subject_number: xxx in ECxxx
        accs (ndarray): shape is (n_chans x n_perms x 3). The last dimension contains accuracy values for held out 
                speech data, shuffled speech data, and shuffled nonspeech data
        accs_test (ndarray): shape is (n_chans). Contains accuracy value for nonspeech data. 
    
        diagonal (bool): whether the invariance analysis used a shrinkage of 1 and was diagonal LDA.
        missing_f0 (bool): missing f0 invariance analysis
    """
    info_str = ""
    if zscore_to_silence is False:
        info_str = info_str + "_zscore_block"
    if missing_f0:
        info_str = info_str + "_missing_f0"
    if diagonal:
        info_str = info_str + "_diagonal"
    if chans is not None:
        info_str = info_str + "_chans"
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_control_test_accs' + info_str + '.mat')
    if chans is not None:
        sio.savemat(filename, {'accs': accs, 'accs_test': accs_test, 'chans': chans})
    else:
        sio.savemat(filename, {'accs': accs, 'accs_test': accs_test})

def load_control_test_accs(subject_number, chans=None, diagonal=True, missing_f0=False, zscore_to_silence=True):
    """Used to load nonspeech control invariance analysis results.

    Args:
        subject_number: xxx in ECxxx
        diagonal (bool): whether the invariance analysis used a shrinkage of 1 and was diagonal LDA.

    Returns:
        (tuple):
            * **accs** (*ndarray*): shape is (n_chans x n_perms x 3). The last dimension contains accuracy values for held out 
                speech data, shuffled speech data, and shuffled nonspeech data
            * **accs_test** (*ndarray*): shape is (n_chans). Contains accuracy value for nonspeech data. 
    """
    info_str = ""
    if zscore_to_silence is False:
        info_str = info_str + "_zscore_block"
    if missing_f0:
        info_str = info_str + "_missing_f0"
    if diagonal:
        info_str = info_str + "_diagonal"
    if chans is not None:
        info_str = info_str + "_chans"
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_control_test_accs' + info_str + '.mat')
    data = sio.loadmat(filename)
    if chans is not None:
        return data['accs'], data['accs_test'], data['chans']
    else:
        return data['accs'], data['accs_test']

def test_invariance_missing_f0(subject_number, solver="lsqr", shrinkage=1, n_perms=1000, zscore_to_silence=True, chans=None):
    Y_mat, sns, sts, sps, Y_mat_plotter = load_Y_mat_sns_sts_sps_for_subject_number(subject_number, zscore_to_silence=zscore_to_silence)
    Y_mat_c, sns_c, sts_c, sps_c, Y_mat_plotter_c = load_Y_mat_sns_sts_sps_for_subject_number(subject_number, missing_f0_stim=True, zscore_to_silence=zscore_to_silence)

    Y_all = np.concatenate([Y_mat, Y_mat_c], axis=2)
    bad_time_indexes = np.isnan(np.sum(Y_all, axis=2))

    print(Y_mat.shape)
    n_chans, n_timepoints, n_trials = Y_mat.shape

    if solver == "svd":
        lda = LinearDiscriminantAnalysis()
    elif solver == "lsqr":
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    accs = np.zeros((n_chans, n_perms, 6))
    accs_test = np.zeros((n_chans, 5))

    n_train_100 = len(sts)

    n_train_60 = int(np.round(n_train_100*0.6))
    n_train_40 = n_train_100 - n_train_60

    ofs_train = sts
    ofs0 = sts_c[sns_c == 0]
    ofs1 = sts_c[sns_c == 1]
    ofs2 = sts_c[sns_c == 2]
    ofs3 = sts_c[sns_c == 3]
    ofs4 = sts_c[sns_c == 4]
    ofs_tests = [ofs0, ofs1, ofs2, ofs3, ofs4]

    for chan in np.arange(n_chans):
        if chans is None or chan in chans:
            Y_train = Y_mat[chan][~bad_time_indexes[chan]].T

            for sn in np.arange(5):
                Y_test = Y_mat_c[chan][~bad_time_indexes[chan]].T[sns_c == sn]
                ofs_test = ofs_tests[sn]

                if Y_train.shape[1] < 1 or Y_test.shape[1] < 1:
                    accs_test[chan, sn] = np.NaN
                else:
                    lda.fit(Y_train, ofs_train)
                    accs_test[chan, sn] = lda.score(Y_test, ofs_test)

    for p in tqdm(np.arange(n_perms)):
        rand_perm_train = np.random.permutation(n_train_100)
        shuffle_train = np.random.permutation(n_train_100)

        shuffle_nonspeech1 = np.random.permutation(len(ofs1))
        shuffle_nonspeech2 = np.random.permutation(len(ofs2))

        for chan in np.arange(n_chans):
            if chans is None or chan in chans:
                Y_train = Y_mat[chan][~bad_time_indexes[chan]].T

                if Y_train.shape[1] < 1 or Y_test.shape[1] < 1:
                    accs[chan, p, :] = np.NaN
                else:
                    lda.fit(Y_train[rand_perm_train][np.arange(n_train_60)], ofs_train[rand_perm_train][np.arange(n_train_60)])
                    Y_speech_test = Y_train[rand_perm_train][np.arange(n_train_60, n_train_100)]
                    Y_speech_shuffle = Y_train[shuffle_train][np.arange(n_train_60, n_train_100)]
                    ofs_train_test = ofs_train[rand_perm_train][np.arange(n_train_60, n_train_100)]

                    sample_inds48 = np.random.randint(0, Y_speech_test.shape[0], size=(48))
                    sample_inds96 = np.random.randint(0, Y_speech_test.shape[0], size=(96))
                    Y_speech_test48 = Y_speech_test[sample_inds48]
                    Y_speech_shuffle48 = Y_speech_shuffle[sample_inds48]
                    ofs_train_test48 = ofs_train_test[sample_inds48]

                    Y_speech_test96 = Y_speech_test[sample_inds96]
                    Y_speech_shuffle96 = Y_speech_shuffle[sample_inds96]
                    ofs_train_test96 = ofs_train_test[sample_inds96]

                    accs[chan, p, 0] = lda.score(Y_speech_test48, ofs_train_test48)
                    accs[chan, p, 1] = lda.score(Y_speech_shuffle48, ofs_train_test48)
                    accs[chan, p, 2] = lda.score(Y_speech_test96, ofs_train_test96)
                    accs[chan, p, 3] = lda.score(Y_speech_shuffle96, ofs_train_test96)

                    Y_nonspeech1 = Y_mat_c[chan][~bad_time_indexes[chan]].T[sns_c == 1]
                    Y_nonspeech2 = Y_mat_c[chan][~bad_time_indexes[chan]].T[sns_c == 2]
                    ofs1_shuffle = ofs1[shuffle_nonspeech1]
                    ofs2_shuffle = ofs2[shuffle_nonspeech2]

                    accs[chan, p, 4] = lda.score(Y_nonspeech1, ofs1_shuffle)
                    accs[chan, p, 5] = lda.score(Y_nonspeech2, ofs2_shuffle)

    if chans is not None:
        return accs, accs_test, chans
    else:
        return accs, accs_test

def test_invariance(Y_mat, sns, sts, sps, of_what="st", to_what="sn", n_perms=1000, solver="svd", shrinkage=1):
    bad_time_indexes = np.isnan(np.sum(Y_mat, axis=2))
    Y_mat_orig = np.copy(Y_mat)
    condition_dict = {'st': sts, 'sn': sns, 'sp': sps}
    condition_labels = {'st':[1, 2, 3, 4], 'sn': [1,2,3,4], 'sp':[1,2,3]}

    ofs = condition_dict[of_what]
    tos = condition_dict[to_what]
    Y_resid = residualize(Y_mat, tos)
    n_chans, n_timepoints, n_trials = Y_mat.shape
    print(Y_mat.shape)

    if solver == "svd":
        lda = LinearDiscriminantAnalysis()
    elif solver == "lsqr":
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)

    test_accs_distribution = False

    if test_accs_distribution:
        accs = np.zeros((n_chans, len(condition_labels[to_what]), n_perms, 8))
        for to_cond in tqdm(condition_labels[to_what]):
            n_train_100 = int(np.sum(tos != to_cond))
            n_test_100 = int(np.sum(tos == to_cond))
            n_test_50 = int(np.round(n_test_100/2))
            n1 = n_train_100 - n_test_50
            ofs1 = ofs[tos != to_cond]
            ofs2 = ofs[tos == to_cond]
            for p in tqdm(np.arange(n_perms)):
                rand_perm_train = np.random.permutation(n_train_100)
                rand_perm_test = np.random.permutation(n_test_100)

                shuffle_train = np.random.permutation(n_train_100)
                shuffle_test = np.random.permutation(n_test_100)

                for chan in np.arange(n_chans):
                    Y_mat_chan = Y_mat[chan][~bad_time_indexes[chan]]
                    Y_resid_chan = Y_resid[chan][~bad_time_indexes[chan]]

                    if Y_mat_chan.shape[0] < 1:
                        accs[chan, to_cond-1, p, :] = np.NaN
                    else:
                        Y_train = Y_mat_chan[:, tos != to_cond].T
                        Y_resid_train = Y_resid_chan[:, tos != to_cond].T
                        Y_test = Y_mat_chan[:, tos == to_cond].T
                        Y_resid_test = Y_resid_chan[:, tos == to_cond].T
                        
                        lda.fit(Y_train[rand_perm_train][np.arange(n1)], ofs1[rand_perm_train][np.arange(n1)])
                        accs[chan, to_cond-1, p, 0] = lda.score(Y_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[rand_perm_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 1] = lda.score(Y_test[rand_perm_test][np.arange(n_test_50)], ofs2[rand_perm_test][np.arange(n_test_50)])
                        accs[chan, to_cond-1, p, 2] = lda.score(Y_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[shuffle_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 3] = lda.score(Y_test[rand_perm_test][np.arange(n_test_50)], ofs2[shuffle_test][np.arange(n_test_50)])

                        lda.fit(Y_resid_train[rand_perm_train][np.arange(n1)], ofs1[rand_perm_train][np.arange(n1)])
                        accs[chan, to_cond-1, p, 4] = lda.score(Y_resid_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[rand_perm_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 5] = lda.score(Y_resid_test[rand_perm_test][np.arange(n_test_50)], ofs2[rand_perm_test][np.arange(n_test_50)])
                        accs[chan, to_cond-1, p, 6] = lda.score(Y_resid_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[shuffle_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 7] = lda.score(Y_resid_test[rand_perm_test][np.arange(n_test_50)], ofs2[shuffle_test][np.arange(n_test_50)])

    else:
        accs = np.zeros((n_chans, len(condition_labels[to_what]), n_perms, 8))
        for to_cond in tqdm(condition_labels[to_what]):
            n_train_100 = int(np.sum(tos != to_cond))
            n_test_100 = int(np.sum(tos == to_cond))
            n1 = n_train_100 - n_test_100
            ofs1 = ofs[tos != to_cond]
            ofs2 = ofs[tos == to_cond]

            for p in tqdm(np.arange(n_perms)):
                rand_perm_train = np.random.permutation(n_train_100)
                shuffle_train = np.random.permutation(n_train_100)
                shuffle_test = np.random.permutation(n_test_100)

                for chan in np.arange(n_chans):
                    Y_mat_chan = Y_mat[chan][~bad_time_indexes[chan]]
                    Y_resid_chan = Y_resid[chan][~bad_time_indexes[chan]]

                    if Y_mat_chan.shape[0] < 1:
                        accs[chan, to_cond-1, p, :] = np.NaN
                    else:
                        Y_train = Y_mat_chan[:, tos != to_cond].T
                        Y_resid_train = Y_resid_chan[:, tos != to_cond].T
                        Y_test = Y_mat_chan[:, tos == to_cond].T
                        Y_resid_test = Y_resid_chan[:, tos == to_cond].T

                        lda.fit(Y_train[rand_perm_train][np.arange(n1)], ofs1[rand_perm_train][np.arange(n1)])
                        accs[chan, to_cond-1, p, 0] = lda.score(Y_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[rand_perm_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 1] = lda.score(Y_test, ofs2) if p == 0 else np.NaN
                        accs[chan, to_cond-1, p, 2] = lda.score(Y_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[shuffle_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 3] = lda.score(Y_test, ofs2[shuffle_test])

                        lda.fit(Y_resid_train[rand_perm_train][np.arange(n1)], ofs1[rand_perm_train][np.arange(n1)])
                        accs[chan, to_cond-1, p, 4] = lda.score(Y_resid_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[rand_perm_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 5] = lda.score(Y_resid_test, ofs2) if p == 0 else np.NaN
                        accs[chan, to_cond-1, p, 6] = lda.score(Y_resid_train[rand_perm_train][np.arange(n1, n_train_100)], ofs1[shuffle_train][np.arange(n1, n_train_100)])
                        accs[chan, to_cond-1, p, 7] = lda.score(Y_resid_test, ofs2[shuffle_test])

    return accs

def save_invariance_test_accs(subject_number, accs, test_accs_distribution=False, of_what="st", to_what="sn", diagonal=False):
    info_str = ""
    if test_accs_distribution == False:
        info_str = info_str + "_single"
    if diagonal:
        info_str = info_str + "_diagonal"
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_invariance_test_of_' + of_what + '_to_' + to_what + '_accs' + info_str + '.mat')
    sio.savemat(filename, {'accs': accs})

def load_invariance_test_accs(subject_number, test_accs_distribution=False, of_what="st", to_what="sn", diagonal=False):
    info_str = ""
    if test_accs_distribution == False:
        info_str = info_str + "_single"
    if diagonal:
        info_str = info_str + "_diagonal"
    filename = os.path.join(results_path, 'EC' + str(subject_number) + '_invariance_test_of_' + of_what + '_to_' + to_what + '_accs' + info_str + '.mat')
    data = sio.loadmat(filename)
    return data['accs']

def residualize(Y_mat, by):
    Y_resid = np.copy(Y_mat)
    for i, cond in enumerate(by):
        Y_resid[:,:,i] = Y_mat[:,:,i] - np.nanmean(Y_mat[:,:,by==cond], axis=2)
    return Y_resid
