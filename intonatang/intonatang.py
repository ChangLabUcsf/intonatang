from __future__ import division, print_function, absolute_import

import os
brain_data_path = os.path.join(os.path.dirname(__file__), 'data', 'brain_imaging')
tokens_path = os.path.join(os.path.dirname(__file__), 'data', 'tokens')
results_path = os.path.join(os.path.dirname(__file__), 'results')

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import seaborn
from scipy.io import wavfile
from sklearn.decomposition import PCA
from cycler import cycler
import pandas as pd

from .intonation_stims import get_pitch_and_intensity, get_continuous_pitch_and_intensity

from .intonation_preanalysis import get_times_hg_for_subject_number, get_bcs, get_gcs, get_stg, get_centers
from .intonation_preanalysis import save_Y_mat_sns_sts_sps_for_subject_number, load_Y_mat_sns_sts_sps_for_subject_number

from .intonation_encoding import single_electrode_encoding_varpart, single_electrode_encoding_all_weights
from .intonation_encoding import save_encoding_results, load_encoding_results
from .intonation_encoding import save_encoding_results_all_weights, load_encoding_results_all_weights

from .nonspeech_control_generation import save_non_linguistic_control_stimuli

from .intonation_invariance import test_invariance, test_invariance_control, save_invariance_test_accs, save_control_test_accs
from .intonation_invariance import load_control_test_accs, load_invariance_test_accs, test_invariance_missing_f0

from .pitch_trf import load_cv_model_fold, get_intonation_tokens_stim, get_abs_and_rel_sig

from . import erps
from . import timit

encoding_colors = ['#ff2f97', '#5674ff', '#3fd400' , '#ae55c6', '#4ea47e', '#d3c26a', '#999999']
encoding_colors_black = ['#ff2f97', '#5674ff', '#3fd400' , 'k']

def generate_all_results():
    subject_numbers = [113, 118, 122, 123, 125, 129, 131]
    nonspeech_subject_numbers = [122, 123, 125, 129, 131]

    for subject_number in subject_numbers:
        save_Y_mat_sns_sts_sps_for_subject_number(subject_number)
        save_Y_mat_sns_sts_sps_for_subject_number(subject_number, zscore_to_silence=False)
        Y_mat, sns, sts, sps, Y_mat_plotter = load_Y_mat_sns_sts_sps_for_subject_number(subject_number)

        r2_varpart, p_varpart, f_varpart = single_electrode_encoding_varpart(Y_mat, sns, sts, sps)
        save_encoding_results(subject_number, r2_varpart, p_varpart, f_varpart)

        f, fp, b, bp, total_r2 = single_electrode_encoding_all_weights(Y_mat, sns, sts, sps)
        save_encoding_results_all_weights(subject_number, f, fp, b, bp, total_r2)

    for subject_number in nonspeech_subject_numbers:
        save_Y_mat_sns_sts_sps_for_subject_number(subject_number, control_stim=True)
        Y_mat, sns, sts, sps, Y_mat_plotter = load_Y_mat_sns_sts_sps_for_subject_number(subject_number, control_stim=True)

        r2_varpart, p_varpart, f_varpart = single_electrode_encoding_varpart(Y_mat, sns, sts, sps, control_stim=True)
        save_encoding_results(subject_number, r2_varpart, p_varpart, f_varpart, control_stim=True)

        f, fp, b, bp, total_r2 = single_electrode_encoding_all_weights(Y_mat, sns, sts, sps, control_stim=True)
        save_encoding_results_all_weights(subject_number, f, fp, b, bp, total_r2, control_stim=True)

        accs, accs_test = test_invariance_control(subject_number)
        save_control_test_accs(subject_number, accs, accs_test)

def load_all_data(subject_numbers=None):
    if subject_numbers is None:
        subject_numbers = [113, 118, 122, 123, 125, 129, 131, 137, 142, 143]
    datas = []
    cats = []
    r_means = []
    r_means_perm = []
    r_maxs = []
    r2s_abs = []
    r2s_rel = []
    wtss = []
    all_psis = []
    for subject_number in subject_numbers:
        r2_varpart, p_varpart, f_varpart = load_encoding_results(subject_number)
        data_varpart = pd.DataFrame(np.nanmax(r2_varpart, 1), columns=['sn', 'st', 'sp', 'sn st', 'sn sp', 'st sp', 'sn st sp'])
        data_varpart['subject_number'] = subject_number

        f, fp, b, bp, r2 = load_encoding_results_all_weights(subject_number)
        sig_elecs_bool = np.nansum(fp < 0.05/(256*101), axis=1) > 2
        not_sig_times = fp > 0.05/(256*101)
        sig_elecs = np.arange(256)
        sig_elecs = sig_elecs[sig_elecs_bool]

        sn_elecs_bool = np.nansum(p_varpart[:,:,0] < 0.05/(256*101), axis=1) > 2
        st_elecs_bool = np.nansum(p_varpart[:,:,1] < 0.05/(256*101), axis=1) > 2
        sp_elecs_bool = np.nansum(p_varpart[:,:,2] < 0.05/(256*101), axis=1) > 2
        data_varpart['sn_sig'] = sn_elecs_bool
        data_varpart['st_sig'] = st_elecs_bool
        data_varpart['sp_sig'] = sp_elecs_bool

        for chan in range(256):
            r2_varpart[chan, not_sig_times[chan], :] = np.NaN

        r_mean = np.nanmean(r2_varpart, axis=1)
        r_max = np.nanmax(r2_varpart, axis=1)
        cat = np.argmax(r_max, axis=1)
        cat[~sig_elecs_bool] = -1
        
        cats.append(cat)
        r_means.append(r_mean)
        r_maxs.append(r_max)

        r_all, r_abs, r_rel, abs_r2, rel_r2, wts_all, wts_abs, wts_rel = load_cv_model_fold(subject_number)
        r2s_abs.append(abs_r2[0, :256])
        r2s_rel.append(rel_r2[0, :256])

        data_varpart['sig_full'] = sig_elecs_bool
        data_varpart['cat'] = cat
        data_varpart['r2_abs'] = abs_r2[0, :256]
        data_varpart['r2_rel'] = rel_r2[0, :256]

        abs_sig, rel_sig = get_abs_and_rel_sig(subject_number)
        data_varpart['rel_sig'] = rel_sig
        data_varpart['abs_sig'] = abs_sig

        datas.append(data_varpart)

        wtss.append(np.mean(wts_all, axis=2))

        average_response, psis = timit.load_average_response_psis_for_subject_number(subject_number)
        all_psis.append(psis)

    cat_all = np.concatenate(cats)
    r_mean_all = np.concatenate(r_means)
    r_max_all = np.concatenate(r_maxs)
    r2s_abs = np.concatenate(r2s_abs)
    r2s_rel = np.concatenate(r2s_rel)
    all_psis = np.concatenate(all_psis, axis=1)
    datas = pd.concat(datas)

    return datas, r_mean_all, r_max_all, cat_all, r2s_abs, r2s_rel, wtss, all_psis

def plot_encoding_summary(ax):
    r_mean_all, r_max_all, cat_all, abs_r2s, rel_r2s = load_all_data()

    for i in range(3):
        ax.bar((10*i)+np.arange(7), np.mean(r_mean_all[cat_all == i], axis=0))

def plot_r2_single_electrode_encoding_all_subsets(centers, r2s, chan, ylabel="adj-r2", ylim=None):
    seaborn.set_context("talk", font_scale=2)
    
    colors = ['#0051e7','#d5cc00','#d9102e','#009f63','#a200fc','#ff9718','#000000']
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.set_prop_cycle(cycler('color', colors) +
                    cycler('lw', [3, 3, 3, 1, 1, 1, 1]) + cycler('alpha', [1, 1, 1, 0.5, 0.5, 0.5, 0.5]))
    hs = ax.plot(centers, r2s[chan,:,:])
    ax.set(xlabel='Time (bins)', xlim=(-10, 300), ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.legend(hs, ['sn', 'st', 'sp', 'sn st', 'sn sp', 'st sp', 'all'], loc='center left', bbox_to_anchor=(1,0.5))
    return fig

def plot_r2_single_electrode_encoding_single_subsets(centers, r2s, p_values, chan, ylim=None, yticks=None, mini=False):
    if not mini: 
        seaborn.set_context("talk")
        fig, ax = plt.subplots(1,1,figsize=(8,4))
        label_font_size=16
    else:
        seaborn.set_context("paper")
        fig, ax = plt.subplots(1,1,figsize=(3,2))
        label_font_size=16
        
    ax.locator_params(axis='y', nbins=7)

    colors = ['#ff2f97','#5674ff','#3fd400']
    ax.set_prop_cycle(cycler('color', colors) +
                           cycler('lw', [1, 1, 1]) + cycler('alpha', [1, 1, 1]))
    r = r2s[chan, :, :][ :, [1, 0, 2]]
    p = p_values[chan,:,:][ :, [1, 0,2]]

    centers = centers/100
    hs = ax.plot(centers, r)
    sig = p < 0.05/(256*101)
    for r1, s in zip(r.T, sig.T):
        c = np.copy(centers)
        rplot = np.copy(r1)
        c[~s] = np.NaN
        rplot[~s] = np.NaN
        ax.plot(c, rplot, linewidth=3)

    ax.set(xlim=(-.25, 2.75))
    ax.set_xlabel("Time (s)", fontsize=label_font_size)
    ax.set_ylabel("Unique R2", fontsize=label_font_size)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    if not mini:
        leg = ax.legend(hs, ['Intonation', 'Sentence', 'Speaker'], loc='center left', bbox_to_anchor=(1,0.5), fontsize=16)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3)
        fig.tight_layout(rect=[0.05, 0.05, 0.68, 0.93])
    else:
        ax.set(yticklabels=[], xticklabels=[])
    seaborn.despine()
    return fig

def plot_betas_intonation(centers, betas, p_values, chan, ylabel="Regression weight", xlabel='Time (s)', ylim=None, mini=False, control_stim=False):
    if mini:
        seaborn.set_context("paper")
        fig, ax = plt.subplots(1,1,figsize=(3,2))
        label_fontsize=16
    else:
        seaborn.set_context("talk")
        fig, ax = plt.subplots(1,1,figsize=(12,5))
        label_fontsize=16

    ax.locator_params(axis='y', nbins=4)
    colors = ['g', 'r', 'm']
    ax.set_prop_cycle(cycler('color', colors))

    if control_stim:
        beta = betas[chan, :, :][ :, [5, 6, 7]]
        p = p_values[chan,:,:][ :, [5, 6, 7]]
    else:
        beta = betas[chan, :, :][ :, [4, 5, 6]]
        p = p_values[chan,:,:][ :, [4, 5, 6]]

    centers = centers/100

    hs = ax.plot(centers, beta)
    sig = p < 0.05/101
    for b, s in zip(beta.T, sig.T):
        c = np.copy(centers)
        bplot = np.copy(b)
        c[~s] = np.NaN
        bplot[~s] = np.NaN
        ax.plot(c, bplot, linewidth=7)

    ax.set(xlim=(-.15, 2.85))
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    if mini == False:
        leg = ax.legend(hs, ['Question vs. Neutral', 'Emphasis 1 vs. Neutral', 'Emphasis 2 vs. Neutral'],
                    loc='center left', bbox_to_anchor=(1,0.5), fontsize=label_fontsize)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(10.0)    
        fig.tight_layout(rect=[0.05, 0.05, 0.6, 0.93])
    else:
        ax.set(yticklabels=[], xticklabels=[])
    seaborn.despine()
    return fig

def plot_betas_speakers(centers, betas, p_values, chan, ylabel="Regression weight", xlabel='Time (s)', ylim=None, mini=False, control_stim=False):
    if mini:
        seaborn.set_context("paper")
        fig, ax = plt.subplots(1,1,figsize=(3,2))
        label_fontsize=16
    else:
        seaborn.set_context("talk")
        fig, ax = plt.subplots(1,1,figsize=(12,5))
        label_fontsize=16

    ax.locator_params(axis='y', nbins=4)
    colors =  ['#7A0071', '#4f8ae0']
    ax.set_prop_cycle(cycler('color', colors))

    if control_stim:
        beta = betas[chan, :, :][ :, [8, 9]]
        p = p_values[chan,:,:][ :, [8, 9]]
    else:
        beta = betas[chan, :, :][ :, [7, 8]]
        p = p_values[chan,:,:][ :, [7, 8]]

    centers = centers/100

    hs = ax.plot(centers, beta)
    sig = p < 0.05/101
    for b, s in zip(beta.T, sig.T):
        c = np.copy(centers)
        bplot = np.copy(b)
        c[~s] = np.NaN
        bplot[~s] = np.NaN
        ax.plot(c, bplot, linewidth=7)

    ax.set(xlim=(-.15, 2.85))
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if ylim is not None:
        ax.set_ylim(ylim)
    if mini == False:
        leg = ax.legend(hs, ['Pitch', 'Formant'],
                    loc='center left', bbox_to_anchor=(1,0.5), fontsize=label_fontsize)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(10.0)    
        fig.tight_layout(rect=[0.05, 0.05, 0.6, 0.93])
    else:
        ax.set(yticklabels=[], xticklabels=[])
    seaborn.despine()
    return fig

def get_sig_elecs_from_full_model(subject_number, alpha=0.05/(256*101)):
    f, fp, b, bp, total_r2 = load_encoding_results_all_weights(subject_number)
    sig_elecs_times = fp < alpha
    sig_elecs = np.sum(sig_elecs_times, axis=1) > 2
    return np.arange(256)[sig_elecs]

def get_vars_for_pie_chart_for_subject_number(subject_number, use_r2=True, alpha=None):
    r2, p, f = load_encoding_results(subject_number)
    _, fp, b, bp, total_r2 = load_encoding_results_all_weights(subject_number)
    radii = np.nansum(total_r2, axis=1)
    radii[radii < 0] = 0
    radii = np.sqrt(radii)
    radii = radii/np.max(radii)
    if alpha is None:
        alpha = 0.05/(256*101)
    sig = fp < alpha
    if use_r2:
        stat = r2
    else:
        stat = f
    stat_zeroed = np.copy(stat)
    for i in range(7):
        stat_ = stat_zeroed[:, :, i]
        stat_[~sig] = 0
    stat_sums = np.sum(stat_zeroed, axis=1)
    stat_sums[stat_sums<0] = 0
    return stat_sums, radii

def plot_pie_chart_for_subject_number(subject_number, use_r2=True, alpha=None, on_brain=False):
    stat_sums, radii = get_vars_for_pie_chart_for_subject_number(subject_number, use_r2=use_r2, alpha=alpha)
    colors = ['#5674ff','#ff2f97','#3fd400', 'k', 'k', 'k', 'k']
    fig, ax = plt.subplots(figsize=(10, 10))
    if on_brain:
        img, xy = get_brain_img_and_xy_for_subject_number(subject_number)
        ax.imshow(img, cmap="Greys_r")
        centers = xy.T
    else:
        centers = np.zeros((256, 2))
        chan = 0
        for i in range(16):
            for j in range(16):
                centers[chan] = [-1*i, j]
                chan = chan+1
    for chan in range(256):
        if on_brain:
            if np.sum(stat_sums[chan]) > 0:
                ax.pie(stat_sums[chan]*10, colors=colors, radius=pie_chart_radius_by_subject_number[subject_number]*radii[chan],
                    center=centers[chan], startangle=90, frame=True, wedgeprops={'linewidth':0})
        else:
            if np.sum(stat_sums[chan]) > 0:
                ax.pie(stat_sums[chan]*10, colors=colors, radius=0.5*radii[chan],
                    center=centers[chan], startangle=90, frame=True, wedgeprops={'linewidth':0})
    ax.axis("off")
    if on_brain:
        for i, p in enumerate(np.sqrt(np.array([0.25, 0.5, 0.75, 1]))):
            ax.pie([10.0, 2.0], colors=colors[::-1], radius=pie_chart_radius_by_subject_number[subject_number] * p,
                  center=[img.shape[1]-120+(i*30), img.shape[0]-20], frame=True, wedgeprops={'linewidth':0})
    return fig, ax, stat_sums

pie_chart_radius_by_subject_number = {113: 10, 118: 14, 122: 11, 123: 11, 125: 11, 129: 12, 131:10, 137: 12, 142: 11, 143: 11}

def get_brain_img_and_xy_for_subject_number(subject_number):
    subject = 'EC' + str(subject_number)
    img_path = os.path.join(brain_data_path, subject + '_brain2D.png')
    img = mpimg.imread(img_path)
    xy = sio.loadmat(os.path.join(brain_data_path, subject + '_elec_pos2D.mat'))['xy']
    return img, xy

def plot_electrode_position_on_brain(subject_number, chan=None, chans=None):
    img, xy = get_brain_img_and_xy_for_subject_number(subject_number)
    fig, ax = plt.subplots(figsize=(1, 0.6))
    ax.axis("off")
    ax.imshow(img, cmap="Greys_r")
    if chan is not None:
        ax.plot(xy[0][chan], xy[1][chan], 'ro', markersize=2)
    if chans is not None:
        for chan in chans:
            ax.plot(xy[0][chan], xy[1][chan], 'ro', markersize=2)
    return fig

import itertools

class Plotter():
    sentence_type_labels = ['Neutral', 'Question', 'Emphasis 1', 'Emphasis 3']
    sentence_number_labels = ['Humans value\ngenuine behavior', 
                              'Movies demand\nminimal energy',
                              'Lawyers give a\nrelevant opinion',
                              'Reindeer are a\nvisual animal']
    speaker_labels =['Low pitch/Low formant', 'High pitch/High formant', 'High pitch/Low formant']
    labels = {'sentence_type': sentence_type_labels,
              'sentence_number': sentence_number_labels,
              'speaker': speaker_labels}    
    def __init__(self, Y_mat, sentence_numbers, sentence_types, speakers, axes_kw=None):
        self.n_chans = Y_mat.shape[0]
        self.n_timepoints = Y_mat.shape[1]
        self.n_conds = Y_mat.shape[2]
        self.Y_mat = Y_mat
        self.speakers = speakers
        self.sentence_types = sentence_types
        self.sentence_numbers = sentence_numbers
        self.sexes = np.array([1 if s < 2 else 2 for s in self.speakers])
        self.axes_kw = axes_kw
       
    def plot(self, chan, axes_by='sentence_number', traces_by='sentence_type', restrict_to=None,
            restrict_trials_to=None, show_acoustics=False, all_individual_plots=False, axes_kw=None):
        if axes_kw is not None:
            self.axes_kw = axes_kw
        Y_mat_chan = np.copy(self.Y_mat[chan, :, :])
        c = {}        
        c['speakers'] = np.copy(self.speakers)
        c['sentence_types'] = np.copy(self.sentence_types)
        c['sentence_numbers'] = np.copy(self.sentence_numbers)
        c['sexs'] = np.copy(self.sexes)
        extra_title = ""

        if restrict_trials_to is not None:
            min_trial = restrict_trials_to[0]
            max_trial = restrict_trials_to[1]
            indexes = np.zeros((self.n_conds))
            indexes[min_trial:max_trial] = 1
            indexes = indexes.astype(np.bool)
            Y_mat_chan = Y_mat_chan[:, indexes]
            c['speakers'] = c['speakers'][indexes]
            c['sentence_types'] = c['sentence_types'][indexes]
            c['sentence_numbers'] = c['sentence_numbers'][indexes]
            c['sexs'] = c['sexs'][indexes]
            extra_title = " trials: " + str(min_trial) + "-" + str(max_trial)
        if restrict_to is not None:
            if restrict_to[0] == 'sex':
                indexes = np.array([s in restrict_to[1] for s in c['sexs']])
            else:
                indexes = np.array([s in restrict_to[1] for s in c[restrict_to[0] + 's']])
            Y_mat_chan = Y_mat_chan[:, indexes]
            c['speakers'] = c['speakers'][indexes]
            c['sentence_types'] = c['sentence_types'][indexes]
            c['sentence_numbers'] = c['sentence_numbers'][indexes]
            c['sexs'] = c['sexs'][indexes]
        
        if all_individual_plots:
            fig, ax = plt.subplots(3, 1, figsize=(12, 20), sharex=True)
        else:
            if axes_by == 'sentence_number':
                fig, axs = plt.subplots(2, 2, figsize=(12,9), sharex=True, sharey=True)
                legend_index = 4

            if axes_by == "sentence_number_with_phonemes":
                fig, axs = plt.subplots(4, 1, figsize=(2, 5), sharey=True, sharex=True)
                legend_index = 10
                axes_by = "sentence_number"
                sentence_with_phonemes = True
            else:
                sentence_with_phonemes = False

            if axes_by == 'sentence_type':
                fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
                legend_index = 2
    
            if axes_by == 'speaker':
                fig, axs = plt.subplots(1, 3, figsize=(22, 6), sharex=True, sharey=True)
                legend_index = 3
     
            if axes_by == 'sex':
                fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
                legend_index = 2
                
            if axes_by == 'none':
                if show_acoustics:
                    fig, axs = plt.subplots(2, 1, figsize=(12,10))
                else:
                    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
                    axs = np.array([axs])
                legend_index = 1
        
        if sentence_with_phonemes is False:
            fig.text(0.45, 0.04, 'Time (s)', ha='center', fontsize=24)
            if axes_by == 'none':
                if show_acoustics:
                    fig.text(0.03, 0.28, 'Neural activity (high-gamma)', va='center', rotation='vertical', fontsize=24)
                    if traces_by=='sentence_number':
                        fig.text(0.03, 0.75, 'Amplitude contour', va='center', rotation='vertical', fontsize=24)
                    elif traces_by == 'sentence_type':
                        fig.text(0.03, 0.75, 'Pitch (Hz)', va='center', rotation='vertical', fontsize=24)
                else:
                    fig.text(0.03, 0.5, 'Neural activity (high-gamma)', va='center', rotation='vertical', fontsize=24)
            else:
                fig.text(0.03, 0.5, 'Neural activity (high-gamma)', va='center', rotation='vertical', fontsize=24)  
            fig.text(0.45, 0.94, 'Channel: ' + str(chan) + ' ' + extra_title, ha='center', fontsize=24)

        axs = axs.flatten()
        for i, ax in enumerate(axs):
            if self.axes_kw is not None:
                ax.set(**self.axes_kw)

            if traces_by == 'sentence_number':
                j_range = 4
                colors = ['navy', 'goldenrod', 'olivedrab', 'palevioletred']
            elif traces_by == 'speaker':
                j_range = 3
                colors = ['b','g','c','r','m','y']
            elif traces_by == 'sentence_type':
                j_range = 4
                colors = ['b','g','r','m']
            elif traces_by == 'sex':
                j_range = 2
                colors = ['b', 'r']
            
            hs = []
            for j in range(j_range):
                if axes_by == 'none' and i == 0 and show_acoustics:
                    xvals = np.arange(0, 2.2, 0.01)
                    pitches, intensities = get_continuous_pitch_and_intensity()
                    if traces_by == 'sentence_number':
                        h1 = ax.plot(xvals, intensities[0] + 0, 'b')
                        h2 = ax.plot(xvals, intensities[1] + 50, 'g')
                        h3 = ax.plot(xvals, intensities[2] + 100, 'r')
                        h4 = ax.plot(xvals, intensities[3] + 150, 'm') 
                        ax.set_yticklabels([])
                        hs = [h1[0], h2[0], h3[0], h4[0]]
                        leg = ax.legend(hs, [Plotter.labels['sentence_number'][k].split()[0] for k in range(4)], fontsize=24, loc='center left', bbox_to_anchor=(1,0.5))
                        for legobj in leg.legendHandles:
                            legobj.set_linewidth(10.0)                              
                    else:
                        h1 = ax.plot(xvals, pitches[0], 'b')
                        h2 = ax.plot(xvals, pitches[1], 'g')
                        h3 = ax.plot(xvals, pitches[2], 'r')
                        h4 = ax.plot(xvals, pitches[3], 'm')
                        h5 = ax.plot(xvals, pitches[4], 'b')
                        h6 = ax.plot(xvals, pitches[5], 'g')
                        h7 = ax.plot(xvals, pitches[6], 'r')
                        h8 = ax.plot(xvals, pitches[7], 'm')
                        hs = [h1[0], h2[0], h3[0], h4[0]]
                        leg = ax.legend(hs, [Plotter.labels['sentence_type'][k] for k in range(4)], fontsize=24, loc='center left', bbox_to_anchor=(1,0.5))
                        for legobj in leg.legendHandles:
                            legobj.set_linewidth(10.0)
                        ax.set_yscale("log")
                        ax.set_ylim((50,400))
                        ax.set(yticks=[50, 100, 200, 400], yticklabels=[50, 100, 200, 400])
                    ax.set_xlim((-0.25, 2.75))
                else:
                    if axes_by == 'none':
                        ax.locator_params(axis='y', nbins=5)
                        to_plot = Y_mat_chan[:, c[traces_by + 's'] == j+1]
                    else:
                        to_plot = Y_mat_chan[:, np.logical_and(c[axes_by + 's'] == i + 1, c[traces_by + 's'] == j+1)]    
                    if to_plot.shape[1] > 0:
                        hg_mean = np.nanmean(to_plot, 1)
                        hg_ste = np.nanstd(to_plot, 1) / np.sqrt(to_plot.shape[1])   
                        xvals = np.arange(0,len(hg_ste))/100 - 0.25
                        if sentence_with_phonemes is False:
                            h = ax.plot(xvals, hg_mean, color=colors[j])
                            hs.append(h[0])

                        ax.fill_between(xvals, hg_mean-hg_ste, hg_mean+hg_ste, color=colors[j], alpha=0.2)
                        if axes_by == 'none':
                            print()
                            #ax.set_title('Total average')
                        else:
                            if sentence_with_phonemes is False:
                                ax.set_title(Plotter.labels[axes_by][i])
                        if i+1 == legend_index:
                            if restrict_to != None and restrict_to[0] == traces_by:
                                ax.legend(hs, [Plotter.labels[traces_by][k-1] for k in np.sort(np.array(restrict_to[1]))], fontsize=16, loc='center left', bbox_to_anchor=(1,0.5))
                            else:
                                leg = ax.legend(hs, [Plotter.labels[traces_by][k] for k in range(j_range)], fontsize=24, loc='center left', bbox_to_anchor=(1,0.5))
                                for legobj in leg.legendHandles:
                                    legobj.set_linewidth(10.0)  
            ax.set(xlim=(-0.25, 2.75), xticks=[0, 0.5, 1, 1.5, 2, 2.5], xticklabels=['0', '0.5', '1', '1.5', '2', '2.5'])
            if sentence_with_phonemes:
                ax.locator_params(axis='y', nbins=5)

        if 'sentence_number' in axes_by or axes_by == 'speaker':
            fig.tight_layout(rect=[0.05, 0.05, 0.8, 0.93])
        elif axes_by == 'none':
            if traces_by == 'speaker':
                fig.tight_layout(rect=[0.05, 0.05, 0.6, 0.93])
            else:
                fig.tight_layout(rect=[0.05, 0.05, 0.68, 0.93])
        elif axes_by == 'sex' or axes_by == 'sentence_type':
            fig.tight_layout(rect=[0.05, 0.05, 0.7, 0.93])

        if sentence_with_phonemes:
            axs = fig.get_axes()
            ax = axs[0]
            yticks = ax.get_yticks()
            y_spacing = yticks[1]-yticks[0]
            print(yticks)
            print(y_spacing)
            ax.set_ylim([yticks[0]+0.4*y_spacing, yticks[-1]+0.4*y_spacing])
            for ax in axs:
                for x in [0, 0.56, 1.08, 1.65, 2.2]:
                    ax.axvline(x=x, color='gray', alpha =0.8)
                for x in [0.26, 0.75, 1.29, 1.42, 1.8, 2.0]:
                    ax.axvline(x=x, color='gray', alpha =0.2)

        return fig

    def add_phoneme_onset_marks(self, fig, phoneme_class):
        axs = fig.get_axes()

        if phoneme_class == "plosive":
            onsets_all = [[1.08, 1.80], [0.56, 1.02, 2], [0.56, 1.78], [0.26]]
        elif phoneme_class == "fricative":
            onsets_all = [[0, 0.48, 0.56, 2.0], [0.26, 0.48], [0.48, 1.42], [1.08, 1.29]]
        elif phoneme_class == "back":
            onsets_all = [[0.65, 1.14, 1.83],[0.82],[0.06, 1.14],[0.06, 0.56, 1.65]]
        elif phoneme_class == "front":
            onsets_all = [[1.67], [0.34,0.62,1.18,2.04], [0.63,1.83], [0.4, 1.17]]
        elif phoneme_class == "nasal":
            onsets_all =[[0.26], [0, 0.75, 1.08,1.42, 1.8], [2.0], [1.80, 2.0]]

        for ax, onsets in zip(axs, onsets_all):
            ylim = ax.get_ylim()
            line_ymin = ylim[0] + 0.75*(ylim[1] - ylim[0])
            line_ymax = ylim[0] + 0.9*(ylim[1] - ylim[0])
            for onset in onsets:
                ax.axes.plot([onset, onset], [line_ymin, line_ymax], color='k')
        
        return fig

def get_sounds(stims):
    sounds = []
    for stim in stims:
        fs, sound = wavfile.read(os.path.join(tokens_path, stim))
        sounds.append(sound)
    return sounds
