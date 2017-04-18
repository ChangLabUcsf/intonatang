from __future__ import division, print_function, absolute_import

import os
intonation_tokens_path = os.path.join(os.path.dirname(__file__), 'data', 'tokens')
timit_tokens_path = os.path.join(os.path.dirname(__file__), 'data', 'timit_tokens')
brain_data_path = os.path.join(os.path.dirname(__file__), 'data', 'brain_imaging')

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import zscore, pearsonr
from scipy.signal import resample
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import matplotlib.image as mpimg
matplotlib.rcParams['pdf.fonttype'] = 42
import librosa
from cycler import cycler


from . import intonatang as tang
from . import timit
from . import pitch_trf as ptrf
from . import erps

import seaborn
seaborn.set_style("ticks", {'xtick.major.size':2, 'ytick.major.size':2, 'ytick.minor.size':0, 'xtick.minor.size':0, 'axes.linewidth': 1})
seaborn.set_context("paper")

icolors = ['b', 'g', 'r', 'm']

pitch_intensity = tang.get_pitch_and_intensity()
pitches, intensities = tang.get_continuous_pitch_and_intensity()
centers = tang.get_centers()

def get_brain_imgs_and_xys(subjects=None):
    if subjects is None:
        subjects = [113, 118, 122, 123, 125, 129, 131]

    img_paths = [os.path.join(brain_data_path, 'EC' + str(subject) + '_brain2D.png') for subject in subjects]
    xy_paths = [os.path.join(brain_data_path, 'EC' + str(subject) + '_elec_pos2D.mat') for subject in subjects]

    imgs = []
    img_shapes = []
    xys = []
    for img_path, xy_path in zip(img_paths, xy_paths):
        img = mpimg.imread(img_path)
        imgs.append(img)
        img_shapes.append(img.shape)

        xys.append(sio.loadmat(xy_path)['xy'])

    return imgs, xys

def fig1():
    fig = plt.figure(figsize=(8, 5.5))

    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[2.5, 3*2], wspace=0.3)
    gs_left = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0], height_ratios=[1, 0.5, 1.5])
    gs_token = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_left[0], height_ratios=[0.5, 1], hspace=0)

    gs_right = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1], height_ratios=[3, 3.2, 3], hspace=0.1)
    gs_rasters_left = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_right[1,0], height_ratios=[0.2, 1, 1, 1])
    gs_rasters_right = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_right[1,1], height_ratios=[0.2, 1, 1, 1])

    # Panel A
    # Amplitude signal (ax1), spectrogram (ax2), and pitch contour (ax3) of an example token
    token = 'sn2_st3_sp2'
    with seaborn.axes_style("white"):
        ax1 = plt.subplot(gs_token[0, 0])
    [fs, sig] = sio.wavfile.read(os.path.join(intonation_tokens_path, token + '.wav'))
    sound = resample(np.atleast_2d(sig).T, (np.float(sig.size)/fs)*16000)
    pitch = pitch_intensity[token]['pitch']
    xvals = np.arange(len(sig))/fs
    ax1.plot([-0.1, 2.4], [0.2, 0.2], color=[0.2,0.2,0.2], linewidth=0.6)
    ax1.plot(xvals, sig, color=[0.2,0.2,0.2], linewidth=0.6)
    ax1.set(yticks=[], xticklabels=[], xlim=(-0.1, 2.4), title='Movies demand minimal energy')

    ax2 = plt.subplot(gs_token[1, 0])
    Pxx, freqs, bins, im = plt.specgram(sound.ravel(), NFFT=256, Fs=16000, noverlap=256*0.75, cmap=plt.get_cmap('Greys'))
    im.axes.set(xlim=(-0.1, 2.4), yticks=[], xticks=[0, 0.5, 1, 1.5, 2], xlabel='Time (s)', ylabel='Frequency (kHz)')
    im.set_clim(-50,80)
    im.axes.yaxis.set_ticks_position("right")
    im.axes.yaxis.set_label_position("right")

    ax3 = im.axes.twinx()
    ax3.plot(np.arange(pitch.shape[0])/100, pitch, color=icolors[2], marker='.', markersize=8)
    ax3.set(xlim=(-0.1, 2.4), ylim=(150, 300), yticks=[150, 200, 250, 300], ylabel="Pitch (Hz)")
    ax3.yaxis.set_ticks_position("left")
    ax3.yaxis.set_label_position("left")
    for ax in [ax1, ax2, ax3]:
        ax.set_frame_on(False)

    # Panel B
    # Pitch contours of intonation conditions for female (ax_female) and male (ax_male) speaker.
    with seaborn.axes_style("white"):
        ax_female = plt.subplot(gs_right[0, 0])
        ax_male = plt.subplot(gs_right[0, 1])

    x_grid = [0, 50, 100, 150, 200, 250]
    y_grid = [50, 100, 200, 400]
    grid_line_kws = {'color': 'lightgray', 'linewidth': 0.6}

    axs_pitch_contours = [ax_female, ax_male]
    for i, ax in enumerate(axs_pitch_contours):
        for xg in x_grid:
            ax.plot([xg, xg], [45, 440], **grid_line_kws)
        for yg in y_grid:
            ax.plot([-25, 275], [yg, yg], **grid_line_kws)
        if i == 0:
            # female pitch contours
            h1 = ax.plot(pitches[0], 'b')
            h2 = ax.plot(pitches[1], 'g')
            h3 = ax.plot(pitches[2], 'r')
            h4 = ax.plot(pitches[3], 'm')
        else:
            # male pitch contours
            h5 = ax.plot(pitches[4], 'b--')
            h6 = ax.plot(pitches[5], 'g--')
            h7 = ax.plot(pitches[6], 'r--')
            h8 = ax.plot(pitches[7], 'm--')
        _ = ax.set(xticks=[0, 50, 100, 150, 200, 250], xlim=(-25,275), xticklabels=['0','0.5','1.0','1.5','2','2.5'])
        ax.set_yscale("log")
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set(ylim=(45, 440), yticks=[50,100,200,400], yticklabels=['50','100','200','400'])
        if i == 0:
            ax.set(ylabel="Pitch (Hz)")
        ax.set_frame_on(False)

    hs = [h1[0], h2[0], h3[0], h4[0]]
    leg = ax.legend(hs, ["Neutral", "Question", "Emphasis 1", "Emphasis 3"], fontsize=8, frameon=True)
    leg.get_frame().set_edgecolor('#ffffff')

    # Panel C
    # Full model encoding R2 on brain
    chan = 53
    f, fp, b, bp, r2 = tang.load_encoding_results_all_weights(113)
    r2_varpart, p_varpart, f_varpart = tang.load_encoding_results(113)
    int_elecs = np.arange(256)[np.sum(p_varpart[:,:,1]<0.05/(256*101), axis=1) > 2]
    sig_elecs = tang.get_sig_elecs_from_full_model(113)
    max_r2 = np.nanmax(r2, axis=1)
    cm = matplotlib.cm.ScalarMappable(cmap=plt.get_cmap('Reds'))
    cm.set_clim(0, 1)
    cm.set_array([0, 1])
    max_r2 = cm.to_rgba(max_r2)

    imgs, xys = get_brain_imgs_and_xys([113])
    ax = plt.subplot(gs_left[2, 0])
    ax.imshow(imgs[0], cmap='Greys_r')

    for i in range(256):
        # all electrode positions
        ax.scatter(xys[0][0][i], xys[0][1][i], color='gray', s=1)
        if i in int_elecs and i in sig_elecs:
            # intonation condition differentitaing electrodes
            ax.scatter(xys[0][0][i], xys[0][1][i], color='k', s=6)
        if i in sig_elecs:
            # significant electrodes
            ax.scatter(xys[0][0][i], xys[0][1][i], color=max_r2[i], s=2.5)
        if i in [chan]:
            # representative electrode used in Panels D and E
            ax.scatter(xys[0][0][i], xys[0][1][i], color='k', s=2)

    cbar = plt.colorbar(cm, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation="horizontal")
    cbar.set_label("R2")
    cm.set_clim([0,1])
    ax.set(xticklabels=[], yticklabels=[], yticks=[], xticks=[])
    ax.set_frame_on(False)

    # Panel D
    # Single trial responses.
    Y_mat, sns, sts, sps, Y_mat_plotter = tang.load_Y_mat_sns_sts_sps_for_subject_number(113)
    raster_clim = (0, 15)

    cmaps = [plt.get_cmap('OrRd'), plt.get_cmap('RdPu'), plt.get_cmap('Greens')]
    # Order of subpanels is Emphasis 1 (st=3), Emphasis 3 (st=4), and Question (st=2)
    for i, stype in enumerate([3, 4, 2]):
        ax = plt.subplot(gs_rasters_left[i+1, 0])
        sentence_order = np.argsort(sns)[::-1]
        indexes = np.logical_and(sts[sentence_order]==stype, sps[sentence_order]==2)
        im = ax.pcolormesh(Y_mat_plotter[:,:,sentence_order][chan, :, indexes], cmap=cmaps[i])
        im.set_clim(raster_clim)
        _ = im.axes.set(xticks=[25, 75, 125, 175, 225, 275], xticklabels=[], yticks=[0, 20], yticklabels=[0, 20])
        edges_bool = np.diff(sns[sentence_order][indexes])
        edges = np.arange(len(edges_bool))[edges_bool.astype(np.bool)] + 1
        for ed in edges:
            ax.plot([0, 300], [ed, ed], color='k', linewidth=0.2)
        
        ax = plt.subplot(gs_rasters_right[i+1, 0])
        indexes = np.logical_and(sts[sentence_order]==stype, sps[sentence_order]==1)
        im = ax.pcolormesh(Y_mat_plotter[:,:,sentence_order][chan, :, indexes], cmap=cmaps[i])
        im.set_clim(raster_clim)
        _ = im.axes.set(xticks=[25, 75, 125, 175, 225, 275], xticklabels=[], yticks=[0, 20], yticklabels=[0, 20])
        edges_bool = np.diff(sns[sentence_order][indexes])
        edges = np.arange(len(edges_bool))[edges_bool.astype(np.bool)] + 1
        for ed in edges:
            ax.plot([0, 300], [ed, ed], color='k', linewidth=0.2)

    # Panel E 
    # Average cortical responses by intonation contour for female (ax at gs_right[2, 0])
    # and male (ax at gs_right[2, 1]) speaker
    ax = plt.subplot(gs_right[2, 0])
    xvals = np.arange(300)/100 - 0.25
    sts_activity = [Y_mat_plotter[chan, :, np.logical_and(sts == i+1, sps == 3)] for i in range(4)]
    sts_means = [np.mean(st, axis=0) for st in sts_activity]
    sts_stes = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in sts_activity]
    ax.fill_between(xvals, sts_means[0]-sts_stes[0], sts_means[0]+sts_stes[0], color='b', alpha=0.2)
    ax.fill_between(xvals, sts_means[1]-sts_stes[1], sts_means[1]+sts_stes[1], color='g', alpha=0.2)
    ax.fill_between(xvals, sts_means[2]-sts_stes[2], sts_means[2]+sts_stes[2], color='r', alpha=0.2)
    ax.fill_between(xvals, sts_means[3]-sts_stes[3], sts_means[3]+sts_stes[3], color='m', alpha=0.2)
    ax.plot(xvals, sts_means[0], 'b')
    ax.plot(xvals, sts_means[1], 'g')
    ax.plot(xvals, sts_means[2], 'r')
    ax.plot(xvals, sts_means[3], 'm')
    _ = ax.set(xticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5], xlim=(-0.25, 2.75), xlabel="Time (s)",
               yticks=[0, 2, 4, 6, 8, 10], ylim=(-1, 11), ylabel="High-gamma (z-score)")
    seaborn.despine(ax=ax)

    ax = plt.subplot(gs_right[2, 1])
    sts_activity = [Y_mat_plotter[chan, :, np.logical_and(sts == i+1, sps == 1)] for i in range(4)]
    sts_means = [np.mean(st, axis=0) for st in sts_activity]
    sts_stes = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in sts_activity]
    ax.fill_between(xvals, sts_means[0]-sts_stes[0], sts_means[0]+sts_stes[0], color='b', alpha=0.2)
    ax.fill_between(xvals, sts_means[1]-sts_stes[1], sts_means[1]+sts_stes[1], color='g', alpha=0.2)
    ax.fill_between(xvals, sts_means[2]-sts_stes[2], sts_means[2]+sts_stes[2], color='r', alpha=0.2)
    ax.fill_between(xvals, sts_means[3]-sts_stes[3], sts_means[3]+sts_stes[3], color='m', alpha=0.2)
    ax.plot(xvals, sts_means[0], 'b--')
    ax.plot(xvals, sts_means[1], 'g--')
    ax.plot(xvals, sts_means[2], 'r--')
    ax.plot(xvals, sts_means[3], 'm--')
    _ = ax.set(xticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5], xlim=(-0.25, 2.75), xlabel="Time (s)",
               yticks=[0, 2, 4, 6, 8, 10], ylim=(-1, 11))
    seaborn.despine(ax=ax)

    return fig

def fig2():
    fig = plt.figure(figsize=(6, 7))
    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1.5, 1])
    gs_top = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs[0], wspace=0.12, hspace=0.12, width_ratios=[1,1,1,0.4])
    gs_bottom = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.3, width_ratios=[1, 1.5])
    gs_brain = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_bottom[0], height_ratios=[0.5, 1], width_ratios=[0.7, 1])
    gs_bar = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_bottom[1], height_ratios=[0.1, 1, 0.3])


    # Panels A-L (gs_top)
    # Average activity by condition (Panels A-I) and main effect unique R2 (Panels J-K)
    r2_varpart, p_varpart, f_varpart = tang.load_encoding_results(113)
    f, fp, b, bp, total_r2 = tang.load_encoding_results_all_weights(113)

    Y_mat, sentence_numbers, sentence_types, speakers, Y_mat_plotter = tang.load_Y_mat_sns_sts_sps_for_subject_number(113)
    Y_mat = Y_mat_plotter

    p_value = 0.05/(256*101)
    full_model_sig = fp < p_value

    colors = ['#ff2f97','#5674ff','#3fd400']
    speaker_colors =  ['#7A0071', '#4f8ae0','#33ff99']

    chans = [167, 133, 36]
    titles = ['Intonation encoding\nelectrode (e1)', 'Sentence encoding\nelectrode (e2)', 'Speaker encoding\nelectrode (e3)']
    line_legend_kws = {'loc': 2, 'bbox_to_anchor': [1.1, 1]}
    hg_line_kws = {'yticks': [0, 2, 4, 6, 8], 'ylim': (-1, 9), 'xticks': [0, 0.5, 1.0, 1.5, 2.0, 2.5], 'xlim':(-0.25, 2.75), 'xticklabels':[]}
    for col, chan in enumerate(chans):
        # Panels J-K
        # The calculation of times where main effects and the full model are significant also happens here.
        ax = plt.subplot(gs_top[3, col])
        ax.locator_params(axis='y', nbins=7)
        ax.set_prop_cycle(cycler('color', colors))
        r = r2_varpart[chan, :, :][ :, [1, 0, 2]]
        hs = ax.plot(centers, r)

        p = p_varpart[chan,:,:][ :, [1, 0, 2]]
        sig = p < p_value

        full_model_sig_timepoints = np.copy(centers)
        full_model_sig_timepoints[~full_model_sig[chan]] = np.NaN
        sig_times = []
        for r1, s in zip(r.T, sig.T):
                c = np.copy(centers)
                rplot = np.copy(r1)
                c[~s] = np.NaN
                c[~full_model_sig[chan]] = np.NaN
                rplot[~s] = np.NaN
                rplot[~full_model_sig[chan]] = np.NaN
                # Plot significant time points for main effects in thicker linewidth
                ax.plot(c, rplot, linewidth=3)
                # Save significant time points for main effects for use in Panels A-I
                sig_times.append(np.copy(c)/100)

        ax.plot(full_model_sig_timepoints, [0.95]*len(full_model_sig_timepoints), 'k')
        ax.set(xlim=(-25, 275), xticks=[0, 50, 100, 150, 200, 250], xticklabels=[0, 0.5, 1.0, 1.5, 2.0, 2.5], xlabel="Time (s)",
               ylim=(0, 1.0), yticks=[0,0.25, 0.5, 0.75,1.0])
        if col == 0:
            ax.set_ylabel("Unique R2")
        if col != 0:
            ax.set(yticklabels=[])
        if col == 2:
            leg = ax.legend(hs, ['Intonation', 'Sentence', 'Speaker'], **line_legend_kws)
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

        # Panels A-I 
        # High-gamma activity averages by intonation, sentence, and speaker
        xvals = np.arange(300)/100 - 0.25

        # Panels A-C: Averages by intonation condition (AKA sentence type)
        ax = plt.subplot(gs_top[0, col])
        ax.set_title(titles[col])
        sts = [Y_mat[chan, :, sentence_types == i+1] for i in range(4)]
        sts_means = [np.mean(st, axis=0) for st in sts]
        sts_stes = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in sts]
        ax.fill_between(xvals, sts_means[0]-sts_stes[0], sts_means[0]+sts_stes[0], color='b', alpha=0.2)
        ax.fill_between(xvals, sts_means[1]-sts_stes[1], sts_means[1]+sts_stes[1], color='g', alpha=0.2)
        ax.fill_between(xvals, sts_means[2]-sts_stes[2], sts_means[2]+sts_stes[2], color='r', alpha=0.2)
        ax.fill_between(xvals, sts_means[3]-sts_stes[3], sts_means[3]+sts_stes[3], color='m', alpha=0.2)
        h1 = ax.plot(xvals, sts_means[0], 'b')
        h2 = ax.plot(xvals, sts_means[1], 'g')
        h3 = ax.plot(xvals, sts_means[2], 'r')
        h4 = ax.plot(xvals, sts_means[3], 'm')
        ax.plot(sig_times[0], [8.4]*len(sig_times[0]), 'k')
        _ = ax.set(**hg_line_kws)
        if col != 0:
            ax.set(yticklabels=[])
        if col == 0:
            ax.set_ylabel('High-gamma (z-score)')
        if col == 2:
            leg = ax.legend([h1[0], h2[0], h3[0], h4[0]], ['Neutral', 'Question', 'Emphasis 1', 'Emphasis 3'], **line_legend_kws)
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

        # Panels D-F: Averages by sentence condition (AKA sentence number)
        ax = plt.subplot(gs_top[1, col])
        sts = [Y_mat[chan, :, sentence_numbers == i+1] for i in range(4)]
        sts_means = [np.mean(st, axis=0) for st in sts]
        sts_stes = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in sts]
        ax.fill_between(xvals, sts_means[0]-sts_stes[0], sts_means[0]+sts_stes[0], color='#3c1718', alpha=0.2)
        ax.fill_between(xvals, sts_means[1]-sts_stes[1], sts_means[1]+sts_stes[1], color='#546a22', alpha=0.2)
        ax.fill_between(xvals, sts_means[2]-sts_stes[2], sts_means[2]+sts_stes[2], color='#5db095', alpha=0.2)
        ax.fill_between(xvals, sts_means[3]-sts_stes[3], sts_means[3]+sts_stes[3], color='#bfc8f1', alpha=0.2)
        h1 = ax.plot(xvals, sts_means[0], color='#3c1718')
        h2 = ax.plot(xvals, sts_means[1], color='#546a22')
        h3 = ax.plot(xvals, sts_means[2], color='#5db095')
        h4 = ax.plot(xvals, sts_means[3], color='#bfc8f1')
        ax.plot(sig_times[1], [8.4]*len(sig_times[1]), 'k')
        _ = ax.set(**hg_line_kws)
        if col != 0:
            ax.set(yticklabels=[])
        if col == 2:
            leg = ax.legend([h1[0], h2[0], h3[0], h4[0]], ['Movies...', 'Humans...', 'Reindeer...', 'Lawyers...'], **line_legend_kws)
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

        # Panels G-I: Averages by speaker condition
        ax = plt.subplot(gs_top[2, col])
        sts = [Y_mat[chan, :, speakers == i+1] for i in range(3)]
        sts_means = [np.mean(st, axis=0) for st in sts]
        sts_stes = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in sts]
        ax.fill_between(xvals, sts_means[0]-sts_stes[0], sts_means[0]+sts_stes[0], color=speaker_colors[0], alpha=0.2)
        ax.fill_between(xvals, sts_means[1]-sts_stes[1], sts_means[1]+sts_stes[1], color=speaker_colors[1], alpha=0.2)
        ax.fill_between(xvals, sts_means[2]-sts_stes[2], sts_means[2]+sts_stes[2], color=speaker_colors[2], alpha=0.2)
        h1 = ax.plot(xvals, sts_means[0],  color=speaker_colors[0])
        h2 = ax.plot(xvals, sts_means[1],  color=speaker_colors[1])
        h3 = ax.plot(xvals, sts_means[2],  color=speaker_colors[2])
        ax.plot(sig_times[2], [8.4]*len(sig_times[2]), 'k')
        _ = ax.set(**hg_line_kws)
        if col != 0:
            ax.set(yticklabels=[])
        if col == 2:
            leg = ax.legend([h1[0], h2[0], h3[0]], ['Male', 'Female 1', 'Female 2'], **line_legend_kws)
            for legobj in leg.legendHandles:
                legobj.set_linewidth(2.0)

    # Panel M
    # Proportion of variance explained as pie-charts on the brain
    imgs, xys = get_brain_imgs_and_xys([113])

    # Whole brain image with inset rectangle and brain inset with pie charts
    ax_brain = plt.subplot(gs_brain[0, 0])
    ax_inset = plt.subplot(gs_brain[1, :])
    brain_inset([ax_brain, ax_inset], 113)

    # Panel N
    # Box plots showing distribution of proportion of variance explained by main effects and interactions
    datas, r_mean_all, r_max_all, cat_all, r2s_abs, r2s_rel, wtss, _ = tang.load_all_data()

    ax1 = plt.subplot(gs_bar[1, 0])
    ax2 = plt.subplot(gs_bar[1, 1])
    ax3 = plt.subplot(gs_bar[1, 2])
    axs = [ax1, ax2, ax3]

    df = pd.DataFrame(r_mean_all, columns=['sn', 'st', 'sp', 'sn st' ,'sn sp' ,'st sp', 'sn st sp'])
    df[df<0] = 0
    df['total_r2'] = df.sum(axis=1)
    for col in ['sn', 'st', 'sp', 'sn st' ,'sn sp' ,'st sp', 'sn st sp']:
        df[col] = df[col]/df['total_r2']
    df['cat'] = cat_all
    df = df[['st', 'sn', 'sp', 'sn st', 'st sp', 'sn sp', 'sn st sp', 'cat']]

    axs, patches = add_encoding_boxplots_to_axs(axs, df)

    # Legend in Panel M
    ax = plt.subplot(gs_brain[0, 1])
    plt.legend(patches, ['Intonation', 'Sentence', 'Speaker', 'All interactions'], loc=2, bbox_to_anchor=(-0.24, 1.1))
    ax.set_frame_on(False)
    ax.set(xticks=[], yticks=[])

    seaborn.despine()
    return fig

def add_encoding_boxplots_to_axs(axs, df, ylim=(-0.05, 1), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], pie_radius=0.1, pie_center=(0.12, 0.75)):
    df2 = pd.melt(df, id_vars=["cat"], var_name="Predictor", value_name='Proportion of total R2')

    boxplot_kws = {'x': 'Predictor', 'y': 'Proportion of total R2', 'palette': tang.encoding_colors, 'linewidth': 0.8, 'fliersize':3, 'orient': 'v'}
    pie_kws = {'colors':tang.encoding_colors_black, 'radius':pie_radius, 'center':pie_center, 'startangle':90, 'frame':True, 'wedgeprops':{'linewidth': 0}}

    # Intonation electrodes
    df3 = df2[df2["cat"] == 1]
    if len(df3) > 0:
        ax = seaborn.boxplot(data=df3, ax=axs[0], **boxplot_kws)
        ax.set(xlabel="")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        pie_means = df[df['cat'] == 1].mean() * 100
        pie_means[pie_means < 0] = 0 
        pie_means = pie_means.values[:-1]
        pie_means = np.concatenate([pie_means[0:3], [np.sum(pie_means[4:])]], axis=0)
        ax1 = ax.twiny()
        ax1.axis("equal")
        patches, texts = ax1.pie(pie_means, **pie_kws)
        ax1.set_frame_on(False)
        ax1.set(xticks=[], xticklabels=[], ylim=ylim, yticks=yticks)
        ax1.set(title='Intonation \n n=' + str(np.sum(df.cat==1)))

    # Sentence electrodes
    df3 = df2[df2["cat"] == 0]
    if len(df3) > 0:
        ax = seaborn.boxplot(data=df3, ax=axs[1], **boxplot_kws)
        ax.set(xticklabels=[], ylabel="")
        pie_means = df[df['cat'] == 0].mean() * 100
        pie_means[pie_means < 0] = 0
        pie_means = pie_means.values[:-1]
        pie_means = np.concatenate([pie_means[0:3], [np.sum(pie_means[4:])]], axis=0)
        ax1 = ax.twiny()
        ax1.axis("equal")
        patches, texts = ax1.pie(pie_means, **pie_kws)
        ax1.set_frame_on(False)
        ax1.set(xticks=[], xticklabels=[], ylim=ylim, yticks=yticks, yticklabels=[], ylabel="")
        ax1.set(title='Sentence \n n=' + str(np.sum(df.cat==0)))

    # Speaker electrodes
    df3 = df2[df2["cat"] == 2]
    if len(df3) > 0:
        ax = seaborn.boxplot(data=df3, ax=axs[2], **boxplot_kws)
        ax.set(xticklabels=[], xlabel="", ylabel="")
        pie_means = df[df['cat'] == 2].mean() * 100
        pie_means[pie_means < 0] = 0 
        pie_means = pie_means.values[:-1]
        pie_means = np.concatenate([pie_means[0:3], [np.sum(pie_means[4:])]], axis=0)
        ax1 = ax.twiny()
        ax1.axis("equal")
        patches, texts = ax1.pie(pie_means, **pie_kws)
        ax1.set_frame_on(False)
        ax1.set(xticks=[], xticklabels=[], ylim=ylim, yticks=yticks, yticklabels=[], ylabel="")
        ax1.set(title='Speaker \n n=' + str(np.sum(df.cat==2)))

    for ax in axs:
        seaborn.despine(ax=ax)
    return axs, patches

def brain_inset(axs, subject_number):
    imgs, xys = get_brain_imgs_and_xys([subject_number])
    inset_xlim, inset_ylim = get_brain_inset_info(subject_number)

    ax_brain, ax_inset = axs
    ax_brain.imshow(imgs[0], cmap='Greys_r')
    ax_brain.add_patch(matplotlib.patches.Rectangle((inset_xlim[0]-10, inset_ylim[1]-10), inset_xlim[1]-inset_xlim[0]+20, inset_ylim[0]-inset_ylim[1]+20, fill=False, linewidth=1))
    ax_brain.set(xticks=[], yticks=[])
    ax_brain.set_frame_on(False)

    stat_sums, radii = tang.get_vars_for_pie_chart_for_subject_number(subject_number)
    colors = ['#5674ff','#ff2f97','#3fd400', 'k', 'k', 'k', 'k']
    ax_inset.imshow(imgs[0], cmap='Greys_r')
    for i in range(256):
        if np.sum(stat_sums[i]) > 0:
            patches, texts = ax_inset.pie(stat_sums[i]*10, colors=colors, radius=pie_chart_radius_by_subject_number[subject_number]*radii[i],
                    center=xys[0].T[i], startangle=90, frame=True, wedgeprops={'linewidth':0, 'clip_on':True})
    ax_inset.set(xticklabels=[], yticklabels=[], yticks=[], xticks=[], xlim=inset_xlim, ylim=inset_ylim)

    for i, p in enumerate(np.sqrt(np.array([0.25, 0.5, 0.75, 1]))):
        ax_inset.pie([10.0, 2.0], colors=colors[::-1], radius=pie_chart_radius_by_subject_number[subject_number] * p,
                  center=[inset_xlim[1]-60+((i**(5/4))*1.4*pie_chart_radius_by_subject_number[subject_number]), inset_ylim[0]+20], frame=True, wedgeprops={'linewidth':0})
    return patches

pie_chart_radius_by_subject_number = {113: 10, 118: 14, 122: 11, 123: 11, 125: 11, 129: 12, 131:10, 137: 12, 142: 11, 143: 11}

def get_brain_inset_info(subject_number):
    if subject_number == 113:
        inset_xlim = (315, 585)
        inset_ylim = (450, 270)
    elif subject_number == 118:
        inset_xlim = (486, 864)
        inset_ylim = (636, 384)
    elif subject_number == 122:
        inset_xlim = (291.5, 588.5)
        inset_ylim = (499, 301)
    elif subject_number == 123:
        inset_xlim = (326.5, 623.5)
        inset_ylim = (504, 306)
    elif subject_number == 125:
        inset_xlim = (381.5, 678.5)
        inset_ylim = (494, 296)
    elif subject_number == 129:
        inset_xlim = (313, 637)
        inset_ylim = (478, 262)
    elif subject_number == 131:
        inset_xlim = (350, 620)
        inset_ylim = (460, 280)
    elif subject_number == 137:
        inset_xlim = (383, 707)
        inset_ylim = (488, 272)
    elif subject_number == 142:
        inset_xlim = (350, 648)
        inset_ylim = (460, 262)
    elif subject_number == 143:
        inset_xlim = (391.5, 688.5)
        inset_ylim = (484, 286)
    return inset_xlim, inset_ylim

def sfig1():
    fig = plt.figure(figsize=(6, 10))

    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 0.7])
    gs_brains = matplotlib.gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=gs[0], width_ratios=[0.7, 1, 0.7, 1], hspace=0.1)
    gs_boxplots = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.3)
    gs_left = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_boxplots[0])
    gs_right = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_boxplots[1])


    for i, subject_number in enumerate([113, 122, 131, 125, 137]):
        ax_brain = plt.subplot(gs_brains[i, 0])
        ax_inset = plt.subplot(gs_brains[i, 1])
        brain_inset([ax_brain, ax_inset], subject_number)
    for i, subject_number in enumerate([118, 123, 143, 129, 142]):
        ax_brain = plt.subplot(gs_brains[i, 2])
        ax_inset = plt.subplot(gs_brains[i, 3])
        patches = brain_inset([ax_brain, ax_inset], subject_number)

    ax = plt.subplot(gs_brains[0, 3])
    patches = [patches[i] for i in [1, 0, 2, 3]]
    plt.legend(patches, ['Intonation', 'Sentence', 'Speaker', 'All interactions'], loc=2, bbox_to_anchor=(1, 1.1))
    ax.set(xticks=[], yticks=[])

    # Left hemisphere subjects
    datas, r_mean_all, r_max_all, cat_all, r2s_abs, r2s_rel, wtss, _ = tang.load_all_data([113, 118, 122, 123, 131, 143])
    ax1 = plt.subplot(gs_left[0])
    ax2 = plt.subplot(gs_left[1])
    ax3 = plt.subplot(gs_left[2])
    axs1 = [ax1, ax2, ax3]
    df = pd.DataFrame(r_mean_all, columns=['sn', 'st', 'sp', 'sn st' ,'sn sp' ,'st sp', 'sn st sp'])
    df[df<0] = 0
    df['total_r2'] = df.sum(axis=1)
    for col in ['sn', 'st', 'sp', 'sn st' ,'sn sp' ,'st sp', 'sn st sp']:
        df[col] = df[col]/df['total_r2']
    df['cat'] = cat_all
    df = df[['st', 'sn', 'sp', 'sn st', 'st sp', 'sn sp', 'sn st sp', 'cat']]
    axs, patches = add_encoding_boxplots_to_axs(axs1, df)

    # Right hemisphere subjects
    datas, r_mean_all, r_max_all, cat_all, r2s_abs, r2s_rel, wtss, _ = tang.load_all_data([125, 129, 137, 142])
    ax1 = plt.subplot(gs_right[0])
    ax2 = plt.subplot(gs_right[1])
    ax3 = plt.subplot(gs_right[2])
    axs2 = [ax1, ax2, ax3]
    df = pd.DataFrame(r_mean_all, columns=['sn', 'st', 'sp', 'sn st' ,'sn sp' ,'st sp', 'sn st sp'])
    df[df<0] = 0
    df['total_r2'] = df.sum(axis=1)
    for col in ['sn', 'st', 'sp', 'sn st' ,'sn sp' ,'st sp', 'sn st sp']:
        df[col] = df[col]/df['total_r2']
    df['cat'] = cat_all
    df = df[['st', 'sn', 'sp', 'sn st', 'st sp', 'sn sp', 'sn st sp', 'cat']]
    axs, patches = add_encoding_boxplots_to_axs(axs2, df)

    for ax in axs1 + axs2:
        seaborn.despine(ax=ax)
    return fig

def fig3():
    fig = plt.figure(figsize=(6, 5))

    gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], left=0.2)
    gs_top = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[0], height_ratios=[0.5, 0.5, 0.8, 1], hspace=0, wspace=0.1)
    gs_bottom = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[0.75, 1], hspace=0.7)
    gs_neural = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_bottom[0], wspace=0.1)
    gs_bottom_row = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_bottom[1], width_ratios=[1,2], wspace=0.1)
    gs_scatters = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_bottom_row[1], wspace=0.6)

    # Panels A-D
    # Example speech and non-speech tokens (A, B) and average high-gamma responses from one electrode (C, D)
    token = 'sn2_st4_sp2'
    wavfiles = [os.path.join(os.path.dirname(__file__), 'data', 'tokens', token +'.wav'),
                os.path.join(os.path.dirname(__file__), 'data', 'tokens_missing_f0', 'purr_stretch_0_female_st4.wav'),
                os.path.join(os.path.dirname(__file__), 'data', 'tokens_missing_f0', 'purr_missing_f0_noise_first_stretch_0_female_st4.wav')]

    subject_number = 142
    chan = 199

    Y_mat__, sns, sts, sps, Y_mat = tang.load_Y_mat_sns_sts_sps_for_subject_number(subject_number, zscore_to_silence=False)
    Y_mat_c__, sns_c, sts_c, sps_c, Y_mat_c = tang.load_Y_mat_sns_sts_sps_for_subject_number(subject_number, missing_f0_stim=True, zscore_to_silence=False)

    # Calculation of average high-gamma responses (mean and ste)
    # and time points where intonation conditions are significantly different.
    hg_sts = [Y_mat[chan, :, sts == i+1] for i in range(4)]
    hg_sts_means = [np.mean(st, axis=0) for st in hg_sts]
    hg_sts_stes = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in hg_sts]

    hg_sts_c = [Y_mat_c[chan, :, np.logical_and(sns_c == 1, sts_c == i+1)] for i in range(4)]
    hg_sts_means_c = [np.mean(st, axis=0) for st in hg_sts_c]
    hg_sts_stes_c = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in hg_sts_c]

    hg_sts_c2 = [Y_mat_c[chan, :, np.logical_and(sns_c == 2, sts_c == i+1)] for i in range(4)]
    hg_sts_means_c2 = [np.mean(st, axis=0) for st in hg_sts_c2]
    hg_sts_stes_c2 = [np.std(st, axis=0)/np.sqrt(st.shape[0]) for st in hg_sts_c2]

    sts_means = [hg_sts_means, hg_sts_means_c, hg_sts_means_c2]
    sts_stes = [hg_sts_stes, hg_sts_stes_c, hg_sts_stes_c2]

    xvals_neural = np.arange(300)/100 - 0.25

    for i in range(3): #loop over columns (different stim conditions)
        # Ampltiude signal (ax0, ax1), pitch contour (ax3), and spectrogram (ax2) of example speech and non-speech tokens
        [fs, sig] = sio.wavfile.read(wavfiles[i])
        ax0 = plt.subplot(gs_top[0, i])
        xvals = np.arange(len(sig))/fs
        xmin = 1.55
        xmax = xmin + 0.05
        xmin_index = np.int(xmin * fs)
        xmax_index = np.int(xmax * fs)
        ax0.plot(xvals[xmin_index:xmax_index], sig[xmin_index:xmax_index], color=[0.2, 0.2, 0.2], linewidth=0.8)
        ax0.set(xticks=[], yticks=[], xlim=(xmin-0.02, xmax+0.02))
        if i == 0:
            ax0.set(title="Speech")
            pitch = pitch_intensity[token]['pitch']
        else:
            ax0.set(title="Non-speech")
            pitch = pitches[3]
        ax1 = plt.subplot(gs_top[1, i])
        sound = resample(np.atleast_2d(sig).T, (np.float(sig.size)/fs)*16000)
        xvals = np.arange(len(sig))/fs
        if i == 2:
            xvals = xvals - 0.25
        ax1.plot([-0.25, 2.75], [0.2, 0.2], color=[0.2,0.2,0.2], linewidth=0.6)
        ax1.plot(xvals, sig, color=[0.2,0.2,0.2], linewidth=0.6)
        ylim = ax1.get_ylim()
        # red signal inset box
        ax1.plot([xmin, xmin], ylim, 'r')
        ax1.plot([xmax, xmax], ylim, 'r')
        ax1.plot([xmin, xmax], [ylim[0], ylim[0]], 'r')
        ax1.plot([xmin, xmax], [ylim[1], ylim[1]], 'r')
        ax1.set(yticks=[], xticklabels=[], xlim=(-0.25, 2.75))

        # Spectrogram
        ax2 = plt.subplot(gs_top[3, i])
        plt.sca(ax2)
        S = librosa.feature.melspectrogram(sig, sr=fs, n_mels=128, fmax=8000, n_fft=441*4, hop_length=441)
        log_S = librosa.logamplitude(S, ref_power=np.max)
        im = librosa.display.specshow(log_S, sr=fs, x_axis="time", y_axis="mel", fmax=8000)
        if i == 2:
            im.axes.set(xlim=(0, 300), yticks=[], xticks=[]) #hard-coding for alignment to deal with noise starting 0.25s before onset.
        else:
            im.axes.set(xlim=(-25, 275), yticks=[], xticks=[])
        if i == 0:
            im.axes.set(ylabel="Frequency\n (Hz)")
        else:
            im.axes.set(ylabel="")
        im.set_cmap(plt.get_cmap("Greys"))
        if i == 1:
            im.set(clim=(-80, 15))

        # Pitch contour
        ax3 = plt.subplot(gs_top[2, i])
        ax3.plot(np.arange(pitch.shape[0]), pitch, color='m', marker='.', markersize=8)
        ax3.set(xlim=(-25, 275), ylim=(50, 300), yticks=[], xticklabels=[])
        ax3.yaxis.tick_left()
        ax3.yaxis.set_label_position("left")
        if i == 0:
            ax3.set_ylabel('Pitch \n(Hz)')
        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_frame_on(False)
        for ax in [ax0, ax1, ax3]:
            ax.set(xticks=[], yticks=[])

        # Average neural responses (ax4)
        ax4 = plt.subplot(gs_neural[0, i])
        ax4.fill_between(xvals_neural, sts_means[i][0]-sts_stes[i][0], sts_means[i][0]+sts_stes[i][0], color='b', alpha=0.8)
        ax4.fill_between(xvals_neural, sts_means[i][1]-sts_stes[i][1], sts_means[i][1]+sts_stes[i][1], color='g', alpha=0.8)
        ax4.fill_between(xvals_neural, sts_means[i][2]-sts_stes[i][2], sts_means[i][2]+sts_stes[i][2], color='r', alpha=0.8)
        ax4.fill_between(xvals_neural, sts_means[i][3]-sts_stes[i][3], sts_means[i][3]+sts_stes[i][3], color='m', alpha=0.8)
        ax4.set(ylim=(-1, 3), yticks=[-1, 0, 1, 2, 3], xlim=(-0.25, 2.75), xlabel="Time (s)")
        if i == 0:
            ax4.set(ylabel="High-gamma (z-score)")
        else:
            ax4.set(yticklabels=[])

    # Panel E
    # Box plot showing accuracies of speech-trained LDA model on speech and non-speech data.
    ax_box = plt.subplot(gs_bottom_row[0])

    accs, accs_test, _ = tang.load_control_test_accs(subject_number, chans=True, missing_f0=True, zscore_to_silence=False)
    accs2 = np.zeros((256, 1000, 6))
    accs2[:, :, 0:2] = accs[:, :, [2,3]]
    accs2[:,:,2] = np.NaN
    accs2[:,0,2] = accs_test[:, 1]
    accs2[:,:,3] = np.NaN
    accs2[:,0,3] = accs_test[:, 2]
    accs2[:, :, 4:6] = accs[:, :, [4, 5]]
    accs = accs2

    ax_box.axhline(0.25, 0, 1, color='gray', alpha=0.5)
    df = pd.melt(pd.DataFrame(accs[chan, :, [0,2,3]].T, columns=['speech', 'non-speech', 'missing f0']), value_name="Classification accuracy")
    df['type']= "Test"

    df_shuf = pd.melt(pd.DataFrame(accs[chan, :, [1,4,5]].T, columns=['speech', 'non-speech', 'missing f0']), value_name="Classification accuracy")
    df_shuf['type']= "Shuffled"
    seaborn.boxplot(data=pd.concat([df,df_shuf]), hue='type', y='Classification accuracy', x='variable', ax=ax_box)
    plt.axes(ax_box)
    plt.legend(bbox_to_anchor=(0.6, 1.05), loc=2)
    ax_box.set_ylim(0, 1)

    # Panel F
    # Scatterplot of accuracies for speech data versus accuracies for non-speech data

    # For five participants who heard original non-speech stimuli, test to see whether speech-trained models for significant electrodes
    # predicted non-speech data as well as held out speech data
    invariant = []
    mean_accs = []
    nonspeech_test_accs = []
    for subject_number in [122, 123, 125, 129, 131]:
        sig_elecs = tang.get_sig_elecs_from_full_model(subject_number)
        accs, accs_test = tang.load_control_test_accs(subject_number)
        mean_accs.append(np.nanmean(accs, axis=1)[sig_elecs, 0])
        nonspeech_test_accs.append(accs_test.T[sig_elecs, 0])
        for sig_elec in sig_elecs:
            value = accs_test[0, sig_elec]
            min_max = np.percentile(accs[sig_elec, :, 0], [2.5, 97.5])
            if value < min_max[1] and value > min_max[0]:
                invariant.append(1)
            else:
                invariant.append(0)

    # For three participants who heard flat-ampltidude, non-speech stimuli, do the same thing and combine
    # these electrodes with the ones from the original non-speech stimuli.
    for subject_number in [137, 142, 143]:
        sig_elecs = tang.get_sig_elecs_from_full_model(subject_number)
        accs, accs_test = tang.load_control_test_accs(subject_number, missing_f0=True, zscore_to_silence=False)
        mean_accs.append(np.nanmean(accs, axis=1)[sig_elecs, 0])
        nonspeech_test_accs.append(accs_test[sig_elecs, 1])
        for sig_elec in sig_elecs:
            value = accs_test[sig_elec, 1]
            min_max = np.percentile(accs[sig_elec, :, 0], [2.5, 97.5])
            if value < min_max[1] and value > min_max[0]:
                invariant.append(1)
            else:
                invariant.append(0)

    invariant = np.array(invariant)
    mean_accs = np.concatenate(mean_accs, axis=0)
    nonspeech_test_accs = np.concatenate(nonspeech_test_accs, axis=0)

    print("Invariant electrodes for non-speech, with f0: {:d}/{:d}, {:.2f}%".format(np.sum(invariant==1), len(invariant), np.mean(invariant)))

    # First scatterplot with non-speech, with f0. 
    ax = plt.subplot(gs_scatters[0])

    ax.axhline(0.25, 0, 1, color='gray', alpha=0.5)
    ax.axvline(0.25, 0, 1, color='gray', alpha=0.5)
    ax.plot([0.2,1],[0.2,1], 'gray', alpha=0.5)
    ax.scatter(mean_accs[invariant==1], nonspeech_test_accs[invariant==1], color='k', marker='.')
    ax.scatter(mean_accs[invariant==0], nonspeech_test_accs[invariant==0], color='red', marker='.')

    ax.axis("square")
    ax.set(xlim=(0.2,1), ylim=(0.2,1), xticks=[0.2,0.4,0.6,0.8,1.0], yticks=[0.2,0.4,0.6,0.8,1.0], 
            xlabel="Accuracy for speech", ylabel="Accuracy for non-speech\nwith f0")

    # For three participants who heard missing f0 stimuli, test to see whether speech-trained models for significant electrodes
    # predicted missing f0 data as well as held out speech data
    invariant = []
    mean_accs = []
    nonspeech_test_accs = []
    for subject_number in [137, 142, 143]:
        sig_elecs = tang.get_sig_elecs_from_full_model(subject_number)
        accs, accs_test = tang.load_control_test_accs(subject_number, missing_f0=True, zscore_to_silence=False)
        mean_accs.append(np.nanmean(accs, axis=1)[sig_elecs])
        nonspeech_test_accs.append(accs_test[sig_elecs, 2])
        for sig_elec in sig_elecs:
            value = accs_test[sig_elec, 2]
            min_max = np.percentile(accs[sig_elec, :, 2], [2.5, 97.5])
            if value < min_max[1] and value > min_max[0]:
                invariant.append(1)
            else:
                invariant.append(0)
    invariant = np.array(invariant)
    mean_accs = np.concatenate(mean_accs, axis=0)
    nonspeech_test_accs = np.concatenate(nonspeech_test_accs, axis=0)

    print("Invariant electrodes for non-speech, missing f0: {:d}/{:d}, {:.2f}%".format(np.sum(invariant==1), len(invariant), np.mean(invariant)))

    # Second scatterplot with non-speech, missing f0.
    ax = plt.subplot(gs_scatters[1])

    ax.axhline(0.25, 0, 1, color='gray', alpha=0.5)
    ax.axvline(0.25, 0, 1, color='gray', alpha=0.5)
    ax.plot([0.2,1],[0.2,1], 'gray', alpha=0.5)
    ax.scatter(mean_accs[invariant==1,0], nonspeech_test_accs[invariant==1], color='k', marker='.')
    ax.scatter(mean_accs[invariant==0,0], nonspeech_test_accs[invariant==0], color='red', marker='.')

    ax.axis("square")
    ax.set(xlim=(0.2,1), ylim=(0.2,1), xticks=[0.2,0.4,0.6,0.8,1.0], yticks=[0.2,0.4,0.6,0.8,1.0], 
            xlabel="Accuracy for speech", ylabel="Accuracy for non-speech\nmissing f0")

    seaborn.despine()
    return fig

def sfig2_scatterplots():
    fig, axs = plt.subplots(3, 1, figsize=(2.1, 4), sharey=True, sharex=True)
    
    datas, _, _, _, _, _, _, all_psis = tang.load_all_data()
    sig = datas.sig_full.values
    ys = np.mean(all_psis, axis=0)[sig]
    axs[0].plot(ys, datas.sn.values[sig], 'ko', markersize=2)
    axs[1].plot(ys, datas.st.values[sig],'ko', markersize=2)
    axs[2].plot(ys, datas.sp.values[sig],'ko', markersize=2)
    axs[0].set_xlim(-2, 26)
    axs[0].set_ylim(-0.06, 1.11)
    axs[0].set_ylabel("Sentence")
    axs[1].set_ylabel("Intonation")
    axs[2].set_ylabel("Speaker")
    axs[2].set_xlabel("Mean PSI")
    seaborn.despine()

    sn_corr = pearsonr(datas.sn.values[sig], ys)
    st_corr = pearsonr(datas.st.values[sig], ys)
    sp_corr = pearsonr(datas.sp.values[sig], ys)
    print("Corrrelation between mean PSI and Sentence R2: {:.2f}, (p={:.2e})".format(*sn_corr))
    print("Corrrelation between mean PSI and Intonation R2: {:.2f}, (p={:.2e})".format(*st_corr))
    print("Corrrelation between mean PSI and Speaker R2: {:.2f}, (p={:.2e})".format(*sp_corr))

    return fig

def sfig3():
    timit_pitch = timit.get_timit_pitch()
    sentences = timit_pitch.index.get_level_values(0)
    sentences = pd.Series(sentences).unique()

    pitch_log_means = np.array([np.nanmean(timit_pitch.loc[sentence].log_hz) for sentence in sentences])
    pitch_rel_means = [np.nanmean(timit_pitch.loc[sentence].rel_pitch_global) for sentence in sentences]
    pitch_log_stds = np.array([np.nanstd(timit_pitch.loc[sentence].log_hz) for sentence in sentences])
    intensity_means = [np.nanmean(timit_pitch.loc[sentence].intensity) for sentence in sentences]
    sex = ['female' if s[0] == 'f' else 'male' for s in sentences]

    flat_timit_pitch = timit_pitch.reset_index()

    abs_bin_edges, rel_bin_edges = ptrf.get_bin_edges_abs_rel(timit_pitch)
    abs_bounds = timit.zscore_abs_pitch(abs_bin_edges[[0, -1]], reverse=True)
    rel_bounds = rel_bin_edges[[0, -1]]


    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.5, 1])

    gs_timit_examples = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.8, 1], hspace=0.1, subplot_spec=gs[0])
    examples = add_timit_example_tokens_to_gridspec(gs_timit_examples, binned_pitch=False)

    gs_below_examples = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[0.5, 1], hspace=0.4)
    gs_speaker = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0.3, hspace=0.4, subplot_spec=gs_below_examples[0])
    gs_pitch_hists = gridspec.GridSpecFromSubplotSpec(2, 2, wspace=0.3, hspace=0.3, subplot_spec=gs_below_examples[1])
    ax_speaker_means = plt.subplot(gs_speaker[0, 0])
    ax_means_vs_stds = plt.subplot(gs_speaker[0, 1])
    ax1 = plt.subplot(gs_pitch_hists[0, 0])
    ax2 = plt.subplot(gs_pitch_hists[0, 1])
    ax3 = plt.subplot(gs_pitch_hists[1, 0])
    ax4 = plt.subplot(gs_pitch_hists[1, 1])


    df = pd.DataFrame({'Mean absolute pitch (Hz)': np.exp(pitch_log_means), 'Sex': sex})
    seaborn.boxplot(x='Mean absolute pitch (Hz)', y='Sex', order=["male", "female"], ax=ax_speaker_means, data=df, linewidth=0.8)

    kws = {'marker': 'o', 'linestyle': 'None', 'markersize': 3}
    pitch_means_hz = np.exp(pitch_log_means)
    pitch_cv_percent = np.exp(pitch_log_stds - 1)*100
    ax_means_vs_stds.plot(pitch_means_hz, pitch_cv_percent, color='k', **kws)
    example_indices = [np.arange(499)[sentences == example][0] for example in examples]
    ax_means_vs_stds.plot(pitch_means_hz[example_indices], pitch_cv_percent[example_indices], color='r', **kws)
    ax_means_vs_stds.set(xlabel="Mean absolute pitch (Hz)", ylabel="Within sentence variability (CV)")

    vline_kws = {'color': 'gray', 'alpha': 0.8}

    pitch = np.exp(flat_timit_pitch.log_hz.values)
    indexes = ~np.isnan(pitch)
    sex = np.array([s[0] for s in flat_timit_pitch.level_0.values])
    pitch = pitch[indexes]

    df = pd.DataFrame({'Absolute pitch (Hz)': pitch, 'sex': sex[indexes]})
    kws = {'bins': range(50, 300, 10), 'kde': False, 'hist_kws': {'alpha':0.8, 'linewidth':0.8}}
    seaborn.distplot(df[df.sex == "m"]['Absolute pitch (Hz)'], ax=ax1, **kws)
    seaborn.distplot(df[df.sex == "f"]['Absolute pitch (Hz)'], ax=ax1, **kws)
    ax1.set(xlim=(45, 305), xlabel="")
    ax1.axvline(abs_bounds[0], **vline_kws)
    ax1.axvline(abs_bounds[1], **vline_kws)
    ax1.set_title("Absolute pitch values")

    abs_bin_edges_hz = timit.zscore_abs_pitch(abs_bin_edges, reverse=True)
    pitch[pitch < abs_bin_edges_hz[0]] = abs_bin_edges_hz[0] + 1
    pitch[pitch > abs_bin_edges_hz[-1]] = abs_bin_edges_hz[-1] - 1
    df = pd.DataFrame({'Absolute pitch (Hz)': pitch, 'sex': sex[indexes]})

    kws = {'bins': abs_bin_edges_hz, 'kde': False, 'hist_kws': {'alpha':0.8, 'linewidth':0.8}}
    seaborn.distplot(df[df.sex == "m"]['Absolute pitch (Hz)'], ax=ax3, **kws)
    seaborn.distplot(df[df.sex == "f"]['Absolute pitch (Hz)'], ax=ax3, **kws)
    ax3.set(xlim=(45, 305))
    ax3.set_title("Binning for absolute pitch")

    pitch = flat_timit_pitch.rel_pitch_global.values
    indexes = ~np.isnan(pitch)
    sex = np.array([s[0] for s in flat_timit_pitch.level_0.values])
    pitch = pitch[indexes]

    df = pd.DataFrame({'Relative pitch (zscore Hz)': pitch, 'sex': sex[indexes]})
    kws = {'bins': np.arange(-4.2, 3.2, 0.25), 'kde': False, 'hist_kws': {'alpha':0.8, 'linewidth':0.8}}
    seaborn.distplot(df[df.sex == "m"]['Relative pitch (zscore Hz)'], ax=ax2, **kws)
    seaborn.distplot(df[df.sex == "f"]['Relative pitch (zscore Hz)'], ax=ax2, **kws)
    ax2.set(xlim=(-4.2, 3.2), xlabel="")
    ax2.axvline(rel_bounds[0], **vline_kws)
    ax2.axvline(rel_bounds[1], **vline_kws)
    ax2.set_title("Relative pitch values")

    pitch[pitch < rel_bin_edges[0]] = rel_bin_edges[0] + 0.01
    pitch[pitch > rel_bin_edges[-1]] = rel_bin_edges[-1] - 0.01
    df = pd.DataFrame({'Relative pitch (zscore Hz)': pitch, 'sex': sex[indexes]})

    kws = {'bins': rel_bin_edges, 'kde': False, 'hist_kws': {'alpha':0.8, 'linewidth':0.8}}
    seaborn.distplot(df[df.sex == "m"]['Relative pitch (zscore Hz)'], ax=ax4, **kws)
    seaborn.distplot(df[df.sex == "f"]['Relative pitch (zscore Hz)'], ax=ax4, **kws)
    ax4.set(xlim=(-4.2, 3.2))
    ax4.set_title("Binning for relative pitch")

    seaborn.despine()

    return fig

def add_timit_example_tokens_to_gridspec(gs, examples=None, titles=None, binned_pitch=True):
    if examples is None:
        examples = ['fadg0_si1279','fcmh0_si2084','msrg0_si1851', 'majp0_si1074','mabc0_si1620']
    if titles is None:
        titles = ['"Bricks are an alternative."' , '"What about a tea room then?"', '"You mean a game with cards?"', '"He gazed away from us as we approached."','"His head flopped back."']

    timit_pitch = timit.get_timit_pitch()
    abs_bin_edges, rel_bin_edges = ptrf.get_bin_edges_abs_rel(timit_pitch)
    stim_pitch_abs= [ptrf.get_pitch_matrix(timit_pitch.loc[ex].abs_pitch.values, abs_bin_edges) for ex in examples]
    stim_pitch_rel= [ptrf.get_pitch_matrix(timit_pitch.loc[ex].rel_pitch_global.values, rel_bin_edges) for ex in examples]

    sigs = [sio.wavfile.read(os.path.join(timit_tokens_path, ex + '.wav')) for ex in examples]
    durations = [np.round(s.shape[0]/fs, decimals=1) - 0.1 for (fs, s) in sigs]

    gs_top = matplotlib.gridspec.GridSpecFromSubplotSpec(2, len(examples), subplot_spec=gs[0], width_ratios=durations, height_ratios=[0.5, 1], hspace=0, wspace=0.05)
    gs_bottom = matplotlib.gridspec.GridSpecFromSubplotSpec(2, len(examples), subplot_spec=gs[1], width_ratios=durations, wspace=0.05)

    for i in range(len(examples)):
        with seaborn.axes_style("white"):
            ax1 = plt.subplot(gs_top[0, i])
        [fs, sig] = sio.wavfile.read(os.path.join(timit_tokens_path, examples[i] + '.wav'))
        sound = resample(np.atleast_2d(sig).T, (np.float(sig.size)/fs)*16000)
        sound = sound[:np.round(durations[i]*16000).astype(np.int)]
        sig = sig[:np.round(durations[i]*fs).astype(np.int)]
        pitch = timit_pitch.loc[examples[i]].pitch
        xvals = np.arange(len(sound))/16000
        ax1.plot(xvals, sound, color=[0.2,0.2,0.2], linewidth=0.6)
        ax1.set(yticks=[], xticks=[], xticklabels=[], xlim=(0, durations[i]), ylim=(-35000, 35000))
        ax1.set_title(titles[i], {'fontsize':8})

        with seaborn.axes_style("white"):
            ax2 = plt.subplot(gs_top[1, i])
        plt.sca(ax2)
        Pxx, freqs, bins, im = plt.specgram(sound.ravel(), NFFT=256, Fs=16000, noverlap=256*0.75,
                                            cmap=plt.get_cmap('Greys'))
        im.axes.set(yticks=[], xticks=[], xlim=(0, durations[i]))
        im.set_clim(-50,80)

        im.axes.yaxis.set_ticks_position("right")
        im.axes.yaxis.set_label_position("right")
        if binned_pitch:
            ax3 = im.axes.twinx()
            ax3.plot(np.arange(pitch.shape[0])/100, pitch, color='orange', marker='.', markersize=8)
            ax3.set_yscale("log")
            ax3.set(ylim=(50, 400), yticks=[50, 100, 200, 400], yticklabels=[50, 100, 200, 400], xticklabels=[])
            if i != 0:
                ax3.set(yticklabels=[])
            ax3.yaxis.tick_left()
            ax3.yaxis.set_label_position("left")
            ax3.set_frame_on(False)

        ax1.set_frame_on(False)
        ax2.set_frame_on(False)

        ax4 = plt.subplot(gs_bottom[0, i])
        ax5 = plt.subplot(gs_bottom[1, i])
        if binned_pitch:
            ax4.pcolormesh(stim_pitch_abs[i].T)
            ax5.pcolormesh(stim_pitch_rel[i].T)
            xticks = np.arange(12.5, durations[i]*100, 25)
            ax4.set(yticks=[], xticks=xticks, xlim=(-0.5, durations[i]*100+0.5), xticklabels=[])
            ax5.set(yticks=[], xticks=xticks, xlim=(-0.5, durations[i]*100+0.5), xticklabels=(xticks-12.5)/100, xlabel="Time (s)")
            if i == 0:
                ax4.set(yticks=(0.5, 3.5, 6.5, 9.5), yticklabels=[90, 150, 200, 250], ylabel="Abs. pitch\n (Hz)")
                ax5.set(yticks=(0.5, 3.5, 6.5, 9.5), yticklabels=[-1.7, -0.5, 0.6, 1.7], ylabel="Rel. pitch\n (z-score)")
        else:
            ax4.plot(timit_pitch.loc[examples[i]].pitch, color="#C13639")
            ax5.plot(timit_pitch.loc[examples[i]].rel_pitch_global, color='#DB7D12')
            ax4.set_yscale("log")
            ax4.set(ylim=(50, 400), yticks=[50, 100, 200, 400], xticks=np.arange(12.5, durations[i]*100, 25), xlim=(0, durations[i]*100), xticklabels=[])
            ax4.yaxis.set_major_formatter(ScalarFormatter())
            ax5.set(ylim=(-3, 3), yticks=[-3, 0, 3], xticks=np.arange(12.5, durations[i]*100, 25),xlim=(0, durations[i]*100), xticklabels=[])
            if i != 0:
                ax4.set(yticklabels=[])
                ax5.set(yticklabels=[])
            else:
                ax4.set_ylabel("Abs. pitch\n (Hz)")
                ax5.set_ylabel("Rel. pitch\n (z-score Hz)")
    return examples

def timit_example_tokens():
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.8])
    add_timit_example_tokens_to_gridspec(gs)

def fig4():
    fig = plt.figure(figsize=(6, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.25], wspace=0.4, height_ratios=[1, 2, 0.8], hspace=0.3)

    gs_timit_examples = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[0.8, 1], hspace=0.1, subplot_spec=gs[0, 0])
    gs_ptrf = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1])
    gs_prediction = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[1, :], wspace=0.08)
    gs_scatter = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, :])

    # Panels A-C
    # TIMIT example tokens with absolute and relative pitch features used in pitch temporal receptive field models
    examples = ['fadg0_si1279', 'mabc0_si1620']
    titles = ['"Bricks are an alternative."' , '"His head flopped back."']
    add_timit_example_tokens_to_gridspec(gs_timit_examples, examples=examples, titles=titles)

    # Panel D
    # Example receptive field
    datas, r_mean_all, r_max_all, cat_all, r2s_abs, r2s_rel, wtss, _ = tang.load_all_data()
    wts = wtss[5]
    chan = 182
    subject_number = 129
    add_ptrf_to_gs(gs_ptrf, wts, chan)

    # Panels E-L
    # Prediction of ptrf model on original intonation stimulus set
    all_preds1, all_preds_ste1, abs_pred1, abs_pred_ste1, rel_pred1, rel_pred_ste1, all_preds2, all_preds_ste2, abs_pred2, abs_pred_ste2, rel_pred2, rel_pred_ste2 = ptrf.predict_response_to_intonation_stims(subject_number, chan)

    Y_mat, sns, sts, sps, Y_mat_plotter = tang.load_Y_mat_sns_sts_sps_for_subject_number(subject_number, zscore_to_silence=False)
    male = [Y_mat_plotter[:, :, np.logical_and(sps==1,sts==i)] for i in range(1,5)]
    female = [Y_mat_plotter[:, :, np.logical_and(np.logical_or(sps==2,sps==3),sts==i)] for i in range(1,5)]
    actual_means_stes = [erps.get_mean_and_ste(sp1_one)[0:2] for sp1_one in male + female]
    actual_means1 = [a[0] for a in actual_means_stes]
    actual_stes1 = [a[1] for a in actual_means_stes]
    speakers = [Y_mat_plotter[:, :, sps==1], Y_mat_plotter[:, :, np.logical_or(sps==2, sps==3)]]
    actual_means_stes = [erps.get_mean_and_ste(sp1_one)[0:2] for sp1_one in speakers]
    actual_means2 = [a[0] for a in actual_means_stes]
    actual_stes2 = [a[1] for a in actual_means_stes]

    actual = np.concatenate(actual_means1, axis=1)
    predicted_abs = np.concatenate(abs_pred1, axis=1)
    predicted_rel = np.concatenate(rel_pred1, axis=1)
    abs_pred_corr = pearsonr(actual[chan], predicted_abs[chan])
    rel_pred_corr = pearsonr(actual[chan], predicted_rel[chan])
    print("Corrrelation between actual and abs-only prediction: {:.2f}, (p={:.2e})".format(*abs_pred_corr))
    print("Corrrelation between actual and rel-only prediction: {:.2f}, (p={:.2e})".format(*rel_pred_corr))

    axs = [] 
    for i in range(0, 4):
        axs_row = []
        for j in range(0, 3):
            axs_row.append(plt.subplot(gs_prediction[i, j]))
        axs.append(axs_row)
    axs = np.array(axs)
    xvals = np.arange(0,300)/100 - 0.25

    # Panels E, F
    # Pitch contours by intonation contour (first two) and average pitch contour by sex (last one)
    plot_pitch(axs[0][0], axs[0][1], axs[0][2])

    # Panels G, I, K
    # Predicted and actual high-gamma by intonation contour for two sexes shown separately
    plot_intonation(xvals, abs_pred1[0:4], abs_pred_ste1[0:4], axs[1][1], chan)
    plot_intonation(xvals, abs_pred1[4:8], abs_pred_ste1[4:8], axs[1][0], chan)
    plot_intonation(xvals, rel_pred1[0:4], rel_pred_ste1[0:4], axs[2][1], chan)
    plot_intonation(xvals, rel_pred1[4:8], rel_pred_ste1[4:8], axs[2][0], chan)
    plot_intonation(xvals, actual_means1[0:4], actual_stes1[0:4], axs[3][1], chan)
    plot_intonation(xvals, actual_means1[4:8], actual_stes1[4:8], axs[3][0], chan)

    # Panels H, J, L
    # Predicted and actual high-gamnma averaged over sex of speaker
    plot_speaker(xvals, abs_pred2[0:3], abs_pred_ste2[0:3], axs[1][2], chan)
    plot_speaker(xvals, rel_pred2[0:3], rel_pred_ste2[0:3], axs[2][2], chan)
    plot_speaker(xvals, actual_means2[0:3], actual_stes2[0:3], axs[3][2], chan)

    axs[0][0].set(title="Female speaker")
    axs[0][1].set(title="Male speaker")
    axs[1][0].set(ylabel="Abs only\nprediction")
    axs[2][0].set(ylabel="Rel only\nprediction")
    axs[3][0].set(ylabel="Actual")

    for ax in axs[1:, :].flatten():
        ax.set(xlim=(-0.25, 2.75), xticks=[0, 0.5, 1, 1.5, 2, 2.5], xticklabels=[], ylim=(-1.8,4), yticks=[-1,0,1,2,3,4])
    for ax in axs[:, 1:].flatten():
        ax.set(yticklabels=[])
    for ax in axs[3,:]:
        ax.set(xticklabels=['0', '0.5', '1', '1.5', '2', '2.5'], xlabel="Time (s)")

    # Panel M
    # Scatter plot of relative and absolute pitch encoding versus intonation condition encoding
    ax_rel = plt.subplot(gs_scatter[0])
    ax_abs = plt.subplot(gs_scatter[1])

    data = datas[datas.sig_full == 1]
    st_all = data.st
    rp_all = data.r2_rel
    ap_all = data.r2_abs
    rel_sig = data.rel_sig
    abs_sig = data.abs_sig

    abs_st_corr = pearsonr(st_all, ap_all)
    rel_st_corr = pearsonr(st_all, rp_all)
    print("Corrrelation between intonation R2 and absolute pitch R2: {:.2f}, (p={:.2e})".format(*abs_st_corr))
    print("Corrrelation between intonation R2 and relative pitch R2: {:.2f}, (p={:.2e})".format(*rel_st_corr))

    kws = {'marker': 'o', 'markersize': 3, 'linewidth': 0}
    ax_rel.plot(rp_all[rel_sig==0], st_all[rel_sig==0], color='k', **kws)
    ax_rel.plot(rp_all[rel_sig==1], st_all[rel_sig==1], color='#DB7D12', **kws)
    ax_rel.set_ylabel('Intonation encoding (R2)')

    ax_abs.plot(ap_all[abs_sig==0], st_all[abs_sig==0], color='k', **kws)
    ax_abs.plot(ap_all[abs_sig==1], st_all[abs_sig==1], color='#C13639', **kws)

    ax_abs.set_xlabel('Absolute pitch encoding (R2)')
    ax_rel.set_xlabel('Relative pitch encoding (R2)')

    for ax in [ax_rel, ax_abs]:
        ax.set_ylim((-0.05, 0.8))
        ax.set_yticks(np.linspace(0, 0.8, 5))
    ax_rel.set_xlim((-0.01, 0.09))
    ax_abs.set_xlim((-0.03, 0.26))

    for ax in axs.flatten().tolist() + [ax_rel, ax_abs]:
        seaborn.despine(ax=ax)
    return fig

def plot_intonation(xvals, means, stes, ax, chan):
    colors = ['b', 'g', 'r', 'm']
    order = [1, 2, 3, 0]
    colors = [colors[i] for i in order]
    means = [means[i] for i in order]
    stes = [stes[i] for i in order]
    for i, (mean_all_chans, ste_all_chans) in enumerate(zip(means, stes)):
        mean = mean_all_chans[chan]
        ste = ste_all_chans[chan]
        ax.fill_between(xvals, mean-ste, mean+ste, color=colors[i], alpha=0.3)
        ax.plot(xvals, mean, color=colors[i])

def plot_speaker(xvals, means, stes, ax, chan):
    colors = ['#7A0071', '#49BFBF']
    for i, (mean_all_chans, ste_all_chans) in enumerate(zip(means, stes)):
        mean = mean_all_chans[chan]
        ste = ste_all_chans[chan]
        ax.fill_between(xvals, mean-ste, mean+ste, color=colors[i], alpha=0.3)
        ax.plot(xvals, mean, color=colors[i])

def plot_pitch(ax1, ax2, ax3, return_handles=False):
    pitches, intensities = tang.get_continuous_pitch_and_intensity()
    x_grid = [0, 50, 100, 150, 200, 250]
    y_grid = [50, 100, 200, 400]
    grid_line_kws = {'color': 'lightgray', 'linewidth': 0.6}
    for ax in [ax1, ax2, ax3]:
        for xg in x_grid:
            ax.plot([xg, xg], [45, 440], **grid_line_kws)
        for yg in y_grid:
            ax.plot([-25, 275], [yg, yg], **grid_line_kws)

    h2 = ax1.plot(pitches[1], 'g')
    h3 = ax1.plot(pitches[2], 'r')
    h4 = ax1.plot(pitches[3], 'm')
    h1 = ax1.plot(pitches[0], 'b')
    h6 = ax2.plot(pitches[5], 'g--')
    h7 = ax2.plot(pitches[6], 'r--')
    h8 = ax2.plot(pitches[7], 'm--')
    h5 = ax2.plot(pitches[4], 'b--')

    h9 = ax3.plot(np.mean(pitches[0:4], axis=0), color='#49BFBF')
    h10 = ax3.plot(np.mean(pitches[4:8], axis=0), color='#7A0071')

    for ax in [ax1, ax2, ax3]:
        _ = ax.set(xticks=[0, 50, 100, 150, 200, 250], xlim=(-25,275), xticklabels=[])
        ax.set_yscale("log")
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set(ylim=(45, 440), yticks=[50,100,200,400], yticklabels=['50','100','200','400'])
        ax.set_frame_on(False)
    ax1.set( ylabel="Pitch (Hz)")

    hs = [[h1[0], h2[0], h3[0], h4[0]], [h9[0], h10[0]]]
    if return_handles:
        return hs

def add_ptrf_to_gs(gs, wts, chan, shorten_ylabel=False):
    gs_abs = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 0.2], subplot_spec=gs[0])
    gs_rel = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1, 0.2], subplot_spec=gs[1])
    ax1 = plt.subplot(gs_abs[0])
    ax2 = plt.subplot(gs_rel[0])
    with seaborn.axes_style("white"):
        cax1 = plt.subplot(gs_abs[1])
        cax2 = plt.subplot(gs_rel[1])
    min_value = np.min(wts[chan].reshape(46,23)[3:43, 0:20])
    max_value = np.max(wts[chan].reshape(46,23)[3:43, 0:20])
    abs_value = np.max(np.abs([min_value, max_value]))
    min_value = -1 * abs_value
    max_value = abs_value
    im1 = ax1.imshow(np.fliplr(np.flipud(wts[chan].reshape(46, 23)[3:43,0:10].T)), cmap=plt.get_cmap('RdBu_r'), aspect="auto")
    im3 = ax2.imshow(np.fliplr(np.flipud(wts[chan].reshape(46, 23)[3:43,10:20].T)), cmap=plt.get_cmap('PuOr_r'), aspect="auto")
    for im in [im1, im3]:
        im.set_clim((-1 * abs_value, abs_value))
    min_tick_value = np.trunc(np.ceil(min_value * 100))/100
    max_tick_value = np.trunc(np.floor(max_value * 100))/100
    plt.colorbar(im1, cax=cax1, ticks=[min_tick_value, 0, max_tick_value])
    plt.colorbar(im3, cax=cax2, ticks=[min_tick_value, 0, max_tick_value])
    for ax in [cax1, cax2]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

    im1.axes.set(xticks=[0, 39], xticklabels=[],
        yticks=(0, 3, 6, 9), yticklabels=[250, 200, 150, 90], ylabel="Absolute pitch\n(Hz)")
    im3.axes.set(xticks=[0, 39], xticklabels=[400, 0], xlabel="Delay (ms)", 
        yticks=(0,3,6,9), yticklabels=[1.7, 0.6, -0.5, -1.7], ylabel="Relative pitch\n(z-score)")
    if shorten_ylabel:
        im1.axes.set(ylabel="Abs. pitch (Hz)")
        im3.axes.set(ylabel="Rel. pitch (z-score)")

def sfig4():
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4.2, 3])

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[0.25, 1], wspace=0.5)
    gs_top_ptrf_padding = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_top[0], height_ratios=[1.2, 2.2, 0.8], hspace=0)
    gs_top_ptrf = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top_ptrf_padding[1])
    gs_top_pred = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs_top[1], height_ratios=[1.2, 1, 1, 1])

    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[0.25, 1], wspace=0.5)
    gs_bottom_ptrf_padding = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom[0], height_ratios=[2.2, 0.8], hspace=0)
    gs_bottom_ptrf = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom_ptrf_padding[0])
    gs_bottom_pred = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_bottom[1])

    ptrf_and_prediction(gs_top_ptrf, gs_top_pred, 113, 167, ylim=(-1.4, 2.8))
    ptrf_and_prediction(gs_bottom_ptrf, gs_bottom_pred, 113, 53, with_stimulus=False)

    return fig

def sfig5():
    fig = plt.figure(figsize=(6, 9.33))
    gs = gridspec.GridSpec(3, 1, height_ratios=[4.2, 3, 1.2])

    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[0.25, 1], wspace=0.5)
    gs_top_ptrf_padding = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_top[0], height_ratios=[1.2, 2.2, 0.8], hspace=0)
    gs_top_ptrf = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top_ptrf_padding[1])
    gs_top_pred = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs_top[1], height_ratios=[1.2, 1, 1, 1])

    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], width_ratios=[0.25, 1], wspace=0.5)
    gs_bottom_ptrf_padding = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom[0], height_ratios=[2.2, 0.8], hspace=0)
    gs_bottom_ptrf = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom_ptrf_padding[0])
    gs_bottom_pred = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_bottom[1])

    gs_scatter = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2])

    ptrf_and_prediction(gs_top_ptrf, gs_top_pred, 129, 202, ylim=(-1.1, 1.6))
    ptrf_and_prediction(gs_bottom_ptrf, gs_bottom_pred, 131, 195, with_stimulus=False, ylim=(-1.4, 4.2))

    datas, r_mean_all, r_max_all, cat_all, r2s_abs, r2s_rel, wtss, _ = tang.load_all_data()

    # Scatter plot of relative and absolute pitch encoding versus intonation condition encoding
    ax_rel = plt.subplot(gs_scatter[0])
    ax_abs = plt.subplot(gs_scatter[1])

    data = datas[datas.sig_full == 1]
    sp_all = data.sp
    rp_all = data.r2_rel
    ap_all = data.r2_abs
    rel_sig = data.rel_sig
    abs_sig = data.abs_sig

    kws = {'marker': 'o', 'markersize': 3, 'linewidth': 0}
    ax_rel.plot(rp_all[rel_sig==0], sp_all[rel_sig==0], color='k', **kws)
    ax_rel.plot(rp_all[rel_sig==1], sp_all[rel_sig==1], color='#DB7D12', **kws)
    ax_rel.set_ylabel('Speaker encoding (R2)')

    ax_abs.plot(ap_all[abs_sig==0], sp_all[abs_sig==0], color='k', **kws)
    ax_abs.plot(ap_all[abs_sig==1], sp_all[abs_sig==1], color='#C13639', **kws)

    ax_abs.set_xlabel('Absolute pitch encoding (R2)')
    ax_rel.set_xlabel('Relative pitch encoding (R2)')

    abs_sp_corr = pearsonr(sp_all, ap_all)
    rel_sp_corr = pearsonr(sp_all, rp_all)
    print("Corrrelation between speaker R2 and absolute pitch R2: {:.2f}, (p={:.2e})".format(*abs_sp_corr))
    print("Corrrelation between speaker R2 and relative pitch R2: {:.2f}, (p={:.2e})".format(*rel_sp_corr))

    for ax in [ax_rel, ax_abs]:
        ax.set_ylim((-0.05, 0.9))
        ax.set_yticks(np.linspace(0, 0.8, 5))
    ax_rel.set_xlim((-0.01, 0.09))
    ax_abs.set_xlim((-0.03, 0.26))

    seaborn.despine(ax=ax_abs)
    seaborn.despine(ax=ax_rel)
    return fig

def ptrf_and_prediction(gs_ptrf, gs_pred, subject_number, chan, with_stimulus=True, ylim=(-1.8, 4)):
    datas, r_mean_all, r_max_all, cat_all, r2s_abs, r2s_rel, wtss, _ = tang.load_all_data([subject_number])
    wts = wtss[0]

    add_ptrf_to_gs(gs_ptrf, wts, chan, shorten_ylabel=True)

    # Prediction of ptrf model on original intonation stimulus set
    all_preds1, all_preds_ste1, abs_pred1, abs_pred_ste1, rel_pred1, rel_pred_ste1, all_preds2, all_preds_ste2, abs_pred2, abs_pred_ste2, rel_pred2, rel_pred_ste2 = ptrf.predict_response_to_intonation_stims(subject_number, chan)

    Y_mat, sns, sts, sps, Y_mat_plotter = tang.load_Y_mat_sns_sts_sps_for_subject_number(subject_number, zscore_to_silence=False)
    male = [Y_mat_plotter[:, :, np.logical_and(sps==1,sts==i)] for i in range(1,5)]
    female = [Y_mat_plotter[:, :, np.logical_and(np.logical_or(sps==2,sps==3),sts==i)] for i in range(1,5)]
    actual_means_stes = [erps.get_mean_and_ste(sp1_one)[0:2] for sp1_one in male + female]
    actual_means1 = [a[0] for a in actual_means_stes]
    actual_stes1 = [a[1] for a in actual_means_stes]
    speakers = [Y_mat_plotter[:, :, sps==1], Y_mat_plotter[:, :, np.logical_or(sps==2, sps==3)]]
    actual_means_stes = [erps.get_mean_and_ste(sp1_one)[0:2] for sp1_one in speakers]
    actual_means2 = [a[0] for a in actual_means_stes]
    actual_stes2 = [a[1] for a in actual_means_stes]

    actual = np.concatenate(actual_means1, axis=1)
    predicted_abs = np.concatenate(abs_pred1, axis=1)
    predicted_rel = np.concatenate(rel_pred1, axis=1)
    abs_pred_corr = pearsonr(actual[chan], predicted_abs[chan])
    rel_pred_corr = pearsonr(actual[chan], predicted_rel[chan])
    print("Prediction results for subject EC{}, chan {}".format(subject_number, chan))
    print("Corrrelation between actual and abs-only prediction: {:.2f}, (p={:.2e})".format(*abs_pred_corr))
    print("Corrrelation between actual and rel-only prediction: {:.2f}, (p={:.2e})".format(*rel_pred_corr))

    ax_row_offset = 0
    if with_stimulus:
        ax_pitch0 = plt.subplot(gs_pred[0, 0])
        ax_pitch1 = plt.subplot(gs_pred[0, 1])
        ax_pitch2 = plt.subplot(gs_pred[0, 2])
        ax_row_offset = 1

    axs = [] 
    for i in range(ax_row_offset, ax_row_offset+3):
        axs_row = []
        for j in range(0, 3):
            axs_row.append(plt.subplot(gs_pred[i, j]))
        axs.append(axs_row)
    axs = np.array(axs)
    xvals = np.arange(0,300)/100 - 0.25

    if with_stimulus:
        # Pitch contours by intonation contour (first two) and average pitch contour by sex (last one)
        hs = plot_pitch(ax_pitch0, ax_pitch1, ax_pitch2, return_handles=True)
        leg1 = ax_pitch1.legend(hs[0], ['Neutral', 'Question', 'Emphasis 1', 'Emphasis 2'], loc=2, bbox_to_anchor=(0.2, 1.2))
        leg2 = ax_pitch2.legend(hs[1], ['Female', 'Male'], loc=2, bbox_to_anchor=(0.2, 1.2))
        for leg in [leg1, leg2]:
            frame = leg.get_frame()
            frame.set_facecolor("white")
        for ax in [ax_pitch0, ax_pitch1, ax_pitch2]:
            seaborn.despine(ax=ax)

    # Predicted and actual high-gamma by intonation contour for two sexes shown separately
    plot_intonation(xvals, abs_pred1[0:4], abs_pred_ste1[0:4], axs[0][1], chan)
    plot_intonation(xvals, abs_pred1[4:8], abs_pred_ste1[4:8], axs[0][0], chan)
    plot_intonation(xvals, rel_pred1[0:4], rel_pred_ste1[0:4], axs[1][1], chan)
    plot_intonation(xvals, rel_pred1[4:8], rel_pred_ste1[4:8], axs[1][0], chan)
    plot_intonation(xvals, actual_means1[0:4], actual_stes1[0:4], axs[2][1], chan)
    plot_intonation(xvals, actual_means1[4:8], actual_stes1[4:8], axs[2][0], chan)

    # Predicted and actual high-gamnma averaged over sex of speaker
    plot_speaker(xvals, abs_pred2[0:3], abs_pred_ste2[0:3], axs[0][2], chan)
    plot_speaker(xvals, rel_pred2[0:3], rel_pred_ste2[0:3], axs[1][2], chan)
    plot_speaker(xvals, actual_means2[0:3], actual_stes2[0:3], axs[2][2], chan)

    if with_stimulus:
        ax_pitch0.set(title="Female speaker")
        ax_pitch1.set(title="Male speaker")
        for ax in [ax_pitch1, ax_pitch2]:
            ax.set(yticklabels=[])
    else:
        axs[0][0].set(title="Female speaker")
        axs[0][1].set(title="Male speaker")
    axs[0][0].set(ylabel="Abs only\nprediction")
    axs[1][0].set(ylabel="Rel only\nprediction")
    axs[2][0].set(ylabel="Actual")

    for ax in axs.flatten():
        ax.set(xlim=(-0.25, 2.75), xticks=[0, 0.5, 1, 1.5, 2, 2.5], xticklabels=[], ylim=ylim)
    for ax in axs[:, 1:].flatten():
        ax.set(yticklabels=[])
    for ax in axs[2,:]:
        ax.set(xticklabels=['0', '0.5', '1', '1.5', '2', '2.5'], xlabel="Time (s)")

    for ax in axs.flatten():
        seaborn.despine(ax=ax)
