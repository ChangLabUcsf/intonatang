from __future__ import print_function, division, absolute_import

import os
subject_data_path = os.path.join(os.path.dirname(__file__), 'data', 'subject_data')
timit_data_path = os.path.join(os.path.dirname(__file__), 'data', 'timit')
timit_pitch_data_path = os.path.join(os.path.dirname(__file__), 'data', 'timit_pitch')
processed_timit_data_path = os.path.join(os.path.dirname(__file__), 'processed_timit_data')
results_path = os.path.join(os.path.dirname(__file__), 'results')

import numpy as np
import scipy.stats as stats
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd
import h5py
import tables
import glob


def generate_all_results(regenerate_processed_timit_data=False):
    if regenerate_processed_timit_data:
        generate_processed_timit_data()

    subject_numbers = [113, 118, 122, 123, 125, 129, 131, 137, 142, 143]
    for subject_number in subject_numbers:
        out = load_h5py_out(subject_number)
        average_response = get_average_response_to_phonemes(out)
        psis = get_psis(out)
        save_average_response_psis_for_subject_number(subject_number, average_response, psis)

def generate_processed_timit_data():
    save_timit_pitch()
    save_timit_phonemes()
    save_timit_pitch_phonetic()


phoneme_order = ['d','b','g','p','k','t','jh','sh','z','s','f','th','dh','v','w','r','l','ae','aa','ay','aw','ow','ax','uw','eh','ey','ih','ux','iy','n','m','ng']
channel_order = [256,240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,255,239,223,207,191,175,159,143,127,111,95,79,63,47,31,15,254,238,222,206,190,174,158,142,126,110,94,78,62,46,30,14,253,237,221,205,189,173,157,141,125,109,93,77,61,45,29,13,252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12,251,235,219,203,187,171,155,139,123,107,91,75,59,43,27,11,250,234,218,202,186,170,154,138,122,106,90,74,58,42,26,10,249,233,217,201,185,169,153,137,121,105,89,73,57,41,25,9,248,232,216,200,184,168,152,136,120,104,88,72,56,40,24,8,247,231,215,199,183,167,151,135,119,103,87,71,55,39,23,7,246,230,214,198,182,166,150,134,118,102,86,70,54,38,22,6,245,229,213,197,181,165,149,133,117,101,85,69,53,37,21,5,244,228,212,196,180,164,148,132,116,100,84,68,52,36,20,4,243,227,211,195,179,163,147,131,115,99,83,67,51,35,19,3,242,226,210,194,178,162,146,130,114,98,82,66,50,34,18,2,241,225,209,193,177,161,145,129,113,97,81,65,49,33,17,1]
phonetic_features = ['dorsal','coronal','labial','high','front','low','back','plosive','fricative','syllabic','nasal','voiced','obstruent','sonorant']
phonetic_dict = {
    'b': ['labial', 'plosive', 'voiced','obstruent'],
    'bcl':['labial', 'plosive', 'voiced','obstruent'],
    'p': ['labial', 'plosive', 'obstruent'],
    'pcl':['labial', 'plosive', 'obstruent'],
    'd': ['coronal', 'plosive', 'voiced','obstruent'],
    'dcl':['coronal', 'plosive', 'voiced','obstruent'],
    't': ['coronal', 'plosive', 'obstruent'],
    'tcl':['coronal', 'plosive', 'obstruent'],  
    'g': ['dorsal', 'plosive', 'voiced','obstruent'],
    'gcl':['dorsal', 'plosive', 'voiced','obstruent'],
    'k': ['dorsal', 'plosive', 'obstruent'],
    'kcl':['dorsal', 'plosive', 'obstruent'],
    'dh': ['fricative', 'obstruent', 'voiced'],
    'th': ['fricative', 'obstruent'],
    'f': ['fricative', 'obstruent', 'labial'],
    's': ['fricative', 'obstruent', 'coronal'],
    'sh': ['fricative', 'obstruent', 'dorsal'],
    'z': ['fricative', 'obstruent', 'coronal', 'voiced'],
    'v': ['fricative', 'obstruent', 'labial', 'voiced'],
    'm': ['nasal', 'obstruent', 'sonorant', 'voiced', 'labial'],
    'n': ['nasal', 'obstruent', 'sonorant', 'voiced', 'coronal'],
    'ng': ['nasal', 'obstruent', 'sonorant', 'voiced', 'dorsal'],

    'aa': ['low', 'back', 'syllabic', 'voiced', 'sonorant'],
    'ao': ['low', 'back', 'syllabic', 'voiced', 'sonorant'],
    'ow': ['syllabic','voiced', 'sonorant'],
    'ax': ['syllabic','voiced','sonorant'],
    'ux': ['high', 'back', 'syllabic', 'voiced', 'sonorant'],
    'uw': ['high', 'back', 'syllabic', 'voiced', 'sonorant'],
    'iy': ['high', 'front', 'syllabic', 'voiced', 'sonorant'],
    'ih': ['high', 'front', 'syllabic', 'voiced', 'sonorant'],
    'ey': ['syllabic', 'voiced', 'sonorant'],
    'eh': ['front', 'low', 'syllabic', 'voiced', 'sonorant'],
    'ae': ['front', 'low', 'syllabic', 'voiced', 'sonorant'],
    'aw': ['syllabic', 'voiced', 'sonorant'],
    'ay': ['syllabic', 'voiced', 'sonorant'],
    'w': ['voiced', 'sonorant'],
    'y': ['voiced', 'sonorant'],
    'r': ['voiced', 'sonorant', 'obstruent'],
    'l': ['voiced', 'sonorant', 'obstruent']
}

def get_timit_erps(subject_number):
    timit_onsets_offsets = get_timit_sent_first_phoneme_start_times()

    out = load_tables_out(subject_number)
    number_of_trials = 0
    for trial in out:
        number_of_trials += trial.ecog.shape[2]

    Y_mat_onset = np.zeros((256, 300, number_of_trials))
    Y_mat_offset = np.zeros((256, 250, number_of_trials))

    females = np.zeros((number_of_trials))

    count = 0
    for trial in out:
        timit_name = trial._v_attrs['timit_name'][0]
        onset = np.round(timit_onsets_offsets[timit_name][0]) + 50
        offset = np.round(timit_onsets_offsets[timit_name][1]) + 50

        ecog = trial.ecog.read()
        for i in range(trial.ecog.shape[2]):
            ecog_trial = ecog[:, :, i]
            if(onset+250 < ecog_trial.shape[1]):

                Y_mat_onset[:, :, count] = ecog_trial[:256, onset-50:onset+250]
                Y_mat_offset[:, :, count] = ecog_trial[:256, offset-100:offset+150]
                    
                if timit_name[0] == 'f':
                    females[count] = 1
                count = count + 1
    Y_mat_onset = Y_mat_onset[:,:,0:count]
    Y_mat_offset = Y_mat_offset[:,:,0:count]
    females = females[:count]

    return Y_mat_onset, Y_mat_offset, females

def plot_timit_onset_erps(Y_mat, indexes1, indexes2, gc=np.arange(256), x_zero=None):
    fig = plot_grid_mean_ste(Y_mat, indexes1, indexes2, gc=gc, x_zero=x_zero)
    return fig

def plot_timit_onset_erps_for_subject(subject_number):
    Y_mat_onset, Y_mat_offset, females = get_timit_erps(subject_number)
    gc = np.arange(256)
    fig = plot_timit_onset_erps(Y_mat_onset, females==0, females==1, gc, x_zero=50)
    return fig

def get_speech_responsive_chans(out):
    number_of_sentences = len(out._f_list_nodes())
    hg_during_silence = np.zeros((256, 5, number_of_sentences))
    hg_during_speech = np.zeros((256, 5, number_of_sentences))
    timit_onsets_offsets = get_timit_sent_first_phoneme_start_times()

    for i, trial in enumerate(out):
        timit_name = trial._v_attrs['timit_name'][0]
        onset = np.round(timit_onsets_offsets[timit_name][0]) + 50
        ecog = trial.ecog.read()
        ecog_trial = ecog[:,:,0]
        rand_indexes_silence = np.random.permutation(30)[0:5].astype('int')
        rand_indexes_speech = (np.random.permutation(60)[0:5] + onset).astype('int')
        hg_during_silence[:,:,i] = ecog_trial[:,rand_indexes_silence]
        hg_during_speech[:,:,i] = ecog_trial[:,rand_indexes_speech]

    hg_during_silence = np.reshape(hg_during_silence, (256, 5 * number_of_sentences))
    hg_during_speech = np.reshape(hg_during_speech, (256, 5 * number_of_sentences))
    sig_chans = []
    for chan in np.arange(256):
        z_stat, p_value = stats.ranksums(hg_during_silence[chan,:], hg_during_speech[chan,:])
        if(p_value < 0.001):
            sig_chans.append(chan)

    return sig_chans

def load_tables_out(subject_number):
    filename = os.path.join(subject_data_path, 'EC' + str(subject_number), 'EC' + str(subject_number) + '_timit.h5')
    f = tables.open_file(filename)
    if subject_number == 113:
        out = f.root.EC113
    elif subject_number == 118:
        out = f.root.EC118
    elif subject_number == 122:
        out = f.root.EC122
    elif subject_number == 123:
        out = f.root.EC123
    elif subject_number == 125:
        out = f.root.EC125
    elif subject_number == 129:
        out = f.root.EC129
    elif subject_number == 131:
        out = f.root.EC131
    elif subject_number == 137:
        out = f.root.EC137
    elif subject_number == 142:
        out = f.root.EC142
    elif subject_number == 143:
        out = f.root.EC143
    return out

def load_h5py_out(subject_number):
    filename = os.path.join(subject_data_path, 'EC' + str(subject_number), 'EC' + str(subject_number) + '_timit.h5')
    f = h5py.File(filename, 'r')  
    return f['EC' + str(subject_number)]

def get_timit_sent_first_phoneme_start_times():
    timit_onsets_offsets = {}
    phn_file_names = glob.glob(os.path.join(timit_data_path, '*.phn'))
    for phn_file in phn_file_names:
        phns = pd.read_csv(phn_file, header=None, delimiter=' ', names=['start_time', 'end_time', 'phn'] )   
        phns['start_time'] = phns['start_time']/16000    
        phns['end_time'] = phns['end_time']/16000

        timit_name = phn_file.split(os.sep)[-1][:-4]

        timit_onsets_offsets[timit_name] = np.array([phns['start_time'].values[1], phns['end_time'].values[-2]]) * 100

    return timit_onsets_offsets

def convert_hz(hz, to="log"):
    hz = np.array(hz)
    assert np.all(hz[~np.isnan(hz)] > 0)
    assert np.all(hz[~np.isnan(hz)] < 20000)
    if to == "log":
        return np.log(hz)
    elif to == "mel":
        return 1127*np.log(1+hz/700)
    elif to == "bark":
        return ((26.81/(1+1960/hz))-0.53)
    elif to == "erb":
        return 11.17 * np.log((hz+312)/(hz+14675)) + 43

def save_timit_pitch():
    """This function saves a pandas dataframe of pitch information for TIMIT sentences.

    This script takes the pitch information (fundamental frequency in Hz in 10ms bins) 
    written in *.wav.txt files (which are output from Praat and manually examined for doubling or halving errors),
    and saves timit_pitch, a dataframe that contains absolute and relative pitch values.

    Relative pitch values are computed as a z-score across each sentence's absolute pitch values. Absolute pitch values are 
    saved with two scalings, one is log Hz and the other is erb-rate. These monotonic, non-linear transformations are based 
    on psychophysical data showing that differences in Hz are perceived differently wtih frequency and lead to pitch values
    that are more linear with respect to pitch perception.
    """
    timit_names = []
    pitch_intensity_tables = []

    wav_txt_file_names = glob.glob(os.path.join(timit_pitch_data_path, '*.wav.txt'))
    for wav_txt_file in wav_txt_file_names:
        pitch_intensity = pd.read_csv(wav_txt_file, delimiter='\t', dtype=np.float64, na_values=['?'])
        pitch_intensity = pitch_intensity.dropna()
        pitch_intensity.loc[pitch_intensity.pitch == 0, 'pitch'] = np.NaN
        pitch_intensity.loc[pitch_intensity.intensity == 0, 'intensity'] = np.NaN
        pitch_intensity['log_hz'] = np.log(pitch_intensity['pitch'])
        pitch_intensity['erb_rate'] = convert_hz(pitch_intensity['pitch'], "erb")
        pitch = pitch_intensity['log_hz']
        pitch_intensity['rel_pitch_global'] = (pitch - np.mean(pitch))/np.std(pitch)
        pitch = pitch_intensity['erb_rate']
        pitch_intensity['rel_pitch_global_erb'] = (pitch - np.mean(pitch))/np.std(pitch)

        timit_name = wav_txt_file.split(os.sep)[-1][:-8]

        timit_names.append(timit_name)
        pitch_intensity_tables.append(pitch_intensity)

    timit_pitch = pd.concat(pitch_intensity_tables, keys=timit_names)
    #print(np.mean(timit_pitch['log_hz']))  # -> 4.9406, (no log: 147.0387)
    #print(np.std(timit_pitch['log_hz']))   # -> 0.3112, (no log: 48.59846)
    timit_pitch['abs_pitch'] = (timit_pitch['log_hz'] - np.mean(timit_pitch['log_hz']))/np.std(timit_pitch['log_hz'])
    timit_pitch['abs_pitch_erb'] = (timit_pitch['erb_rate'] - np.mean(timit_pitch['erb_rate']))/np.std(timit_pitch['erb_rate'])
    timit_pitch['abs_pitch_change'] = timit_pitch['abs_pitch'].diff()
    timit_pitch['abs_pitch_erb_change'] = timit_pitch['abs_pitch_erb'].diff()
    #print(np.mean(timit_pitch.intensity)) # -> 63.000
    #print(np.std(timit_pitch.intensity)) # -> 15.537
    timit_pitch['zscore_intensity'] = (timit_pitch.intensity - np.mean(timit_pitch.intensity))/np.std(timit_pitch.intensity)

    filename = os.path.join(processed_timit_data_path, 'timit_pitch.h5')
    timit_pitch.to_hdf(filename, 'timit_pitch')
    return timit_pitch

def zscore_intensity(intensity):
    return (intensity - 63.000)/15.537

def zscore_abs_pitch(pitch, reverse=False):
    if reverse:
        return np.exp((pitch * 0.3112) + 4.9406)
    else:
        pitch = np.log(pitch)
        return (pitch - 4.9406)/0.3112

def get_timit_pitch():
    filename = os.path.join(processed_timit_data_path, 'timit_pitch.h5')
    timit_pitch = pd.read_hdf(filename, 'timit_pitch')
    return timit_pitch

def save_timit_phonemes():
    phoneme_tables = []
    timit_names = []

    phn_file_names = glob.glob(os.path.join(timit_data_path, '*.phn'))
    for phn_file in phn_file_names:
        phns = pd.read_csv(phn_file, header=None, delimiter=' ', names=['start_time', 'end_time', 'phn'] )   
        phns['start_time'] = phns['start_time']/16000
        phns['end_time'] = phns['end_time']/16000

        timit_name = phn_file.split(os.sep)[-1][:-4]

        phoneme_tables.append(phns)
        timit_names.append(timit_name)

    timit_phonemes = pd.concat(phoneme_tables, keys=timit_names, names=['timit_name', 'phoneme_index'])
    filename = os.path.join(processed_timit_data_path, 'timit_phonemes.h5')
    timit_phonemes.to_hdf(filename, 'timit_phonemes')
    return timit_phonemes

def get_timit_phonemes():
    filename = os.path.join(processed_timit_data_path, 'timit_phonemes.h5')
    timit_phonemes = pd.read_hdf(filename, 'timit_phonemes')
    return timit_phonemes

def save_timit_pitch_phonetic():
    """This function combines timit_pitch with timit_phonemes and needs to be run after save_timit_pitch and save_timit_phonemes

    """
    timit_pitch = get_timit_pitch()
    timit_phonemes = get_timit_phonemes()

    timit_phonemes['start_index'] = (timit_phonemes['start_time'] * 100).round()
    timit_phonemes['end_index'] = (timit_phonemes['end_time'] * 100).round()
    
    timit_pitch['phn'] = 'h#'

    for i, row in enumerate(timit_phonemes.iterrows()):
        if row[1]['phn'] != 'h#':
            timit_name = row[0][0]
            phn = row[1]['phn']
            start_index = row[1]['start_index'] - 1
            end_index = row[1]['end_index'] - 1 
            
            for index in np.arange(start_index, end_index):
                timit_pitch.set_value((timit_name, index), 'phn', phn)
    
    for feat in phonetic_features:
        timit_pitch[feat] = 0
        
    a = timit_phonemes['phn'].unique()
    not_included = set(a) - set(phonetic_dict.keys())

    for row in timit_pitch.iterrows():
        if row[1]['phn'] not in not_included:
            for val in phonetic_dict[row[1]['phn']]:
                timit_pitch.set_value(row[0], val, 1)

    filename = os.path.join(processed_timit_data_path, 'timit_pitch_phonetic.h5')
    timit_pitch.to_hdf(filename, 'timit_pitch_phonetic')
    return timit_pitch

def get_timit_pitch_phonetic():
    filename = os.path.join(processed_timit_data_path, 'timit_pitch_phonetic.h5')
    timit_pitch = pd.read_hdf(filename, 'timit_pitch_phonetic')
    return timit_pitch

def get_average_response_to_phonemes(out, phoneme_order=phoneme_order):
    """Returns the average response over all instances of each phoneme in TIMIT
    """
    timit_phonemes = get_timit_phonemes()
    names = [i[0] for i in out.items()] #get names of sentences that were recorded for the specific subject.
    timit_phonemes = timit_phonemes[timit_phonemes.index.get_level_values(0).isin(names)]

    #response is 500ms, 100ms before phoneme onset to 400ms after phoneme onset.
    average_response = np.zeros((256, len(phoneme_order), 50))
    for p_index, phoneme in enumerate(phoneme_order):
        phonemes = timit_phonemes[timit_phonemes.phn == phoneme]
        average_response_phoneme = np.zeros((256, 50, len(phonemes)))
        for i, trial in enumerate(phonemes.iterrows()):
            timit_name = trial[0][0]
            start_time = trial[1].start_time
            start_index = int(round((start_time - 0.1)*100) + 50) #+50 to account for 500ms offset in the neural data from generating the out file
            average_response_phoneme[:,:,i] = out[timit_name]['ecog'][:][:256,start_index:start_index+50,0]
        average_response[:,p_index,:] = np.mean(average_response_phoneme,2)

    return average_response

def get_psis(out, phoneme_order=phoneme_order):
    # timit_phonemes is a dataframe containing information about phoneme onsets in timit sentences
    timit_phonemes = get_timit_phonemes()
    names = [i[0] for i in out.items()] # timit sentences that are in a given subject's out data file
    timit_phonemes = timit_phonemes[timit_phonemes.index.get_level_values(0).isin(names)]

    psis = np.zeros((256, len(phoneme_order))) 

    #Get the distribution of high-gamma values at 110ms after phoneme onset for each electrode for each phoneme.
    #The phoneme is the key used in the dict activitiy_distributions
    activity_distributions = {}
    for phoneme in phoneme_order:
        # all instances of a specific phoneme in the set of timit sentences a subject heard
        phoneme_instances = timit_phonemes[timit_phonemes.phn == phoneme]

        activity_phoneme = np.zeros((256, len(phoneme_instances)))
        for i, trial in enumerate(phoneme_instances.iterrows()):
            timit_name = trial[0][0]
            start_time = trial[1].start_time
            index = np.int(round((start_time + 0.11)*100) + 50)

            activity_phoneme[:, i] = out[timit_name]['ecog'][:][:256,index,0].flatten()

        activity_distributions[phoneme] = activity_phoneme

    phonemes = set(phoneme_order)

    for p_index, phoneme1 in enumerate(phoneme_order):
        dist1 = activity_distributions[phoneme1]

        for phoneme2 in phonemes - set([phoneme1]):
            dist2 = activity_distributions[phoneme2]

            for chan in np.arange(256):
                z_stat, p_value = stats.ranksums(dist1[chan,:], dist2[chan,:])
                if(p_value < 0.001):
                    psis[chan, p_index] = psis[chan, p_index] + 1

    return psis.T

def save_average_response_psis_for_subject_number(subject_number, average_response, psis):
    filename = os.path.join(results_path, "EC" + str(subject_number) + "_timit_average_response_psis.mat")
    sio.savemat(filename, {'average_response': average_response, 'psis': psis})

def load_average_response_psis_for_subject_number(subject_number):
    filename = os.path.join(results_path, "EC" + str(subject_number) + "_timit_average_response_psis.mat")
    data = sio.loadmat(filename)
    return data['average_response'], data['psis']

def plot_nima_fig1(average_response, psis, cm_choice=plt.get_cmap('bwr'), five_chans=[245,130,165,131,69]):
    resp_axes_positions = [[0.058, 0.1, 0.11, 0.75],
                            [0.218, 0.1, 0.11, 0.75],
                            [0.378, 0.1, 0.11, 0.75],
                            [0.538, 0.1, 0.11, 0.75],
                            [0.698, 0.1, 0.11, 0.75]] 

    psi_axes_positions = [[0.058+0.111, 0.1, 0.02, 0.75],
                            [0.218+0.111, 0.1, 0.02, 0.75],
                            [0.378+0.111, 0.1, 0.02, 0.75],
                            [0.538+0.111, 0.1, 0.02, 0.75],
                            [0.698+0.111, 0.1, 0.02, 0.75]]

    min_value = np.min(average_response[five_chans,:,:])
    max_value = np.max(average_response[five_chans,:,:])


    fig = plt.figure(figsize=(7,7))
    for i, resp_axes_position in enumerate(resp_axes_positions):
        ax = fig.add_axes(resp_axes_position)
        im1 = ax.imshow(average_response[five_chans[i],:,:], interpolation='none', aspect='auto', vmin=-1*0.8*max_value, vmax=0.8*max_value, cmap=cm_choice)
        add_resp_axes_styles(ax)
    for i, psi_axes_position in enumerate(psi_axes_positions):
        ax = fig.add_axes(psi_axes_position)
        im2 = ax.imshow(np.atleast_2d(psis[:,five_chans[i]]).T, interpolation='none', aspect='auto', cmap=plt.get_cmap('Greys'), vmin=0, vmax=len(phoneme_order))
        ax.set_axis_off()

    ax1 = fig.add_axes([0.86, 0.1, 0.05, 0.3])
    plt.colorbar(im1, cax=ax1, label='high-gamma (z-score)')
    ax2 = fig.add_axes([0.86, 0.5, 0.05, 0.3])
    plt.colorbar(im2, cax=ax2, label='psi')

    return fig

def plot_erps_for_five_chans(Y_mat_onset, indexes1, indexes2, chans, x_zero=None):
    axes_positions =  [[0.058, 0.2, 0.13, 0.65],
                            [0.218, 0.2, 0.13, 0.65],
                            [0.378, 0.2, 0.13, 0.65],
                            [0.538, 0.2, 0.13, 0.65],
                            [0.698, 0.2, 0.13, 0.65]] 

    Y_mat = Y_mat_onset[chans,:,:]

    average_hg1 = np.mean(Y_mat[:,:,indexes1], 2)
    ste_hg1 = np.std(Y_mat[:,:,indexes1], 2)/np.sqrt(sum(indexes1))
    average_hg2 = np.mean(Y_mat[:,:,indexes2], 2)
    ste_hg2 = np.std(Y_mat[:,:,indexes2], 2)/np.sqrt(sum(indexes2))   
    min_value = np.min(average_hg2)
    max_value = np.max(average_hg2)    

    min_value = -1.5
    max_value = 3.5    

    fig = plt.figure(figsize=(10, 2))
    for i, pos in enumerate(axes_positions):
        ax = fig.add_axes(pos)
        ax.plot(average_hg1[i], 'b')
        ax.fill_between(np.arange(Y_mat.shape[1]), average_hg1[i] + ste_hg1[i], average_hg1[i] - ste_hg1[i], color='b', alpha=0.2)
        ax.plot(average_hg2[i], 'r')
        ax.fill_between(np.arange(Y_mat.shape[1]), average_hg2[i] + ste_hg2[i], average_hg2[i] - ste_hg2[i], color='r', alpha=0.2)

        ax.set_xticklabels(['','0','','1','','2'])
        ax.set_xlabel('Time (s)')
        if i != 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('High-gamma (z-score)')
        if x_zero is not None:
            ax.plot([x_zero, x_zero], [min_value, max_value], color='k', alpha=0.4)
        ax.set_ylim(min_value, max_value)
        ax.set_title(str(chans[i]), {'fontsize':20})  
    return fig

def add_resp_axes_styles(ax):
    ax.set_yticks(np.arange(len(phoneme_order)))
    h = ax.set_yticklabels(phoneme_order)
    ax.set_xticks([0,10,20,30,40,50])
    ax.set_xticklabels(['','0','','0.2','','0.4'])
    ax.set_xlabel('Time (s)')
    ymin,ymax = ax.get_ylim()
    ax.plot([10,10],[ymin,ymax], 'k--')
    return ax

def plot_grid_phoneme_response(average_response, gc):
    min_value = np.min(average_response)
    max_value = np.max(average_response)*0.7

    fig = plt.figure(figsize=(20,20))
    channel_order = [256,240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,255,239,223,207,191,175,159,143,127,111,95,79,63,47,31,15,254,238,222,206,190,174,158,142,126,110,94,78,62,46,30,14,253,237,221,205,189,173,157,141,125,109,93,77,61,45,29,13,252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12,251,235,219,203,187,171,155,139,123,107,91,75,59,43,27,11,250,234,218,202,186,170,154,138,122,106,90,74,58,42,26,10,249,233,217,201,185,169,153,137,121,105,89,73,57,41,25,9,248,232,216,200,184,168,152,136,120,104,88,72,56,40,24,8,247,231,215,199,183,167,151,135,119,103,87,71,55,39,23,7,246,230,214,198,182,166,150,134,118,102,86,70,54,38,22,6,245,229,213,197,181,165,149,133,117,101,85,69,53,37,21,5,244,228,212,196,180,164,148,132,116,100,84,68,52,36,20,4,243,227,211,195,179,163,147,131,115,99,83,67,51,35,19,3,242,226,210,194,178,162,146,130,114,98,82,66,50,34,18,2,241,225,209,193,177,161,145,129,113,97,81,65,49,33,17,1]
    for i in range(256):
        if i in gc:
            ax = fig.add_subplot(16,16,channel_order[i])
            ax.imshow(average_response[i,:,:], aspect='auto', cmap=plt.get_cmap('bwr'), interpolation='None', vmin=-1*max_value, vmax=max_value)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(0.2,0.8, str(i), transform=ax.transAxes)
    fig.tight_layout()
    return fig

def plot_grid_mean_ste(Y_mat, indexes1, indexes2, gc, x_zero=None):
    fig = plt.figure(figsize=(20,20))
    channel_order = [256,240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,255,239,223,207,191,175,159,143,127,111,95,79,63,47,31,15,254,238,222,206,190,174,158,142,126,110,94,78,62,46,30,14,253,237,221,205,189,173,157,141,125,109,93,77,61,45,29,13,252,236,220,204,188,172,156,140,124,108,92,76,60,44,28,12,251,235,219,203,187,171,155,139,123,107,91,75,59,43,27,11,250,234,218,202,186,170,154,138,122,106,90,74,58,42,26,10,249,233,217,201,185,169,153,137,121,105,89,73,57,41,25,9,248,232,216,200,184,168,152,136,120,104,88,72,56,40,24,8,247,231,215,199,183,167,151,135,119,103,87,71,55,39,23,7,246,230,214,198,182,166,150,134,118,102,86,70,54,38,22,6,245,229,213,197,181,165,149,133,117,101,85,69,53,37,21,5,244,228,212,196,180,164,148,132,116,100,84,68,52,36,20,4,243,227,211,195,179,163,147,131,115,99,83,67,51,35,19,3,242,226,210,194,178,162,146,130,114,98,82,66,50,34,18,2,241,225,209,193,177,161,145,129,113,97,81,65,49,33,17,1]

    #average_hg1 = np.mean(Y_mat[:,:,indexes1], 2)
    #ste_hg1 = np.std(Y_mat[:,:,indexes1], 2)/np.sqrt(sum(indexes1))
    average_hg2 = np.mean(Y_mat[:,:,indexes2], 2)
    ste_hg2 = np.std(Y_mat[:,:,indexes2], 2)/np.sqrt(sum(indexes2))
    min_value = np.min(average_hg2)
    max_value = np.max(average_hg2)

    for i in range(256):
        if i in gc:
            ax = fig.add_subplot(16,16,channel_order[i])
            #ax.plot(average_hg1[i], 'b')
            #ax.fill_between(np.arange(Y_mat.shape[1]), average_hg1[i] + ste_hg1[i], average_hg1[i] - ste_hg1[i], color='b', alpha=0.2)
            ax.plot(average_hg2[i], 'b')
            ax.fill_between(np.arange(Y_mat.shape[1]), average_hg2[i] + ste_hg2[i], average_hg2[i] - ste_hg2[i], color='b', alpha=0.2)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if x_zero is not None:
                ax.plot([x_zero, x_zero], [min_value, max_value], color='k', alpha=0.4)
            ax.set_ylim(min_value, max_value)

            ax.text(0.2,0.8, str(i), transform=ax.transAxes)
    fig.tight_layout()
    return fig
