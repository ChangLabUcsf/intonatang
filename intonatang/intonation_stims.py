from __future__ import division, print_function, absolute_import

import os
pitch_info_path = os.path.join(os.path.dirname(__file__), 'data', 'pitch')

import numpy as np
import pandas as pd
import itertools


def get_pitch_and_intensity():
    """Returns dict of pitch and intensity values over time for each intonation token.

    The pitch and intensity values were extracted from each stimulus token using Praat.
    The sampling frequency matches the high-gamma sampling frequency of 100Hz.

    Pitch values are NaN when there are unvoiced segments.
    """
    sentence_numbers = ['sn1_', 'sn2_', 'sn3_', 'sn4_']
    sentence_types = ['st1_', 'st2_', 'st3_', 'st4_']
    speakers = ['sp1', 'sp2', 'sp3']
    stims_tuple = list(itertools.product(sentence_numbers, sentence_types, speakers))
    pitch_intensity_prosody = {}
    for stim_tuple in stims_tuple:
        path = os.path.join(pitch_info_path, "".join(stim_tuple) + ".wav.txt")
        pitch = pd.read_table(path, na_values=0.0, index_col=0)['pitch'].values
        intensity = pd.read_table(path, na_values=0.0, index_col=0)['intensity'].values
        pitch_intensity_prosody["".join(stim_tuple)] = {'pitch': pitch, 'intensity': intensity}
    return pitch_intensity_prosody

def get_continuous_pitch_and_intensity():
    """Returns pitches for each intonation/speaker condition and intensities for each sentence condition.

    Ordering of pitches is female Neutral, female Question, female Emphasis 1, female Emphasis 3, male Neutral, male Question, male Emphasis 1, and male Emphasis 3.

    Ordering of intensities is sentences 1-4. 
    """
    p = get_pitch_and_intensity()
    pitches_df = pd.read_hdf(os.path.join(pitch_info_path, 'pitch_contours'), 'pitch_contours')
    pitches = pitches_df.values.T

    stims = ['sn1_st1_sp2', 'sn2_st1_sp2', 'sn3_st1_sp2', 'sn4_st1_sp2']
    intensities = np.vstack([p[s]['intensity'] for s in stims])
    return pitches, intensities

stims1_1 = ['sn3_st3_sp2.wav', 'sn1_st2_sp1.wav', 'sn3_st2_sp1.wav', 'sn2_st2_sp1.wav', 'sn2_st4_sp3.wav', 'sn3_st2_sp3.wav', 'sn4_st2_sp1.wav', 'sn4_st3_sp2.wav', 'sn2_st3_sp2.wav', 'sn2_st4_sp2.wav', 'sn1_st4_sp2.wav', 'sn1_st1_sp1.wav', 'sn1_st3_sp2.wav', 'sn4_st2_sp2.wav', 'sn4_st4_sp1.wav', 'sn4_st4_sp3.wav', 'sn3_st3_sp1.wav', 'sn3_st4_sp1.wav', 'sn3_st4_sp2.wav', 'sn2_st1_sp3.wav', 'sn1_st3_sp3.wav', 'sn4_st2_sp3.wav', 'sn2_st4_sp1.wav', 'sn3_st3_sp3.wav', 'sn2_st3_sp3.wav', 'sn4_st4_sp2.wav', 'sn4_st1_sp1.wav', 'sn4_st1_sp2.wav', 'sn3_st4_sp3.wav', 'sn4_st3_sp1.wav', 'sn2_st2_sp2.wav', 'sn3_st1_sp3.wav', 'sn1_st4_sp1.wav', 'sn1_st4_sp3.wav', 'sn1_st2_sp3.wav', 'sn1_st3_sp1.wav', 'sn2_st1_sp2.wav', 'sn4_st3_sp3.wav', 'sn2_st3_sp1.wav', 'sn2_st2_sp3.wav', 'sn3_st1_sp1.wav', 'sn1_st2_sp2.wav', 'sn3_st1_sp2.wav', 'sn3_st2_sp2.wav', 'sn4_st1_sp3.wav', 'sn1_st1_sp2.wav', 'sn1_st1_sp3.wav', 'sn2_st1_sp1.wav']
stims1_2 = ['sn1_st2_sp3.wav', 'sn3_st3_sp3.wav', 'sn2_st3_sp1.wav', 'sn3_st1_sp1.wav', 'sn2_st3_sp2.wav', 'sn1_st3_sp2.wav', 'sn4_st1_sp2.wav', 'sn3_st1_sp2.wav', 'sn2_st4_sp1.wav', 'sn2_st1_sp3.wav', 'sn1_st1_sp3.wav', 'sn1_st1_sp2.wav', 'sn1_st4_sp2.wav', 'sn3_st1_sp3.wav', 'sn3_st2_sp3.wav', 'sn3_st4_sp2.wav', 'sn2_st2_sp2.wav', 'sn4_st4_sp1.wav', 'sn4_st4_sp2.wav', 'sn1_st4_sp3.wav', 'sn4_st3_sp3.wav', 'sn2_st2_sp3.wav', 'sn4_st1_sp3.wav', 'sn1_st4_sp1.wav', 'sn1_st3_sp1.wav', 'sn4_st4_sp3.wav', 'sn3_st3_sp2.wav', 'sn2_st3_sp3.wav', 'sn2_st2_sp1.wav', 'sn4_st2_sp1.wav', 'sn2_st4_sp3.wav', 'sn4_st3_sp2.wav', 'sn3_st4_sp1.wav', 'sn3_st2_sp1.wav', 'sn1_st3_sp3.wav', 'sn1_st1_sp1.wav', 'sn4_st1_sp1.wav', 'sn2_st1_sp2.wav', 'sn4_st2_sp3.wav', 'sn4_st2_sp2.wav', 'sn3_st2_sp2.wav', 'sn2_st1_sp1.wav', 'sn3_st3_sp1.wav', 'sn4_st3_sp1.wav', 'sn1_st2_sp1.wav', 'sn2_st4_sp2.wav', 'sn1_st2_sp2.wav', 'sn3_st4_sp3.wav']
stims2_1 = ['sn2_st4_sp3.wav', 'sn3_st2_sp3.wav', 'sn1_st2_sp1.wav', 'sn4_st4_sp3.wav', 'sn3_st3_sp1.wav', 'sn4_st2_sp3.wav', 'sn1_st3_sp1.wav', 'sn1_st4_sp1.wav', 'sn2_st2_sp3.wav', 'sn3_st2_sp2.wav', 'sn2_st3_sp1.wav', 'sn2_st3_sp2.wav', 'sn4_st2_sp2.wav', 'sn3_st4_sp1.wav', 'sn3_st3_sp3.wav', 'sn3_st3_sp2.wav', 'sn1_st3_sp3.wav', 'sn4_st1_sp3.wav', 'sn4_st4_sp2.wav', 'sn2_st1_sp2.wav', 'sn1_st4_sp2.wav', 'sn4_st2_sp1.wav', 'sn3_st1_sp1.wav', 'sn3_st1_sp2.wav', 'sn2_st2_sp2.wav', 'sn1_st2_sp3.wav', 'sn1_st3_sp2.wav', 'sn1_st1_sp3.wav', 'sn1_st1_sp1.wav', 'sn4_st3_sp2.wav', 'sn1_st2_sp2.wav', 'sn1_st1_sp2.wav', 'sn4_st1_sp1.wav', 'sn4_st1_sp2.wav', 'sn4_st4_sp1.wav', 'sn3_st1_sp3.wav', 'sn4_st3_sp3.wav', 'sn3_st2_sp1.wav', 'sn2_st1_sp3.wav', 'sn1_st4_sp3.wav', 'sn3_st4_sp2.wav', 'sn2_st4_sp2.wav', 'sn2_st3_sp3.wav', 'sn2_st4_sp1.wav', 'sn2_st1_sp1.wav', 'sn2_st2_sp1.wav', 'sn3_st4_sp3.wav', 'sn4_st3_sp1.wav']
stims2_2 = ['sn4_st4_sp1.wav', 'sn2_st1_sp2.wav', 'sn3_st3_sp1.wav', 'sn4_st3_sp2.wav', 'sn3_st3_sp2.wav', 'sn3_st1_sp2.wav', 'sn4_st1_sp3.wav', 'sn1_st3_sp1.wav', 'sn1_st2_sp1.wav', 'sn4_st4_sp3.wav', 'sn3_st4_sp3.wav', 'sn2_st4_sp2.wav', 'sn2_st3_sp1.wav', 'sn3_st2_sp2.wav', 'sn1_st1_sp2.wav', 'sn4_st2_sp2.wav', 'sn3_st2_sp3.wav', 'sn4_st3_sp3.wav', 'sn3_st3_sp3.wav', 'sn1_st2_sp3.wav', 'sn3_st4_sp2.wav', 'sn2_st1_sp3.wav', 'sn3_st4_sp1.wav', 'sn4_st2_sp1.wav', 'sn3_st1_sp1.wav', 'sn3_st2_sp1.wav', 'sn4_st2_sp3.wav', 'sn4_st3_sp1.wav', 'sn1_st1_sp3.wav', 'sn2_st2_sp2.wav', 'sn2_st2_sp3.wav', 'sn2_st4_sp1.wav', 'sn1_st4_sp2.wav', 'sn4_st1_sp1.wav', 'sn3_st1_sp3.wav', 'sn2_st3_sp2.wav', 'sn1_st1_sp1.wav', 'sn4_st1_sp2.wav', 'sn1_st3_sp2.wav', 'sn2_st4_sp3.wav', 'sn2_st3_sp3.wav', 'sn1_st3_sp3.wav', 'sn4_st4_sp2.wav', 'sn2_st1_sp1.wav', 'sn1_st2_sp2.wav', 'sn1_st4_sp1.wav', 'sn1_st4_sp3.wav', 'sn2_st2_sp1.wav']
stims3_1 = ['sn3_st3_sp2.wav', 'sn4_st2_sp3.wav', 'sn2_st3_sp1.wav', 'sn1_st1_sp1.wav', 'sn1_st3_sp3.wav', 'sn1_st4_sp3.wav', 'sn1_st2_sp3.wav', 'sn3_st2_sp2.wav', 'sn2_st3_sp2.wav', 'sn3_st4_sp2.wav', 'sn3_st1_sp2.wav', 'sn4_st2_sp2.wav', 'sn3_st4_sp1.wav', 'sn4_st4_sp1.wav', 'sn4_st1_sp1.wav', 'sn1_st2_sp1.wav', 'sn3_st3_sp3.wav', 'sn4_st3_sp2.wav', 'sn4_st3_sp3.wav', 'sn3_st4_sp3.wav', 'sn3_st2_sp3.wav', 'sn3_st3_sp1.wav', 'sn3_st1_sp1.wav', 'sn1_st1_sp3.wav', 'sn2_st1_sp1.wav', 'sn4_st1_sp2.wav', 'sn2_st2_sp2.wav', 'sn2_st4_sp3.wav', 'sn1_st1_sp2.wav', 'sn1_st3_sp2.wav', 'sn2_st2_sp3.wav', 'sn2_st1_sp3.wav', 'sn1_st3_sp1.wav', 'sn1_st4_sp2.wav', 'sn4_st4_sp3.wav', 'sn2_st4_sp1.wav', 'sn2_st2_sp1.wav', 'sn1_st4_sp1.wav', 'sn3_st2_sp1.wav', 'sn4_st2_sp1.wav', 'sn2_st3_sp3.wav', 'sn2_st4_sp2.wav', 'sn4_st3_sp1.wav', 'sn1_st2_sp2.wav', 'sn3_st1_sp3.wav', 'sn2_st1_sp2.wav', 'sn4_st1_sp3.wav', 'sn4_st4_sp2.wav']
stims3_2 = ['sn4_st4_sp2.wav', 'sn3_st3_sp1.wav', 'sn4_st3_sp3.wav', 'sn4_st3_sp1.wav', 'sn3_st4_sp2.wav', 'sn3_st4_sp3.wav', 'sn1_st3_sp3.wav', 'sn2_st4_sp2.wav', 'sn3_st3_sp3.wav', 'sn4_st2_sp2.wav', 'sn2_st1_sp3.wav', 'sn2_st3_sp3.wav', 'sn4_st1_sp3.wav', 'sn1_st3_sp1.wav', 'sn1_st2_sp1.wav', 'sn2_st1_sp1.wav', 'sn2_st1_sp2.wav', 'sn3_st4_sp1.wav', 'sn3_st1_sp2.wav', 'sn2_st3_sp1.wav', 'sn1_st1_sp2.wav', 'sn1_st4_sp1.wav', 'sn4_st2_sp3.wav', 'sn1_st4_sp2.wav', 'sn2_st4_sp1.wav', 'sn3_st2_sp1.wav', 'sn3_st1_sp1.wav', 'sn1_st2_sp3.wav', 'sn4_st1_sp2.wav', 'sn3_st2_sp2.wav', 'sn4_st2_sp1.wav', 'sn4_st4_sp1.wav', 'sn3_st1_sp3.wav', 'sn3_st2_sp3.wav', 'sn4_st1_sp1.wav', 'sn1_st1_sp3.wav', 'sn1_st2_sp2.wav', 'sn1_st1_sp1.wav', 'sn2_st2_sp2.wav', 'sn4_st4_sp3.wav', 'sn2_st4_sp3.wav', 'sn2_st3_sp2.wav', 'sn4_st3_sp2.wav', 'sn1_st3_sp2.wav', 'sn1_st4_sp3.wav', 'sn3_st3_sp2.wav', 'sn2_st2_sp3.wav', 'sn2_st2_sp1.wav']
stims4_1 = ['sn4_st2_sp3.wav', 'sn3_st4_sp1.wav', 'sn4_st1_sp1.wav', 'sn3_st2_sp2.wav', 'sn2_st1_sp1.wav', 'sn3_st3_sp1.wav', 'sn4_st2_sp2.wav', 'sn3_st2_sp3.wav', 'sn3_st1_sp1.wav', 'sn4_st3_sp1.wav', 'sn1_st3_sp2.wav', 'sn3_st1_sp2.wav', 'sn2_st2_sp2.wav', 'sn4_st3_sp3.wav', 'sn1_st4_sp2.wav', 'sn3_st3_sp2.wav', 'sn1_st1_sp1.wav', 'sn1_st3_sp3.wav', 'sn4_st4_sp2.wav', 'sn3_st4_sp3.wav', 'sn4_st1_sp2.wav', 'sn3_st2_sp1.wav', 'sn3_st3_sp3.wav', 'sn1_st4_sp1.wav', 'sn2_st3_sp1.wav', 'sn2_st2_sp3.wav', 'sn2_st3_sp3.wav', 'sn1_st3_sp1.wav', 'sn2_st1_sp3.wav', 'sn4_st2_sp1.wav', 'sn1_st1_sp2.wav', 'sn2_st4_sp1.wav', 'sn2_st4_sp3.wav', 'sn4_st4_sp1.wav', 'sn2_st1_sp2.wav', 'sn2_st4_sp2.wav', 'sn1_st2_sp1.wav', 'sn1_st2_sp2.wav', 'sn1_st1_sp3.wav', 'sn1_st2_sp3.wav', 'sn4_st1_sp3.wav', 'sn3_st1_sp3.wav', 'sn3_st4_sp2.wav', 'sn2_st3_sp2.wav', 'sn4_st4_sp3.wav', 'sn2_st2_sp1.wav', 'sn1_st4_sp3.wav', 'sn4_st3_sp2.wav']
stims4_2 = ['sn3_st4_sp2.wav', 'sn1_st2_sp1.wav', 'sn1_st1_sp1.wav', 'sn2_st1_sp2.wav', 'sn2_st3_sp2.wav', 'sn4_st2_sp2.wav', 'sn3_st2_sp1.wav', 'sn3_st3_sp3.wav', 'sn4_st3_sp2.wav', 'sn1_st3_sp2.wav', 'sn1_st4_sp1.wav', 'sn3_st3_sp1.wav', 'sn2_st3_sp3.wav', 'sn2_st3_sp1.wav', 'sn4_st2_sp3.wav', 'sn2_st4_sp3.wav', 'sn1_st1_sp2.wav', 'sn1_st4_sp2.wav', 'sn4_st3_sp3.wav', 'sn2_st4_sp1.wav', 'sn4_st1_sp2.wav', 'sn3_st1_sp1.wav', 'sn2_st1_sp3.wav', 'sn2_st1_sp1.wav', 'sn4_st2_sp1.wav', 'sn3_st3_sp2.wav', 'sn2_st2_sp3.wav', 'sn1_st3_sp1.wav', 'sn4_st4_sp2.wav', 'sn3_st2_sp3.wav', 'sn4_st4_sp3.wav', 'sn2_st2_sp1.wav', 'sn3_st1_sp3.wav', 'sn1_st3_sp3.wav', 'sn2_st4_sp2.wav', 'sn3_st4_sp3.wav', 'sn4_st1_sp1.wav', 'sn3_st1_sp2.wav', 'sn3_st4_sp1.wav', 'sn4_st3_sp1.wav', 'sn4_st1_sp3.wav', 'sn1_st4_sp3.wav', 'sn1_st1_sp3.wav', 'sn1_st2_sp2.wav', 'sn3_st2_sp2.wav', 'sn4_st4_sp1.wav', 'sn2_st2_sp2.wav', 'sn1_st2_sp3.wav']
stims5_1 = ['sn3_st4_sp2.wav', 'sn2_st2_sp2.wav', 'sn4_st3_sp1.wav', 'sn1_st1_sp1.wav', 'sn4_st2_sp3.wav', 'sn3_st3_sp3.wav', 'sn2_st3_sp1.wav', 'sn1_st2_sp1.wav', 'sn2_st1_sp2.wav', 'sn1_st4_sp3.wav', 'sn1_st3_sp3.wav', 'sn2_st4_sp3.wav', 'sn3_st2_sp2.wav', 'sn1_st2_sp2.wav', 'sn3_st2_sp1.wav', 'sn4_st1_sp3.wav', 'sn3_st1_sp2.wav', 'sn3_st1_sp1.wav', 'sn3_st4_sp1.wav', 'sn3_st3_sp1.wav', 'sn4_st2_sp2.wav', 'sn4_st4_sp1.wav', 'sn2_st2_sp3.wav', 'sn1_st4_sp2.wav', 'sn4_st4_sp2.wav', 'sn4_st1_sp2.wav', 'sn1_st2_sp3.wav', 'sn1_st3_sp2.wav', 'sn1_st4_sp1.wav', 'sn4_st2_sp1.wav', 'sn1_st3_sp1.wav', 'sn4_st3_sp2.wav', 'sn4_st3_sp3.wav', 'sn1_st1_sp2.wav', 'sn2_st1_sp1.wav', 'sn2_st1_sp3.wav', 'sn2_st4_sp1.wav', 'sn2_st2_sp1.wav', 'sn3_st1_sp3.wav', 'sn2_st3_sp2.wav', 'sn2_st4_sp2.wav', 'sn2_st3_sp3.wav', 'sn1_st1_sp3.wav', 'sn4_st1_sp1.wav', 'sn3_st2_sp3.wav', 'sn4_st4_sp3.wav', 'sn3_st4_sp3.wav', 'sn3_st3_sp2.wav']
stims5_2 = ['sn1_st1_sp2.wav', 'sn1_st1_sp1.wav', 'sn1_st3_sp2.wav', 'sn1_st4_sp2.wav', 'sn2_st3_sp2.wav', 'sn3_st2_sp2.wav', 'sn2_st1_sp1.wav', 'sn4_st4_sp1.wav', 'sn2_st2_sp3.wav', 'sn4_st3_sp2.wav', 'sn4_st4_sp3.wav', 'sn4_st4_sp2.wav', 'sn4_st1_sp2.wav', 'sn1_st1_sp3.wav', 'sn3_st4_sp3.wav', 'sn4_st3_sp3.wav', 'sn3_st2_sp1.wav', 'sn2_st4_sp1.wav', 'sn4_st2_sp2.wav', 'sn3_st3_sp2.wav', 'sn2_st4_sp2.wav', 'sn2_st1_sp3.wav', 'sn3_st4_sp1.wav', 'sn4_st1_sp1.wav', 'sn1_st3_sp3.wav', 'sn3_st2_sp3.wav', 'sn3_st4_sp2.wav', 'sn1_st2_sp2.wav', 'sn2_st3_sp3.wav', 'sn3_st3_sp1.wav', 'sn1_st3_sp1.wav', 'sn1_st4_sp1.wav', 'sn3_st1_sp3.wav', 'sn2_st2_sp1.wav', 'sn1_st2_sp1.wav', 'sn3_st1_sp1.wav', 'sn4_st3_sp1.wav', 'sn1_st2_sp3.wav', 'sn2_st1_sp2.wav', 'sn4_st2_sp3.wav', 'sn3_st3_sp3.wav', 'sn4_st2_sp1.wav', 'sn2_st2_sp2.wav', 'sn4_st1_sp3.wav', 'sn1_st4_sp3.wav', 'sn2_st4_sp3.wav', 'sn2_st3_sp1.wav', 'sn3_st1_sp2.wav']
stims1_1.extend(stims1_2)
stims1 = stims1_1
stims2_1.extend(stims2_2)
stims2 = stims2_1
stims3_1.extend(stims3_2)
stims3 = stims3_1
stims4_1.extend(stims4_2)
stims4 = stims4_1
stims5_1.extend(stims5_2)
stims5 = stims5_1

purr_stims1_1 = ['purr_female_st4_sn2.wav', 'purr_male_st3_sn2.wav', 'purr_female_st1_sn3.wav', 'purr_female_st4_sn4.wav', 'purr_female_st1_sn2.wav', 'purr_male_st1_sn3.wav', 'purr_female_st4_sn1.wav', 'purr_female_st4.wav', 'purr_female_st2_sn4.wav', 'purr_female_st1_sn1.wav', 'purr_female_st3_sn2.wav', 'purr_female_st3.wav', 'purr_female_st1_sn4.wav', 'purr_male_st2.wav', 'purr_female_st3_sn4.wav', 'purr_male_st2_sn1.wav', 'purr_male_st3.wav', 'purr_male_st4_sn2.wav', 'purr_male_st2_sn3.wav', 'purr_female_st1.wav', 'purr_female_st3_sn1.wav', 'purr_male_st1_sn1.wav', 'purr_female_st2_sn1.wav', 'purr_male_st2_sn2.wav', 'purr_male_st3_sn3.wav', 'purr_female_st2_sn3.wav', 'purr_male_st4_sn1.wav', 'purr_female_st2_sn2.wav', 'purr_female_st3_sn3.wav', 'purr_female_st4_sn3.wav', 'purr_female_st2.wav', 'purr_male_st1_sn2.wav', 'purr_male_st4_sn3.wav', 'purr_male_st2_sn4.wav', 'purr_male_st1_sn4.wav', 'purr_male_st4.wav', 'purr_male_st3_sn4.wav', 'purr_male_st1.wav', 'purr_male_st3_sn1.wav', 'purr_male_st4_sn4.wav']
purr_stims1_2 = ['purr_female_st3.wav', 'purr_female_st3_sn3.wav', 'purr_female_st4.wav', 'purr_female_st2_sn4.wav', 'purr_male_st2_sn3.wav', 'purr_male_st3_sn3.wav', 'purr_male_st1_sn1.wav', 'purr_female_st1_sn1.wav', 'purr_male_st2.wav', 'purr_male_st4.wav', 'purr_male_st2_sn4.wav', 'purr_male_st1_sn4.wav', 'purr_female_st1_sn4.wav', 'purr_female_st4_sn1.wav', 'purr_male_st3_sn2.wav', 'purr_female_st3_sn4.wav', 'purr_female_st2_sn2.wav', 'purr_male_st3_sn4.wav', 'purr_female_st2_sn1.wav', 'purr_male_st4_sn4.wav', 'purr_male_st4_sn2.wav', 'purr_male_st3_sn1.wav', 'purr_male_st1.wav', 'purr_female_st4_sn4.wav', 'purr_female_st3_sn2.wav', 'purr_male_st3.wav', 'purr_female_st1_sn3.wav', 'purr_female_st1_sn2.wav', 'purr_male_st1_sn2.wav', 'purr_female_st4_sn3.wav', 'purr_female_st1.wav', 'purr_female_st3_sn1.wav', 'purr_female_st2_sn3.wav', 'purr_male_st4_sn1.wav', 'purr_male_st2_sn2.wav', 'purr_male_st2_sn1.wav', 'purr_female_st4_sn2.wav', 'purr_male_st4_sn3.wav', 'purr_male_st1_sn3.wav', 'purr_female_st2.wav']
purr_stims1_3 = ['purr_female_st1_sn1.wav', 'purr_male_st2_sn3.wav', 'purr_male_st4_sn1.wav', 'purr_female_st4_sn3.wav', 'purr_female_st2_sn2.wav', 'purr_female_st4.wav', 'purr_male_st4_sn4.wav', 'purr_female_st1.wav', 'purr_male_st2_sn4.wav', 'purr_female_st4_sn4.wav', 'purr_male_st4_sn3.wav', 'purr_male_st4_sn2.wav', 'purr_male_st2_sn1.wav', 'purr_male_st4.wav', 'purr_male_st1_sn3.wav', 'purr_male_st1_sn4.wav', 'purr_female_st1_sn3.wav', 'purr_female_st4_sn2.wav', 'purr_female_st3_sn1.wav', 'purr_female_st2_sn3.wav', 'purr_male_st3_sn4.wav', 'purr_male_st1_sn2.wav', 'purr_male_st1.wav', 'purr_female_st3_sn4.wav', 'purr_male_st3_sn3.wav', 'purr_female_st1_sn4.wav', 'purr_male_st2.wav', 'purr_female_st2_sn1.wav', 'purr_male_st1_sn1.wav', 'purr_male_st3.wav', 'purr_female_st2_sn4.wav', 'purr_male_st3_sn2.wav', 'purr_male_st2_sn2.wav', 'purr_female_st4_sn1.wav', 'purr_female_st2.wav', 'purr_female_st1_sn2.wav', 'purr_male_st3_sn1.wav', 'purr_female_st3_sn3.wav', 'purr_female_st3_sn2.wav', 'purr_female_st3.wav']
purr_stims2_1 = ['purr_female_st2.wav', 'purr_male_st1.wav', 'purr_male_st2_sn2.wav', 'purr_female_st4_sn3.wav', 'purr_female_st2_sn1.wav', 'purr_female_st3_sn3.wav', 'purr_male_st1_sn4.wav', 'purr_female_st3_sn4.wav', 'purr_female_st2_sn4.wav', 'purr_female_st3_sn1.wav', 'purr_female_st2_sn2.wav', 'purr_male_st4_sn3.wav', 'purr_male_st3_sn2.wav', 'purr_female_st1_sn3.wav', 'purr_male_st2_sn1.wav', 'purr_male_st2.wav', 'purr_male_st3_sn3.wav', 'purr_male_st3.wav', 'purr_male_st2_sn4.wav', 'purr_female_st1_sn4.wav', 'purr_female_st3.wav', 'purr_male_st2_sn3.wav', 'purr_female_st4.wav', 'purr_male_st3_sn1.wav', 'purr_female_st1_sn1.wav', 'purr_male_st4_sn2.wav', 'purr_male_st3_sn4.wav', 'purr_female_st4_sn1.wav', 'purr_female_st4_sn2.wav', 'purr_female_st1_sn2.wav', 'purr_female_st2_sn3.wav', 'purr_female_st1.wav', 'purr_male_st1_sn1.wav', 'purr_male_st1_sn3.wav', 'purr_female_st3_sn2.wav', 'purr_female_st4_sn4.wav', 'purr_male_st4.wav', 'purr_male_st4_sn4.wav', 'purr_male_st1_sn2.wav', 'purr_male_st4_sn1.wav']
purr_stims2_2 = ['purr_female_st2.wav', 'purr_male_st2_sn2.wav', 'purr_male_st2_sn1.wav', 'purr_male_st4_sn2.wav', 'purr_male_st4.wav', 'purr_male_st4_sn3.wav', 'purr_male_st1_sn1.wav', 'purr_male_st2_sn4.wav', 'purr_female_st3_sn1.wav', 'purr_female_st2_sn1.wav', 'purr_female_st3_sn2.wav', 'purr_female_st1_sn3.wav', 'purr_female_st3.wav', 'purr_male_st3.wav', 'purr_male_st2_sn3.wav', 'purr_male_st1_sn2.wav', 'purr_male_st1_sn3.wav', 'purr_female_st1_sn4.wav', 'purr_female_st2_sn2.wav', 'purr_male_st4_sn4.wav', 'purr_male_st3_sn2.wav', 'purr_female_st1.wav', 'purr_male_st1.wav', 'purr_male_st4_sn1.wav', 'purr_female_st3_sn3.wav', 'purr_male_st2.wav', 'purr_male_st3_sn3.wav', 'purr_female_st1_sn1.wav', 'purr_female_st3_sn4.wav', 'purr_female_st4_sn3.wav', 'purr_female_st4_sn4.wav', 'purr_female_st4.wav', 'purr_female_st2_sn3.wav', 'purr_female_st4_sn1.wav', 'purr_female_st2_sn4.wav', 'purr_male_st3_sn4.wav', 'purr_female_st1_sn2.wav', 'purr_male_st3_sn1.wav', 'purr_male_st1_sn4.wav', 'purr_female_st4_sn2.wav']
purr_stims2_3 = ['purr_male_st1_sn3.wav', 'purr_male_st3_sn4.wav', 'purr_male_st1_sn2.wav', 'purr_female_st4_sn4.wav', 'purr_female_st2_sn1.wav', 'purr_female_st4_sn2.wav', 'purr_female_st3_sn2.wav', 'purr_male_st3_sn2.wav', 'purr_female_st2_sn4.wav', 'purr_female_st3_sn4.wav', 'purr_male_st4_sn3.wav', 'purr_male_st4_sn1.wav', 'purr_female_st3_sn1.wav', 'purr_male_st2_sn1.wav', 'purr_female_st1_sn3.wav', 'purr_male_st1_sn4.wav', 'purr_female_st4_sn3.wav', 'purr_female_st1_sn1.wav', 'purr_male_st3_sn1.wav', 'purr_male_st2_sn4.wav', 'purr_female_st3_sn3.wav', 'purr_male_st3.wav', 'purr_female_st3.wav', 'purr_male_st3_sn3.wav', 'purr_female_st1_sn2.wav', 'purr_female_st1_sn4.wav', 'purr_male_st2.wav', 'purr_male_st4_sn4.wav', 'purr_female_st2_sn3.wav', 'purr_male_st1_sn1.wav', 'purr_female_st1.wav', 'purr_male_st4.wav', 'purr_female_st2.wav', 'purr_female_st4.wav', 'purr_female_st4_sn1.wav', 'purr_male_st1.wav', 'purr_male_st2_sn3.wav', 'purr_male_st4_sn2.wav', 'purr_male_st2_sn2.wav', 'purr_female_st2_sn2.wav']
purr_stims3_1 = ['purr_female_st1_sn4.wav', 'purr_male_st3_sn2.wav', 'purr_female_st4_sn1.wav', 'purr_female_st2_sn1.wav', 'purr_female_st3_sn4.wav', 'purr_male_st3.wav', 'purr_male_st4_sn2.wav', 'purr_female_st4_sn3.wav', 'purr_male_st2_sn2.wav', 'purr_female_st3_sn1.wav', 'purr_female_st2_sn3.wav', 'purr_female_st2_sn2.wav', 'purr_male_st1.wav', 'purr_male_st2_sn4.wav', 'purr_male_st1_sn2.wav', 'purr_female_st1_sn2.wav', 'purr_female_st3_sn3.wav', 'purr_male_st2_sn1.wav', 'purr_female_st3.wav', 'purr_female_st4_sn2.wav', 'purr_female_st3_sn2.wav', 'purr_male_st3_sn4.wav', 'purr_female_st1_sn3.wav', 'purr_male_st4.wav', 'purr_male_st2.wav', 'purr_female_st4.wav', 'purr_male_st3_sn3.wav', 'purr_male_st2_sn3.wav', 'purr_female_st4_sn4.wav', 'purr_female_st2_sn4.wav', 'purr_male_st4_sn1.wav', 'purr_female_st1_sn1.wav', 'purr_male_st4_sn3.wav', 'purr_male_st1_sn1.wav', 'purr_male_st1_sn4.wav', 'purr_male_st1_sn3.wav', 'purr_female_st1.wav', 'purr_male_st4_sn4.wav', 'purr_female_st2.wav', 'purr_male_st3_sn1.wav']
purr_stims3_2 = ['purr_female_st2_sn4.wav', 'purr_female_st2_sn1.wav', 'purr_female_st1_sn2.wav', 'purr_male_st2.wav', 'purr_male_st2_sn3.wav', 'purr_female_st4_sn1.wav', 'purr_female_st1_sn4.wav', 'purr_male_st1_sn4.wav', 'purr_female_st3_sn4.wav', 'purr_male_st4_sn3.wav', 'purr_male_st3_sn1.wav', 'purr_female_st2_sn2.wav', 'purr_female_st3_sn2.wav', 'purr_male_st3_sn2.wav', 'purr_female_st4.wav', 'purr_female_st1_sn1.wav', 'purr_female_st1.wav', 'purr_female_st4_sn3.wav', 'purr_male_st2_sn4.wav', 'purr_male_st4.wav', 'purr_male_st1.wav', 'purr_male_st3_sn4.wav', 'purr_male_st2_sn2.wav', 'purr_male_st1_sn1.wav', 'purr_female_st2.wav', 'purr_male_st3_sn3.wav', 'purr_male_st4_sn2.wav', 'purr_male_st1_sn2.wav', 'purr_female_st3.wav', 'purr_female_st4_sn4.wav', 'purr_female_st3_sn1.wav', 'purr_male_st4_sn1.wav', 'purr_male_st4_sn4.wav', 'purr_female_st3_sn3.wav', 'purr_male_st2_sn1.wav', 'purr_female_st4_sn2.wav', 'purr_male_st1_sn3.wav', 'purr_female_st1_sn3.wav', 'purr_female_st2_sn3.wav', 'purr_male_st3.wav']
purr_stims3_3 = ['purr_male_st1_sn3.wav', 'purr_male_st1_sn4.wav', 'purr_female_st2_sn2.wav', 'purr_male_st3_sn4.wav', 'purr_female_st3.wav', 'purr_female_st2_sn1.wav', 'purr_female_st3_sn1.wav', 'purr_male_st3_sn2.wav', 'purr_female_st1_sn4.wav', 'purr_male_st2_sn4.wav', 'purr_female_st1_sn1.wav', 'purr_male_st1.wav', 'purr_female_st4.wav', 'purr_female_st2_sn3.wav', 'purr_male_st4_sn4.wav', 'purr_female_st1_sn2.wav', 'purr_female_st4_sn1.wav', 'purr_female_st4_sn3.wav', 'purr_male_st1_sn2.wav', 'purr_male_st4_sn1.wav', 'purr_male_st2_sn1.wav', 'purr_male_st2_sn3.wav', 'purr_male_st3.wav', 'purr_female_st4_sn4.wav', 'purr_female_st3_sn4.wav', 'purr_female_st1.wav', 'purr_female_st2.wav', 'purr_male_st4_sn2.wav', 'purr_male_st2_sn2.wav', 'purr_male_st3_sn3.wav', 'purr_female_st1_sn3.wav', 'purr_male_st4.wav', 'purr_male_st2.wav', 'purr_female_st2_sn4.wav', 'purr_male_st3_sn1.wav', 'purr_male_st1_sn1.wav', 'purr_female_st3_sn2.wav', 'purr_male_st4_sn3.wav', 'purr_female_st4_sn2.wav', 'purr_female_st3_sn3.wav']
purr_stims4_1 = ['purr_female_st4_sn2.wav', 'purr_male_st4_sn3.wav', 'purr_female_st1.wav', 'purr_male_st3_sn1.wav', 'purr_male_st3_sn2.wav', 'purr_female_st3.wav', 'purr_male_st2_sn2.wav', 'purr_male_st4_sn1.wav', 'purr_male_st2_sn3.wav', 'purr_male_st3.wav', 'purr_female_st1_sn1.wav', 'purr_male_st3_sn4.wav', 'purr_female_st4_sn4.wav', 'purr_female_st2_sn4.wav', 'purr_male_st3_sn3.wav', 'purr_female_st4_sn3.wav', 'purr_female_st2.wav', 'purr_male_st1_sn2.wav', 'purr_female_st1_sn2.wav', 'purr_female_st4.wav', 'purr_female_st3_sn1.wav', 'purr_female_st3_sn4.wav', 'purr_male_st2.wav', 'purr_male_st1.wav', 'purr_male_st1_sn3.wav', 'purr_female_st2_sn3.wav', 'purr_male_st2_sn1.wav', 'purr_female_st3_sn3.wav', 'purr_female_st1_sn3.wav', 'purr_female_st1_sn4.wav', 'purr_female_st2_sn1.wav', 'purr_female_st4_sn1.wav', 'purr_male_st4_sn2.wav', 'purr_male_st1_sn4.wav', 'purr_male_st4.wav', 'purr_male_st4_sn4.wav', 'purr_male_st1_sn1.wav', 'purr_female_st3_sn2.wav', 'purr_male_st2_sn4.wav', 'purr_female_st2_sn2.wav']
purr_stims4_2 = ['purr_male_st3_sn2.wav', 'purr_female_st1_sn1.wav', 'purr_female_st4_sn3.wav', 'purr_female_st4_sn4.wav', 'purr_male_st3_sn1.wav', 'purr_male_st1_sn4.wav', 'purr_female_st2_sn4.wav', 'purr_male_st1.wav', 'purr_male_st4_sn4.wav', 'purr_female_st3_sn2.wav', 'purr_female_st4.wav', 'purr_female_st2_sn3.wav', 'purr_male_st1_sn1.wav', 'purr_male_st3.wav', 'purr_female_st2_sn1.wav', 'purr_female_st4_sn2.wav', 'purr_male_st4_sn2.wav', 'purr_female_st1_sn4.wav', 'purr_male_st1_sn2.wav', 'purr_female_st4_sn1.wav', 'purr_female_st1.wav', 'purr_male_st2_sn2.wav', 'purr_female_st3_sn1.wav', 'purr_male_st2_sn3.wav', 'purr_female_st1_sn2.wav', 'purr_male_st4.wav', 'purr_female_st3.wav', 'purr_male_st4_sn1.wav', 'purr_female_st3_sn4.wav', 'purr_female_st2_sn2.wav', 'purr_male_st3_sn4.wav', 'purr_male_st1_sn3.wav', 'purr_female_st2.wav', 'purr_female_st3_sn3.wav', 'purr_female_st1_sn3.wav', 'purr_male_st2_sn4.wav', 'purr_male_st2_sn1.wav', 'purr_male_st4_sn3.wav', 'purr_male_st3_sn3.wav', 'purr_male_st2.wav']
purr_stims4_3 = ['purr_male_st1.wav', 'purr_female_st1_sn2.wav', 'purr_male_st4_sn2.wav', 'purr_male_st3.wav', 'purr_female_st1_sn4.wav', 'purr_female_st2_sn3.wav', 'purr_female_st4_sn2.wav', 'purr_male_st1_sn3.wav', 'purr_female_st2_sn2.wav', 'purr_female_st1_sn3.wav', 'purr_male_st4.wav', 'purr_female_st3_sn1.wav', 'purr_female_st3_sn3.wav', 'purr_male_st2_sn1.wav', 'purr_male_st3_sn2.wav', 'purr_female_st4_sn3.wav', 'purr_female_st4_sn1.wav', 'purr_male_st1_sn2.wav', 'purr_male_st4_sn4.wav', 'purr_female_st4_sn4.wav', 'purr_female_st2_sn4.wav', 'purr_male_st2_sn4.wav', 'purr_female_st3.wav', 'purr_female_st3_sn2.wav', 'purr_male_st1_sn4.wav', 'purr_female_st1_sn1.wav', 'purr_male_st3_sn1.wav', 'purr_female_st1.wav', 'purr_female_st4.wav', 'purr_male_st1_sn1.wav', 'purr_male_st4_sn1.wav', 'purr_female_st2_sn1.wav', 'purr_female_st2.wav', 'purr_male_st4_sn3.wav', 'purr_male_st3_sn3.wav', 'purr_female_st3_sn4.wav', 'purr_male_st3_sn4.wav', 'purr_male_st2_sn2.wav', 'purr_male_st2.wav', 'purr_male_st2_sn3.wav']
purr_stims1_1.extend(purr_stims1_2)
purr_stims1_1.extend(purr_stims1_3)
purr_stims1 = purr_stims1_1
purr_stims2_1.extend(purr_stims2_2)
purr_stims2_1.extend(purr_stims2_3)
purr_stims2 = purr_stims2_1
purr_stims3_1.extend(purr_stims3_2)
purr_stims3_1.extend(purr_stims3_3)
purr_stims3 = purr_stims3_1
purr_stims4_1.extend(purr_stims4_2)
purr_stims4_1.extend(purr_stims4_3)
purr_stims4 = purr_stims4_1

missing_f0_stims1_1 = ['male_st3.wav', 'female_st2.wav', 'female_st3.wav', 'female_st3.wav', 'female_st4.wav', 'female_st1.wav', 'male_st2.wav', 'female_st3.wav', 'female_st4.wav', 'male_st1.wav', 'female_st1.wav', 'male_st4.wav', 'male_st1.wav', 'male_st3.wav', 'female_st1.wav', 'male_st2.wav', 'male_st4.wav', 'male_st2.wav', 'male_st3.wav', 'female_st2.wav', 'male_st4.wav', 'female_st2.wav', 'male_st1.wav', 'female_st4.wav']
missing_f0_stims1_2 = ['male_st3.wav', 'female_st4.wav', 'female_st3.wav', 'male_st4.wav', 'male_st4.wav', 'male_st2.wav', 'female_st3.wav', 'male_st3.wav', 'female_st1.wav', 'female_st1.wav', 'male_st1.wav', 'male_st1.wav', 'male_st2.wav', 'male_st1.wav', 'female_st2.wav', 'female_st4.wav', 'male_st4.wav', 'female_st2.wav', 'female_st1.wav', 'female_st2.wav', 'male_st3.wav', 'female_st4.wav', 'female_st3.wav', 'male_st2.wav']
missing_f0_stims1_3 = ['male_st2.wav', 'female_st1.wav', 'male_st1.wav', 'male_st2.wav', 'male_st2.wav', 'female_st4.wav', 'female_st1.wav', 'male_st4.wav', 'male_st1.wav', 'female_st3.wav', 'female_st2.wav', 'male_st1.wav', 'female_st2.wav', 'female_st3.wav', 'female_st3.wav', 'male_st3.wav', 'female_st4.wav', 'male_st4.wav', 'male_st3.wav', 'female_st1.wav', 'male_st4.wav', 'female_st4.wav', 'male_st3.wav', 'female_st2.wav']
missing_f0_stims1_4 = ['female_st1.wav', 'female_st4.wav', 'male_st4.wav', 'male_st2.wav', 'male_st3.wav', 'female_st4.wav', 'female_st1.wav', 'male_st3.wav', 'male_st4.wav', 'female_st2.wav', 'female_st4.wav', 'male_st3.wav', 'male_st1.wav', 'male_st1.wav', 'male_st2.wav', 'female_st3.wav', 'female_st2.wav', 'female_st3.wav', 'male_st2.wav', 'male_st1.wav', 'female_st2.wav', 'female_st3.wav', 'female_st1.wav', 'male_st4.wav']
missing_f0_stims2_1 = ['male_st3.wav', 'female_st2.wav', 'female_st4.wav', 'female_st1.wav', 'male_st4.wav', 'male_st2.wav', 'male_st1.wav', 'male_st2.wav', 'male_st4.wav', 'female_st2.wav', 'female_st1.wav', 'male_st2.wav', 'female_st1.wav', 'male_st3.wav', 'female_st4.wav', 'male_st1.wav', 'male_st4.wav', 'female_st3.wav', 'male_st1.wav', 'female_st4.wav', 'female_st2.wav', 'female_st3.wav', 'male_st3.wav', 'female_st3.wav']
missing_f0_stims2_2 = ['female_st3.wav', 'female_st2.wav', 'male_st4.wav', 'male_st1.wav', 'male_st3.wav', 'male_st1.wav', 'female_st1.wav', 'male_st2.wav', 'female_st2.wav', 'female_st4.wav', 'female_st3.wav', 'female_st4.wav', 'male_st3.wav', 'male_st4.wav', 'female_st1.wav', 'male_st2.wav', 'female_st2.wav', 'female_st1.wav', 'male_st4.wav', 'male_st1.wav', 'male_st3.wav', 'male_st2.wav', 'female_st3.wav', 'female_st4.wav']
missing_f0_stims2_3 = ['female_st2.wav', 'female_st2.wav', 'female_st4.wav', 'male_st2.wav', 'male_st3.wav', 'female_st4.wav', 'male_st4.wav', 'male_st4.wav', 'male_st1.wav', 'female_st2.wav', 'male_st4.wav', 'female_st3.wav', 'male_st1.wav', 'female_st3.wav', 'female_st1.wav', 'female_st1.wav', 'male_st3.wav', 'male_st2.wav', 'female_st4.wav', 'male_st1.wav', 'female_st1.wav', 'female_st3.wav', 'male_st3.wav', 'male_st2.wav']
missing_f0_stims2_4 = ['female_st4.wav', 'male_st4.wav', 'male_st2.wav', 'male_st3.wav', 'female_st4.wav', 'female_st3.wav', 'female_st4.wav', 'male_st1.wav', 'male_st3.wav', 'male_st3.wav', 'female_st1.wav', 'male_st2.wav', 'male_st4.wav', 'female_st2.wav', 'male_st1.wav', 'female_st2.wav', 'female_st1.wav', 'female_st1.wav', 'male_st4.wav', 'male_st1.wav', 'male_st2.wav', 'female_st3.wav', 'female_st2.wav', 'female_st3.wav']
missing_f0_stims3_1 = ['female_st4.wav', 'female_st1.wav', 'female_st2.wav', 'female_st3.wav', 'male_st2.wav', 'male_st3.wav', 'male_st4.wav', 'male_st3.wav', 'female_st2.wav', 'male_st4.wav', 'male_st2.wav', 'female_st1.wav', 'male_st1.wav', 'female_st1.wav', 'male_st4.wav', 'male_st1.wav', 'female_st3.wav', 'male_st1.wav', 'female_st3.wav', 'male_st2.wav', 'female_st4.wav', 'female_st2.wav', 'male_st3.wav', 'female_st4.wav']
missing_f0_stims3_2 = ['male_st1.wav', 'male_st3.wav', 'male_st4.wav', 'female_st4.wav', 'female_st3.wav', 'male_st2.wav', 'male_st2.wav', 'male_st1.wav', 'female_st2.wav', 'male_st3.wav', 'male_st2.wav', 'female_st1.wav', 'female_st3.wav', 'female_st3.wav', 'female_st4.wav', 'female_st4.wav', 'female_st1.wav', 'male_st1.wav', 'male_st4.wav', 'female_st1.wav', 'female_st2.wav', 'male_st4.wav', 'male_st3.wav', 'female_st2.wav']
missing_f0_stims3_3 = ['female_st1.wav', 'female_st4.wav', 'female_st1.wav', 'male_st3.wav', 'male_st3.wav', 'female_st4.wav', 'male_st2.wav', 'male_st1.wav', 'female_st2.wav', 'female_st4.wav', 'male_st2.wav', 'male_st1.wav', 'male_st4.wav', 'male_st1.wav', 'male_st2.wav', 'female_st2.wav', 'male_st4.wav', 'female_st3.wav', 'female_st1.wav', 'female_st3.wav', 'female_st3.wav', 'male_st4.wav', 'male_st3.wav', 'female_st2.wav']
missing_f0_stims3_4 = ['female_st2.wav', 'female_st1.wav', 'male_st1.wav', 'male_st4.wav', 'male_st3.wav', 'female_st3.wav', 'female_st4.wav', 'female_st4.wav', 'male_st3.wav', 'male_st4.wav', 'female_st2.wav', 'male_st4.wav', 'female_st4.wav', 'male_st1.wav', 'male_st3.wav', 'female_st1.wav', 'female_st2.wav', 'female_st3.wav', 'male_st2.wav', 'male_st2.wav', 'male_st1.wav', 'female_st3.wav', 'female_st1.wav', 'male_st2.wav']
missing_f0_stims1_1.extend(missing_f0_stims1_2)
missing_f0_stims1_1.extend(missing_f0_stims1_3)
missing_f0_stims1_1.extend(missing_f0_stims1_4)
missing_f0_stims1_1.extend(missing_f0_stims2_1)
missing_f0_stims1_1.extend(missing_f0_stims2_2)
missing_f0_stims2_3.extend(missing_f0_stims2_4)
missing_f0_stims2_3.extend(missing_f0_stims3_1)
missing_f0_stims2_3.extend(missing_f0_stims3_2)
missing_f0_stims2_3.extend(missing_f0_stims3_3)
missing_f0_stims2_3.extend(missing_f0_stims3_4)
missing_f0_stims_1and2 = missing_f0_stims1_1
missing_f0_stims_3and4 = missing_f0_stims2_3
missing_f0_stims1 = ['nostretch_' + s for s in missing_f0_stims_1and2]
missing_f0_stims2 = ['stretch_' + s for s in missing_f0_stims_1and2]
missing_f0_stims3 = ['nostretch_' + s for s in missing_f0_stims_3and4]
missing_f0_stims4 = ['stretch_' + s for s in missing_f0_stims_3and4]

def get_sn_st_sp_from_stims(stims):
    """Returns ndarrays of sentence, intonation, and speaker conditions for a given list of stims.
    """
    sn = []
    st = []
    sp = []
    for s in stims:
        sn.append(int(s[2]))
        st.append(int(s[6]))
        sp.append(int(s[10]))
    return np.array(sn), np.array(st), np.array(sp)

def get_sn_st_sp_from_purr_stims(stims):
    """Returns ndarrays of sentence, intonation, and speaker conditions for a given list of control (purr) stims.

    For this set of non-speech controls, sentence conditions refer to different amplitude contours, taken to match
    the amplitude contours of the sentence conditions in the speech stimuli.
    """
    sn = []
    st = []
    sp = []
    for s in stims:
        s = s + "_555"
        sn.append(int(s.split('_')[3][2]))
        st.append(int(s.split('_')[2][2]))
        sp.append(1 if s.split('_')[1] == "male" else 2)
    return np.array(sn), np.array(st), np.array(sp)

def get_sn_st_sp_from_missing_f0_stims(stims):
    """Returns ndarrays of intonation and speaker conditions for a given list of missing f0 stims.

    For this set of non-speech controls, sentence conditions do not relate to sentence conditions for other sets of stimuli.
    The sentence condition here refers to the type of missing fundamental control. 

    sn: composition of stimulus (h: harmonic, f0: fundamental frequency, noise: pink noise masker, stretch: factor the pitch range was changed by)
    0: 4h + 5h + 6h, stretch = 1
    1: f0 + 2h + 3h, stretch = 1
    2: 4h + 5h + 6h + noise, stretch = 1
    3: 4h + 5h + 6h + noise, stretch = 0.5
    4: 4h + 5h + 6h + noise, stretch = 2
    """
    sn = []
    st = []
    sp = []
    for i, s in enumerate(stims):
        if s.split('_')[0] == "stretch":
            if i < 48:
                sn.append(3)
            elif i < 96:
                sn.append(2)
            else:
                sn.append(4)
        elif s.split('_')[0] == "nostretch":
            if i < 48:
                sn.append(1)
            elif i < 96:
                sn.append(0)
            else:
                sn.append(2)
        st.append(int(s.split('_')[2][2]))
        sp.append(1 if s.split('_')[1] == "male" else 2)
    return np.array(sn), np.array(st), np.array(sp)

__all__ = ['stims1', 'stims2', 'stims3', 'stims4', 'stims5',
    'purr_stims1', 'purr_stims2', 'purr_stims3', 'purr_stims4',
    'missing_f0_stims1', 'missing_f0_stims2', 'missing_f0_stims3', 'missing_f0_stims4',
    'get_sn_st_sp_from_purr_stims', 'get_sn_st_sp_from_stims', 'get_sn_st_sp_from_missing_f0_stims']
