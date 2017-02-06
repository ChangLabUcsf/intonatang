from __future__ import print_function, division, absolute_import

import os
data_path = os.path.join(os.path.dirname(__file__), 'data')

import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import itertools
import pandas as pd
from scipy.signal import sweep_poly
from scipy import signal

from .intonation_stims import get_continuous_pitch_and_intensity

def save_non_linguistic_control_stimuli(missing_f0=False, add_noise=None, stretch_factor=0):
    pitches, _ = get_continuous_pitch_and_intensity()
    stim_prefixes = ['female_st1', 'female_st2', 'female_st3', 'female_st4', 'male_st1', 'male_st2', 'male_st3', 'male_st4']
    polydeg = 16
    fs = 44100
    t = np.arange(np.int(1.1*fs))/fs
    t_extra = np.arange(np.int(1.1*fs))/fs
    amplitude_ratio = [5000, 5000, 5000] #original [24000, 6000, 1500]
    if add_noise is not None:
        amplitude_ratio = np.array(amplitude_ratio)/1

    info = "" if not missing_f0 else "_missing_f0"
    if add_noise is not None:
        info = info + "_noise"
    if add_noise == 'first':
        info = info + "_first"


    pitches = np.array([stretch(p, stretch_factor) for p in pitches])
    info = info + "_stretch_" + str(stretch_factor)

    for i in range(8):
        pitch1 = pitches[i, 0:110]
        pitch2 = pitches[i, 110:]

        ys_all = []
        for harmonic in np.arange(0,6):
            z1 = np.polyfit(np.arange(110)/100, (harmonic+1)*pitch1, polydeg)
            z2 = np.polyfit(np.arange(110)/100, (harmonic+1)*pitch2, polydeg)

            phase = _sweep_poly_phase(t_extra, z1)
            y = np.concatenate([sweep_poly(t, z1), sweep_poly(t, z2, np.rad2deg(phase[-1]))])
            ys_all.append(y)

        ys = [ys_all[0], ys_all[1], ys_all[2]] if not missing_f0 else [ys_all[3], ys_all[4], ys_all[5]]

        if i > 3:
            if add_noise is not None:
                amplitude_ratio = [6500, 6500, 6500]
            else:
                amplitude_ratio = [6500, 6500, 6500]
        s= ys[0]*amplitude_ratio[0] + ys[1]*amplitude_ratio[1] + ys[2]*amplitude_ratio[2]
        if add_noise == 'first':
            s = add_sec_to_beg(s)
            s = s + (2000*noise_longer_filtered[:len(s)]).astype(np.int16)
        elif add_noise == 'together':
            s = s + (2000*noise_filtered).astype(np.int16)
        s= np.asarray(apply_cos_squared_ramp(s), dtype=np.int16)

        wavfile.write('missing_f0/purr' + info + "_" + stim_prefixes[i] + '.wav', fs, s)

noises = sio.loadmat(os.path.join(data_path, 'pink_noise.mat'))
noise = noises['x']
noise_longer = noises['x2']

b, a = signal.butter(5, [0.001, 0.24], 'bandpass')
noise_filtered = signal.lfilter(b, a, noise[:,0])
noise_longer_filtered = signal.lfilter(b, a, noise_longer[:,0])

def add_sec_to_beg(sound, fs=44100, seconds=0.25):
    original_len = len(sound)
    new_sound = np.zeros((int(np.round(original_len + seconds*fs))), dtype=np.int16)
    new_sound[int(np.round(seconds*fs)):] = sound
    return new_sound

def stretch(a, factor=1):
    min_value = np.nanmin(a)
    max_value = np.nanmax(a)
    b = [item + factor * (item-min_value) for item in a]
    return np.array(b)

def apply_cos_squared_ramp(signal, fs=44100, ms=5):
    time_steps = np.int(44100 * (ms/1000))
    time = np.arange(time_steps)/(time_steps/(np.pi/2))
    ramp_off = np.cos(time)**2
    ramp_on = np.copy(ramp_off[::-1])
    signal_ramped = np.concatenate([signal[:time_steps]*ramp_on, signal[time_steps:-1*time_steps], signal[-1*time_steps:]*ramp_off])
    return signal_ramped

# The following code is from scipy.signal (v0.18.1)
from numpy import polyint, polyval, pi
def _sweep_poly_phase(t, poly):
    """From scipy.signal (v.0.18.1)

    Calculate the phase used by sweep_poly to generate its output.
    See `sweep_poly` for a description of the arguments.
    """
    # polyint handles lists, ndarrays and instances of poly1d automatically.
    intpoly = polyint(poly)
    phase = 2 * pi * polyval(intpoly, t)
    return phase
