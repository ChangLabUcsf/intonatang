Pre-analysis
============

Pre-analysis is the creation of neural activity ndarrays 
(``Y_mat``: with dimensions ``n_chans`` x ``n_timepoints`` x ``n_trials``)) and lists of stimulus conditions.

After finding stimulus onset times and putting subject metadata into ``intonation_subject_data.py``, the pre-analysis
pipeline can be run with ``save_Y_mat_sns_sts_sps_for_subject_number``.

1. For each block, find stimulus onsets using cross-correlation between stimulus waveform and recorded speaker/microphone channel (This uses Liberty's event dectection code). These are then saved to the .mat file for that block in the variable ``times``.
2. For each subject, get metadata from ``intonation_subject_data.py`` about what the block numbers were and which stimulus set was played during that block.
3. For each subject, load .mat data (``hg``, ``times``, ``bcs``, ``badTimeSegments``) for each of their blocks along with stimulus set information.
4. To get ``Y_mat`` for each subject:
    * neural activity time-locked to stimulus onset is extracted from the hg time series.
    * a moving average is used to reduce the number of samples in time and reduce timepoint by timepoint variability.
    * trials that overlap with bad time segments (marked from preprocessing) are excluded.

At the end of pre-analysis, a Y_mat.mat file is created that contains five variables

    1. ``Y_mat`` (*ndarray*): ndarray with neural activity that is time-averaged in a moving window. Default settings use a window size of 60ms moving in 30ms steps that 
        starts with a center of -150ms before stimulus onset (window from -180ms to -130ms) and ends with a center of +2850ms after stimulus onset (+650ms after stimulus offset).
    2. ``sns`` (*list*): list of sentence conditions which are integers 1, 2, 3, or 4. (or 1, 2, 3, 4, 5 for non-speech control)
    3. ``sts`` (*list*): list of intonation conditions (sts is short for "sentence types") which are integers 1, 2, 3, or 4 for both speech and non-speech controls.
    4. ``sps`` (*list*): list of speaker conditions which are integers 1, 2, 3 for speech and 1, 2 for non-speech control.
    5. ``Y_mat_plotter`` (*ndarray*): neural activity ndarray used by the Plotter. With default settings, this contains the high-gamma
        from 250ms before stimulus onset to 550ms after stimulus offset for a total window duration of 3s (stimulus duration is 2.2s).
        dimensions are ``n_chans`` x ``n_timepoints`` x ``n_trials``


Two toggleable options for generating the Y_mat.mat files using ``save_Y_mat_sns_sts_sps_for_subject_number`` are the two parameters:

    1. ``control_stim`` (*bool*): whether to generate the Y_mat.mat file for this subject's non-speech control blocks.
    2. ``zscore_to_silence`` (*bool*): whether to z-score neural activity to a silent baseline (otherwise z-score to the entire block). 
        The silent baseline consists of silent periods within the intertrial interval that exclude the first 500ms after stimulus offset.

.. automodule:: intonatang.intonation_preanalysis
   :members:
