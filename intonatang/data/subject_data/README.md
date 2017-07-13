# Neural data

This folder contains all the neural data for each subject. 

Within each subject's folder, there is neural data from both intonation (speech and nonspeech) and TIMIT tasks. 

The intonation data is saved by block and consists of a .mat file for each block. Each data file is named ECXXX_BXX.mat indicating the subject, ECXXX, and block number, BXX. The data files are organized into subdirectories, first by subject and then by block (e.g. subject_data/EC113/EC113_B13/EC113_B13.mat). 

The following variables are in the .mat files:

* badTimeSegments - (n_badTimeSegments x 2)

    This variable contains manually marked time segments containing epileptiform, electrical, or movement artifacts. Each row indicates one bad time segment, with the start time and end time in seconds.

* bcs - (n_bcs)

    This array contains manually marked bad channels. These channels from the ECoG grid either had continuous epileptiform activity or signal indistinguishable from noise. The channels are indexed from 0.

* ECXXX_BXX_hg_100Hz - (n_chans x n_timepoints)

    This variable contains the mean high-gamma analytic amplitude signal for each channel, sampled at 100Hz. The mean is taken across 8 bands between 70-150Hz. The variable name contains the subject number, ECXXX, and block number BXX. (Note for EC125_B1044. )

* ECXXX_BXX_log_hg_100Hz - (n_chans x n_timepoints)

    This variable contains the mean of the natural logarithm of the high-gamma analytic amplitude signal for each channel. The log is taken for each of the 8 bands between 70-150Hz and then averaged.

* experiment -

    This variable holds the experiment type and is either "Speech", "Non-speech control", or "Non-speech missing f0 control".

* sentence_numbers - (n_trials)

    The integers in this array are the sentence number condition for each trial in this block. The sentence number conditions depend on the experiment type.

    1. For the "Speech" experiment, the four sentences indicated by 1, 2, 3, and 4.

        | sn | Sentence                        |
        |----|---------------------------------|
        | 1  | Humans value genuine behavior   |
        | 2  | Movies demand minimal energy    |
        | 3  | Lawyers give a relevant opinion |
        | 4  | Reindeer are a visual animal    |

    2. For the "Non-speech control" experiment, the sentence number conditions indicate which sentence from the main experiment the amplitude contour for the control stimuli came from. A sentence number of 5 means that the amplitude contour was flat. 

    3. For the "Non-speech missing f0 control", the sentence number holds information about the composition of the stimulus (which harmonics were present), whether noise was added, and how much the pitch range was stretched. Refer to the table below.

        | sn | composition of stimulus | noise    | stretch       |
        |----|-------------------------|----------|---------------|
        | 0  | 4h + 5h + 6h            | no noise | stretch = 1   |
        | 1  | f0 + 2h + 3h            | no noise | stretch = 1   |
        | 2  | 4h + 5h + 6h            | noise    | stretch = 1   |
        | 3  | 4h + 5h + 6h            | noise    | stretch = 0.5 |
        | 4  | 4h + 5h + 6h            | noise    | stretch = 2   |

* sentence_types - (n_trials)

    The sentence type is the intonation contour condition. Across all experiment types, a sentence type of 1 is Neutral, 2 is Question, 3 is Emphasis 1, and 4 is Emphasis 3. 

* speakers - (n_trial)

    The speaker condition for the "Speech" experiment is an integer between 1 and 3. 1 is the low-formant, low-pitch male speaker. 2 is the high-formant, high-pitch female speaker. 3 is the low-formant, high-pitch female speaker. The absolute pitch values of speakers 2 and 3 match, while the formant values of speaker 1 and 3 match.

* stims - (n_trils)

    This array holds the stimulus name that was played for each trial. The names refer to the wav files in the tokens, tokens_nonspeech, and tokens_missing_f0 folders, for the "Speech", "Non-speech control", and "Non-speech missing f0 control" experiments, respectively.

* times - (n_trials)

    This array contains the onset times of each trial in seconds.

TIMIT data is saved into one hdf5 file per subject. This file contains data from multiple blocks of TIMIT. (Note that TIMIT data is currently only provided for EC113)
