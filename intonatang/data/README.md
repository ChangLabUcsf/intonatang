# Data

* [brain imaging](#brain_imaging)
* [pitch](#pitch)
* [subject data](#subject_data)
* [timit](#timit)
* [timit pitch](#timit_pitch)
* [timit tokens](#timit_tokens)
* [tokens](#tokens)
* [tokens_missing_f0](#tokens_missing_f0)
* [tokens_nonspeech](#tokens_nonspeech)
* [pink_noise.mat]

Below is an overview of what's contained in each directory. More details can be found in the README.md files within each directory.

## brain_imaging

This directory contains information about the location of each ECoG grid electrode.

## pitch

In this folder, there is information about the pitch of the intonation, experimental speech and non-speech tokens. 

## subject_data

This folder contains all the neural data for each subject. 

Within each subject's folder, there is neural data from both intonation (speech and nonspeech) and TIMIT tasks. 

The intonation data is saved by block and consists of a .mat file for each block. Each data file is named ECXXX_BXX.mat indicating the subject, ECXXX, and block number, BXX. The data files are organized into subdirectories, first by subject and then by block (e.g. subject_data/EC113/EC113_B13/EC113_B13.mat).

## timit

This directory contains the phonetic transcription of TIMIT stimuli.

## timit_pitch

The 499 text files in this directory contain pitch information for the TIMIT stimuli.

## timit_tokens

The subset of TIMIT stimuli used in this experiment.

## tokens

Synthesized set of speech stimuli used in the main experiment that independently varies intonation contour, phonetic content, and speaker.

## tokens_missing_f0

Stimuli for the non-speech missing fundamental experiment.

## tokens_nonspeech

Set of non-speech stimuli that preserves intonational pitch contour but removes spectral content related to phonetic features.
