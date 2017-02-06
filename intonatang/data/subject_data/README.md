This folder contains all the neural data for each subject. 

Within each subject's folder, there is neural data from both intonation (speech and nonspeech)
and TIMIT tasks. 

The intonation data is saved by block and consists of a .mat file for each block
The metadata mapping from block number to experimental task information is currently coded into
intonation_subject_data.py. 

TIMIT data is saved into one hdf5 file per subject. This file contains data from multiple blocks
of TIMIT.
