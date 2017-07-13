# Pitch information for intonation stimuli

In this folder, there are 48 text files and 1 hdf5 file with information about the pitch of the experimental speech and non-speech tokens. 

The text files contain pitch and intensity information for the speech signals in the tokens folder. Each .txt file is named according to naming scheme described for the set of speech stimuli in tokens. Each file has three columns, time (s), pitch (Hz), and intensity (dB). A pitch value of 0 indicates that there was no voicing (and therefore no pitch) at that time point. The sampling rate for these measurements is 100Hz.

The pitch_contours hdf5 file (created by the Python package pandas's to_hdf function) contains the continuous pitch contours of each intonation condition for high and low absolute pitch.
