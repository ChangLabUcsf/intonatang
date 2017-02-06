This folder contains processed data relating to the TIMIT dataset and the pitch temporal
receptive field analysis. 

The three HDF files, timit_pitch.h5, timit_phonemes.h5, and timit_pitch_phonetic.h5 are created by 
timit.py. They are generated during generate_all_results and can also be individually created
by the functions, save_timit_pitch, save_timit_phonemes, and save_timit_pitch_phonetic. 

The timit_pitch_shuffle_*.h5 files are created by pitch_trf.py. They are generated during 
generate_all_results.
