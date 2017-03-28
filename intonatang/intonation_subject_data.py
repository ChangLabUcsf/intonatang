from __future__ import division, print_function, absolute_import

import numpy as np
from .intonation_stims import *

def get_blocks_for_subject_number(subject_number, control_stim=False, missing_f0_stim=False):
    """Returns list of block numbers for each subject. 
    
    Order of blocks is chronological and matched with list of stims returned from get_stims_for_subject_number

    Args:
        subject_number: xxx in ECxxx
        control_stim: flag for getting data from non-speech blocks
        missing_f0_stim: flag for getting data from missing_f0 blocks
    """
    assert (control_stim and missing_f0_stim) is False

    if subject_number == 113:
        if control_stim:
            blocks = []
        elif missing_f0_stim:
            blocks = []
        else:
            blocks = [13, 20, 21]
    elif subject_number == 118:
        if control_stim:
            blocks = []
        elif missing_f0_stim:
            blocks = []
        else:
            blocks = [3, 7, 13]
    elif subject_number == 122:
        if control_stim:
            blocks = [33, 45]
        elif missing_f0_stim:
            blocks = []
        else:
            blocks = [30, 40, 43, 53]
    elif subject_number == 123:
        if control_stim:
            blocks = [11, 16]
        elif missing_f0_stim:
            blocks = []
        else:
            blocks = [4, 5, 10]
    elif subject_number == 125:
        if control_stim:
            blocks = [30]
        elif missing_f0_stim:
            blocks = []
        else:
            blocks = [13, 1044]
    elif subject_number == 129:
        if control_stim:
            blocks = [40, 42]
        elif missing_f0_stim:
            blocks = []
        else:
            blocks = [10, 16, 37]
    elif subject_number == 131:
        if control_stim:
            blocks = [54, 59]
        elif missing_f0_stim:
            blocks = []
        else:
            blocks = [47, 48]
    elif subject_number == 137:
        if control_stim:
            blocks = []
        elif missing_f0_stim:
            blocks = [9, 11]
        else:
            blocks = [7, 10]
    elif subject_number == 142:
        if control_stim:
            blocks = []
        elif missing_f0_stim:
            blocks = [38, 40]
        else:
            blocks = [36, 37]
    elif subject_number == 143:
        if control_stim:
            blocks = []
        elif missing_f0_stim:
            blocks = [10, 12, 14]
        else:
            blocks = [9, 11, 13]

    return blocks

def get_stims_for_subject_number(subject_number, control_stim=False, missing_f0_stim=False):
    """Returns list of stims for each subject. 

    Order of stims is matched with order of blocks from get_blocks_for_subject_number

    Args:
        subject_number: xxx in ECxxx
        control_stim: flag for getting data from non-speech blocks
        missing_f0_stim: flag for getting data from missing_f0 blocks
    """
    assert (control_stim and missing_f0_stim) is False

    if subject_number == 113:
        if control_stim:
            stims_list = []
        elif missing_f0_stim:
            stims_list = []
        else:
            stims_list = [stims1, stims5, stims2]
    elif subject_number == 118:
        if control_stim:
            stims_list = []
        elif missing_f0_stim:
            stims_list = []
        else:
            stims_list = [stims1, stims2, stims3]
    elif subject_number == 122:
        if control_stim:
            stims_list = [purr_stims1, purr_stims2]
        elif missing_f0_stim:
            stims_list = []
        else:
            stims_list = [stims1, stims2, stims3, stims4]
    elif subject_number == 123:
        if control_stim:
            stims_list = [purr_stims1, purr_stims2]
        elif missing_f0_stim:
            stims_list = []
        else:
            stims_list = [stims1, stims2, stims3]
    elif subject_number == 125:
        if control_stim:
            stims_list = [purr_stims1]
        elif missing_f0_stim:
            stims_list = []
        else:
            stims_list = [stims1, stims2]
    elif subject_number == 129:
        if control_stim:
            stims_list = [purr_stims1, purr_stims2]
        elif missing_f0_stim:
            stims_list = []
        else:
            stims_list = [stims1, stims2, stims3]
    elif subject_number == 131:
        if control_stim:
            stims_list = [purr_stims1, purr_stims2]
        elif missing_f0_stim:
            stims_list = []
        else:
            stims_list = [stims1, stims2]
    elif subject_number == 137:
        if control_stim:
            stims_list = []
        elif missing_f0_stim:
            stims_list = [missing_f0_stims1, missing_f0_stims2]
        else:
            stims_list = [stims1, stims2]
    elif subject_number == 142:
        if control_stim:
            stims_list = []
        elif missing_f0_stim:
            stims_list = [missing_f0_stims1, missing_f0_stims2]
        else:
            stims_list = [stims1, stims2]
    elif subject_number == 143:
        if control_stim:
            stims_list = []
        elif missing_f0_stim:
            stims_list = [missing_f0_stims1, missing_f0_stims2, missing_f0_stims4]
        else:
            stims_list = [stims1, stims2, stims3]

    return stims_list

def get_sentence_numbers_sentence_types_speakers_for_stims_list(stims_list, control_stim=False, missing_f0_stim=False):
    """Returns lists of sentence, intonation, and speaker conditions (integers from 1 to 4, 1 to 4, and 1 to 3, respectively)

    This is a subfunction used to get sns, sts, and sps given a list of stims.

    Args:
        stims_list: returned from get_stims_for_subject_number
        control_stim: flag for getting data from non-speech blocks
        missing_f0_stim: flag for getting data from missing_f0 blocks
    """
    assert (control_stim and missing_f0_stim) is False

    sns_blocks = []
    sts_blocks = []
    sps_blocks = []
    for stims in stims_list:
        if control_stim:
            sns, sts, sps = get_sn_st_sp_from_purr_stims(stims)
        elif missing_f0_stim:
            sns, sts, sps = get_sn_st_sp_from_missing_f0_stims(stims)
        else:
            sns, sts, sps = get_sn_st_sp_from_stims(stims)
        sns_blocks.append(sns)
        sts_blocks.append(sts)
        sps_blocks.append(sps)
    return np.concatenate(sns_blocks), np.concatenate(sts_blocks), np.concatenate(sps_blocks)

