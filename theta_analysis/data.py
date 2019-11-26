# Keiland's version
import numpy as np

# TODO:
# fix path's issue

def load_data(dataPaths, rat_name):
    """
    Load processed Numpy data arrays.
    path ex. 'D:\Documents\Data\processedHPC\{}\{}_trial_info.npy'
    """
    trial_info = np.load((dataPaths['trial'].format(rat_name, rat_name.lower())))
    spike_data = np.load(dataPaths['spike'].format(rat_name, rat_name.lower()))
    lfp_data = np.load(dataPaths['lfp'].format(rat_name, rat_name.lower()))
    lfp_data = np.swapaxes(lfp_data, 1, 2)
    return trial_info, spike_data, lfp_data


def load_data2(dataPaths, rat_name, event):
    """
    Load processed Numpy data arrays.
    path ex. 'D:\Documents\Data\processedHPC\{}\{}_trial_info.npy'
    """
    trial_info = np.load(dataPaths['trial'].format(rat_name, event))
    spike_data = np.load(dataPaths['spike'].format(rat_name, event))
    lfp_data = np.load(dataPaths['lfp'].format(rat_name, event))
    lfp_data = np.swapaxes(lfp_data, 1, 2)
    return trial_info, spike_data, lfp_data


def filter_trials(trial_info):
    """
    Get indices of correct in-sequence trials of odors A to D.
    """
    rat_correct = trial_info[:, 0] == 1
    in_sequence = trial_info[:, 1] == 1
    not_odor_e = trial_info[:, 3] < 5
    select = rat_correct & in_sequence & not_odor_e
    return select

def filter_trials2(trial_info, inseq, corr):
    """
    Get indices of correct in-sequence trials of odors A to D.
    if inseq is 1 then inseq, if 0 then outseq
    if corr is 1 then corr, if 0 then incorr
    """
    rat_correct = trial_info[:, 0] == corr
    in_sequence = trial_info[:, 1] == inseq
    not_odor_e = trial_info[:, 3] < 5
    select = rat_correct & in_sequence & not_odor_e
    return select

def clean_data(trial_info, spike_data, lfp_data):
    """
    Clean up trials, remove cells that do not fire, and label targets.
    """
    #trial_indices = filter_trials(trial_info)
    trial_indices = filter_trials(trial_info)
    spike_data = spike_data[trial_indices]
    lfp_data = lfp_data[trial_indices]
    total_spike = np.sum(np.sum(spike_data[:, :, 200:300], axis=2), axis=0) # select 0s to 2s spikes
    spike_data = spike_data[:, total_spike > 0, :]
    target = trial_info[trial_indices, 3] - 1
    return target, spike_data, lfp_data

def clean_data2(trial_info, spike_data, lfp_data, inseq, corr):
    """
    Clean up trials, remove cells that do not fire, and label targets.
    """
    #trial_indices = filter_trials(trial_info)
    trial_indices = filter_trials2(trial_info, inseq, corr)
    spike_data = spike_data[trial_indices]
    lfp_data = lfp_data[trial_indices]
    total_spike = np.sum(np.sum(spike_data[:, :, 200:300], axis=2), axis=0) # select 0s to 2s spikes
    spike_data = spike_data[:, total_spike > 0, :]
    target = trial_info[trial_indices, 3] - 1
    return target, spike_data, lfp_data

def clean_data3(trial_info, spike_data, lfp_data):
    """
    Clean up trials, and label targets.
    """
    #trial_indices = filter_trials(trial_info)
    trial_indices = filter_trials(trial_info)
    spike_data = spike_data[trial_indices]
    lfp_data = lfp_data[trial_indices]
    total_spike = np.sum(np.sum(spike_data[:, :, 200:300], axis=2), axis=0) # select 0s to 2s spikes
    #spike_data = spike_data[:, total_spike > 0, :]
    target = trial_info[trial_indices, 3] - 1
    return target, spike_data, lfp_data


def reference_tetrode(rat_name):
    """
    Return the index of reference tetrode.
    TODO: Convert this into a dictionary...
    """
    if rat_name == 'Superchris':
        tetrode_index = 3
    if rat_name == 'Stella':
        tetrode_index = 3
    if rat_name == 'Barat':
        tetrode_index = 2
    if rat_name == 'Buchanan':
        tetrode_index = 11
    if rat_name == 'Mitt':
        tetrode_index = 2
    return tetrode_index


