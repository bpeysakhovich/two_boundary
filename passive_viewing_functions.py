import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import math
from scipy import stats
import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # "Support Vector Classifier"
import shutil
import os
from sklearn.metrics import roc_auc_score
import itertools
from os import path

# STIMULUS CODE TIMES
SAMP_ON = 23
TEST_ON = 25
STRT_TRIAL = 9
STIM_ON = 23
PV_STIM_ON = 25
PV_STIM_OFF = 26
END_TRIAL = 18

# STIMULUS DIRECTIONS
dirs = [247.5, 225, 202.5, 67.5, 45, 22.5, 157.5, 135, 112.5, 337.5, 315, 292.5]

def passive_viewing_analysis_main(fname, window, bin_size, savepath, area, monkey, exclude_stim1):
    # Load data
    bhv = sio.loadmat(fname)['data']['BHV']
    neuro = sio.loadmat(fname)['data']['NEURO']

    monkey_name = fname[0:7]
    date = fname[8:18]

    # Extract relevant behavioral variables for all the trials
    errors = bhv[0][0][0][0]["TrialError"]
    corr_trials = np.where(errors == 0)[0]
    trial_nums = len(bhv[0][0][0][0]["TrialNumber"][0])
    conditions = bhv[0][0][0][0]["ConditionNumber"][0]

    # Extract relevant neural data for all the trials
    n_neurons = len([i for i in neuro[0][0][0][0][1][0][0].dtype.names if 'wf' not in i])
    spikes_all = neuro[0][0][0][0]["Neuron"][0][0]
    code_times = neuro[0][0][0][0]['CodeTimes']
    code_numbers= np.array([i[0] for i in neuro[0][0][0][0]['CodeNumbers']])

    strt_trial_indx = np.where(code_numbers == STRT_TRIAL)[0]
    strt_trial_time = code_times[strt_trial_indx]
    end_trial_indx = np.where(code_numbers == END_TRIAL)[0]

    # Get PV trial information
    trials_pv = np.where(conditions == 73)[0]
    corr_trials_pv = [i for i, val in enumerate(trials_pv) if val in corr_trials]
    [pv_dirs, trial_counts_per_dir] = get_passive_viewing_dirs(bhv, trials_pv, corr_trials_pv)

    # Get DMC trial information
    trials_dmc = np.where(conditions != 73)[0]
    corr_trials_dmc = [i for i in trials_dmc if i in corr_trials]

    # Only take datasets with >= 10 trials per direction
    if all(i >= 10 for i in trial_counts_per_dir):

        for i_neuron in range(n_neurons):
            neuron_name = neuro[0][0][0][0][1][0][0].dtype.names[i_neuron]

            # If neuron is task-responsive
            if path.exists('E:\\two_boundary\\data\\' + monkey + '\\' + area + '\\good_neurons\\task_responsive\\' + date + '_' + neuron_name + '.mat'):

                # Get current neuron's spikes
                spikes = spikes_all[i_neuron].flatten()

                # Passive viewing
                [spikes_raw_mean_pv, spikes_binned_mean_pv, window_mean_pv, window_sem_pv] = get_spikes_pv(spikes, trials_pv, corr_trials_pv, pv_dirs, window, code_times, code_numbers, strt_trial_indx, end_trial_indx, exclude_stim1, bin_size)

                # DMC
                [spikes_raw_mean_dmc, spikes_binned_mean_dmc, window_mean_dmc, window_sem_dmc] = get_spikes_dmc(spikes, corr_trials_dmc, pv_dirs, window, code_times, code_numbers, conditions, strt_trial_indx, end_trial_indx, strt_trial_time, exclude_stim1, bin_size)

                make_dirs(savepath)

                if exclude_stim1:
                    fname_data = savepath + date + '_' + neuron_name + '_stim1_excl' + '.mat'
                else:
                    fname_data = savepath + date + '_' + neuron_name + '_stim1_incl' + '.mat'

                sp.io.savemat(fname_data,  {'spikes_raw_mean_pv': spikes_raw_mean_pv, 'spikes_binned_mean_pv': spikes_binned_mean_pv, 'window_mean_pv': window_mean_pv, 'window_sem_pv': window_sem_pv,
                                            'spikes_raw_mean_dmc': spikes_raw_mean_dmc, 'spikes_binned_mean_dmc': spikes_binned_mean_dmc, 'window_mean_dmc': window_mean_dmc, 'window_sem_dmc': window_sem_dmc})

def get_spikes_dmc(spikes, corr_trials_dmc, pv_dirs, window, code_times, code_numbers, conditions, strt_trial_indx, end_trial_indx, strt_trial_time, exclude_stim1, bin_size):

   # Pre-allocate
    spikes_raw_dmc = [[] for i in range(12)]
    spikes_binned_dmc = [[] for i in range(12)]

    # Loop through trials
    for i_trial in corr_trials_dmc:

        if code_numbers[strt_trial_indx[i_trial]+1] != 14:

            if strt_trial_time[i_trial] > spikes[0] and strt_trial_time[i_trial] < spikes[-1]:

                # Compute when the stimulus for this trial came on
                stim_on_indx = np.where(code_numbers[strt_trial_indx[i_trial]:end_trial_indx[i_trial]] == 23)[0]
                stim_on_indx = stim_on_indx + strt_trial_indx[i_trial]

                # Find spike times for this trial based on stim. on index, compute histogram
                stim_on_time = code_times[stim_on_indx][0]

                t1 = spikes[spikes > stim_on_time + (window[0])]
                t2 =  spikes[spikes < stim_on_time + (window[-1]+1)]
                spike_times = np.intersect1d(t1, t2) - stim_on_time

                hist_spks = np.histogram(spike_times, window)[0]

                # Store the histogram for this trial
                dir_num = int(np.ceil(conditions[i_trial]/6)) - 1
                spikes_raw_dmc[dir_num].append(hist_spks)
                spikes_binned_dmc[dir_num].append(np.convolve(hist_spks, np.ones(100), 'same'))

            # Take averages by direction
            spikes_raw_mean = [np.array(i).squeeze().mean(axis=0) for i in spikes_raw_dmc]
            spikes_binned_mean  = [np.array(i).squeeze().mean(axis=0) for i in spikes_binned_dmc]

            dmc_sample_avg_spikes = np.zeros_like(dirs)
            sem_dmc_sample_spikes = np.zeros_like(dirs)

            for i_dir in range(12):
                tmp = np.array(spikes_raw_dmc[i_dir])*1000
                mean_across_trials_dmc = np.mean(tmp, 0)

                dmc_sample_avg_spikes[i_dir] = np.mean(mean_across_trials_dmc)
                sem_dmc_sample_spikes[i_dir] = stats.sem(mean_across_trials_dmc)

    return spikes_raw_mean, spikes_binned_mean, dmc_sample_avg_spikes, sem_dmc_sample_spikes

def get_spikes_pv(spikes, trials_pv, corr_trials_pv, pv_dirs, window, code_times, code_numbers, strt_trial_indx, end_trial_indx, exclude_stim1, bin_size):

    # Pre-allocate
    spikes_raw_pv = [[] for i in range(12)]
    spikes_binned_pv = [[] for i in range(12)]

    for i_trial, trial_num in enumerate(trials_pv):
        curr_stims = pv_dirs[i_trial]
        stim_on_inds = np.where(code_numbers[strt_trial_indx[trial_num]:end_trial_indx[trial_num]] == 25)[0]
        stim_on_inds = stim_on_inds + strt_trial_indx[trial_num]

        # For incomplete trials, get rid of last stimulus (likely not seen for full 400ms)
        if i_trial not in corr_trials_pv:
            stim_on_inds = stim_on_inds[0:-1]
            curr_stims = curr_stims[0:-1]

        if len(stim_on_inds) > 0 and exclude_stim1:
            stim_on_inds = stim_on_inds[1:]
            curr_stims = curr_stims[1:]

        n_stimuli = len(stim_on_inds)

        if n_stimuli > 0:
            stim_on_times = code_times[stim_on_inds].flatten()

            for i_stim, stim_on in enumerate(stim_on_times):
                t1 = spikes[spikes > stim_on+(window[0])]
                t2 =  spikes[spikes < stim_on+(window[-1]+1)]
                spike_times = np.intersect1d(t1, t2) - stim_on

                hist_spks = np.histogram(spike_times, window)[0]

                dir_num = int(curr_stims[i_stim])
                spikes_raw_pv[dir_num].append(hist_spks)
                spikes_binned_pv[dir_num].append(np.convolve(hist_spks, np.ones(bin_size), 'same'))

    spikes_raw_mean = [np.array(i).squeeze().mean(axis=0) for i in spikes_raw_pv]
    spikes_binned_mean = [np.array(i).squeeze().mean(axis=0) for i in spikes_binned_pv]

    # Get mean firing rate for entire window
    window_mean_pv = np.zeros_like(dirs)
    window_sem_pv = np.zeros_like(dirs)

    for i_dir in range(12):
        tmp = np.array(spikes_raw_mean[i_dir])
        mean_across_trials = np.mean(tmp, 0)*1000

        window_mean_pv[i_dir] = np.mean(mean_across_trials)
        window_sem_pv[i_dir] = stats.sem(mean_across_trials)

    return spikes_raw_mean, spikes_binned_mean, window_mean_pv, window_sem_pv

def get_passive_viewing_dirs(bhv, trials_pv, corr_trials_pv):

    # PV stimulus names in BHV file
    im_set = ('Stim_Filename_1', 'Stim_Filename_2', 'Stim_Filename_3', 'Stim_Filename_4', 'Stim_Filename_5')

    # Pre-allocate
    pv_dirs = [[] for i in range(len(trials_pv))]

    # Get direction shown on each trial from stimulus filename
    for i, i_trial in enumerate(trials_pv):
        for i_im in im_set:
            if len(bhv[0][0][0][0]["UserVars"][0][i_trial][i_im]) > 0:
                if len(bhv[0][0][0][0]["UserVars"][0][i_trial][i_im][0]) > 4:
                    file_split1 = (bhv[0][0][0][0]["UserVars"][0][i_trial][i_im][0]).split('\\')
                    file_split2 = file_split1[-1].split('_')
                    pv_dirs[i].append(int(file_split2[-2])-1)

    # Count number of PV trials per dir
    merged = list(itertools.chain.from_iterable(pv_dirs))
    counts = []
    for i_value in range (len(dirs)):
        counts.append(merged.count(i_value))

    return pv_dirs, counts
