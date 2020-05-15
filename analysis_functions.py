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


""" GENERAL UTILITY FUNCTIONS """

def get_epoch_means(all_spikes, indx):

    n_trials = np.size(all_spikes, 0)
    all_trials = np.zeros(n_trials)
    for i_trial in range(n_trials):
        all_trials[i_trial] = np.mean(all_spikes[i_trial, indx])*1000

    mean_all_trials = np.mean(all_trials)
    std_all_trials = np.std(all_trials)

    return mean_all_trials, all_trials

def epoch_inds(bins, time1, time2):
    t1 = np.where(bins >= time1)
    t2 = np.where(bins < time2)
    indx = np.intersect1d(t1, t2)

    return indx

def make_dirs(path):

    import os

    if not os.path.exists(path):
        os.makedirs(path)

def max_consecutive_vals(data, value, thresh):

    indx_thresh = []

    counter = 0
    longest_run = 0
    for i, val in enumerate(data[0:-1]):
        if val == value and data[i+1] == value:
            if counter == 0:
                counter = 2
            else:
                counter += 1

            if counter >= thresh and not indx_thresh:
                indx_thresh = i

            if counter > longest_run:
                longest_run = counter
        else:
            counter = 0

    return longest_run, indx_thresh

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x, y)

def shuffle_pev(data_group1, data_group2, n_perms):

    true_diff = abs(np.mean(data_group1)-np.mean(data_group2))
    all_means = np.zeros(n_perms)

    combined = np.hstack([data_group1, data_group2])
    n_samples = int(len(combined)/2)

    for i_perm in range(n_perms):
        perm = np.random.permutation(combined)

        group1 = perm[0:n_samples]
        group2 = perm[n_samples:len(perm)]

        all_means[i_perm] = (np.mean(group1)-np.mean(group2))

    pval = sum(abs(all_means) >= abs(true_diff))/n_perms

    return pval

def shuffle_decoder(data_group1, data_group2, n_perms):

    pvals = np.zeros(3)
    for i in range(3):

        true_diff = abs(np.median(data_group1[i])-np.median(data_group2[i]))
        all_means = np.zeros(n_perms)

        combined = np.hstack([data_group1[i], data_group2[i]])
        n_samples = int(len(combined)/2)

        for i_perm in range(n_perms):
            perm = np.random.permutation(combined)

            group1 = perm[0:n_samples]
            group2 = perm[n_samples:len(perm)]

            all_means[i_perm] = (np.median(group1)-np.median(group2))

        pvals[i] = sum(abs(all_means) >= abs(true_diff))/n_perms

    return pvals


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontweight='bold', fontsize = 18)
    ax.set_xlim(0.25, len(labels) + 0.75)

def label_diff(i,j,text,X,Y):
    x = (X[i]+X[j])/2
    y = 1.1*max(Y[i], Y[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'linewidth':2}
    ax.annotate(text, xy=(X[i],y+7), zorder=10)
    ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)


""" PSTH AND TASK_RESPONSIVE CELLS FUNCTIONS """
### Generatae psth
## Output: raw_spikes, mean_psth, mean_psth_sm, psth_binned, trials_per_dir
def get_psth(spikes, errors, trial_starts, conditions, code_numbers, code_times, stim_on, bins, std = 30, bin_size = 1):

    spikes = np.array(spikes)

    from scipy.ndimage import gaussian_filter1d
    n_conds = 72
    n_dirs = 12

    raw_spikes = [[] for i in range(n_conds)]
    psth_binned = [[] for i in range(n_conds)]
    n_trials = len(conditions)


    for i_trial in range(n_trials):

        condition_num = conditions[i_trial]-1
        dir_num = int(np.ceil(condition_num/6))-1
        cat_num = int(np.floor(condition_num/37))

        trial_start_indx = trial_starts[i_trial]
        trial_start_time = code_times[trial_start_indx]

        if errors[i_trial] == 0 and condition_num != 73 and code_numbers[trial_start_indx+1] != 14 and code_numbers[trial_start_indx+1] != 1:

            if trial_start_time > spikes[0] & trial_start_time < spikes[-1]:

                if i_trial != n_trials-1:
                    samp_on_indx = [i+trial_start_indx for i, val in enumerate(code_numbers[trial_start_indx:trial_starts[i_trial+1]]) if val == stim_on][0]
                else:
                    samp_on_indx = [i+trial_start_indx for i, val in enumerate(code_numbers[trial_start_indx:]) if val == stim_on][0]
                samp_on_time = code_times[samp_on_indx]

                t1 = spikes[spikes > samp_on_time-abs(bins[0])]
                t2 =  spikes[spikes < samp_on_time+(bins[-1]+1)]
                spike_times = np.intersect1d(t1, t2) - samp_on_time

                binned = np.histogram(spike_times, bins)[0]
                raw_spikes[condition_num].append(binned)

                psth_binned[condition_num].append(np.convolve(binned, np.ones(bin_size), 'same'))

    mean_psth = np.zeros([n_conds, len(bins)-1])
    for i, curr_spikes in enumerate(raw_spikes):
        mean_psth[i] = np.mean(curr_spikes, 0)*1000

    n_timepoints = np.shape(mean_psth)[1]

    mean_psth_sm = np.zeros([n_conds, n_timepoints])
    for i_cond, curr_spikes in enumerate(mean_psth):
        mean_psth_sm[i_cond] = gaussian_filter1d(mean_psth[i_cond], std)


    trials_per_dir = np.zeros(n_dirs)
    for i, val in enumerate(np.arange(0, 72, 6)):
        trials_per_dir[i] = len(np.vstack([ii for ii in raw_spikes[val:val+6] if len(ii)> 0]))

    return raw_spikes, mean_psth, mean_psth_sm, psth_binned, trials_per_dir


def get_task_responsive_cells(filelist, bins, trial_epochs, currfigpath):

    if not os.path.exists('task_responsive'):
        os.makedirs('task_responsive')

    if not os.path.exists('unresponsive'):
        os.makedirs('unresponsive')

    figpath_resp = currfigpath + 'task_responsive\\'
    figpath_unresp = currfigpath + 'unresponsive\\'

    if not os.path.exists(figpath_resp):
        os.makedirs(figpath_resp)

    if not os.path.exists(figpath_unresp):
        os.makedirs(figpath_unresp)

    bins = np.array(bins)

    indx_baseline = epoch_inds(bins, -400, -200)
    indx_samp = epoch_inds(bins, 0, trial_epochs[0])
    indx_delay1 = epoch_inds(bins, trial_epochs[0], trial_epochs[1]-600)
    indx_delay2 = epoch_inds(bins, trial_epochs[1]-600, trial_epochs[1])
    indx_test1 = epoch_inds(bins, trial_epochs[1], trial_epochs[2])

    all_pvals = []
    mean_fr = []

    for file in filelist:
        data = sio.loadmat(file)
        all_spikes = data['raw_spikes']
        all_spikes = np.array(np.squeeze(all_spikes))
        all_spikes = np.vstack([item for sublist in all_spikes for item in sublist])

        [mean_baseline, all_baseline] = get_epoch_means(all_spikes, indx_baseline)
        [mean_samp, all_samp] = get_epoch_means(all_spikes, indx_samp)
        [mean_delay1, all_delay1] = get_epoch_means(all_spikes, indx_delay1)
        [mean_delay2, all_delay2] = get_epoch_means(all_spikes, indx_delay2)
        [mean_test1, all_test1] = get_epoch_means(all_spikes, indx_test1)

        [f, pval] = stats.f_oneway(all_baseline, all_samp, all_delay1, all_delay2, all_test1)
        curr_fr = np.mean([all_baseline, all_samp, all_delay1, all_delay2, all_test1])
        mean_per_epoch = [np.mean(i) for i in [all_samp, all_delay1, all_delay2, all_test1]]
        mean_fr.append(max(mean_per_epoch))
        all_pvals.append(pval)

        if pval <= 0.01 and max(mean_per_epoch) > 0.5:
            destination_mat = 'task_responsive\\' + file
            destination_png = figpath_resp + file[0:-3] + 'png'
        else:
            destination_mat = 'unresponsive\\' + file
            destination_png = figpath_unresp + file[0:-3] + 'png'

        figfile = currfigpath + file[0:-3] + 'png'

        shutil.copy(file, destination_mat)
        shutil.copy(figfile, destination_png)

    all_pvals = np.array(all_pvals)
    mean_fr = np.array(mean_fr)

    #sigp = np.where(all_pvals <= 0.01)
    #firing_thresh = np.where(mean_fr > 1

    return all_pvals, mean_fr

""" DECODING FUNCTIONS """

def category_decoder(n_iter, time_points, n_neurons_decoder, n_neurons_total, n_trials, filelist):
    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"

    clf = SVC(kernel='linear')

    train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11], [0, 1, 2, 9, 10, 11], [3, 4, 5, 6, 7, 8]]
    test_dir_all = [[3, 4, 5, 9, 10, 11], [0, 1, 2, 6, 7, 8], [3, 4, 5, 6, 7, 8], [0, 1, 2, 9, 10, 11]]

    perf_all = np.zeros([n_iter, len(time_points)])

    for iteration in range(0, n_iter):
        indx = np.random.randint(0, 4)
        train_dir = train_dir_all[indx]
        test_dir = test_dir_all[indx]

        train_data_c1 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
        train_data_c2 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]

        test_data_c1 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
        test_data_c2 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]

        neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

        for i, neuron in enumerate(neurons):
            file = filelist[neuron]
            data_by_cond = sio.loadmat(file)['binned_spikes'][0]

            tmpdirs = np.arange(0, 72, 6)
            data = [[] for i in range(12)]
            for ii, val in enumerate(tmpdirs):
                data[ii] = np.vstack([ii for ii in data_by_cond[val:val+6] if len(ii)> 0])

            all_trials_train_c1 = np.vstack([data[train_dir[0]], data[train_dir[1]], data[train_dir[2]]])
            all_trials_train_c2 = np.vstack([data[train_dir[3]], data[train_dir[4]], data[train_dir[5]]])

            all_trials_test_c1 = np.vstack([data[test_dir[0]], data[test_dir[1]], data[test_dir[2]]])
            all_trials_test_c2 = np.vstack([data[test_dir[3]], data[test_dir[4]], data[test_dir[5]]])

            train_trials_c1 = np.random.choice(len(all_trials_train_c1), n_trials, replace=False)
            train_trials_c2 = np.random.choice(len(all_trials_train_c2), n_trials, replace=False)

            test_trials_c1 = np.random.choice(len(all_trials_test_c1), n_trials, replace=False)
            test_trials_c2 = np.random.choice(len(all_trials_test_c2), n_trials, replace=False)

            c1_train = np.vstack([all_trials_train_c1[i] for i in train_trials_c1])
            c2_train = np.vstack([all_trials_train_c2[i] for i in train_trials_c2])

            c1_test = np.vstack([all_trials_test_c1[i] for i in test_trials_c1])
            c2_test = np.vstack([all_trials_test_c2[i] for i in test_trials_c2])

            for i_trial in range(n_trials):
                train_data_c1[i_trial][i] = c1_train[i_trial]
                train_data_c2[i_trial][i] = c2_train[i_trial]

                test_data_c1[i_trial][i] = c1_test[i_trial]
                test_data_c2[i_trial][i] = c2_test[i_trial]

        perf = np.zeros([len(time_points)])

        for i, timepoint in enumerate(time_points):

            x = np.zeros([n_neurons_decoder, n_trials*2])
            y = np.zeros([n_trials*2])

            for i_trial in range(n_trials):
                x[:, i_trial] = train_data_c1[i_trial][:, timepoint]
                y[i_trial] = 1

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials] = train_data_c2[i_trial][:, timepoint]
                y[i_trial+n_trials] = 2

            x_test = np.zeros([n_neurons_decoder, n_trials*2])
            y_test = np.zeros([n_trials*2])

            for i_trial in range(n_trials):
                x_test[:, i_trial] = test_data_c1[i_trial][:, timepoint]
                y_test[i_trial] = 1

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials] = test_data_c2[i_trial][:, timepoint]
                y_test[i_trial+n_trials] = 2

            clf.fit(x.T, y)

            if i == 240:
                clf240 = clf

            pred = clf.predict(x_test.T)
            perf[i] = sum(pred == y_test)/len(y_test)*100

        perf_all[iteration] = perf

    return perf_all, clf240


def direction_decoder(n_iter, time_points, n_neurons_decoder, n_neurons_total):
    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"

    clf = SVC(kernel='linear')
    perf_all = np.zeros([n_iter, len(time_points)])
    train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]
    test_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]

    for iteration in range(0, n_iter):
        indx = np.random.randint(0, 2)
        train_dir = train_dir_all[indx]
        test_dir = train_dir_all[indx]

        train_data_c1 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        train_data_c2 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        train_data_c3 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        train_data_c4 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        train_data_c5 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        train_data_c6 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]

        test_data_c1 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        test_data_c2 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        test_data_c3 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        test_data_c4 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        test_data_c5 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]
        test_data_c6 = [np.zeros([n_neurons_decoder, 3549-100]) for i in range(n_trials)]

        neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

        for i, neuron in enumerate(neurons):
            file = filelist[neuron]
            data_by_cond = sio.loadmat(file)['binned_spikes'][0]

            tmpdirs = np.arange(0, 72, 6)
            data = [[] for i in range(12)]
            for ii, val in enumerate(tmpdirs):
                data[ii] = np.vstack([ii for ii in data_by_cond[val:val+6] if len(ii)> 0])

            trials_c1 = np.random.choice(len(data[train_dir[0]]), n_trials*2, replace=False)
            trials_c2 = np.random.choice(len(data[train_dir[1]]), n_trials*2, replace=False)
            trials_c3 = np.random.choice(len(data[train_dir[2]]), n_trials*2, replace=False)
            trials_c4 = np.random.choice(len(data[train_dir[3]]), n_trials*2, replace=False)
            trials_c5 = np.random.choice(len(data[train_dir[4]]), n_trials*2, replace=False)
            trials_c6 = np.random.choice(len(data[train_dir[5]]), n_trials*2, replace=False)

            c1_train = np.vstack([data[train_dir[0]][i] for i in trials_c1[0:n_trials]])
            c2_train = np.vstack([data[train_dir[1]][i] for i in trials_c2[0:n_trials]])
            c3_train = np.vstack([data[train_dir[2]][i] for i in trials_c3[0:n_trials]])
            c4_train = np.vstack([data[train_dir[3]][i] for i in trials_c4[0:n_trials]])
            c5_train = np.vstack([data[train_dir[4]][i] for i in trials_c5[0:n_trials]])
            c6_train = np.vstack([data[train_dir[5]][i] for i in trials_c6[0:n_trials]])

            c1_test = np.vstack([data[train_dir[0]][i] for i in trials_c1[n_trials:]])
            c2_test = np.vstack([data[train_dir[1]][i] for i in trials_c2[n_trials:]])
            c3_test = np.vstack([data[train_dir[2]][i] for i in trials_c3[n_trials:]])
            c4_test = np.vstack([data[train_dir[3]][i] for i in trials_c4[n_trials:]])
            c5_test = np.vstack([data[train_dir[4]][i] for i in trials_c5[n_trials:]])
            c6_test = np.vstack([data[train_dir[5]][i] for i in trials_c6[n_trials:]])

            for i_trial in range(n_trials):
                train_data_c1[i_trial][i_neuron] = c1_train[i_trial]
                train_data_c2[i_trial][i_neuron] = c2_train[i_trial]
                train_data_c3[i_trial][i_neuron] = c3_train[i_trial]
                train_data_c4[i_trial][i_neuron] = c4_train[i_trial]
                train_data_c5[i_trial][i_neuron] = c5_train[i_trial]
                train_data_c6[i_trial][i_neuron] = c6_train[i_trial]

                test_data_c1[i_trial][i_neuron] = c1_test[i_trial]
                test_data_c2[i_trial][i_neuron] = c2_test[i_trial]
                test_data_c3[i_trial][i_neuron] = c3_test[i_trial]
                test_data_c4[i_trial][i_neuron] = c4_test[i_trial]
                test_data_c5[i_trial][i_neuron] = c5_test[i_trial]
                test_data_c6[i_trial][i_neuron] = c6_test[i_trial]

        perf = np.zeros([len(time_points)])

        for i, timepoint in enumerate(time_points):

            x = np.zeros([n_neurons_decoder, n_trials*6])
            y = np.zeros([n_trials*6])

            for i_trial in range(n_trials):
                x[:, i_trial] = train_data_c1[i_trial][:, timepoint]
                y[i_trial] = 1

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials] = train_data_c2[i_trial][:, timepoint]
                y[i_trial+n_trials] = 2

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*2] = train_data_c3[i_trial][:, timepoint]
                y[i_trial+n_trials*2] = 3

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*3] = train_data_c4[i_trial][:, timepoint]
                y[i_trial+n_trials*3] = 4

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*4] = train_data_c5[i_trial][:, timepoint]
                y[i_trial+n_trials*4] = 5

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*5] = train_data_c6[i_trial][:, timepoint]
                y[i_trial+n_trials*5] = 6

            x_test = np.zeros([n_neurons_decoder, n_trials*6])
            y_test = np.zeros([n_trials*6])

            for i_trial in range(n_trials):
                x_test[:, i_trial] = test_data_c1[i_trial][:, timepoint]
                y_test[i_trial] = 1

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials] = test_data_c2[i_trial][:, timepoint]
                y_test[i_trial+n_trials] = 2

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*2] = test_data_c3[i_trial][:, timepoint]
                y_test[i_trial+n_trials*2] = 3

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*3] = test_data_c4[i_trial][:, timepoint]
                y_test[i_trial+n_trials*3] = 4

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*4] = test_data_c5[i_trial][:, timepoint]
                y_test[i_trial+n_trials*4] = 5

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*5] = test_data_c6[i_trial][:, timepoint]
                y_test[i_trial+n_trials*5] = 6

            clf.fit(x.T, y)

            pred = clf.predict(x_test.T)
            perf[i] = sum(pred == y_test)/len(y_test)*100

        perf_all[iteration] = perf
    return perf_all

def category_decoder_epoch(all_data, n_iter, n_neurons_decoder, n_neurons_total, n_trials):
    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"

    clf = SVC(kernel='linear')

    train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11], [0, 1, 2, 9, 10, 11], [3, 4, 5, 6, 7, 8]]
    test_dir_all = [[3, 4, 5, 9, 10, 11], [0, 1, 2, 6, 7, 8], [3, 4, 5, 6, 7, 8], [0, 1, 2, 9, 10, 11]]

    perf = np.zeros(n_iter)

    for iteration in range(0, n_iter):
        indx = np.random.randint(0, 4)
        train_dir = train_dir_all[indx]
        test_dir = test_dir_all[indx]

        train_data_c1 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        train_data_c2 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]

        test_data_c1 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        test_data_c2 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]

        neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

        for i, data in enumerate(all_data):

            all_trials_train_c1 = np.hstack([data[train_dir[0]], data[train_dir[1]], data[train_dir[2]]])
            all_trials_train_c2 = np.hstack([data[train_dir[3]], data[train_dir[4]], data[train_dir[5]]])

            all_trials_test_c1 = np.hstack([data[test_dir[0]], data[test_dir[1]], data[test_dir[2]]])
            all_trials_test_c2 = np.hstack([data[test_dir[3]], data[test_dir[4]], data[test_dir[5]]])

            train_trials_c1 = np.random.choice(len(all_trials_train_c1), n_trials, replace=False)
            train_trials_c2 = np.random.choice(len(all_trials_train_c2), n_trials, replace=False)

            test_trials_c1 = np.random.choice(len(all_trials_test_c1), n_trials, replace=False)
            test_trials_c2 = np.random.choice(len(all_trials_test_c2), n_trials, replace=False)

            c1_train = np.hstack([all_trials_train_c1[i] for i in train_trials_c1])
            c2_train = np.hstack([all_trials_train_c2[i] for i in train_trials_c2])

            c1_test = np.hstack([all_trials_test_c1[i] for i in test_trials_c1])
            c2_test = np.hstack([all_trials_test_c2[i] for i in test_trials_c2])

            for i_trial in range(n_trials):
                train_data_c1[i_trial][i] = c1_train[i_trial]
                train_data_c2[i_trial][i] = c2_train[i_trial]

                test_data_c1[i_trial][i] = c1_test[i_trial]
                test_data_c2[i_trial][i] = c2_test[i_trial]

        x = np.zeros([n_neurons_decoder, n_trials*2])
        y = np.zeros([n_trials*2])

        for i_trial in range(n_trials):
            x[:, i_trial] = train_data_c1[i_trial][:]
            y[i_trial] = 1

        for i_trial in range(n_trials):
            x[:, i_trial+n_trials] = train_data_c2[i_trial][:]
            y[i_trial+n_trials] = 2

        x_test = np.zeros([n_neurons_decoder, n_trials*2])
        y_test = np.zeros([n_trials*2])

        for i_trial in range(n_trials):
            x_test[:, i_trial] = test_data_c1[i_trial][:]
            y_test[i_trial] = 1

        for i_trial in range(n_trials):
            x_test[:, i_trial+n_trials] = test_data_c2[i_trial][:]
            y_test[i_trial+n_trials] = 2

        clf.fit(x.T, y)

        pred = clf.predict(x_test.T)
        perf[iteration] = sum(pred == y_test)/len(y_test)*100


    return perf

def direction_decoder_epoch(all_data, n_iter, n_neurons_decoder, n_neurons_total, n_trials):
    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"

    clf = SVC(kernel='linear')
    train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]
    test_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]

    perf = np.zeros(n_iter)
    for iteration in range(n_iter):
        indx = np.random.randint(0, 2)
        train_dir = train_dir_all[indx]
        test_dir = train_dir_all[indx]

        train_data_c1 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        train_data_c2 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        train_data_c3 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        train_data_c4 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        train_data_c5 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        train_data_c6 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]

        test_data_c1 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        test_data_c2 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        test_data_c3 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        test_data_c4 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        test_data_c5 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]
        test_data_c6 = [np.zeros([n_neurons_decoder]) for i in range(n_trials)]

        neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

        for i, data in enumerate(all_data):

            trials_c1 = np.random.choice(len(data[train_dir[0]]), n_trials*2, replace=False)
            trials_c2 = np.random.choice(len(data[train_dir[1]]), n_trials*2, replace=False)
            trials_c3 = np.random.choice(len(data[train_dir[2]]), n_trials*2, replace=False)
            trials_c4 = np.random.choice(len(data[train_dir[3]]), n_trials*2, replace=False)
            trials_c5 = np.random.choice(len(data[train_dir[4]]), n_trials*2, replace=False)
            trials_c6 = np.random.choice(len(data[train_dir[5]]), n_trials*2, replace=False)



            c1_train = np.vstack([data[train_dir[0]][i] for i in trials_c1[0:n_trials]])
            c2_train = np.vstack([data[train_dir[1]][i] for i in trials_c2[0:n_trials]])
            c3_train = np.vstack([data[train_dir[2]][i] for i in trials_c3[0:n_trials]])
            c4_train = np.vstack([data[train_dir[3]][i] for i in trials_c4[0:n_trials]])
            c5_train = np.vstack([data[train_dir[4]][i] for i in trials_c5[0:n_trials]])
            c6_train = np.vstack([data[train_dir[5]][i] for i in trials_c6[0:n_trials]])

            c1_test = np.vstack([data[train_dir[0]][i] for i in trials_c1[n_trials:]])
            c2_test = np.vstack([data[train_dir[1]][i] for i in trials_c2[n_trials:]])
            c3_test = np.vstack([data[train_dir[2]][i] for i in trials_c3[n_trials:]])
            c4_test = np.vstack([data[train_dir[3]][i] for i in trials_c4[n_trials:]])
            c5_test = np.vstack([data[train_dir[4]][i] for i in trials_c5[n_trials:]])
            c6_test = np.vstack([data[train_dir[5]][i] for i in trials_c6[n_trials:]])

            for i_trial in range(n_trials):
                train_data_c1[i_trial][i] = c1_train[i_trial]
                train_data_c2[i_trial][i] = c2_train[i_trial]
                train_data_c3[i_trial][i] = c3_train[i_trial]
                train_data_c4[i_trial][i] = c4_train[i_trial]
                train_data_c5[i_trial][i] = c5_train[i_trial]
                train_data_c6[i_trial][i] = c6_train[i_trial]

                test_data_c1[i_trial][i] = c1_test[i_trial]
                test_data_c2[i_trial][i] = c2_test[i_trial]
                test_data_c3[i_trial][i] = c3_test[i_trial]
                test_data_c4[i_trial][i] = c4_test[i_trial]
                test_data_c5[i_trial][i] = c5_test[i_trial]
                test_data_c6[i_trial][i] = c6_test[i_trial]

        x = np.zeros([n_neurons_decoder, n_trials*6])
        y = np.zeros([n_trials*6])

        for i_trial in range(n_trials):
            x[:, i_trial] = train_data_c1[i_trial][:]
            y[i_trial] = 1

        for i_trial in range(n_trials):
            x[:, i_trial+n_trials] = train_data_c2[i_trial][:]
            y[i_trial+n_trials] = 2

        for i_trial in range(n_trials):
            x[:, i_trial+n_trials*2] = train_data_c3[i_trial][:]
            y[i_trial+n_trials*2] = 3

        for i_trial in range(n_trials):
            x[:, i_trial+n_trials*3] = train_data_c4[i_trial][:]
            y[i_trial+n_trials*3] = 4

        for i_trial in range(n_trials):
            x[:, i_trial+n_trials*4] = train_data_c5[i_trial][:]
            y[i_trial+n_trials*4] = 5

        for i_trial in range(n_trials):
            x[:, i_trial+n_trials*5] = train_data_c6[i_trial][:]
            y[i_trial+n_trials*5] = 6

        x_test = np.zeros([n_neurons_decoder, n_trials*6])
        y_test = np.zeros([n_trials*6])

        for i_trial in range(n_trials):
            x_test[:, i_trial] = test_data_c1[i_trial][:]
            y_test[i_trial] = 1

        for i_trial in range(n_trials):
            x_test[:, i_trial+n_trials] = test_data_c2[i_trial][:]
            y_test[i_trial+n_trials] = 2

        for i_trial in range(n_trials):
            x_test[:, i_trial+n_trials*2] = test_data_c3[i_trial][:]
            y_test[i_trial+n_trials*2] = 3

        for i_trial in range(n_trials):
            x_test[:, i_trial+n_trials*3] = test_data_c4[i_trial][:]
            y_test[i_trial+n_trials*3] = 4

        for i_trial in range(n_trials):
            x_test[:, i_trial+n_trials*4] = test_data_c5[i_trial][:]
            y_test[i_trial+n_trials*4] = 5

        for i_trial in range(n_trials):
            x_test[:, i_trial+n_trials*5] = test_data_c6[i_trial][:]
            y_test[i_trial+n_trials*5] = 6


        clf.fit(x.T, y)

        pred = clf.predict(x_test.T)
        perf[iteration] = sum(pred == y_test)/len(y_test)*100

    return perf

def category_decoder_PV(n_iter, time_points, n_neurons_decoder, n_neurons_total, n_trials, all_data):

    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"
    neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)
    clf = SVC(kernel='linear')
    time_points = np.arange(0, len(window))
    decoder_timepoints = np.arange(0, len(window), 10)
    train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11], [0, 1, 2, 9, 10, 11], [3, 4, 5, 6, 7, 8]]
    test_dir_all = [[3, 4, 5, 9, 10, 11], [0, 1, 2, 6, 7, 8], [3, 4, 5, 6, 7, 8], [0, 1, 2, 9, 10, 11]]

    perf_all = np.zeros([n_iter, len(decoder_timepoints)])

    for iteration in range(0, n_iter):

        indx = np.random.randint(0, 4)
        train_dir = train_dir_all[indx]
        test_dir = test_dir_all[indx]

        train_data_c1 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        train_data_c2 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]

        test_data_c1 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        test_data_c2 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]

        for i, neuron in enumerate(neurons):
            data = all_data[neuron]

            all_trials_train_c1 = np.vstack([data[train_dir[0]], data[train_dir[1]], data[train_dir[2]]])
            all_trials_train_c2 = np.vstack([data[train_dir[3]], data[train_dir[4]], data[train_dir[5]]])

            all_trials_test_c1 = np.vstack([data[test_dir[0]], data[test_dir[1]], data[test_dir[2]]])
            all_trials_test_c2 = np.vstack([data[test_dir[3]], data[test_dir[4]], data[test_dir[5]]])

            train_trials_c1 = np.random.choice(len(all_trials_train_c1), n_trials, replace=False)
            train_trials_c2 = np.random.choice(len(all_trials_train_c2), n_trials, replace=False)

            test_trials_c1 = np.random.choice(len(all_trials_test_c1), n_trials, replace=False)
            test_trials_c2 = np.random.choice(len(all_trials_test_c2), n_trials, replace=False)

            c1_train = np.vstack([all_trials_train_c1[i] for i in train_trials_c1])
            c2_train = np.vstack([all_trials_train_c2[i] for i in train_trials_c2])

            c1_test = np.vstack([all_trials_test_c1[i] for i in test_trials_c1])
            c2_test = np.vstack([all_trials_test_c2[i] for i in test_trials_c2])

            for i_trial in range(n_trials):
                train_data_c1[i_trial][i] = c1_train[i_trial]
                train_data_c2[i_trial][i] = c2_train[i_trial]

                test_data_c1[i_trial][i] = c1_test[i_trial]
                test_data_c2[i_trial][i] = c2_test[i_trial]


        perf = np.zeros([len(decoder_timepoints)])
        for i, timepoint in enumerate(decoder_timepoints):

            x = np.zeros([n_neurons_decoder, n_trials*2])
            y = np.zeros([n_trials*2])

            for i_trial in range(n_trials):
                x[:, i_trial] = train_data_c1[i_trial][:, timepoint]
                y[i_trial] = 1

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials] = train_data_c2[i_trial][:, timepoint]
                y[i_trial+n_trials] = 2

            x_test = np.zeros([n_neurons_decoder, n_trials*2])
            y_test = np.zeros([n_trials*2])

            for i_trial in range(n_trials):
                x_test[:, i_trial] = test_data_c1[i_trial][:, timepoint]
                y_test[i_trial] = 1

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials] = test_data_c2[i_trial][:, timepoint]
                y_test[i_trial+n_trials] = 2

            clf.fit(x.T, y)

            pred = clf.predict(x_test.T)
            perf[i] = sum(pred == y_test)/len(y_test)*100

        perf_all[iteration] = perf

    return perf_all

def direction_decoder_PV(n_iter, time_points, n_neurons_decoder, n_neurons_total, n_trials, all_data):
    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"

    time_points = np.arange(0, len(window))
    decoder_timepoints = np.arange(0, len(window), 10)

    clf = SVC(kernel='linear')
    perf_all = np.zeros([n_iter, len(decoder_timepoints)])
    train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]
    test_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]


    for iteration in range(0, n_iter):
        indx = np.random.randint(0, 2)
        train_dir = train_dir_all[indx]
        test_dir = train_dir_all[indx]

        train_data_c1 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        train_data_c2 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        train_data_c3 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        train_data_c4 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        train_data_c5 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        train_data_c6 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]

        test_data_c1 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        test_data_c2 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        test_data_c3 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        test_data_c4 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        test_data_c5 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]
        test_data_c6 = [np.zeros([n_neurons_decoder, len(time_points)-1]) for i in range(n_trials)]

        neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

        for i_neuron, neuron in enumerate(neurons):

            data = all_data[neuron]

            trials_c1 = np.random.choice(len(data[train_dir[0]]), n_trials*2, replace=False)
            trials_c2 = np.random.choice(len(data[train_dir[1]]), n_trials*2, replace=False)
            trials_c3 = np.random.choice(len(data[train_dir[2]]), n_trials*2, replace=False)
            trials_c4 = np.random.choice(len(data[train_dir[3]]), n_trials*2, replace=False)
            trials_c5 = np.random.choice(len(data[train_dir[4]]), n_trials*2, replace=False)
            trials_c6 = np.random.choice(len(data[train_dir[5]]), n_trials*2, replace=False)

            c1_train = np.vstack([data[train_dir[0]][i] for i in trials_c1[0:n_trials]])
            c2_train = np.vstack([data[train_dir[1]][i] for i in trials_c2[0:n_trials]])
            c3_train = np.vstack([data[train_dir[2]][i] for i in trials_c3[0:n_trials]])
            c4_train = np.vstack([data[train_dir[3]][i] for i in trials_c4[0:n_trials]])
            c5_train = np.vstack([data[train_dir[4]][i] for i in trials_c5[0:n_trials]])
            c6_train = np.vstack([data[train_dir[5]][i] for i in trials_c6[0:n_trials]])

            c1_test = np.vstack([data[train_dir[0]][i] for i in trials_c1[n_trials:]])
            c2_test = np.vstack([data[train_dir[1]][i] for i in trials_c2[n_trials:]])
            c3_test = np.vstack([data[train_dir[2]][i] for i in trials_c3[n_trials:]])
            c4_test = np.vstack([data[train_dir[3]][i] for i in trials_c4[n_trials:]])
            c5_test = np.vstack([data[train_dir[4]][i] for i in trials_c5[n_trials:]])
            c6_test = np.vstack([data[train_dir[5]][i] for i in trials_c6[n_trials:]])

            for i_trial in range(n_trials):
                train_data_c1[i_trial][i_neuron] = c1_train[i_trial]
                train_data_c2[i_trial][i_neuron] = c2_train[i_trial]
                train_data_c3[i_trial][i_neuron] = c3_train[i_trial]
                train_data_c4[i_trial][i_neuron] = c4_train[i_trial]
                train_data_c5[i_trial][i_neuron] = c5_train[i_trial]
                train_data_c6[i_trial][i_neuron] = c6_train[i_trial]

                test_data_c1[i_trial][i_neuron] = c1_test[i_trial]
                test_data_c2[i_trial][i_neuron] = c2_test[i_trial]
                test_data_c3[i_trial][i_neuron] = c3_test[i_trial]
                test_data_c4[i_trial][i_neuron] = c4_test[i_trial]
                test_data_c5[i_trial][i_neuron] = c5_test[i_trial]
                test_data_c6[i_trial][i_neuron] = c6_test[i_trial]

        perf = np.zeros([len(decoder_timepoints)])

        for i, timepoint in enumerate(decoder_timepoints):

            x = np.zeros([n_neurons_decoder, n_trials*6])
            y = np.zeros([n_trials*6])

            for i_trial in range(n_trials):
                x[:, i_trial] = train_data_c1[i_trial][:, timepoint]
                y[i_trial] = 1

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials] = train_data_c2[i_trial][:, timepoint]
                y[i_trial+n_trials] = 2

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*2] = train_data_c3[i_trial][:, timepoint]
                y[i_trial+n_trials*2] = 3

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*3] = train_data_c4[i_trial][:, timepoint]
                y[i_trial+n_trials*3] = 4

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*4] = train_data_c5[i_trial][:, timepoint]
                y[i_trial+n_trials*4] = 5

            for i_trial in range(n_trials):
                x[:, i_trial+n_trials*5] = train_data_c6[i_trial][:, timepoint]
                y[i_trial+n_trials*5] = 6

            x_test = np.zeros([n_neurons_decoder, n_trials*6])
            y_test = np.zeros([n_trials*6])

            for i_trial in range(n_trials):
                x_test[:, i_trial] = test_data_c1[i_trial][:, timepoint]
                y_test[i_trial] = 1

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials] = test_data_c2[i_trial][:, timepoint]
                y_test[i_trial+n_trials] = 2

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*2] = test_data_c3[i_trial][:, timepoint]
                y_test[i_trial+n_trials*2] = 3

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*3] = test_data_c4[i_trial][:, timepoint]
                y_test[i_trial+n_trials*3] = 4

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*4] = test_data_c5[i_trial][:, timepoint]
                y_test[i_trial+n_trials*4] = 5

            for i_trial in range(n_trials):
                x_test[:, i_trial+n_trials*5] = test_data_c6[i_trial][:, timepoint]
                y_test[i_trial+n_trials*5] = 6

            clf.fit(x.T, y)

            pred = clf.predict(x_test.T)
            perf[i] = sum(pred == y_test)/len(y_test)*100

        perf_all[iteration] = perf
    return perf_all

""" ROC ANALYSIS """

def get_WC_BC_ROC(data, pairs_indx, step_size = 10):

    n_timepoints_total = len(data[0][0])
    n_timepoints_stepped = len(np.arange(0, n_timepoints_total, 10))

    n_pairs = len(pairs_indx)

    roc_all = np.zeros([n_pairs, n_timepoints_stepped])

    for i_pair in range(n_pairs):
        dir1 = data[pairs_indx[i_pair][0]]
        dir2 = data[pairs_indx[i_pair][1]]

        for i, timepoint in enumerate(np.arange(0, n_timepoints_total, step_size)):
            y_true = np.hstack([np.zeros(np.size(dir1, 0)), np.ones(np.size(dir2, 0))])
            y_data = np.hstack([dir1[:, timepoint], dir2[:, timepoint]])

            curr_roc = abs(roc_auc_score(y_true, y_data) - 0.5)
            roc_all[i_pair, i] = curr_roc

    mean_roc = np.mean(roc_all, 0)

    return roc_all, mean_roc


def get_WC_BC_ROC_epoch(data, pairs_indx):

    n_pairs = len(pairs_indx)

    roc_all = np.zeros(n_pairs)

    for i_pair in range(n_pairs):
        dir1 = data[pairs_indx[i_pair][0]]
        dir2 = data[pairs_indx[i_pair][1]]

        y_true = np.hstack([np.zeros(np.size(dir1, 0)), np.ones(np.size(dir2, 0))])
        y_data = np.hstack([dir1, dir2])

        curr_roc = abs(roc_auc_score(y_true, y_data) - 0.5)
        roc_all[i_pair] = curr_roc

    mean_roc = np.mean(roc_all, 0)

    return roc_all, mean_roc

""" PASSIVE VIEWING FUNCTIONS """
def passive_viewing_analysis(fname, area, monkey, timepoints, savepath):

    dirs = np.array([247.5, 225, 202.5, 67.5, 45, 22.5, 157.5, 135, 112.5, 337.5, 315, 292.5])

    center_inds = [i for i, val in enumerate(dirs) if isinstance(val, int)]
    border_inds = [3, 8, 6, 9, 0, 5, 2, 11]


    BCD_pairs = [[22.5, 157.5], [112.5, 247.5], [202.5, 337.5], [292.5, 67.5], [67.5, 112.5], [157.5, 202.5], [247.5, 292.5], [337.5, 22.5]]
    BCD_pairs_indx = [[np.where(dirs == i[0])[0][0], np.where(dirs == i[1])[0][0]] for i in BCD_pairs]

    WCD_pairs = [[337.5, 112.5], [157.5, 292.5], [67.5, 202.5], [247.5, 22.5], [22.5, 67.5], [202.5, 247.5], [112.5, 157.5], [292.5, 337.5]]
    WCD_pairs_indx = [[np.where(dirs == i[0])[0][0], np.where(dirs == i[1])[0][0]] for i in WCD_pairs]


    pv_mean_rate = []
    dmc_mean_rate = []

    dmc_border = []
    dmc_center = []

    corrs = []

    rCTI_pv = []
    rCTI_dmc = []

    dmc = []
    pv = []

    spikes_binned_dmc_all = []
    spikes_binned_pv_all = []


    im_set = ('Stim_Filename_1', 'Stim_Filename_2', 'Stim_Filename_3', 'Stim_Filename_4', 'Stim_Filename_5')

    data  = sio.loadmat(fname)['data']

    bhv   = data['BHV']
    neuro = data['NEURO']

    # Load the file specified by fn
    monkey_name = fname[0:7]
    date        = fname[8:18]

    # Extract relevant behavioral variables for all the trials
    error     = bhv[0][0][0][0]["TrialError"]
    trial_num = len(bhv[0][0][0][0]["TrialNumber"][0])
    condition = bhv[0][0][0][0]["ConditionNumber"][0]


    # Do the same for neural data
    if area == 'MT':
        num_neurons = len(neuro[0][0][0][0]["Neuron"][0][0])//2
    else:
        num_neurons = len(neuro[0][0][0][0]["Neuron"][0][0])
    spikes_all  = neuro[0][0][0][0]["Neuron"][0][0]
    code_time   = neuro[0][0][0][0]['CodeTimes']
    code_num    = neuro[0][0][0][0]['CodeNumbers']
    code_num    = np.array([i[0] for i in code_num])


    strt_trial_indx = np.where(code_num == STRT_TRIAL)[0]
    strt_trial_time = code_time[strt_trial_indx]

    end_trial_indx = np.where(code_num == END_TRIAL)[0]

    #Get direction from PV filenames
    pv_trials = np.where (condition == 73)[0]
    #correct_trials = np.where(error == 0)[0]
    #pv_0_trials = [value for value in pv_trials if value in correct_trials]
    pv_0_trials = [value for value in pv_trials]
    pass_view_dirs = [[] for i in range(len(pv_0_trials))]

    for i, i_trial in enumerate (pv_0_trials):
        for i_im in im_set:
            if len(bhv[0][0][0][0]["UserVars"][0][i_trial][i_im]) > 0:
                if len(bhv[0][0][0][0]["UserVars"][0][i_trial][i_im][0]) > 4:
                    file_split1 = (bhv[0][0][0][0]["UserVars"][0][i_trial][i_im][0]).split('\\')
                    file_split2 = file_split1[-1].split('_')
                    pass_view_dirs[i].append(int(file_split2[-2])-1)

    pass_view_dirs = [row if len(row) == 5 else [] if len(row) < 2 else row[0:-1] for row in pass_view_dirs]

    #Count number of PV trials per dir
    merged = list(itertools.chain.from_iterable(pass_view_dirs))
    counts = []
    for i_value in range (len(dirs)):
        counts.append(merged.count(i_value))
    #only take Data sets with >= 5  trials per direction
    if all(i >= 10 for i in counts):


        #DMC task
        # Correct trial IDs (DMC)
        correct_inds   = set(np.where(error == 0)[0])
        condition_inds = set(np.where(condition != 73)[0])
        trial_inds = correct_inds.intersection(condition_inds)

        for i_neuron in range(num_neurons):
            spikes_binned_pv = [[] for i in range(len(dirs))]
            spikes_100_pv = [[] for i in range(len(dirs))]
            neuron_spikes = spikes_all[i_neuron].flatten()
            neuron_name   = neuro[0][0][0][0][1][0][0].dtype.names[i_neuron]
            rating        = neuro[0][0][0][0][-1][i_neuron][1][0][0]

            for i, trial_number in enumerate(pv_0_trials):

                n_stimuli = len(pass_view_dirs[i])

                if n_stimuli > 0:

                    stim1_on_indx = np.where(code_num[strt_trial_indx[trial_number]:end_trial_indx[trial_number]] == PV_STIM_ON)[0]
                    if len(stim1_on_indx) > 1:
                        stim1_on_indx = stim1_on_indx[0]
                    stim1_on_indx = stim1_on_indx + strt_trial_indx[trial_number]
                    stim1_on_time = code_time[stim1_on_indx].flatten()[0]

                    for i_stim in range(n_stimuli):

                        stim_on_time = stim1_on_time + (i_stim*600)

                        spks = neuron_spikes[(neuron_spikes - stim_on_time > timepoints[0]) & (neuron_spikes - stim_on_time < timepoints[-1]+1)] - stim_on_time
                        hist_spks = np.histogram(spks, timepoints)[0]

                        dir_num = int(pass_view_dirs[i][i_stim])
                        spikes_binned_pv[dir_num].append(hist_spks)
                        spikes_100_pv[dir_num].append(np.convolve(hist_spks, np.ones(100), 'same'))

            avg_binned = [np.array(i).squeeze().mean(axis=0) for i in spikes_binned_pv]

            pv_flat_avg = np.zeros_like(dirs)
            sem_pv_flat_avg = np.zeros_like(dirs)

            for i_dir in range(12):
                tmp = np.array(spikes_binned_pv[i_dir])*1000
                mean_across_trials = np.mean(tmp, 1)

                pv_flat_avg[i_dir] = np.mean(mean_across_trials)
                sem_pv_flat_avg[i_dir] = stats.sem(mean_across_trials)



            #DMC--------
            # Set up storage for spikes by direction
            spikes_binned_dmc = [[] for i in range(len(dirs))]
            spikes_100_dmc = [[] for i in range(len(dirs))]

            # Loop through trials
            for i_trial in trial_inds:
                if code_num[strt_trial_indx[i_trial]+1] != 14:

                    # Compute when the stimulus for this trial came on
                    stim_on_indx = np.where(code_num[strt_trial_indx[i_trial]:end_trial_indx[i_trial]] == STIM_ON)[0]
                    stim_on_indx = stim_on_indx + strt_trial_indx[i_trial]

                    # Find spike times for this trial based on stim. on index, compute histogram
                    stim_on_time = code_time[stim_on_indx].flatten()[0]
                    spks         = neuron_spikes[(neuron_spikes - stim_on_time > timepoints[0]) & (neuron_spikes - stim_on_time < timepoints[-1]+1)] - stim_on_time
                    hist_spks    = np.histogram(spks, timepoints)[0]

                    # Store the histogram for this trial
                    dir_num = int(np.ceil(condition[i_trial]/6)) - 1
                    spikes_binned_dmc[dir_num].append(hist_spks)
                    spikes_100_dmc[dir_num].append(np.convolve(hist_spks, np.ones(100), 'same'))


            # Take averages by direction
            dmc_avg_spikes = np.array([np.array(i).squeeze().mean(axis=0) for i in spikes_binned_dmc])

            #Take avg across entirety of sample period DMC

            #dmc_sample_avg_spikes = np.array([np.mean(i, axis = 0)*1000 for i in dmc_avg_spikes])
            #sem_dmc_sample_spikes = np.array([stats.sem(i)*1000 for i in dmc_avg_spikes])

            dmc_sample_avg_spikes = np.zeros_like(dirs)
            sem_dmc_sample_spikes = np.zeros_like(dirs)

            for i_dir in range(12):
                tmp = np.array(spikes_binned_dmc[i_dir])*1000
                mean_across_trials_dmc = np.mean(tmp, 1)

                dmc_sample_avg_spikes[i_dir] = np.mean(mean_across_trials_dmc)
                sem_dmc_sample_spikes[i_dir] = stats.sem(mean_across_trials_dmc)


            #Order directions & directional outputs from 0-360

            #order stuff
            ord_ind = np.argsort(dirs)
            dmc_sample_avg_spikes_ord = [dmc_sample_avg_spikes[i] for i in ord_ind]
            dmc_sem_sample_spikes_ord = [sem_dmc_sample_spikes[i] for i in ord_ind]
            pv_flat_avg_ord = [pv_flat_avg[i] for i in ord_ind]
            sem_pv_flat_avg_ord = [sem_pv_flat_avg[i] for i in ord_ind]
            ord_dirs = [dirs[i] for i in ord_ind]

            #find max to make y axis even
            max_fr = np.maximum(np.amax(dmc_sample_avg_spikes_ord),np.amax(pv_flat_avg_ord))

            objects = [str(i) for i in ord_dirs]
            y_pos = np.arange(len(objects))


            if path.exists('E:\\two_boundary\\data\\' + monkey + '\\' + area + '\\good_neurons\\task_responsive\\' + date + '_' + neuron_name + '.mat'): #rating >= 2.5:


                ############# rCTI

                epoch_mean_pv = [[] for i in range(12)]
                for ii in range(12):
                    data = spikes_binned_pv[ii]
                    epoch_mean_pv[ii] = ([np.array(i).squeeze().mean(axis=0)*1000 for i in data])

                epoch_mean_dmc = [[] for i in range(12)]
                for ii in range(12):
                    data = spikes_binned_dmc[ii]
                    epoch_mean_dmc[ii] = ([np.array(i).squeeze().mean(axis=0)*1000 for i in data])

                [WCD_roc_all, WCD_roc_pv] = get_WC_BC_ROC_epoch(epoch_mean_pv, WCD_pairs_indx)
                [BCD_roc_all, BCD_roc_pv] = get_WC_BC_ROC_epoch(epoch_mean_pv, BCD_pairs_indx)

                [WCD_roc_all, WCD_roc_dmc] = get_WC_BC_ROC_epoch(epoch_mean_dmc, WCD_pairs_indx)
                [BCD_roc_all, BCD_roc_dmc] = get_WC_BC_ROC_epoch(epoch_mean_dmc, BCD_pairs_indx)

                rCTI_pv.append(BCD_roc_pv-WCD_roc_pv)
                rCTI_dmc.append(BCD_roc_dmc-WCD_roc_dmc)

                dmc.append(epoch_mean_dmc)
                pv.append(epoch_mean_pv)
                #print(BCD_roc_pv-WCD_roc_pv)
                #print(BCD_roc_dmc-WCD_roc_dmc)

                ###################


                pv_mean_rate.append(np.mean(pv_flat_avg))
                dmc_mean_rate.append(np.mean(dmc_sample_avg_spikes))

                dmc_mean_rate.append(np.mean(dmc_sample_avg_spikes))

                dmc_border.append(np.mean(np.array(dmc_sample_avg_spikes_ord)[border_inds]))
                dmc_center.append(np.mean(np.array(dmc_sample_avg_spikes_ord)[center_inds]))

                corrs.append(np.corrcoef(pv_flat_avg, dmc_sample_avg_spikes)[0][1])

                spikes_binned_dmc_all.append(spikes_100_dmc)
                spikes_binned_pv_all.append(spikes_100_pv)

                '''
                fig = plt.figure(figsize=(5, 3))
                ax = fig.add_axes([1,1,1,1])

                ax.errorbar(y_pos, dmc_sample_avg_spikes_ord, dmc_sem_sample_spikes_ord, fmt='-o', color = 'k', label = 'DMC')
                #ax.errorbar(y_pos, pv_flat_avg_ord, sem_pv_flat_avg_ord,  fmt='--o', color = 'lightslategrey', label = 'PV')
                ax.set(xticks = y_pos, xticklabels = objects, ylim =( 0, max_fr+(max_fr*.3)))

                ax.set_xlabel('Direction', fontsize = 20)
                ax.set_ylabel('Firing rate (Hz)', fontsize = 20)

                #ax.legend(frameon = False, fontsize = 20)
                plt.xticks(fontsize=22, rotation=0)
                plt.yticks(fontsize=22, rotation=0)

                xtick_dirs = [ord_dirs[i] for i in np.arange(1, 11, 3)]
                ax.set_xticks(np.arange(1, 11, 3))
                ax.set(xticklabels = xtick_dirs)

                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax.axvspan(0, 2, alpha = 0.1, color = '#3891A6', zorder=1)
                ax.axvspan(6, 8, alpha = 0.1, color = '#3891A6', zorder=2)

                ax.axvspan(3, 5, alpha = 0.1, color = '#DB6C79', zorder=1)
                ax.axvspan(9, 11, alpha = 0.1, color = '#DB6C79', zorder=2)

                plt.tick_params(labelsize=20)

                #plt.show()

                fname_fig = savepath + 'PV_{}_{}_{}_{}.png'.format(area, monkey_name, date, neuron_name)

                fig.savefig(fname_fig, bbox_inches='tight', dpi=500)

                plt.close()
                '''

            else:
                epoch_mean_dmc = np.nan
                epoch_mean_pv = np.nan

    return pv_mean_rate, dmc_mean_rate, spikes_binned_dmc_all, spikes_binned_pv_all, corrs, dmc, pv
