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
from sklearn.model_selection import StratifiedShuffleSplit
from random import sample
from sklearn.svm import SVC # "Support Vector Classifier"
import shutil
import os
from sklearn.metrics import roc_auc_score
import itertools
from os import path
from scipy.ndimage import gaussian_filter1d
from constant_variables import *
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

""" GENERAL UTILITY FUNCTIONS """

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def org_data_by_dir(data_by_cond):

    n_conds_per_dir = int(N_CONDS_DMC/N_DIRS)

    tmpdirs = np.arange(0, N_CONDS_DMC, n_conds_per_dir)
    data_by_dir = [[] for i in range(N_DIRS)]
    for ii, val in enumerate(tmpdirs):
        data_by_dir[ii] = np.vstack([ii for ii in data_by_cond[val:val+n_conds_per_dir] if len(ii)> 0])

    return data_by_dir

def org_data_by_cat(data_by_cond):

    n_conds_per_cat = int(N_CONDS_DMC/2)

    tmpdirs = np.arange(0, N_CONDS_DMC, n_conds_per_cat)
    data_by_cat = [[] for i in range(2)]
    for ii, val in enumerate(tmpdirs):
        data_by_cat[ii] = np.vstack([ii for ii in data_by_cond[val:val+n_conds_per_cat] if len(ii)> 0])

    return data_by_cat

def org_data_by_test(data):

    data_by_test = [[] for i in range(4)]
    data_by_test_and_sample_cat = [[] for i in range(8)]

    # 45, 225, 135, 315
    test_conds = [[2,8,15,20,26,31,40,42,46,48,52,54,58,60,64,66,70,72],
                  [1,7,13,20,26,32,39,41,45,47,51,53,57,59,63,65,69,71],
                  [3,5,9,11,15,17,22,24,28,30,34,36,37,43,49,56,62,68],
                  [4,6,10,12,16,18,21,23,27,29,33,35,38,44,50,55,61,67]]

    test_conds_sample_cat = [[1,7,13,20,26,32], [39,41,45,47,51,53,57,59,63,65,69,71],
                         [2,8,15,20,26,31], [40,42,46,48,52,54,58,60,64,66,70,72],
                         [37,43,49,56,62,68], [3,5,9,11,15,17,22,24,28,30,34,36],
                         [38,44,50,55,61,67], [4,6,10,12,16,18,21,23,27,29,33,35]]

    for i, conds in enumerate(test_conds):
        data_by_test[i] = np.vstack([data[ii-1] for ii in conds if len(data[ii-1]) > 0])

    for i, conds in enumerate(test_conds_sample_cat):
        data_by_test_and_sample_cat[i] = np.vstack([data[ii-1] for ii in conds if len(data[ii-1]) > 0])

    return data_by_test, data_by_test_and_sample_cat


def plot_trial_epochs(TRIAL_EPOCHS, ax):
    ax.plot([0, 0], [0, 100], '--k', lw = 1.3)
    ax.plot([TRIAL_EPOCHS[0], TRIAL_EPOCHS[0]], [0, 100], '--k', lw = 1.3)
    ax.plot([TRIAL_EPOCHS[1], TRIAL_EPOCHS[1]], [0, 100], '--k', lw = 1.3)
    ax.plot([TRIAL_EPOCHS[2], TRIAL_EPOCHS[2]], [0, 100], '--k', lw = 1.3)
    ax.plot([TRIAL_EPOCHS[3], TRIAL_EPOCHS[3]], [0, 100], '--k', lw = 1.3)
    ax.plot([TRIAL_EPOCHS[4], TRIAL_EPOCHS[4]],  [0, 100], '--k', lw = 1.3)

    return ax

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
    for i, val in enumerate(data[:-1]):
        if val >= value and data[i+1] >= value:
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

def shuffle_decoder(data_group1, data_group2, n_perms, brain_areas):

    pvals = {}

    for area in brain_areas:
        true_diff = abs(np.mean(data_group1[area])-np.mean(data_group2[area]))
        all_means = np.zeros(n_perms)

        combined = np.hstack([data_group1[area], data_group2[area]])
        n_samples = int(len(combined)/2)

        for i_perm in range(n_perms):
            perm = np.random.permutation(combined)

            group1 = perm[0:n_samples]
            group2 = perm[n_samples:len(perm)]

            all_means[i_perm] = (np.mean(group1)-np.mean(group2))

        pvals[area] = sum(abs(all_means) >= abs(true_diff))/n_perms

    return pvals

def shuffle_decoder_epoch_pv(data, n_perms, brain_areas, comparison = 'mean'):

    pvals = {}

    for area in brain_areas:
        if comparison == 'mean':
            true_diff = abs(np.mean(data[area]['DMC'])-np.mean(data[area]['PV']))
        elif comparison == 'median':
            true_diff = abs(np.median(data[area]['DMC'])-np.median(data[area]['PV']))

        all_means = np.zeros(n_perms)

        combined = np.hstack([data[area]['DMC'], data[area]['PV']])
        n_samples = int(len(combined)/2)

        for i_perm in range(n_perms):
            perm = np.random.permutation(combined)

            group1 = perm[0:n_samples]
            group2 = perm[n_samples:len(perm)]

            if comparison == 'mean':
                all_means[i_perm] = (np.mean(group1)-np.mean(group2))
            elif comparison == 'median':
                all_means[i_perm] = (np.median(group1)-np.median(group2))

        pvals[area] = sum(abs(all_means) >= abs(true_diff))/n_perms

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


def get_sigtext(curr_pval):
    if curr_pval > 0.05:
        sigtext = 'n.s'
    elif curr_pval <= 0.05 and curr_pval > 0.01:
        sigtext = '*'
    elif curr_pval <= 0.01 and curr_pval > 0.005:
        sigtext = '**'
    elif curr_pval <= 0.005 and curr_pval > 0.005:
        sigtext = '***'
    else:
        sigtext = '****'

    return sigtext

""" PSTH AND TASK_RESPONSIVE CELLS FUNCTIONS """
### Generatae psth
## Output: raw_spikes, mean_psth, mean_psth_sm, psth_binned, trials_per_dir
def get_psth(spikes, errors, trial_starts, conditions, code_numbers, code_times, bins, std = 30, bin_size = 1):

    spikes = np.array(spikes)

    n_conds = 72

    raw_spikes = [[] for i in range(n_conds)]
    psth_binned = [[] for i in range(n_conds)]
    n_trials = len(conditions)


    for i_trial in range(n_trials):

        condition_num = conditions[i_trial]
        dir_num = int(np.ceil(condition_num/6))-1
        cat_num = int(np.floor(condition_num/37))

        trial_start_indx = trial_starts[i_trial]
        trial_start_time = code_times[trial_start_indx]

        if errors[i_trial] == 0 and condition_num != PV_COND and code_numbers[trial_start_indx+1] != MS_STIM_ON and code_numbers[trial_start_indx+1] != 1:

            if trial_start_time > spikes[0] and trial_start_time < spikes[-1]:

                if i_trial != n_trials-1:
                    samp_on_indx = [i+trial_start_indx for i, val in enumerate(code_numbers[trial_start_indx:trial_starts[i_trial+1]]) if val == SAMP_ON][0]
                else:
                    samp_on_indx = [i+trial_start_indx for i, val in enumerate(code_numbers[trial_start_indx:]) if val == SAMP_ON][0]
                samp_on_time = code_times[samp_on_indx]

                t1 = spikes[spikes > samp_on_time-abs(bins[0])]
                t2 =  spikes[spikes < samp_on_time+(bins[-1]+1)]
                spike_times = np.intersect1d(t1, t2) - samp_on_time

                binned = np.histogram(spike_times, bins)[0]
                raw_spikes[condition_num-1].append(binned)

                psth_binned[condition_num-1].append(np.convolve(binned, np.ones(bin_size), 'same'))

    mean_psth = np.zeros([n_conds, len(bins)-1])
    for i, curr_spikes in enumerate(raw_spikes):
        mean_psth[i] = np.mean(curr_spikes, 0)*1000

    n_timepoints = np.shape(mean_psth)[1]

    mean_psth_sm = np.zeros([n_conds, n_timepoints])
    for i_cond, curr_spikes in enumerate(mean_psth):
        mean_psth_sm[i_cond] = gaussian_filter1d(mean_psth[i_cond], std)

    trials_per_dir = np.zeros(N_DIRS)
    for i, val in enumerate(np.arange(0, 72, 6)):
        if len([ii for ii in raw_spikes[val:val+6] if len(ii)> 0]) > 0:
            trials_per_dir[i] = len(np.vstack([ii for ii in raw_spikes[val:val+6] if len(ii)> 0]))
        else:
            trials_per_dir[i] = 0

    return raw_spikes, mean_psth, mean_psth_sm, psth_binned, trials_per_dir


def get_task_responsive_cells(filelist, bins, currfigpath):

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
    indx_samp = epoch_inds(bins, 0, TRIAL_EPOCHS[0])
    indx_delay1 = epoch_inds(bins, TRIAL_EPOCHS[0], TRIAL_EPOCHS[1]-600)
    indx_delay2 = epoch_inds(bins, TRIAL_EPOCHS[1]-600, TRIAL_EPOCHS[1])
    indx_test1 = epoch_inds(bins, TRIAL_EPOCHS[1], TRIAL_EPOCHS[2])

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

        if pval <= 0.05 and max(mean_per_epoch) > 0.5:
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

def category_decoder(all_data, n_iter, timepoints, n_neurons_decoder, n_neurons_total, n_trials_per_dir, equal_n_per_dir = True):

    # equal_n_per_dir: ensure that the training/testing set contains and equal number of trials per direction within a single category
    clf = SVC(kernel='linear')

    # Run decoder every 10ms
    n_timepoints = len(timepoints)-1
    decoder_timepoints = np.arange(0, n_timepoints, 10)

    # Number trials per class (category)
    n_dirs_per_class = 3
    n_trials_train = n_trials_per_dir*n_dirs_per_class

    # Define quadrants for training/testing (4 possible combinations)
    all_configs = {}
    all_configs['train_data_c1'] = [[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]]
    all_configs['train_data_c2'] = [[6, 7, 8], [9, 10, 11], [9, 10, 11], [6, 7, 8]]
    all_configs['test_data_c1'] = [[3, 4, 5], [0, 1, 2], [3, 4, 5], [0, 1, 2]]
    all_configs['test_data_c2'] = [[9, 10, 11], [6, 7, 8], [6, 7, 8], [9, 10, 11]]
    groups = ['train_data_c1', 'train_data_c2', 'test_data_c1', 'test_data_c2']

    # Pre-allocate to save decoder performance
    perf_all = np.zeros([n_iter, len(decoder_timepoints)])

    for iteration in range(0, n_iter):

        # Choose subset of neurons (if sub-sampling)
        neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

        # Choose a train-test config
        indx = np.random.randint(0, 4)
        curr_config = {}
        for ii, group in enumerate(groups):
            curr_config[group] = all_configs[group][indx]

        # Pre-allocate training and testing arrays
        svm_data = {}
        for group in groups:
            svm_data[group] = [np.zeros([n_neurons_decoder, n_trials_train]) for i in range(n_timepoints)]

        for i_neuron, neuron in enumerate(neurons):
            if n_neurons_total == 1:
                data = all_data
            else:
                data = all_data[neuron]

             # Organize neuron's data by category and training/testing trials
            for group in groups:

                if equal_n_per_dir:
                    min_nt = min([len(i) for i in [data[curr_config[group][0]], data[curr_config[group][1]], data[curr_config[group][2]]]])

                    if min_nt >= n_trials_per_dir:
                        inds = np.vstack([np.random.choice(len(data[ii]), n_trials_per_dir, replace= False) for ii in curr_config[group]])
                    else:
                        inds = np.vstack([np.random.choice(len(data[i]), n_trials_per_dir, replace= False) if len(data[i]) >= n_trials_per_dir else np.random.choice(len(data[i]), n_trials_per_dir, replace= True) for i in curr_config[group]])

                    X = np.vstack([data[val][inds[i]] for i, val in enumerate(curr_config[group])])

                else:
                    curr_data = np.vstack([data[curr_config[group][0]], data[curr_config[group][1]], data[curr_config[group][2]]])
                    X, Xt, y, yt = train_test_split(curr_data, np.zeros([len(curr_data)]), train_size = n_trials_per_class)

                if n_timepoints == 1:
                    svm_data[group][0][i_neuron][:] = np.hstack(X)
                else:
                    for ii in range(n_timepoints):
                        svm_data[group][ii][i_neuron][:] = X[:, ii]

        # Run decoder
        perf = np.ones([len(decoder_timepoints)])

        for i_timepoint, timepoint in enumerate(decoder_timepoints):

            x_train = np.hstack([svm_data['train_data_c1'][timepoint], svm_data['train_data_c2'][timepoint]])
            y_train = np.hstack([np.zeros([n_trials_train]), np.ones([n_trials_train])])

            x_test = np.hstack([svm_data['test_data_c1'][timepoint], svm_data['test_data_c2'][timepoint]])
            y_test = np.hstack([np.zeros([n_trials_train]), np.ones([n_trials_train])])

            clf.fit(x_train.T, y_train)

            pred = clf.predict(x_test.T)
            perf[i_timepoint] = sum(pred == y_test)/len(y_test)*100

        perf_all[iteration] = perf

    return perf_all

def direction_decoder(all_data, n_iter, timepoints, n_neurons_decoder, n_neurons_total, n_trials, k_fold = 4, ind_per_cat = False):

    # ind_per_cat:
        # True: independent decoders per category
        # False: include all directions in single decoder
    clf = SVC(kernel='linear')

    # Run decoder every 10ms
    n_timepoints = len(timepoints)-1
    decoder_timepoints = np.arange(0, n_timepoints, 10)

    # Pre-allocate to save decoder performance
    perf_all = np.zeros([n_iter, len(decoder_timepoints)])

    # Define directions for training/testing
    if ind_per_cat:
        all_configs = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
    else:
        all_configs = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    groups = ['train', 'test']

    n_dirs_decoder = len(all_configs[0])
    n_trials_test = int(n_trials/k_fold) * n_dirs_decoder
    n_trials_train = (n_trials - int(n_trials/k_fold)) * n_dirs_decoder

    for iteration in range(n_iter):

        # Choose subset of neurons (if sub-sampling)
        #subsampled_data = sample(all_data, n_neurons_decoder)
        neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

        # Choose category to tets on
        indx = np.random.randint(0, 2)
        curr_dirs = all_configs[indx]

        # Pre-allocate training and testing arrays
        svm_data = {}
        svm_data['train'] = [np.zeros([n_neurons_decoder, n_trials_train]) for i in range(n_timepoints)]
        svm_data['test'] = [np.zeros([n_neurons_decoder, n_trials_test]) for i in range(n_timepoints)]

        for i_neuron, neuron in enumerate(neurons):
            data = all_data[neuron]

            # Organize neuron's data by category and training/testing trials
            X_data, X_data_sorted = {}, {}
            y_data, y_data_sorted = {}, {}

            min_nt = min([len(data[i]) for i in curr_dirs])

            if min_nt >= n_trials:
                inds = np.vstack([np.random.choice(len(data[i]), min_nt, replace= False) for i in curr_dirs])
                Y = np.hstack([np.full(min_nt, i) for i in curr_dirs])
            else:
                inds = np.vstack([np.random.choice(len(data[i]), n_trials, replace= False) if len(data[i]) >= n_trials else np.random.choice(len(data[i]), n_trials, replace= True) for i in curr_dirs])
                Y = np.hstack([np.full(n_trials, i) for i in curr_dirs])

            if n_timepoints == 1:
                X = np.hstack([data[val][inds[i]] for i, val in enumerate(curr_dirs)])
            else:
                X = np.vstack([data[val][inds[i]] for i, val in enumerate(curr_dirs)])

            #X_data['train'], X_data['test'], y_data['train'], y_data['test'] = train_test_split(X, Y, train_size = n_trials_train, test_size = n_trials_test, stratify=Y)

            sss = StratifiedShuffleSplit(n_splits = 1, train_size = n_trials_train, test_size = n_trials_test)
            for train_index, test_index in sss.split(X, Y):
                X_data['train'], X_data['test'] = X[train_index], X[test_index]
                y_data['train'], y_data['test'] = Y[train_index], Y[test_index]

            for group in groups:
                sort = sorted([(e, i) for i, e in enumerate(y_data[group])])
                y_data_sorted[group] = [i[0] for i in sort]
                sorted_inds = [i[1] for i in sort]
                X_data_sorted[group] = X_data[group][sorted_inds]

                if n_timepoints == 1:
                    svm_data[group][0][i_neuron][:] = np.hstack(X_data_sorted[group])
                else:
                    for ii in range(n_timepoints):
                        svm_data[group][ii][i_neuron][:] = X_data_sorted[group][:, ii]

        # Run decoder
        perf = np.zeros([len(decoder_timepoints)])

        for i_timepoint, timepoint in enumerate(decoder_timepoints):

            clf.fit(svm_data['train'][timepoint].T, y_data_sorted['train'])

            pred = clf.predict(svm_data['test'][timepoint].T)
            perf[i_timepoint] = sum(pred == y_data_sorted['test'])/len(y_data_sorted['test'])*100

        perf_all[iteration] = perf


    return perf_all

def run_decoder_pv_epoch(filelist, decoder_type, n_trials, n_iter, timepoints, indx1, indx2):

    # Pre-allocate
    spikes_all = {'PV':[], 'DMC':[]}
    perf_all = {}

    n_neurons = len(filelist)

    for file in filelist:
        data = sio.loadmat(file)

        spikes_all['PV'].append([np.mean(data['spikes_binned_pv'].flatten()[i][:, indx1:indx2], 1) for i in range(N_DIRS)])
        spikes_all['DMC'].append([np.mean(data['spikes_binned_dmc'].flatten()[i][:, indx1:indx2], 1) for i in range(N_DIRS)])

    for task in tasks:
        if decoder_type == 'cat':
            perf_all[task] = np.hstack(category_decoder(spikes_all[task], n_iter, timepoints, n_neurons, n_neurons, n_trials))
        elif decoder_type == 'dir':
            perf_all[task] = np.hstack(direction_decoder(spikes_all[task], n_iter, timepoints, n_neurons, n_neurons, n_trials))

    return perf_all

def run_decoder_dmc(area, filelist, bins, n_iter, decoder_type, n_trials, n_neurons_decoder, n_neurons_total):

    timepoints = np.arange(0, len(bins))

    # Pre-allocate
    decoder_all, decoder_mean, decoder_std = {}, {}, {}
    spikes_all = []

    for file in filelist:
        data = sio.loadmat(file)
        data_by_cond = data['binned_spikes'].flatten()
        data_by_dir = org_data_by_dir(data_by_cond)

        spikes_all.append(data_by_dir)

    if decoder_type == 'cat':
        perf = category_decoder(spikes_all, n_iter, timepoints, n_neurons_decoder, n_neurons_total, n_trials)
    elif decoder_type == 'dir':
        perf = direction_decoder(spikes_all, n_iter, timepoints, n_neurons_decoder, n_neurons_total, n_trials)

    perf_mean = np.mean(perf, 0)
    perf_std = np.std(perf, 0)

    decoder_mean[area] = perf_mean
    decoder_std[area] = perf_std
    decoder_all[area] = perf

    return perf, perf_mean, perf_std

""" ROC ANALYSIS """

def get_WC_BC_ROC(data, pairs_indx, step_size = 10):

    n_timepoints_total = len(data[0][0])
    n_timepoints_stepped = len(np.arange(0, n_timepoints_total, 10))

    n_pairs = len(pairs_indx)

    roc_all = np.zeros([n_pairs, n_timepoints_stepped])

    for i_pair in range(n_pairs):

        dir1_all = data[pairs_indx[i_pair][0]]
        dir2_all = data[pairs_indx[i_pair][1]]

        min_nt = min([len(dir1_all), len(dir2_all)])

        dir1 = dir1_all[np.random.choice(len(dir1_all), min_nt, replace = False), :]
        dir2 = dir2_all[np.random.choice(len(dir2_all), min_nt, replace = False), :]

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

def passive_viewing_analysis_main(fname, window, bin_size, savepath, area, monkey, exclude_stim1, min_ntrials):
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
    trials_pv = np.where(conditions == PV_COND)[0]
    corr_trials_pv = [i for i, val in enumerate(trials_pv) if val in corr_trials]
    [pv_dirs, trial_counts_per_dir] = get_passive_viewing_dirs(bhv, trials_pv, corr_trials_pv)

    # Get DMC trial information
    trials_dmc = np.where(conditions != PV_COND)[0]
    corr_trials_dmc = [i for i in trials_dmc if i in corr_trials]

    # Only take datasets with >= 5 trials per direction
    if all(i >= min_ntrials for i in trial_counts_per_dir):

        for i_neuron in range(n_neurons):
            neuron_name = neuro[0][0][0][0][1][0][0].dtype.names[i_neuron]

            # If neuron is task-responsive
            if path.exists('E:\\two_boundary\\data\\' + monkey + '\\' + area + '\\good_neurons\\task_responsive\\' + date + '_' + neuron_name + '.mat'):

                # Get current neuron's spikes
                spikes = spikes_all[i_neuron].flatten()

                # Passive viewing
                [spikes_raw_pv, spikes_binned_pv, window_mean_pv, window_sem_pv] = get_spikes_pv(spikes, trials_pv, corr_trials_pv, pv_dirs, window, code_times, code_numbers, strt_trial_indx, end_trial_indx, exclude_stim1, bin_size)

                # DMC
                [spikes_raw_dmc, spikes_binned_dmc, window_mean_dmc, window_sem_dmc] = get_spikes_dmc(spikes, corr_trials_dmc, pv_dirs, window, code_times, code_numbers, conditions, strt_trial_indx, end_trial_indx, strt_trial_time, exclude_stim1, bin_size)

                if exclude_stim1:
                    curr_path = savepath + 'stim1_excluded\\' + 'window_' + str(window[0]) + '_' + str(window[-1]) + '\\'
                else:
                    curr_path = savepath + 'stim1_included\\'+ 'window_' + str(window[0]) + '_' + str(window[-1]) + '\\'

                make_dirs(curr_path)
                fname_data = curr_path + date + '_' + neuron_name + '.mat'

                sp.io.savemat(fname_data,  {'spikes_raw_pv': spikes_raw_pv, 'spikes_binned_pv': spikes_binned_pv, 'window_mean_pv': window_mean_pv, 'window_sem_pv': window_sem_pv,
                                            'spikes_raw_dmc': spikes_raw_dmc, 'spikes_binned_dmc': spikes_binned_dmc, 'window_mean_dmc': window_mean_dmc, 'window_sem_dmc': window_sem_dmc})

def get_spikes_dmc(spikes, corr_trials_dmc, pv_dirs, window, code_times, code_numbers, conditions, strt_trial_indx, end_trial_indx, strt_trial_time, exclude_stim1, bin_size):

   # Pre-allocate
    spikes_raw_dmc = [[] for i in range(N_DIRS)]
    spikes_binned_dmc = [[] for i in range(N_DIRS)]

    # Loop through trials
    for i_trial in corr_trials_dmc:

        if code_numbers[strt_trial_indx[i_trial]+1] != MS_STIM_ON:

            if (strt_trial_time[i_trial] > spikes[0]) and (strt_trial_time[i_trial] < spikes[-1]):

                # Compute when the stimulus for this trial came on
                stim_on_indx = np.where(code_numbers[strt_trial_indx[i_trial]:end_trial_indx[i_trial]] == SAMP_ON)[0]
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

            window_mean_dmc = np.zeros_like(DIRS)
            window_sem_dmc = np.zeros_like(DIRS)

            for i_dir in range(N_DIRS):
                tmp = np.array(spikes_raw_dmc[i_dir])*1000
                mean_across_trials_dmc = np.mean(tmp, 0)

                window_mean_dmc[i_dir] = np.mean(mean_across_trials_dmc)
                window_sem_dmc[i_dir] = stats.sem(mean_across_trials_dmc)

    return spikes_raw_dmc, spikes_binned_dmc, window_mean_dmc, window_sem_dmc

def get_spikes_pv(spikes, trials_pv, corr_trials_pv, pv_dirs, window, code_times, code_numbers, strt_trial_indx, end_trial_indx, exclude_stim1, bin_size):

    # Pre-allocate
    spikes_raw_pv = [[] for i in range(N_DIRS)]
    spikes_binned_pv = [[] for i in range(N_DIRS)]

    for i_trial, trial_num in enumerate(trials_pv):
        curr_stims = pv_dirs[i_trial]
        stim_on_inds = np.where(code_numbers[strt_trial_indx[trial_num]:end_trial_indx[trial_num]] == PV_STIM_ON)[0]
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
    window_mean_pv = np.zeros_like(DIRS)
    window_sem_pv = np.zeros_like(DIRS)

    for i_dir in range(N_DIRS):
        tmp = np.array(spikes_raw_mean[i_dir])
        mean_across_trials = np.mean(tmp, 0)*1000

        window_mean_pv[i_dir] = np.mean(mean_across_trials)
        window_sem_pv[i_dir] = stats.sem(mean_across_trials)

    return spikes_raw_pv, spikes_binned_pv, window_mean_pv, window_sem_pv

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
                    if len(file_split2) == 5:
                        pv_dirs[i].append(int(file_split2[-1][:-8]))
                    else:
                        pv_dirs[i].append(int(file_split2[-2])-1)

    # Count number of PV trials per dir
    merged = list(itertools.chain.from_iterable(pv_dirs))
    counts = []
    for i_value in range(N_DIRS):
        counts.append(merged.count(i_value))

    return pv_dirs, counts
