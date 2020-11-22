import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from analysis_functions import *
from constant_variables import *

def plot_psth(raw_spikes, time_x, fname, sm_std):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_axes([1,1,1,1])

    tmpdirs = np.arange(0, N_CONDS_DMC, 6)
    mean_per_dir = np.zeros((N_DIRS, len(time_x)))
    for i, val in enumerate(tmpdirs):
        mean_per_dir[i] = (gaussian_filter1d(np.nanmean(np.vstack([i for i in raw_spikes[val:val+6] if len(i)> 0]), 0), sm_std))

    for i_dir in range(N_DIRS):
        if i_dir < 6:
            if i_dir == 0:
                ax.plot(time_x, mean_per_dir[i_dir], lw = 2, color = plot_colors['Category1'], label = 'Category 1')
            else:
                ax.plot(time_x, mean_per_dir[i_dir], lw = 2, color = plot_colors['Category1'])
        elif i_dir >= 6:
            if i_dir == 6:
                ax.plot(time_x, mean_per_dir[i_dir], lw = 2, color = plot_colors['Category2'], label = 'Category 2')
            else:
                ax.plot(time_x, mean_per_dir[i_dir], lw = 2, color = plot_colors['Category2'])

    ax.plot([0, 0], [0, 100], 'lightgrey', lw = 1)
    for i in range(4):
        ax.plot([TRIAL_EPOCHS[i], TRIAL_EPOCHS[i]], [0, 100], 'lightgrey', lw = 1)

    max_fr = np.nanmax(mean_per_dir[:, 200:2500])*1.2
    max_y = int(np.ceil(max_fr/5.0)*5.0)
    plt.ylim((0, max_y))
    #plt.ylim((0, max_fr+(max_fr*.1)))
    #plt.xlim((-500, test2))
    plt.xlim((-300, 2300))

    #ax.set_title('Time from sample onset (ms)', fontsize = 15)
    ax.set_xlabel('Time from sample onset (ms)', fontsize = 20)
    ax.set_ylabel('Firing rate (Hz)', fontsize = 20)
    ax.legend(frameon = False, fontsize = 20)

    ax.axvspan(0, TRIAL_EPOCHS[0], alpha = 0.4, color = 'lightgrey', zorder=1)
    ax.axvspan(TRIAL_EPOCHS[1], TRIAL_EPOCHS[2], alpha = 0.4, color = 'lightgrey', zorder=2)

    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tick_params(labelsize=20)

    #plt.show()
    fig.savefig(fname, bbox_inches='tight')

    plt.close(fig)
    '''
    textstr = 'neuron rating: {}\ndir 1: {} trials\ndir 2: {} trials\ndir 3: {} trials\
    \ndir 4: {} trials\ndir 5: {} trials\ndir 6: {} trials\ndir 7: {} trials\
    \ndir 8: {} trials\ndir 9: {} trials\ndir 10: {} trials\ndir 11: {} trials\
    \ndir 12: {} trials'.format(rating, trials_per_dir[0], trials_per_dir[1], trials_per_dir[2],\
    trials_per_dir[3], trials_per_dir[4], trials_per_dir[5], trials_per_dir[6], trials_per_dir[7]\
    , trials_per_dir[8], trials_per_dir[9], trials_per_dir[10], trials_per_dir[11])

    ax.text(-.25, 0.4, textstr, transform=ax.transAxes, fontsize=13)

    plt.subplots_adjust(left=0.25)
    '''

def plot_pv_dmc_violin(input_data, ylim, chance_level, savepath, brain_areas, tasks, save = False):
    violin_colors = [plot_colors['DMC'], plot_colors['PV']]

    fig, ax =  plt.subplots(1, 3, figsize=(20, 6))
    x1, x2 = 1, 2

    for i, area in enumerate(brain_areas):

        data = [input_data[area]['DMC'], input_data[area]['PV']]

        parts = ax[i].violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for ii, pc in enumerate(parts['bodies']):
            pc.set_facecolor(violin_colors[ii])
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

        inds = np.arange(1, len(medians) + 1)
        ax[i].scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax[i].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax[i].vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        ax[i].set_title(area, fontsize = 25, y = 1.08, fontweight='bold')

        ax[i].plot([0, 3], [chance_level, chance_level], '--k', lw = 1.3, zorder = 0)
        ax[i].tick_params(labelsize = 18)

        curr_pval = input_data['pvals'][area]
        sigtext = get_sigtext(curr_pval)

        if abs(np.max(data) + 5)-50 > 5:
            y = np.max(data) + 5
        else:
            y = 55

        h, col = 1.5, 'k'
        ax[i].plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax[i].text((x1+x2)*.5, y+h, sigtext, ha='center', va='bottom', color=col, fontsize = 20)

    # Hide the right and top spines
    for i in range(3):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        #ax[i].set_ylim(0, 0.0016)

    ax[0].set_ylabel('Performance', fontsize = 20)

    ax[0].set_ylim(ylim)
    ax[1].set_ylim(ylim)
    ax[2].set_ylim(ylim)

    # set style for the axes
    labels = ['DMC', 'PV']
    for ax in [ax[0], ax[1], ax[2]]:
        set_axis_style(ax, labels)

    plt.show()

    if save:
        for i in save_filetype:
            savename = savepath + i
            fig.savefig(savename, bbox_inches='tight')

def plot_pv_decoder(input_data, ylim, chance_level, savepath, brain_areas, save = False):

    x = np.arange(-100, 400, 10)
    fig, ax =  plt.subplots(1, 3, figsize=(20, 5))

    for i, area in enumerate(brain_areas):

        for task in tasks:
            currmean = np.mean(input_data[area][task], 0)
            currstd = np.std(input_data[area][task], 0)

            ax[i].plot(x, currmean, color =  'darkslategrey',  lw = 3, label = task)
            ax[i].fill_between(x, currmean-currstd, currmean+currstd, color = plot_colors[task],  lw = 1., alpha = 0.5, zorder=20)

        ax[i].plot([-400, 3000], [chance_level, chance_level], '--k', lw = 1.3)

        ax[i].set_xlim(-100, 400)
        ax[i].set_ylim(ylim)
        ax[i].tick_params(labelsize=25)
        ax[i].set_title(area, fontsize = 25, y=1.08, fontweight='bold')

    # Hide the right and top spines
    for i in range(3):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        #ax[i].set_ylim(0, 0.0016)

    ax[0].set_ylabel('Classifier accuracy (%)', fontsize = 25)
    ax[1].set_xlabel('Time from stimulus onset (ms)', fontsize = 25)
    ax[1].legend(frameon = False, fontsize = 20)

    plt.show()

    if save:
        for i in save_filetype:
            savename = savepath + i
            fig.savefig(savename, bbox_inches='tight')

def plot_pv_tuning_curve(window_mean_dmc, window_mean_pv, window_sem_dmc, window_sem_pv, neuron_name, savefigpath):

    # Order directions & directional outputs from 0-360
    ord_ind = np.argsort(DIRS)
    window_mean_dmc = [window_mean_dmc[i] for i in ord_ind]
    window_sem_dmc = [window_sem_dmc[i] for i in ord_ind]
    window_mean_pv = [window_mean_pv[i] for i in ord_ind]
    window_sem_pv = [window_sem_pv[i] for i in ord_ind]
    ord_dirs = [DIRS[i] for i in ord_ind]

    #find max to make y axis even
    max_fr = np.maximum(np.amax(window_mean_dmc),np.amax(window_mean_pv))

    objects = [str(i) for i in ord_dirs]
    y_pos = np.arange(len(objects))

    ### PLOTTING
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([1,1,1,1])

    ax.errorbar(y_pos, window_mean_dmc, window_sem_dmc, fmt='-o', color = 'k', label = 'DMC')
    ax.errorbar(y_pos, window_mean_pv, window_sem_pv,  fmt='--o', color = 'lightslategrey', label = 'PV')
    #ax.set(xticks = y_pos, xticklabels = objects, ylim =( 0, max_fr+(max_fr*.3)))

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

    for i in save_filetype:
        savepath = savefigpath + neuron_name + '.' + i
        fig.savefig(savepath, bbox_inches='tight')
    plt.close()


def plot_rCTI(rCTI_mean, rCTI_std, n_neurons, brain_areas, bins_rCTI, figpath, monkey, save_fig = False):

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([1,1,1,1])

    ax.plot([0, 0], [0, 100], 'lightgrey', lw = 1)

    for i in range(4):
        ax.plot([TRIAL_EPOCHS[i], TRIAL_EPOCHS[i]], [0, 100], 'lightgrey', lw = 1)

    ax.plot([-500, 3000], [0, 0], '--k', lw = 1.3)
    plt.ylim([-0.015, 0.05])
    plt.xlim([-300, TEST1])

    ax.axvspan(0, SAMPLE_STIM, alpha = 0.4, color = 'lightgrey', zorder=1)
    ax.axvspan(DELAY1, TEST1, alpha = 0.4, color = 'lightgrey', zorder=2)

    for area in brain_areas:

        currmean = rCTI_mean[area]
        currstd = rCTI_std[area]

        ax.plot(bins_rCTI, currmean, color = plot_colors[area],  lw = 3, label = "{0}, n = {1}".format(area, n_neurons[area]))
        ax.fill_between(bins_rCTI, currmean-currstd, currmean+currstd, color = plot_colors[area],  lw = 1., alpha = 0.5, zorder=20)

    ax.set_xlabel('Time from sample onset (ms)', fontsize = 20)
    ax.set_ylabel('mean rCTI', fontsize = 20)
    ax.legend(frameon = False, fontsize = 20)

    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tick_params(labelsize=20)

    if save_fig:
        for i in save_filetype:
            savepath = figpath + 'rCTI\\' + monkey + '_rCTI_sliding_window.' + i
            fig.savefig(savepath, bbox_inches='tight')


def plot_decoder(decoder_all, brain_areas, monkey, bins, save_fig = False):

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([1,1,1,1])

    ax.plot([0, 0], [0, 100], 'lightgrey', lw = 1)
    for i in range(4):
        ax.plot([TRIAL_EPOCHS[i], TRIAL_EPOCHS[i]], [0, 100], 'lightgrey', lw = 1)

    ax.plot([-400, 3000], [50, 50], '--k', lw = 1.3)
    plt.ylim([0, 100])
    plt.xlim([-300, TEST1])
    #plt.xlim([2800, test2])

    ax.axvspan(0, SAMPLE_STIM, alpha = 0.4, color = 'lightgrey', zorder=1)
    ax.axvspan(DELAY1, TEST1, alpha = 0.4, color = 'lightgrey', zorder=2)

    for area in brain_areas:

        currmean = decoder_all[area][0][0]['decoder_mean'][0][0]
        currstd = decoder_all[area][0][0]['decoder_std'][0][0]

        ax.plot(bins, currmean, color = plot_colors[area],  lw = 3, label = area)
        ax.fill_between(bins, currmean-currstd, currmean+currstd, color = plot_colors[area],  lw = 1., alpha = 0.5, zorder=20)


    ax.set_xlabel('Time from sample onset (ms)', fontsize = 20)
    ax.set_ylabel('Decoder accuracy (%)', fontsize = 20)
    ax.legend(frameon = False, fontsize = 20)

    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tick_params(labelsize=20)

    if save_fig:
        for i in save_filetype:
            savepath = figpath + 'category_decoding\\' + monkey + '_cat_decoder_count_matched.' + i
            fig.savefig(savepath, bbox_inches='tight')
