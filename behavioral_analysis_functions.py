import numpy as np
import scipy.io as sio
from constant_variables import *

# ordered by num
ord_ind = np.argsort(DIRS)
ord_dirs = [DIRS[i] for i in ord_ind]

# Get mean accuracy per direction, per session
def perf_per_dir(filelist):

    n_files = len(filelist)

    # Pre-allocate to store  accuracy per direction directions for each file
    acc_all = [np.zeros([N_DIRS]) for i in range(n_files)]

    for i_file, file in enumerate(filelist):

        acc = np.zeros([2, N_DIRS])

        bhv = sio.loadmat(file)['data']['BHV']
        code_numbers = bhv[0][0][0][0][7][0]
        conditions = bhv[0][0][0][0][3][0]
        errors = bhv[0][0][0][0][5]
        dir_nums = [int(np.ceil(i/6))-1 for i in conditions]

        n_trials = len(errors)

        for i_trial in range(n_trials):
            if code_numbers[i_trial][1][0] != MS_STIM_ON and conditions[i_trial] != PV_COND:
                if errors[i_trial] == 0:
                    acc[0, dir_nums[i_trial]] += 1
                    acc[1, dir_nums[i_trial]] += 1
                elif errors[i_trial] == 6:
                    acc[1, dir_nums[i_trial]] += 1

        acc_all[i_file] = acc[0]/acc[1]*100

    acc_mean = np.mean(acc_all, 0)
    acc_std = np.std(acc_all, 0)

    acc_mean = [x for _,x in sorted(zip(DIRS, acc_mean))]
    acc_std = [x for _,x in sorted(zip(DIRS, acc_std))]

    return acc_all, acc_mean, acc_std

# Get accuracy when test stimulus is in same vs different quadrant as sample stimulus
def beh_same_opp_quad(filelist):

    accuracies = [[[] for i in range (2)] for j in range(len(filelist))]

    for fn in filelist:

        data = sio.loadmat(fn)['data']
        bhv   = data['BHV']

        # Load the file specified by fn
        monkey_name = fn[0:7]
        date        = fn[8:18]

        # Extract relevant behavioral variables for all the trials
        error     = [i[0] for i in bhv[0][0][0][0]["TrialError"]]
        trial_num = len(bhv[0][0][0][0]["TrialNumber"][0])
        condition = bhv[0][0][0][0]["ConditionNumber"][0]

        error_match_sameqd = []
        error_match_oppqd = []

        #define  and group match conditions
        cond_test1_sameqd = [1,7,13,19,25,31,38,44,50,56,62,68]
        cond_test1_oppqd = [2,8,14,20,26,32,37,43,49,55,61,67]

        #pull only complete trials
        for i in range(trial_num):
            if error[i] ==5 or error[i] == 6 or error[i]==0:
                if condition[i] in cond_test1_sameqd:
                    error_match_sameqd.append(error[i])
                if condition[i] in cond_test1_oppqd:
                    error_match_oppqd.append(error[i])

        #compute accuracies per session
        accuracy_sameqd = error_match_sameqd.count(0)/len(error_match_sameqd)
        accuracy_oppqd = error_match_oppqd.count(0)/len(error_match_oppqd)

        ind = filelist.index(fn)
        accuracies[ind][0].append(accuracy_sameqd)
        accuracies[ind][1].append(accuracy_oppqd)

    accuracies = (np.asarray(accuracies)).T

    return accuracies
