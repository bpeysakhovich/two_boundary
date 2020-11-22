global SAMP_ON, TEST_ON, STRT_TRIAL, STIM_ON, PV_STIM_ON, PV_STIM_OFF, END_TRIAL, MS_STIM_ON
global PV_COND
global N_CONDS_DMC
global tasks
global DIRS, n_dirs, CENTER_DIRS
global plot_colors
global save_filetype

# STIMULUS CODE TIMES
SAMP_ON = 23
TEST_ON = 25
STRT_TRIAL = 9
STIM_ON = 23
PV_STIM_ON = 25
PV_STIM_OFF = 26
END_TRIAL = 18
MS_STIM_ON = 14

# TASK TIMING
SAMPLE_STIM = 550
DELAY1 = 1200 + SAMPLE_STIM
TEST1 = 550 + DELAY1
DELAY2 = 175 + TEST1
TEST2 = 550 + DELAY2

TRIAL_EPOCHS = [SAMPLE_STIM, DELAY1, TEST1, DELAY2, TEST2]

# Condition number for passive viewing
PV_COND = 73

# Num. conditions for DMC
N_CONDS_DMC = 72

# Task names
tasks = ['PV', 'DMC']

# STIMULUS DIRECTIONS
DIRS = [247.5, 225, 202.5, 67.5, 45, 22.5, 157.5, 135, 112.5, 337.5, 315, 292.5]
N_DIRS = len(DIRS)

CENTER_DIRS = [1, 4, 7, 10]

# PLOTTING COLORS
plot_colors = {'LIP':'#A7ACD9', 'MST':'#60712F', 'MT':'#FFCA47', 'Category1': '#3891A6', 'Category2': '#DB6C79', 'DMC': 'darkslategrey', 'PV': 'darkgrey'}
plot_colors_SC = {'LIP':'#F95700FF', 'SC':'#00A4CCFF',  'Category1': '#3891A6', 'Category2': '#DB6C79', 'DMC': 'darkslategrey', 'PV': 'darkgrey'}

# Extensions for saving figures
save_filetype = ['png', 'pdf']
