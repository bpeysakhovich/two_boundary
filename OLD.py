def category_decoder_PV(n_iter, window, n_neurons_decoder, n_neurons_total, n_trials, all_data):

    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"
    neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)
    clf = SVC(kernel='linear')
    time_points = window
    decoder_timepoints = np.arange(0, len(window)-1, 10)
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


def category_decoder(n_iter, time_points, n_neurons_decoder, n_neurons_total, n_trials, filelist):

    import scipy.io as sio
    import numpy as np
    from sklearn.svm import SVC # "Support Vector Classifier"

    clf = SVC(kernel='linear')
    n_trials_train = int(np.ceil(n_trials*0.75))
    n_trials_test = n_trials - n_trials_train

    train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11], [0, 1, 2, 9, 10, 11], [3, 4, 5, 6, 7, 8]]
    test_dir_all = [[3, 4, 5, 9, 10, 11], [0, 1, 2, 6, 7, 8], [3, 4, 5, 6, 7, 8], [0, 1, 2, 9, 10, 11]]

    perf_all = np.zeros([n_iter, len(time_points)])

    for iteration in range(0, n_iter):
        indx = np.random.randint(0, 4)
        train_dir = train_dir_all[indx]
        test_dir = test_dir_all[indx]

        train_data_c1 = [np.zeros([n_neurons_decoder, 3550]) for i in range(n_trials_train)]
        train_data_c2 = [np.zeros([n_neurons_decoder, 3550]) for i in range(n_trials_train)]

        test_data_c1 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials_test)]
        test_data_c2 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials_test)]

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

            train_trials_c1 = np.random.choice(len(all_trials_train_c1), n_trials_train, replace=False)
            train_trials_c2 = np.random.choice(len(all_trials_train_c2), n_trials_train, replace=False)

            test_trials_c1 = np.random.choice(len(all_trials_test_c1), n_trials_test, replace=False)
            test_trials_c2 = np.random.choice(len(all_trials_test_c2), n_trials_test, replace=False)

            c1_train = np.vstack([all_trials_train_c1[i] for i in train_trials_c1])
            c2_train = np.vstack([all_trials_train_c2[i] for i in train_trials_c2])

            c1_test = np.vstack([all_trials_test_c1[i] for i in test_trials_c1])
            c2_test = np.vstack([all_trials_test_c2[i] for i in test_trials_c2])

            for i_trial in range(n_trials_train):
                train_data_c1[i_trial][i] = c1_train[i_trial]
                train_data_c2[i_trial][i] = c2_train[i_trial]

            for i_trial in range(n_trials_test):
                test_data_c1[i_trial][i] = c1_test[i_trial]
                test_data_c2[i_trial][i] = c2_test[i_trial]


        perf = np.zeros([len(time_points)])

        for i, timepoint in enumerate(time_points):

            x = np.zeros([n_neurons_decoder, n_trials_train*2])
            y = np.zeros([n_trials_train*2])

            for i_trial in range(n_trials_train):
                x[:, i_trial] = train_data_c1[i_trial][:, timepoint]
                y[i_trial] = 1

            for i_trial in range(n_trials_train):
                x[:, i_trial+n_trials_train] = train_data_c2[i_trial][:, timepoint]
                y[i_trial+n_trials_train] = 2

            x_test = np.zeros([n_neurons_decoder, n_trials_test*2])
            y_test = np.zeros([n_trials_test*2])

            for i_trial in range(n_trials_test):
                x_test[:, i_trial] = test_data_c1[i_trial][:, timepoint]
                y_test[i_trial] = 1

            for i_trial in range(n_trials_test):
                x_test[:, i_trial+n_trials_test] = test_data_c2[i_trial][:, timepoint]
                y_test[i_trial+n_trials_test] = 2

            clf.fit(x.T, y)

            if i == 240:
                clf240 = clf

            pred = clf.predict(x_test.T)
            perf[i] = sum(pred == y_test)/len(y_test)*100

        perf_all[iteration] = perf



    return perf_all, clf240





    def direction_decoder(n_iter, time_points, n_neurons_decoder, n_neurons_total, n_trials, filelist):
        import scipy.io as sio
        import numpy as np
        from sklearn.svm import SVC # "Support Vector Classifier"

        clf = SVC(kernel='linear')

        train_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]
        test_dir_all = [[0, 1, 2, 6, 7, 8], [3, 4, 5, 9, 10, 11]]

        perf_all = np.zeros([n_iter, len(time_points)])

        for iteration in range(0, n_iter):
            indx = np.random.randint(0, 2)
            train_dir = train_dir_all[indx]
            test_dir = test_dir_all[indx]

            train_data_c1 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            train_data_c2 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            train_data_c3 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            train_data_c4 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            train_data_c5 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            train_data_c6 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]

            test_data_c1 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            test_data_c2 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            test_data_c3 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            test_data_c4 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            test_data_c5 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]
            test_data_c6 = [np.zeros([n_neurons_decoder, 3549]) for i in range(n_trials)]

            neurons = np.random.choice(n_neurons_total, n_neurons_decoder, replace=False)

            for i, neuron in enumerate(neurons):
                file = filelist[neuron-1]
                data_by_cond = sio.loadmat(file)['binned_spikes'][0]

                tmpdirs = np.arange(0, 72, 6)
                data = [[] for i in range(12)]

                for ii, val in enumerate(tmpdirs):
                    data[ii] = np.vstack([ii for ii in data_by_cond[val:val+6] if len(ii)> 0])

                if min([len(indx) for indx in data]) >= n_trials*2:
                    trials_c1 = np.random.choice(len(data[train_dir[0]]), n_trials*2, replace=False)
                    trials_c2 = np.random.choice(len(data[train_dir[1]]), n_trials*2, replace=False)
                    trials_c3 = np.random.choice(len(data[train_dir[2]]), n_trials*2, replace=False)
                    trials_c4 = np.random.choice(len(data[train_dir[3]]), n_trials*2, replace=False)
                    trials_c5 = np.random.choice(len(data[train_dir[4]]), n_trials*2, replace=False)
                    trials_c6 = np.random.choice(len(data[train_dir[5]]), n_trials*2, replace=False)

                else:
                    trials_c1 = np.random.choice(len(data[train_dir[0]]), n_trials*2, replace=True)
                    trials_c2 = np.random.choice(len(data[train_dir[1]]), n_trials*2, replace=True)
                    trials_c3 = np.random.choice(len(data[train_dir[2]]), n_trials*2, replace=True)
                    trials_c4 = np.random.choice(len(data[train_dir[3]]), n_trials*2, replace=True)
                    trials_c5 = np.random.choice(len(data[train_dir[4]]), n_trials*2, replace=True)
                    trials_c6 = np.random.choice(len(data[train_dir[5]]), n_trials*2, replace=True)

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
