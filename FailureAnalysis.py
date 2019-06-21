import MovieLensData as MD
import numpy as np
import Utils
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def few_ratings():
    #X = MD.load_user_item_matrix_1m()
    X = MD.load_user_item_matrix_1m()
    X = Utils.normalize(X)
    #T = MD.load_gender_vector_1m()
    T = MD.load_gender_vector_1m()
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]
    test_data = list(zip(X_test, T_test))

    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    #Utils.ROC_plot(X_test, T_test, model)
    roc = True
    min_rating = [0, 27, 282, 390]
    for index, max_rating in enumerate([26, 33, 389, 1000]):
        selected_X = []
        selected_T = []
        for user, label in test_data:
            counter = 0
            for rating in user:
                if rating > 0:
                    counter += 1
            if min_rating[index] <= counter <= max_rating:
                selected_X.append(user)
                selected_T.append(label)
        # resample:

        """
        sampled_X = []
        sampled_T = []
        for i in range(1000):
            sample_id = np.random.randint(len(selected_X))
            sampled_X.append(selected_X[sample_id])
            sampled_T.append(selected_T[sample_id])
        selected_X = sampled_X
        selected_T = sampled_T
        if len(selected_X)> 100:
            data = list(zip(selected_X, selected_T))
            np.random.shuffle(data)
            selected_X = []
            selected_T = []
            for x, y in list(data)[:100]:
                selected_X.append(x)
                selected_T.append(y)
        """
        probs = model.predict_proba(selected_X)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(selected_T, preds)
        roc_auc = metrics.auc(fpr, tpr)
        if roc:
            # method I: plt
            plt.subplot(2, 2, index+1)
            plt.title('Receiver Operating Characteristic with users having rated between ' + str(min_rating[index]) + " and " + str(max_rating) + ' making N=' + str(len(selected_X)))
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

        # print the confusion matrix:
        print("For max rating =", max_rating, ":")
        Y = model.predict(selected_X)
        TPR, TNR, FPR, FNR, precision, accuracy = Utils.performance_measures(Y, T)
        print("TPR:", TPR, "TNR:", TNR, "FPR:", FPR, "FNR:", FNR, "precision:", precision, "accuracy:", accuracy)
    if roc:
        plt.show()


def lot_ratings():
    X = MD.load_user_item_matrix_1m()
    X = Utils.normalize(X)
    T = MD.load_gender_vector_1m()
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]
    test_data = list(zip(X_test, T_test))

    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    # Utils.ROC_plot(X_test, T_test, model)
    roc = True
    min_rating = [162, 211, 282, 390]

    for index, max_rating in enumerate([210, 281, 389, 1000]):
        selected_X = []
        selected_T = []
        for user, label in test_data:
            counter = 0
            for rating in user:
                if rating > 0:
                    counter += 1
            if min_rating[index] <= counter <= max_rating:
                selected_X.append(user)
                selected_T.append(label)
        probs = model.predict_proba(selected_X)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(selected_T, preds)
        roc_auc = metrics.auc(fpr, tpr)

        if roc:
            # method I: plt
            plt.subplot(2, 2, index + 1)
            plt.title(
                'Receiver Operating Characteristic with users having rated between ' + str(max_rating) + " and " + str(
                    min_rating[index]) + ' making N=' + str(len(selected_X)))
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
        # print the confusion matrix:
        print("For max rating =", max_rating, ":")
        Y = model.predict(selected_X)
        TPR, TNR, FPR, FNR, precision, accuracy = Utils.performance_measures(Y, T)
        print("TPR:", TPR, "TNR:", TNR, "FPR:", FPR, "FNR:", FNR, "precision:", precision, "accuracy:", accuracy)

    if roc:
        plt.show()


def loyal_ratings():
    X = MD.load_user_item_matrix_1m()
    X = Utils.normalize(X)
    T = MD.load_gender_vector_1m()
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]
    test_data = list(zip(X_test, T_test))
    Test_indecies = range(int(0.8 * len(X)), len(X))
    print(len(X_test))
    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    # Utils.ROC_plot(X_test, T_test, model)
    roc = True
    upper_bound = [0.17,0.19,0.42,1]
    for index, percent_loyal in enumerate([0.0, 0.17, 0.35, 0.42]):
        test_ids = [i+1 for i in Test_indecies]
        selected_ids = Utils.is_loyal(test_ids, loyal_percent_lower=percent_loyal, loyal_percent_upper=upper_bound[index])
        selected_indecies = [i-1 for i in selected_ids]
        selected_X = X[selected_indecies]
        selected_T = T[selected_indecies]

        probs = model.predict_proba(selected_X)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(selected_T, preds)
        roc_auc = metrics.auc(fpr, tpr)

        if roc:
            # method I: plt
            plt.subplot(2, 2, index + 1)
            plt.title('Receiver Operating Characteristic with users having a loyality between ' + str(
                percent_loyal) + ' and ' + str(upper_bound[index]) + ' making N=' + str(len(selected_X)))
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
        # print the confusion matrix:
        print("For loyality =", percent_loyal, ":")
        Y = model.predict(selected_X)
        TPR, TNR, FPR, FNR, precision, accuracy = Utils.performance_measures(Y, T)
        print("TPR:", TPR, "TNR:", TNR, "FPR:", FPR, "FNR:", FNR, "precision:", precision, "accuracy:", accuracy)
    if roc:
        plt.show()


loyal_ratings()