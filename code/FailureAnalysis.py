import MovieLensData as MD
import numpy as np
import Utils
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def few_ratings():
    X = MD.load_user_item_matrix_1m()
    X = Utils.normalize(X)
    T = MD.load_gender_vector_1m()
    X_train, T_train = X[0:int(0.9 * len(X))], T[0:int(0.9 * len(X))]
    X_test, T_test = X[int(0.9 * len(X)):], T[int(0.9 * len(X)):]
    test_data = list(zip(X_test, T_test))

    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    #Utils.ROC_plot(X_test, T_test, model)

    for index, max_rating in enumerate([20, 50, 100, 200]):
        selected_X = []
        selected_T = []
        for user, label in test_data:
            counter = 0
            for rating in user:
                if rating > 0:
                    counter += 1
            if counter <= max_rating:
                selected_X.append(user)
                selected_T.append(label)
        probs = model.predict_proba(selected_X)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(selected_T, preds)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        plt.subplot(2, 2, index+1)
        plt.title('Receiver Operating Characteristic with useres having rated less than ' + str(max_rating) + ' making N=' + str(len(selected_X)))
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    plt.show()


def lot_ratings():
    X = MD.load_user_item_matrix_1m()
    X = Utils.normalize(X)
    T = MD.load_gender_vector_1m()
    X_train, T_train = X[0:int(0.9 * len(X))], T[0:int(0.9 * len(X))]
    X_test, T_test = X[int(0.9 * len(X)):], T[int(0.9 * len(X)):]
    test_data = list(zip(X_test, T_test))

    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    # Utils.ROC_plot(X_test, T_test, model)

    for index, max_rating in enumerate([100, 200, 300, 500]):
        selected_X = []
        selected_T = []
        for user, label in test_data:
            counter = 0
            for rating in user:
                if rating > 0:
                    counter += 1
            if counter > max_rating:
                selected_X.append(user)
                selected_T.append(label)
        probs = model.predict_proba(selected_X)
        preds = probs[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(selected_T, preds)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        plt.subplot(2, 2, index + 1)
        plt.title('Receiver Operating Characteristic with useres having rated more than ' + str(max_rating) + ' making N=' + str(len(selected_X)))
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
    plt.show()

few_ratings()