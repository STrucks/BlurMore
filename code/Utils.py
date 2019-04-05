import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.fixes import signature

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def confusion_matrix(y, t, size=2, image=False):
    print("Confusion Matrix")
    #print(y,t)

    matrix = np.zeros(shape=(size, size))
    for ys, ts in zip(y, t):
        matrix[int(ys), int(ts)] += 1
    print(matrix)
    if image:
        plt.imshow(matrix, cmap='hot', interpolation='nearest')


def classification_accuracy(Y, T):
    correct = 0
    for y, t in zip(Y,T):
        if y == t:
            correct += 1
    return correct/len(T)


def one_hot(x, length):
    out = np.zeros(shape=(length,))
    out[x] = 1
    return np.asarray(out)


def plot_line(x, y, show=True, legend=[], xlabel="", ylabel="", ):
    plt.plot(x, y)
    if show:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.show()


def performance_measures(Y, T):
    tp, fp, fn, tn = 0, 0, 0, 0

    for y, t in zip(Y,T):
        if y == 0 and t == 0:
            tp += 1
        elif y == 1 and t == 0:
            fn += 1
        elif y == 0 and t == 1:
            fp += 1
        elif y == 1 and t == 1:
            tn += 1

    TPR = tp/(tp+fn) # also called sensitivity/Recall
    FNR = fn/(tp+fn) # also called miss rate
    TNR = tn/(tn+fp) # specificity/selectivity
    FPR = fp/(fp+tn)

    print("tp:", tp, ",fp:", fp, "\nfn:", fn, ",tn:", tn)

    precision = tp/(tp+fp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    return TPR, TNR, FPR, FNR, precision, accuracy


def ROC_cv(X, T, classifier):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from scipy import interp

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=10)

    tprs = []
    aucs = []
    p_r_aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    recalls = []
    precisions = []
    i = 0
    for train, test in cv.split(X, T):
        classifier.fit(X[train], T[train])
        Y = classifier.predict(X[test])
        TPR, TNR, FPR, FNR, precision, accuracy = performance_measures(Y, T[test])
        recalls.append(TPR)
        precisions.append(precision)
        probas_ = classifier.predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(T[test], probas_[:, 1])
        # print("thresholds", thresholds)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        #plt.subplot(1,2,1)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        """
        plt.subplot(1,2,2)
        prec, recall, _ = precision_recall_curve(T[test], Y)

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.plot(recall, prec, color='b', alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        #plt.fill_between(recall, prec, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            np.average(prec)))
        """

        i += 1

    #plt.subplot(1,2,1)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    print("CV recall:", np.average(recalls), "CV precision:", np.average(precisions))