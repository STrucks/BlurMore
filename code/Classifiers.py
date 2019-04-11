from Utils import confusion_matrix, classification_accuracy, one_hot, plot_line, performance_measures, ROC_cv, ROC_multiclass
from sklearn.model_selection import cross_val_score
from Models import MLP, MLP_classifier
import numpy as np


def svm_classifier(X, T):
    from sklearn.svm import SVC
    X_train, T_train = X[0:int(0.9 * len(X))], T[0:int(0.9 * len(X))]
    X_test, T_test = X[int(0.9 * len(X)):], T[int(0.9 * len(X)):]

    random_state = np.random.RandomState(0)
    classifier = SVC(kernel='linear', probability=True, random_state=random_state)
    #classifier= SVC(kernel='rbf', C=1, probability=True, random_state=random_state)

    ROC_cv(X, T, classifier)

    #return scores.mean()
    return 0


def log_reg(X, T, multiclass = False):
    X_train, T_train = X[0:int(0.9 * len(X))], T[0:int(0.9 * len(X))]
    X_test, T_test = X[int(0.9 * len(X)):], T[int(0.9 * len(X)):]
    from sklearn.linear_model import LogisticRegression

    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    if multiclass:
        ROC_multiclass(X, T, model)
    else:
        ROC_cv(X, T, model)

    return 0 #scores.mean()


def mlp_classifier(X, T):
    nr_output = 2

    model = MLP_classifier(n_hidden=500, n_output=nr_output)
    ROC_cv(X, T, model)


def MLP_classifier2(X, T, n_items, nr_output=2):
    from chainer.optimizers import Adam
    from chainer import Variable
    import chainer.functions as F
    from chainer import optimizers
    from chainer import iterators
    import random as r
    data = list(zip(X, T))
    r.shuffle(data)
    train = data[0:int(len(data)*0.8)]
    test = data[int(len(data)*0.8):]
    print(len(train), len(test))
    batchsize = 100
    max_label = int(max(T))+1

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

    model = MLP(n_hidden=int(n_items/2), n_output=nr_output)
    gpu_id = -1  # Set to -1 if you use CPU
    if gpu_id >= 0:
        model.to_gpu(gpu_id)
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    import numpy as np
    from chainer.dataset import concat_examples
    from chainer.backends.cuda import to_cpu

    max_epoch = 30
    train_accs = []
    train_accs_epochs = []
    val_acc_epochs = []
    while train_iter.epoch < max_epoch:

        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        image_train, target_train = concat_examples(train_batch, gpu_id)
        image_train = Variable(image_train).data.astype(np.float32)
        target_train = Variable(target_train).data.astype(np.float32)
        OH_T = np.asarray([one_hot(int(x), max_label) for x in target_train])
        OH_T = Variable(OH_T).data.astype(np.float32)
        # Calculate the prediction of the network
        prediction_train = model(image_train)
        final_pred = np.zeros(shape=(len(prediction_train),))
        for i in range(len(prediction_train)):
            dummy = list(prediction_train[i].data)
            final_pred[i] = dummy.index(max(dummy))
        # Calculate the loss with MSE
        loss = F.mean_squared_error(prediction_train, OH_T)
        # Calculate the gradients in the network
        model.cleargrads()
        loss.backward()
        # Update all the trainable parameters
        optimizer.update()

        train_acc = classification_accuracy(final_pred, target_train)
        train_accs.append(train_acc)
        # --------------------- until here ---------------------

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch

            # Display the training loss
            print('epoch:{:02d} train_loss:{:.04f}'.format(train_iter.epoch, float(to_cpu(loss.array))), end='')
            print(" train acc: {:f} ".format(sum(train_accs)/len(train_accs)), end='')
            train_accs_epochs.append(sum(train_accs)/len(train_accs))
            train_accs = []
            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                image_test, target_test = concat_examples(test_batch, gpu_id)
                image_test = Variable(image_test).data.astype(np.float32)
                target_test = Variable(target_test).data.astype(np.float32)
                OH_T = np.asarray([one_hot(int(x), max_label) for x in target_test])
                OH_T = Variable(OH_T).data.astype(np.float32)

                target_test = Variable(target_test).data.astype(np.float32)
                # Forward the test data
                prediction_test = model(image_test)
                final_pred = np.zeros(shape=(len(prediction_test),))
                for i in range(len(prediction_test)):
                    dummy = list(prediction_test[i].data)
                    final_pred[i] = dummy.index(max(dummy))

                # Calculate the loss
                loss_test = F.mean_squared_error(prediction_test, OH_T)
                #loss_test = F.mean_squared_error(prediction_test, OH_T)
                test_losses.append(to_cpu(loss_test.array))

                # Calculate the accuracy
                #prediction_test = Variable(prediction_test).data.astype(np.int)
                target_test = Variable(target_test).data.astype(np.int)

                accuracy = classification_accuracy(final_pred, target_test.data)
                #print(prediction_test, target_test)
                test_accuracies.append(accuracy)
                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(np.mean(test_losses), np.mean(test_accuracies)))
            val_acc_epochs.append(np.mean(test_accuracies))
    confusion_matrix(final_pred, target_test.data, size=max_label)

    #print(train_accs_epochs, val_acc_epochs)
    #plot_line(range(max_epoch), train_accs_epochs, show=False)
    #plot_line(range(max_epoch), val_acc_epochs, legend=['train accuracy', 'validation accuracy'], xlabel='epoch', ylabel='accuracy')
    X_train, T_train = X[0:int(0.9 * len(X))], T[0:int(0.9 * len(X))]
    X_test, T_test = X[int(0.9 * len(X)):], T[int(0.9 * len(X)):]
    X_test = Variable(X_test).data.astype(np.float32)
    plot_ROC = True
    if plot_ROC:
        import sklearn.metrics as metrics
        # calculate the fpr and tpr for all thresholds of the classification
        preds = model(X_test)
        final_pred = np.zeros(shape=(len(preds),))
        for i in range(len(preds)):
            dummy = list(preds[i].data)
            final_pred[i] = int(dummy.index(max(dummy)))
        #print(len(T_test), len(final_pred), len(X_test))
        fpr, tpr, threshold = metrics.roc_curve(T_test, final_pred)
        roc_auc = metrics.auc(fpr, tpr)

        # method I: plt
        import matplotlib.pyplot as plt
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return np.mean(test_accuracies)


def random_forest(X, T):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    ROC_cv(X, T, model)
    return 0


def naive_bayes(X, T):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    ROC_cv(X, T, model)
    return 0


def multinomial_bayes(X, T):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    ROC_cv(X, T, model)
    return 0


def bernoulli_bayes(X, T):
    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB()
    for i, row in enumerate(X):
        for j, value in enumerate(row):
            if value > 0:
                X[i,j] = 1

    ROC_cv(X, T, model)
    return 0


def prior(X, T):
    from Models import Prior_classifier
    model = Prior_classifier()
    ROC_cv(X, T, model)
    return 0


def random(X, T):
    from Models import Random_classifier
    model = Random_classifier()
    ROC_cv(X, T, model)
    return 0