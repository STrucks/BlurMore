from chainer import ChainList
from chainer import Chain
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import chainer.functions as F
from chainer import optimizers
from chainer import iterators
import random as r
from chainer.optimizers import Adam
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import numpy as np
from Utils import one_hot


class MLP_classifier():

    def __init__(self, n_output=2, n_hidden=200):
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.mlp = MLP(n_output, n_hidden)

    def fit(self, X_train, T_train):
        self.mlp = MLP(self.n_output, self.n_hidden)
        print("start fitting")
        train = list(zip(X_train, T_train))
        batchsize = 100
        max_label = int(max(T_train)) + 1

        train_iter = iterators.SerialIterator(train, batchsize)

        gpu_id = -1  # Set to -1 if you use CPU
        if gpu_id >= 0:
            self.mlp.to_gpu(gpu_id)
        optimizer = optimizers.Adam(alpha=0.001)
        optimizer.setup(self.mlp)

        max_epoch = 30
        while train_iter.epoch < max_epoch:

            # ---------- One iteration of the training loop ----------
            train_batch = train_iter.next()
            image_train, target_train = concat_examples(train_batch, gpu_id)
            image_train = Variable(image_train).data.astype(np.float32)
            target_train = Variable(target_train).data.astype(np.float32)
            OH_T = np.asarray([one_hot(int(x), max_label) for x in target_train])
            OH_T = Variable(OH_T).data.astype(np.float32)
            # Calculate the prediction of the network
            prediction_train = self.mlp(image_train)
            final_pred = np.zeros(shape=(len(prediction_train),))
            for i in range(len(prediction_train)):
                dummy = list(prediction_train[i].data)
                final_pred[i] = dummy.index(max(dummy))
            # Calculate the loss with MSE
            loss = F.mean_squared_error(prediction_train, OH_T)
            # Calculate the gradients in the network
            self.mlp.cleargrads()
            loss.backward()
            # Update all the trainable parameters
            optimizer.update()

            # --------------------- until here ---------------------

            # Check the validation accuracy of prediction after every epoch
            if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
                # Display the training loss
                print('epoch:{:02d} train_loss:{:.04f}'.format(train_iter.epoch, float(to_cpu(loss.array))))
        return self.mlp

    def predict_proba(self, X):
        X = Variable(X).data.astype(np.float32)
        return self.mlp(X).data

    def predict(self, X):
        X = Variable(X).data.astype(np.float32)
        prediction_train = self.mlp(X)
        final_pred = np.zeros(shape=(len(prediction_train),))
        for i in range(len(prediction_train)):
            dummy = list(prediction_train[i].data)
            final_pred[i] = dummy.index(max(dummy))
        return final_pred


class MLP(Chain):
    """Multilayer perceptron"""
    nr_hidden = 200

    def __init__(self, n_output=2, n_hidden=5):
        #super(MLP, self).__init__(L.Linear(None, n_hidden),L.Linear(n_hidden, n_hidden),L.Linear(n_hidden, n_hidden),L.Linear(n_hidden, n_output))
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_output)
        self.nr_hidden = n_hidden

    def __call__(self, x):
        """
        for layer in self.children():
            x = F.relu(layer(x))
        activation = F.softmax(x).data
        return activation.astype(np.float32)
        """
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return F.softmax(self.l3(h))


class Prior_classifier():

    def __init__(self, nr_classes=2):
        self.majority_class = 0
        self.nr_classes = nr_classes

    def fit(self, X, T):
        self.frequnecies = {}
        for t in T:
            if t not in self.frequnecies:
                self.frequnecies[t] = 0
            else:
                self.frequnecies[t] +=1
        values = [v for v in self.frequnecies.values()]
        index_max = values.index(max(values))
        keys = [key for key in self.frequnecies.keys()]
        self.majority_class = keys[index_max]
        print("majority class:", self.majority_class)
        self.other_classes = list(range(self.nr_classes))
        self.other_classes.remove(self.majority_class)
        self.total_frequency = len(T)

    def predict(self, X):
        Y = []
        for x in X:
            outcome = np.random.randint(0, self.total_frequency)
            if outcome < self.frequnecies[self.majority_class]:
                #print(outcome, self.frequnecies[self.majority_class])
                outcome = self.majority_class
            else:
                outcome = self.other_classes[np.random.randint(0, self.nr_classes-1)]
            Y.append(outcome)
        return np.asarray(Y)

    def predict_proba(self, X):
        Y = np.zeros(shape=(len(X), self.nr_classes))
        Y[:, self.majority_class] = self.frequnecies[self.majority_class]/len(X)
        shared_prob = 1-(self.frequnecies[self.majority_class]/len(X))
        Y[:, self.other_classes] = shared_prob/(self.nr_classes-1)
        return Y


class Random_classifier():
    def __init__(self, nr_classes=2):
        self.nr_classes = nr_classes

    def fit(self, X, T):
        # there is nothing to fit
        return 0

    def predict(self, X):
        Y = []
        for x in X:
            out_come = np.random.randint(0, self.nr_classes)
            Y.append(out_come)
        return Y

    def predict_proba(self, X):
        Y = np.zeros(shape=(len(X), self.nr_classes))
        Y[:, :] = 1/self.nr_classes
        return Y
