from sklearn.datasets import load_digits
import numpy as np
from Classifiers import MLP_classifier
from Utils import one_hot


def load_MNIST_raw():
    digits = load_digits()
    data = [np.reshape(img, newshape=(1, 8*8)) for img in digits['images']]
    data = [d[0] for d in data]
    labels = digits['target']
    OH_labels = [one_hot(x, 10) for x in labels]
    return data, labels, OH_labels


if __name__ == '__main__':
    X, T, _ = load_MNIST_raw()
    model = MLP_classifier(X, T, 8*8, nr_output=10)
    print(model)
