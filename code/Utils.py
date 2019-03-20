import pickle
import matplotlib.pyplot as plt
import numpy as np


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

