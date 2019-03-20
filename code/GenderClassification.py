from MovieLensData import load_user_item_matrix, load_gender_vector, load_user_item_matrix_100k
import Classifiers
from Utils import one_hot
import numpy as np



if __name__ == '__main__':
    # load the data, It needs to be in the form N x M where N_i is the ith user and M_j is the jth item. Y, the target,
    # is the gender of every user
    import timeit
    start = timeit.default_timer()

    max_user = 900
    max_item = 1000
    X = load_user_item_matrix_100k(max_user=max_user, max_item=max_item)
    T = load_gender_vector(max_user=max_user)
    #OH_T = [one_hot(int(x), 2) for x in T]
    accs = []
    for i in range(1):
        accs.append(Classifiers.MLP_classifier(X, T, max_item))

    print(accs, "\n", sum(accs)/len(accs))

    stop = timeit.default_timer()
    print('Time: ', stop - start)

