#from MovieLensData import load_user_item_matrix, load_gender_vector, load_user_item_matrix_100k, load_user_item_matrix_1m, load_gender_vector_1m
import MovieLensData as MD
import Classifiers
from Utils import one_hot
import numpy as np



if __name__ == '__main__':
    # load the data, It needs to be in the form N x M where N_i is the ith user and M_j is the jth item. Y, the target,
    # is the gender of every user
    import timeit
    start = timeit.default_timer()

    max_user = 6040
    max_item = 1682
    X = MD.load_user_item_matrix_1m()#max_user=max_user, max_item=max_item)
    T = MD.load_gender_vector_1m()#max_user=max_user)

    print(X.shape, T.shape)
    #OH_T = [one_hot(int(x), 2) for x in T]

    Classifiers.svm_classifier(X, T)

    accs = []
    for i in range(1):
        accs.append(0)

    print(accs, "\n", sum(accs)/len(accs))

    stop = timeit.default_timer()
    print('Time: ', stop - start)

