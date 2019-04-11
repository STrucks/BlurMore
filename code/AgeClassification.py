#from MovieLensData import load_user_item_matrix, load_gender_vector, load_user_item_matrix_100k, load_user_item_matrix_1m, load_gender_vector_1m
import MovieLensData as MD
import Classifiers
from Utils import one_hot
import numpy as np
from Utils import feature_selection, normalize, chi2_selection, normalize2
from sklearn.feature_selection import f_regression, f_classif


def one_million(classifier):
    max_user = 6040
    max_item = 3952
    X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    T = MD.load_age_vector_1m(boarder=15)  # max_user=max_user)
    X = normalize(X)
    #X = feature_selection(X, T, f_regression)
    #X = chi2_selection(X, T)
    classifier(X, T)


def one_hundert_k(classifier):
    X = MD.load_user_item_matrix_100k()  # max_user=max_user, max_item=max_item)
    #X = normalize(X)
    T = MD.load_age_vector_100k()  # max_user=max_user)
    #X = chi2_selection(X, T)

    classifier(X, T)


def one_hundert_k_obfuscated(classifier):
    X = MD.load_user_item_matrix_100k_masked()  # max_user=max_user, max_item=max_item)
    T = MD.load_age_vector_100k()  # max_user=max_user)
    #X = chi2_selection(X, T)

    classifier(X, T)


if __name__ == '__main__':
    # load the data, It needs to be in the form N x M where N_i is the ith user and M_j is the jth item. Y, the target,
    # is the gender of every user
    import timeit
    start = timeit.default_timer()

    #max_user = 6040
    #max_item = 3952
    #X = MD.load_user_item_matrix_1m()#max_user=max_user, max_item=max_item)
    #T = MD.load_gender_vector_1m()#max_user=max_user)

    #print(X.shape, T.shape)
    #OH_T = [one_hot(int(x), 2) for x in T]
    #Classifiers.log_reg(X, T)
    #Classifiers.MLP_classifier(X, T, max_item)

    one_million(Classifiers.log_reg)

    stop = timeit.default_timer()
    print('Time: ', stop - start)

