import MovieLensData as MD
import Classifiers
import numpy as np

def one_million(classifier):
    max_user = 6040
    max_item = 3952
    X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    T = MD.load_occupation_vector_1m()  # max_user=max_user)
    #print(T)
    #X = MD.feature_selection(X, T, f_regression)
    #X = MD.chi2_selection(X, T)
    classifier(X, T, multiclass=True)


def one_hundert_k(classifier):
    X = MD.load_user_item_matrix_100k()  # max_user=max_user, max_item=max_item)
    T = MD.load_occupation_vector_100k()  # max_user=max_user)
    #X = MD.chi2_selection(X, T)

    classifier(X, T, multiclass=True)


def one_hundert_k_obfuscated(classifier):
    X = MD.load_user_item_matrix_100k_masked()  # max_user=max_user, max_item=max_item)
    T = MD.load_gender_vector()  # max_user=max_user)
    #X = MD.chi2_selection(X, T)

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

    one_hundert_k(Classifiers.svm_classifier)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
