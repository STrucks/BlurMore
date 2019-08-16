import MovieLensData as MD
import Utils
import Classifiers
import numpy as np

def one_million(classifier):
    max_user = 6040
    max_item = 3952
    #X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    X = MD.load_user_item_matrix_1m_masked(file_index=71)
    T = MD.load_user_genre_matrix_1m(one_hot=True, top=5)
    T = np.argwhere(T==1)[:,1]
    print(min(T), max(T))
    """
    Note that we loose class 13 (Romance. it seems that no one has romance as favourite genre. This kinda makes sense 
    because it correlates so much with drama and comedy.
    """
    import collections
    import matplotlib.pyplot as plt
    counter = collections.Counter(T)
    #plt.bar(counter.keys(), counter.values())
    #plt.xlabel("T")
    #plt.ylabel('frequency')
    #plt.show()
    print(counter)

    X = Utils.normalize(X)
    #print(T)
    #X = MD.feature_selection(X, T, f_regression)
    #X = MD.chi2_selection(X, T)
    classifier(X, T, multiclass=True, nr_classes=17)


def one_hundert_k(classifier):
    X = MD.load_user_item_matrix_100k()  # max_user=max_user, max_item=max_item)
    T = MD.load_occupation_vector_100k()  # max_user=max_user)
    #X = MD.chi2_selection(X, T)

    classifier(X, T, multiclass=True)


def one_hundert_k_obfuscated(classifier):
    X = MD.load_user_item_matrix_100k_masked()  # max_user=max_user, max_item=max_item)
    T = MD.load_gender_vector_100k()  # max_user=max_user)
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

    one_million(Classifiers.log_reg)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
