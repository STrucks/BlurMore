#from MovieLensData import load_user_item_matrix, load_gender_vector, load_user_item_matrix_100k, load_user_item_matrix_1m, load_gender_vector_1m
import MovieLensData as MD
import Classifiers
from Utils import one_hot
import Utils
import numpy as np
from Utils import feature_selection, normalize, chi2_selection, normalize2
from sklearn.feature_selection import f_regression, f_classif, chi2
import matplotlib.pyplot as plt
import Models


def one_million(classifier):
    X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    # X = MD.load_user_genre_matrix_100k_obfuscated()
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    #X, T = Utils.balance_data(X, T)

    X = Utils.normalize(X)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    # print(X)
    print("before", X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    #X_train, _ = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)

    # X = Utils.normalize(X)
    # X = Utils.standardize(X)
    # X = chi2_selection(X, T)

    classifier(X_train, T_train)
    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    #model = Models.Dominant_Class_Classifier()
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    Utils.ROC_plot(X_test, T_test, model)


def one_million_obfuscated(classifier):
    #X2 = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    X1 = MD.load_user_item_matrix_1m()
    X2 = MD.load_user_item_matrix_1m_masked(file_index=72)  # max_user=max_user, max_item=max_item)

    #X1, T = Utils.balance_data(X1, T)
    #X2, T2 = Utils.balance_data(X2, T)
    print(X1)
    #X1 = Utils.normalize(X1)
    #X2 = Utils.normalize(X2)
    print(X1)
    print(X2)
    X_train, T_train = X1[0:int(0.8 * len(X1))], T[0:int(0.8 * len(X1))]
    X_test, T_test = X2[int(0.8 * len(X2)):], T[int(0.8 * len(X2)):]
    print(list(X1[0,:]))
    print(list(X2[0,:]))
    # print(X)
    print("before", X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    # X_train, _ = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)
    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)

    Utils.ROC_cv_obf(X1, X2, T, model)

    model = LogisticRegression(penalty='l2', random_state=random_state)
    #model.fit(X_train, T_train)
    #Utils.ROC_plot(X_test, T_test, model)


def one_hundert_k(classifier):
    X = MD.load_user_item_matrix_100k()  # max_user=max_user, max_item=max_item)
    #X = MD.load_user_genre_matrix_100k()
    T = MD.load_gender_vector_100k()  # max_user=max_user)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    # print(X)
    print(X_train.shape)
    # X = Utils.remove_significant_features(X, T)
    #X_train = Utils.random_forest_selection(X_train, T_train)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    print(X_train.shape)

    # X = Utils.normalize(X)
    # X = Utils.standardize(X)
    # X = chi2_selection(X, T)

    classifier(X_train, T_train)


def one_hundert_k_obfuscated(classifier):
    X1 = MD.load_user_item_matrix_100k()
    X2 = MD.load_user_item_matrix_100k_masked(file_index=1)  # max_user=max_user, max_item=max_item)
    X3 = MD.load_user_item_matrix_100k_masked(file_index=-1)

    T = MD.load_gender_vector_100k()  # max_user=max_user)
    X_train, T_train = X3[0:int(0.8 * len(X3))], T[0:int(0.8 * len(X3))]
    X_test, T_test = X1[int(0.8 * len(X1)):], T[int(0.8 * len(X1)):]

    from sklearn.linear_model import LogisticRegression
    random_state = np.random.RandomState(0)
    model = LogisticRegression(penalty='l2', random_state=random_state)
    model.fit(X_train, T_train)
    Utils.ROC_plot(X_test, T_test, model)


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

    one_million_obfuscated(Classifiers.log_reg)

    stop = timeit.default_timer()
    print('Time: ', stop - start)


# SVM AUC: 0.79 +- 0.02 CV recall: 0.8173577335277402 CV precision: 0.8321544547101147
# svm rbf AUC: 0.86 +-0.02 CV recall: 0.96028671470078 CV precision: 0.7988469786349818
# log reg AUC: 0.81 CV recall: 0.8323666201934845 CV precision: 0.8386711692966065
# MLP AUC:  CV recall: 0.8549914326156598 CV precision: 0.7159403232729369
# naive bayes AUC: 0.56 +- 0.02 CV recall: 0.23274603292855547 CV precision: 0.8463640876689859
# MN bayes AUC: 0.81 +- 0.03 CV recall: 0.768882302231777 CV precision: 0.8897811875641992
# b bayes AUC: 0.77 +- 0.03 CV recall: 0.47657113057545153 CV precision: 0.8824519431822766



# ------------------- with chi2 feature selection
# SVM AUC: 0.78 +- 0.02 CV recall: 0.8018848245548685 CV precision: 0.8282078648271979 (same for p<0.1 and p<0.001)
# svm rbf AUC: 0.85 +- 0.02 CV recall: 0.9528985430125264 CV precision: 0.8050422639305786
# log reg AUC: 0.80 CV recall: 0.8252141846085077 CV precision: 0.8378892449626741
# MLP AUC:
# naive bayes AUC:
# MN bayes AUC:
# b bayes AUC:

# ------------------ different feature selections:
# chi2 : SVM AUC: 0.78 +- 0.02 CV recall: 0.8018848245548685 CV precision: 0.8282078648271979
# f_classif: SVM AUC: 0.78 +- 0.02 CV recall: 0.8002820319068551 CV precision: 0.8277579831118793
# f_regression SVM AUC: 0.78 +- 0.02 CV recall: 0.8002820319068551 CV precision: 0.8277579831118793

