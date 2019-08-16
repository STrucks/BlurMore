import Classifiers
import RealFakeData as RFData
import numpy as np
import scipy.stats as ss
import MovieLensData as MD
import Utils


def real_vs_fake():
    X, T = RFData.load_real_fake_data_ML_1m(file_index=49)
    #X, T = RFData.load_real_fake_data_ML_100k()
    #print(type(Y[0]))
    # Classifiers.log_reg(X, Y)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    Classifiers.log_reg(X_train, T_train)

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X_train[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X_train, T_train):
        x, t = X_train[train], T_train[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(X_train[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))

    values = coefs[:, 2]
    index_zero = np.where(values == np.min(np.abs(values)))
    top_male = index_zero[0][0]
    top_female = index_zero[0][-1]
    L_m = coefs[:top_male, 1]
    R_m = 3952 - coefs[:top_male, 0]
    C_m = np.abs(coefs[:top_male, 2])
    L_f = coefs[coefs.shape[0] - top_female:, 1]
    L_f = list(reversed(L_f))
    R_f = coefs[coefs.shape[0] - top_female:, 0]
    R_f = list(reversed(R_f))
    C_f = coefs[coefs.shape[0] - top_female:, 2]
    C_f = list(reversed(np.abs(C_f)))
    id_index, index_id = MD.load_movie_id_index_dict()
    movies = []
    with open("ml-1m/movies.dat", 'r') as f:
        for line in f.readlines():
            movies.append(line.replace("\n", ""))

    for index, val in enumerate(L_m[0:min(10,len(L_m))]):
        print(index, movies[id_index[int(val)+1]], C_m[index])
    for index, val in enumerate(L_f[0:min(10,len(L_f))]):
        print(index, movies[id_index[int(val)+1]], C_f[index])


def real_vs_fake_masked():
    X1, T = RFData.load_real_fake_data_ML_1m(file_index=23)
    X2,_ = RFData.load_real_fake_data_ML_1m(file_index=29)
    # print(type(Y[0]))
    # Classifiers.log_reg(X, Y)
    X_train, T_train = X1[0:int(0.8 * len(X1))], T[0:int(0.8 * len(X1))]
    X_test, T_test = X2[int(0.8 * len(X2)):], T[int(0.8 * len(X2)):]
    print(list(X1[0, :]))
    print(list(X2[0, :]))
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


def real_vs_fake_flixster():
    X, T = RFData.load_real_fake_data_flixster(file_index=4)
    # X, T = RFData.load_real_fake_data_ML_100k()
    # print(type(Y[0]))
    # Classifiers.log_reg(X, Y)
    #X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    #X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    Classifiers.log_reg(X, T)

def real_vs_fake_libimseti():
    import LibimSeTiData as LD
    X, T = RFData.load_real_fake_data_libimseti(file_index=11)
    # X, T = RFData.load_real_fake_data_ML_100k()
    # print(type(Y[0]))
    # Classifiers.log_reg(X, Y)
    #X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    #X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    Classifiers.log_reg(X, T)

#real_vs_fake()
real_vs_fake_masked()
