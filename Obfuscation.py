import MovieLensData as MD
import numpy as np
import Utils
import Classifiers
import matplotlib.pyplot as plt
import scipy.stats as ss


def blurMe_1m():
    sample_mode = list(['random', 'sampled', 'greedy'])[0]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    top = -1
    X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    #X = Utils.normalize(X)
    avg_ratings = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)


    # 1: get the set of most correlated movies, L_f and L_m:
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

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
        #print(len(model.coef_[0]),len(X_train[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i+1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    if top == -1:
        values = coefs[:,2]
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

    else:
        L_m = coefs[:top, 1]
        R_m = 3952-coefs[:top, 0]
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0]-top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0]-top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0]-top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    #print(R_f)
    """
    movie_dict = MD.load_movie_id_dictionary_1m()
    print("males")
    for id in L_m:
        print(movie_dict[int(id)])

    print("females")
    for id in L_f:
        print(movie_dict[int(id)])
    """

    # Now, where we have the two lists, we can start obfuscating the data:
    X = MD.load_user_item_matrix_1m()
    X_obf = MD.load_user_item_matrix_1m()
    #X = Utils.normalize(X)
    #X_obf = Utils.normalize(X_obf)
    p = 0.2
    prob_m = [p / sum(C_m) for p in C_m]
    prob_f = [p / sum(C_f) for p in C_f]
    for index, user in enumerate(X):
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        #print(k)
        if T[index] == 1:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                elif sample_mode == 'sampled':
                    movie_id = L_m[np.random.choice(range(len(L_m)), p=prob_m)]
                elif sample_mode == 'greedy':
                    movie_id = L_m[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_m):
                        safety_counter = 100
                if X_obf[index, int(movie_id)-1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id)-1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                elif sample_mode == 'sampled':
                    movie_id = L_f[np.random.choice(range(len(L_f)), p=prob_f)]
                elif sample_mode == 'greedy':
                    movie_id = L_f[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_f):
                        safety_counter = 100
                if X_obf[index, int(movie_id) - 1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id)-1]
                    added += 1
                safety_counter += 1

    # output the data in a file:
    with open("ml-1m/blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(top) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user+1) + "::" + str(index_movie+1) + "::" + str(int(np.round(rating))) + "::000000000\n")
    return X_obf


def blurMe_100k():
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]

    # 1: get the set of most correlated movies, L_f and L_m:
    X = MD.load_user_item_matrix_100k()  # max_user=max_user, max_item=max_item)
    avg_ratings = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)

    T = MD.load_gender_vector_100k()  # max_user=max_user)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []

    for train, test in cv.split(X_train, T_train):
        x, t = X_train[train], T_train[train]
        random_state = np.random.RandomState(0)
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        coefs.append(model.coef_)

    coefs = np.average(coefs, axis=0)[0]
    coefs = [[coefs[i], i + 1] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    L_m = coefs[:10, 1]
    L_f = coefs[coefs.shape[0] - 10:, 1]
    L_f = list(reversed(L_f))

    print(L_f)
    print("------")
    print(L_m)

    """
    movie_dict = MD.load_movie_id_dictionary_1m()
    print("males")
    for id in L_m:
        print(movie_dict[int(id)])

    print("females")
    for id in L_f:
        print(movie_dict[int(id)])
    """

    # Now, where we have the two lists, we can start obfuscating the data:
    X = MD.load_user_item_matrix_100k()
    X_obf = MD.load_user_item_matrix_100k()
    p = 0.1
    prob_m = [p / sum(L_m) for p in L_m]
    prob_f = [p / sum(L_f) for p in L_f]
    for index, user in enumerate(X):
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        #print(k)
        if T[index] == 1:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                elif sample_mode == 'sampled':
                    movie_id = L_m[np.random.choice(range(len(L_m)), p=prob_m)]
                elif sample_mode == 'greedy':
                    movie_id = L_m[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_m):
                        safety_counter = 100
                if X_obf[index, int(movie_id)-1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id)]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                # select a random movie:
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                elif sample_mode == 'sampled':
                    movie_id = L_f[np.random.choice(range(len(L_f)), p=prob_f)]
                elif sample_mode == 'greedy':
                    movie_id = L_f[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_f):
                        safety_counter = 100
                if X_obf[index, int(movie_id) - 1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id)]
                    added += 1
                safety_counter += 1

    # output the data in a file:
    with open("ml-100k/blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + ".csv", 'w') as f:
        f.write("user_id,item_id,rating")
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "," + str(index_movie + 1) + "," + str(int(rating)) + "\n")
    return X_obf


def rating_add_1m():
    # add a percentage of random ratings to a user:
    X = MD.load_user_item_matrix_1m()
    X_obf = MD.load_user_item_matrix_1m()
    percentage = 0.05
    for user_index, user in enumerate(X):
        nr_ratings = 0
        for rating in user:
            if rating > 0:
                nr_ratings += 1

        added = 0
        safety_counter = 0
        while added < nr_ratings*percentage and safety_counter < 100:
            index = np.random.randint(0,len(user))
            if X_obf[user_index, index] > 0:
                safety_counter += 1
                continue
            else:
                X_obf[user_index, index] = np.random.randint(1,6)

    # output the data in a file:
    with open("ml-1m/random_added_obfuscated_" + str(percentage) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                        int(rating)) + "::000000000\n")
    return X_obf


def rating_swap_1m():
    plot = False
    low_bound, high_bound = 100, 1500
    # swap 0 ratings with non zero ratings:
    X = np.transpose(MD.load_user_item_matrix_1m())
    X_obf = np.transpose(MD.load_user_item_matrix_1m())
    nr_ratings = []
    for item in X:
        nr_rating = 0
        for rating in item:
            if rating > 0:
                nr_rating += 1
        nr_ratings.append(nr_rating)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    if plot:
        #plt.subplot(1,2,1)
        ax1.bar(range(1,len(X)+1), nr_ratings)
        ax1.set_xlabel("movie id")
        ax1.set_ylabel("nr ratings")

    # we want to remove ratings from movies that have more than 1500 ratings:
    amount_removed = 0
    for item_index, item in enumerate(X):
        if nr_ratings[item_index] > high_bound:
            indecies = np.argwhere(X[item_index,:] > 0)[:,0]
            indecies = np.random.choice(indecies, size=(nr_ratings[item_index]-high_bound,), replace=False)
            amount_removed += len(indecies)
            for i in indecies:
                X_obf[item_index, i] = 0
    """ To check if the removal is working
    
    nr_ratings = []
    for item in X_obf:
        nr_rating = 0
        for rating in item:
            if rating > 0:
                nr_rating += 1
        nr_ratings.append(nr_rating)
    if plot:
        plt.bar(range(1, len(X) + 1), nr_ratings)
        plt.xlabel("movie id")
        plt.ylabel("nr ratings")
        plt.show()
    
    """
    # now we want to add ratings to movies with a small number of ratings:
    print(np.asarray(nr_ratings))
    indecies = np.argwhere(np.asarray(nr_ratings) < low_bound)[:,0]
    print(indecies)
    nr_few_rated_movies = len(indecies)
    nr_to_be_added = amount_removed/nr_few_rated_movies
    print(nr_to_be_added)
    for item_index, item in enumerate(X):
        if nr_ratings[item_index] < low_bound:
            indecies = np.argwhere(X[item_index,:] == 0)[:,0]
            indecies = np.random.choice(indecies, size=(int(nr_to_be_added),), replace=False)
            for i in indecies:
                X_obf[item_index, i] = np.random.randint(1,6)

    """ To check if the removal and adding is working
    """
    nr_ratings = []
    for item in X_obf:
        nr_rating = 0
        for rating in item:
            if rating > 0:
                nr_rating += 1
        nr_ratings.append(nr_rating)
    if plot:
        #plt.subplot(1,2,2)
        ax2.bar(range(1, len(X) + 1), nr_ratings)
        ax2.set_xlabel("movie id")
        ax2.set_ylabel("nr ratings")
        plt.show()

    X_obf = np.transpose(X_obf)

    # output the data in a file:
    with open("ml-1m/rebalanced_(" + str(low_bound) + "," + str(high_bound) + ").dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                        int(rating)) + "::000000000\n")

    return X_obf


def blurMePP():
    top = -1
    notice_factor = 2
    X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
    # X = Utils.normalize(X)
    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    T = MD.load_gender_vector_1m()  # max_user=max_user)
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

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
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    if top == -1:
        values = coefs[:,2]
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

    else:
        L_m = coefs[:top, 1]
        R_m = 3952-coefs[:top, 0]
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0]-top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0]-top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0]-top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    # Now, where we have the two lists, we can start obfuscating the data:
    X = MD.load_user_item_matrix_1m()
    X_obf = MD.load_user_item_matrix_1m()
    p = 0.05
    total_added = 0
    for index, user in enumerate(X):
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        # print(k)
        added = 0
        if T[index] == 1:
            safety_counter = 0
            while added < k and safety_counter < 100:
                if greedy_index >= len(L_m):
                    safety_counter = 100
                    continue
                movie_id = L_m[greedy_index]
                greedy_index += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id)-1]])
                if rating_count > max_count[int(movie_id)-1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            safety_counter = 0
            while added < k and safety_counter < 100:
                if greedy_index >= len(L_f):
                    safety_counter = 100
                    continue
                movie_id = L_f[greedy_index]
                greedy_index += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 200 ratings equally:
    nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    nr_remove = total_added/nr_many_ratings

    for user_index, user in enumerate(X):
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:,0], size=(int(nr_remove),), replace=False)
            X_obf[user_index, to_be_removed_indecies] = 0
    # output the data in a file:
    with open("ml-1m/blurmepp_obfuscated_" + str(p) + "_" + str(notice_factor) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(
                        str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(int(np.round(rating))) + "::000000000\n")
    return X_obf


blurMePP()