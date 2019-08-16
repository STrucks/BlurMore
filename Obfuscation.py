import MovieLensData as MD
import numpy as np
import Utils
import Classifiers
import matplotlib.pyplot as plt
import scipy.stats as ss


def blurMe_1m():
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    top = -1
    p = 0.01
    dataset = ['ML', 'Fx', 'Li'][0]
    if dataset == 'ML':
        X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
        T = MD.load_gender_vector_1m()  # max_user=max_user)
    elif dataset == 'Fx':
        import FlixsterData as FD
        X, T, _ = FD.load_flixster_data_subset()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
    #X = Utils.normalize(X)

    avg_ratings = np.zeros(shape=X.shape[0])
    for index, user in enumerate(X):
        ratings = []
        for rating in user:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[index] = 0
        else:
            avg_ratings[index] = np.average(ratings)

    """ AVERAGE ACROSS MOVIE
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
    """


    # 1: get the set of most correlated movies, L_f and L_m:
    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]
    print("lists")
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
    id_index, index_id = MD.load_movie_id_index_dict()
    movies = []
    with open("ml-1m/movies.dat", 'r') as f:
        for line in f.readlines():
            movies.append(line.replace("\n", ""))

    for index, val in enumerate(L_m[0:10]):
        print(index, movies[id_index[int(val)]], C_m[index])
    for index, val in enumerate(L_f[0:10]):
        print(index, movies[id_index[int(val)]], C_f[index])

    
    movie_dict = MD.load_movie_id_dictionary_1m()
    print("males")
    for id in L_m:
        print(movie_dict[int(id)])

    print("females")
    for id in L_f:
        print(movie_dict[int(id)])
    """
    print("obfuscation")
    # Now, where we have the two lists, we can start obfuscating the data:
    #X = MD.load_user_item_matrix_1m()
    X_obf = np.copy(X)

    #X = Utils.normalize(X)
    #X_obf = Utils.normalize(X_obf)
    prob_m = []#[p / sum(C_m) for p in C_m]
    prob_f = []#[p / sum(C_f) for p in C_f]
    print("obfuscation")
    for index, user in enumerate(X):
        print(index)
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
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(index)]
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
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(index)]
                    added += 1
                safety_counter += 1

    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-1m/"
        with open(output_file + "blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        import FlixsterData as FD
        output_file = "Flixster/"
        user_id2index, user_index2id = FD.load_user_id_index_dict()
        movie_id2index, movie_index2id = FD.load_movie_id_index_dict()

        with open(output_file + "FX_blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(user_index2id[index_user]) + "::" + str(movie_index2id[index_movie]) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurme_obfuscated_" + str(p) + "_" + sample_mode + "_" + rating_mode + "_top" + str(
                top) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user+1) + "::" + str(index_movie+1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

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
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2
    p = 0.1
    dataset = ['ML', 'Fx', 'Li'][2]
    if dataset == 'ML':
        X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
        T = MD.load_gender_vector_1m()  # max_user=max_user)
    elif dataset == 'Fx':
        import FlixsterData as FD
        X, T, _ = FD.load_flixster_data_subset()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
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
    #X = MD.load_user_item_matrix_1m()
    #np.random.shuffle(X)
    #print(X.shape)
    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index_m = 0
        greedy_index_f = 0
        # print(k)
        added = 0
        if T[index] == 1:
            safety_counter = 0
            while added < k and safety_counter < 1000:
                if greedy_index_m >= len(L_m):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_m[greedy_index_m]
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                greedy_index_m += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id)-1]])
                if rating_count > max_count[int(movie_id)-1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            safety_counter = 0
            while added < k and safety_counter < 1000:
                if greedy_index_f >= len(L_f):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_f[greedy_index_f]
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                greedy_index_f += 1
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
    print(nr_many_ratings)
    nr_remove = total_added/nr_many_ratings

    for user_index, user in enumerate(X):
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:,0], size=(int(nr_remove),), replace=False)
            X_obf[user_index, to_be_removed_indecies] = 0

    # finally, shuffle the user vectors:
    #np.random.shuffle(X_obf)
    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-1m/"
        with open(output_file + "blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        import FlixsterData as FD
        output_file = "Flixster/"
        user_id2index, user_index2id = FD.load_user_id_index_dict()
        movie_id2index, movie_index2id = FD.load_movie_id_index_dict()

        with open(output_file + "FX_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(user_index2id[index_user]) + "::" + str(movie_index2id[index_movie]) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user+1) + "::" + str(index_movie+1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")


    return X_obf


def blurMeAgain():
    import RealFakeData as RF
    top = -1
    rating_mode = "avg"
    sample_mode = list(['random', 'sampled', 'greedy'])[0]
    id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2

    p = 0.01
    # 17 for blurme 10% greedy
    file = 17
    dataset = ['ML', 'Fx', 'Li'][0]
    if dataset == 'ML':
        first_order_data = MD.load_user_item_matrix_1m_masked(file_index=71)  # max_user=max_user, max_item=max_item)
        BlurMeLabels = MD.load_gender_vector_1m()  # max_user=max_user)
        X, T = RF.load_real_fake_data_ML_1m(file_index=file)
    elif dataset == 'Fx':
        import FlixsterData as FD
        X, T = FD.load_flixster_data_subset()
    else:
        X, T = 0, 0
    # X = Utils.normalize(X)

    avg_ratings = np.zeros(shape=first_order_data.shape[0])
    for index, user in enumerate(first_order_data):
        ratings = []
        for rating in user:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[index] = 0
        else:
            avg_ratings[index] = np.average(ratings)

    """ AVERAGE ACROSS MOVIE
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
    """

    # 1: get the set of most correlated movies, L_f and L_m:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
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

    else:
        L_m = coefs[:top, 1]
        R_m = 3952 - coefs[:top, 0]
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0] - top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    # print(R_f)

    id_index, index_id = MD.load_movie_id_index_dict()
    movies = []
    with open("ml-1m/movies.dat", 'r') as f:
        for line in f.readlines():
            movies.append(line.replace("\n", ""))

    for index, val in enumerate(L_m[0:10]):
        print(index, movies[id_index[int(val)]], C_m[index])
    for index, val in enumerate(L_f[0:10]):
        print(index, movies[id_index[int(val)]], C_f[index])

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
    X_obf = np.copy(first_order_data)
    # X = Utils.normalize(X)
    # X_obf = Utils.normalize(X_obf)
    prob_m = [p / sum(C_m) for p in C_m]
    prob_f = [p / sum(C_f) for p in C_f]
    for index, user in enumerate(first_order_data):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        # print(k)
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
                if X_obf[index, int(movie_id) - 1] == 0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(index)]
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
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int(index)]
                    added += 1
                safety_counter += 1

    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-1m/"
        with open(output_file + "blurme_x2_obfuscated_" + str(p) + "_" + sample_mode + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
    elif dataset == 'Fx':
        output_file = "Flixster/"
    else:
        output_file = "whatever/"


    return X_obf


def blurMoreAgain():
    import RealFakeData as RF
    top = -1
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2

    p = 0.05
    # 23 for blurmore 10% greedy; 17 for BlurMe 10% greedy
    file = 17
    dataset = ['ML', 'Fx', 'Li'][0]
    if dataset == 'ML':
        BlurMoreData = MD.load_user_item_matrix_1m_masked(file_index=71)  # max_user=max_user, max_item=max_item)
        BlurMoreLabels = MD.load_gender_vector_1m()  # max_user=max_user)
        X, T = RF.load_real_fake_data_ML_1m(file_index=file)
    elif dataset == 'Fx':
        import FlixsterData as FD
        X, T = FD.load_flixster_data_subset()
    else:
        X, T = 0, 0
    # X = Utils.normalize(X)
    avg_ratings = np.zeros(shape=BlurMoreData.shape[1])
    initial_count = np.zeros(shape=BlurMoreData.shape[1])
    for item_id in range(BlurMoreData.shape[1]):
        ratings = []
        for rating in BlurMoreData[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    #T = MD.load_gender_vector_1m()  # max_user=max_user)
    #X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    #X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
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

    else:
        L_m = coefs[:top, 1]
        R_m = 3952 - coefs[:top, 0]
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0] - top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    id_index, index_id = MD.load_movie_id_index_dict()
    movies = []
    with open("ml-1m/movies.dat", 'r') as f:
        for line in f.readlines():
            movies.append(line.replace("\n", ""))

    for index, val in enumerate(L_m[0:10]):
        print(index, movies[id_index[int(val)]], C_m[index])
    for index, val in enumerate(L_f[0:10]):
        print(index, movies[id_index[int(val)]], C_f[index])

    # Now, where we have the two lists, we can start obfuscating the data:
    #X = MD.load_user_item_matrix_1m()
    # np.random.shuffle(X)
    X_obf = np.copy(BlurMoreData)
    total_added = 0
    for index, user in enumerate(BlurMoreData):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        # print(k)
        added = 0

        safety_counter = 0
        while added < k and safety_counter < 1000:
            if greedy_index >= len(L_f):
                safety_counter = 1000
                continue
            if sample_mode == 'greedy':
                movie_id = L_f[greedy_index]
            if sample_mode == 'random':
                movie_id = L_f[np.random.randint(0, len(L_f))]
            greedy_index += 1
            rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
            if rating_count > max_count[int(movie_id) - 1]:
                continue

            if X_obf[index, int(movie_id) - 1] == 0:
                X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                added += 1
            safety_counter += 1
        total_added += added

    # Now remove ratings from users for movies from the list of movies that corr with obfuscation

    for index, user in enumerate(BlurMoreData):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        # print(k)
        added = 0

        safety_counter = 0
        while added < k and safety_counter < 1000:
            if greedy_index >= len(L_m):
                safety_counter = 1000
                continue
            if sample_mode == 'greedy':
                movie_id = L_m[greedy_index]
            if sample_mode == 'random':
                movie_id = L_m[np.random.randint(0, len(L_m))]
            greedy_index += 1

            if X_obf[index, int(movie_id) - 1] != 0:
                X_obf[index, int(movie_id) - 1] = 0
                added += 1
            safety_counter += 1
        total_added -= added

    output_file = ""
    if dataset == 'ML':
        output_file = "ml-1m/"
        with open(output_file + "blurme_x2_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")
    elif dataset == 'Fx':
        output_file = "Flixster/"
    else:
        output_file = "whatever/"


    return X_obf


def blurMeBetter():
    top = -1
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    p = 0.05
    id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2
    certainty_threshold = 0.8
    dataset = ['ML', 'Fx', 'Li'][0]
    if dataset == 'ML':
        X = MD.load_user_item_matrix_1m()  # max_user=max_user, max_item=max_item)
        T = MD.load_gender_vector_1m()  # max_user=max_user)
    elif dataset == 'Fx':
        import FlixsterData as FD
        X, T, _ = FD.load_flixster_data_subset()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
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
    #X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    #X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    certainty = np.zeros(shape=(len(X),))
    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(X_train[0]))
        avg_coefs += model.coef_[0]
        x_test = X[test]

        class_prob = np.max(model.predict_proba(x_test),axis=1)
        #correct, so that 1 means the classifier is very sure and 0 means it is not sure
        class_prob -= 0.5
        class_prob *= 2
        certainty[test] = class_prob
        # set certainty to 0 for all missclassifications:
        t_pred = model.predict(x_test)
        t_test = T[test]
        for index, (pred, target) in enumerate(zip(t_pred, t_test)):
            #print(pred, target, index, test)
            if pred != target:
                certainty[test[index]] = 0

    """ plot certainty scores
    print("-------------------------")
    import matplotlib.pyplot as plt
    plt.bar(range(0,50), certainty[0:50])
    plt.xlabel("user")
    plt.ylabel("certainty score")
    plt.show()
    """
    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))
    if top == -1:
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

    else:
        L_m = coefs[:top, 1]
        R_m = 3952 - coefs[:top, 0]
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0] - top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    # Now, where we have the two lists, we can start obfuscating the data:
    #X = MD.load_user_item_matrix_1m()
    # np.random.shuffle(X)
    X_obf = np.copy(X)
    total_added = 0
    nr_skipped_users= 0
    for index, user in enumerate(X):
        if certainty[index] < certainty_threshold:
            nr_skipped_users+=1
            print(index, nr_skipped_users)
            continue
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
            while added < k and safety_counter < 1000:
                if greedy_index >= len(L_m):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_m[greedy_index]
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                greedy_index += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            safety_counter = 0
            while added < k and safety_counter < 1000:
                if greedy_index >= len(L_f):
                    safety_counter = 1000
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_f[greedy_index]
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                greedy_index += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                if X_obf[index, int(movie_id) - 1] == 0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        total_added += added
    print("nr of skipped users:", nr_skipped_users)
    # Now remove ratings from users that have more than 200 ratings equally:
    nr_many_ratings = 0
    for user in X:
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            nr_many_ratings += 1
    nr_remove = total_added / nr_many_ratings

    for user_index, user in enumerate(X):
        rating_count = sum([1 if x > 0 else 0 for x in user])
        if rating_count > 200:
            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0], size=(int(nr_remove),),
                                                      replace=False)
            X_obf[user_index, to_be_removed_indecies] = 0

    # finally, shuffle the user vectors:
    # np.random.shuffle(X_obf)
    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-1m/"
        with open(output_file + "blurmebetter_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_c" + str(certainty_threshold) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")
    elif dataset == 'Fx':
        import FlixsterData as FD
        output_file = "Flixster/"
        user_id2index, user_index2id = FD.load_user_id_index_dict()
        movie_id2index, movie_index2id = FD.load_movie_id_index_dict()

        with open(output_file + "FX_blurmebetter_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_c" + str(certainty_threshold) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(user_index2id[index_user]) + "::" + str(movie_index2id[index_movie]) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
    else:
        with open("libimseti/LST_blurmebetter_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_c" + str(certainty_threshold) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user+1) + "::" + str(index_movie+1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    return X_obf

blurMoreAgain()