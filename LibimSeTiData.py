import numpy as np


def load_libimseti_data(max_user=400, max_item=66730):
    user_id2gender = load_user_gender()
    X = []
    T = []
    cur_id = "1"
    with open("libimseti/ratings.dat", 'r') as f:
        user = np.zeros(shape=(max_item,))
        for line in f.readlines():
            if len(line)<3:
                continue
            else:
                user_id, item_id, rating = line.split(",")
                if user_id not in user_id2gender:
                    continue
                if int(item_id) >= max_item:
                    continue
                if user_id != cur_id:
                    if sum(user)>0:
                        X.append(user)
                        T.append(user_id2gender[cur_id])
                        if len(X) >= max_user:
                            return np.asarray(X), np.asarray(T)
                    cur_id = user_id
                    user = np.zeros(shape=(max_item,))


                # if the user is not in the id_index dict, it means that it does not have a valid gender or age
                # check if the user and movie is in the subset:

                user[int(item_id)-1] = int(rating)
    return np.asarray(X), np.asarray(T)

def load_libimseti_data2():
    import pandas as pd
    from scipy import sparse as sp
    names = ['user_id', 'item_id', 'rating']
    df_dating = pd.read_csv('libimseti/ratings.dat', sep=',', names=names)
    ratings_df_dating = df_dating.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

    n_users = df_dating.user_id.unique().shape[0]
    n_items = df_dating.item_id.unique().shape[0]

    print(str(n_users), 'users')
    print(str(n_items), 'items')

    item_idx_map = {v: k for k, v in enumerate(df_dating.item_id.unique())}
    user_idx_map = {v: k for k, v in enumerate(df_dating.user_id.unique())}

    df_dating_wo_nan = df_dating.dropna(axis=0)
    ratings_dating = sp.coo_matrix(
    (df_dating_wo_nan.rating,
    (df_dating_wo_nan.user_id.map(user_idx_map),
     df_dating_wo_nan.item_id.map(item_idx_map))),
    shape = (n_users, n_items),
    )
    print(df_dating_wo_nan)


def load_user_id_index_dict():
    """
    loads a dictionary for user id <-> index for users having a valig gender, age and ratings
    :return: id2index, index2id dictionary
    """
    # first we have to find out which users (id) have rated movies
    valid_user_ids = []
    with open("libimseti/ratings.dat", 'r') as f:
        for line in f.readlines():
            if len(line) < 3:
                continue
            else:
                user_id, movie_id, rating = line.split(",")
                #if user_id not in valid_user_ids:
                valid_user_ids.append(user_id)
    valid_user_ids = set(valid_user_ids)
    id2index = {}
    index2id = {}
    index = 0
    with open("libimseti/gender.dat", 'r') as f:
        for line in f.readlines():
            user_id, gender = line.replace("\n", "").split(",")
            if user_id in valid_user_ids:
                if gender == "F" or gender == "M":
                    id2index[user_id] = index
                    index2id[index] = user_id
                    index+=1
    return id2index, index2id


def load_user_gender():
    profiles = {}
    with open("libimseti/gender.dat", 'r') as f:
        for index, line in enumerate(f.readlines()):
            userid, gender = line.replace("\n", "").split(",")
            if gender == "F" or gender == "M":
                profiles[userid] = gender

    return profiles


def load_movies():
    movies = {}
    with open("Flixster/movie.txt", 'r') as f:
        for index, line in enumerate(f.readlines()[1:]):
            line = line.replace("\n", "")
            begin = line.rfind(",")
            movie, movie_id = line[0:begin], line[begin+1:]
            movies[movie_id] = movie
    return movies


def load_movie_id_index_dict():
    """

    :return: id2index, index2id dictionary
    """
    id2index = {}
    index2id = {}
    valid_items = []
    index = 0
    with open("libimseti/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, item_id, rating = line.replace("\n", "").split(",")
            if item_id not in valid_items:
                valid_items.append(item_id)
                id2index[item_id] = index
                index2id[index] = item_id
                index += 1

    return id2index, index2id


def export_subset(nr_users=400):
    X, T = load_libimseti_data(max_user=nr_users)
    #user_id2index, user_index2id = load_user_id_index_dict()
    #movie_id2index, movie_index2id = load_movie_id_index_dict()
    with open("libimseti/subset_" + str(nr_users) + "_ratings.txt", 'w') as f:
        for user_index, user in enumerate(X[0:nr_users, :]):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(int(user_index)+1) + "::" + str(int(index_movie)+1) + "::" + str(rating) + "::000000\n")
    with open("libimseti/subset_" + str(nr_users) + "_gender.txt", 'w') as f:
        for user_index, gender in enumerate(T[0:nr_users]):
            f.write(str(int(user_index) + 1) + "," + gender +"\n")


def load_libimseti_data_subset(file="libimseti/subset_400_ratings.txt", small=False):
    genders = []
    with open("libimseti/subset_400_gender.txt", 'r') as f:
        for line in f.readlines():
            user_index, gender = line.replace("\n", "").split(",")
            genders.append(gender)

    X = np.zeros(shape=(400, 66730))
    T = np.zeros(shape=(400,))
    with open(file, 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            X[user_id-1, movie_id-1] = np.round(rating)
            if genders[user_id-1] == "F":
                T[user_id-1] = 1
            else:
                T[user_id-1] = 0
    if small:
        valid_movie_ind = np.argwhere(np.sum(X, axis=0) != 0)[:,0]
        X = X[:,valid_movie_ind]
        #movie_ids = movie_index2id[valid_movie_ind]
        return X, T, valid_movie_ind
    else:
        return X, T, 0


def load_libimseti_data_subset_masked(file_index=-1, small=False, valid_movies=[]):
    genders = []
    with open("libimseti/subset_400_gender.txt", 'r') as f:
        for line in f.readlines():
            user_index, gender = line.replace("\n", "").split(",")
            genders.append(gender)

    X = np.zeros(shape=(400, 66730))
    T = np.zeros(shape=(400,))

    files = [
        "libimseti/LST_blurme_obfuscated_0.01_greedy_avg_top-1.dat",
        "libimseti/LST_blurme_obfuscated_0.05_greedy_avg_top-1.dat",
        "libimseti/LST_blurme_obfuscated_0.1_greedy_avg_top-1.dat",
        "libimseti/LST_blurmepp_obfuscated_greedy_0.01_2.dat",
        "libimseti/LST_blurmepp_obfuscated_greedy_0.05_2.dat",
        "libimseti/LST_blurmepp_obfuscated_greedy_0.1_2.dat",#5
        "libimseti/LST_blurmebetter_obfuscated_greedy_0.01_2_c0.8.dat",
        "libimseti/LST_blurmebetter_obfuscated_greedy_0.05_2_c0.8.dat",
        "libimseti/LST_blurmebetter_obfuscated_greedy_0.1_2_c0.8.dat",
        "libimseti/LST_blurmebetter_obfuscated_greedy_0.01_2_c0.99.dat",
        "libimseti/LST_blurmebetter_obfuscated_greedy_0.05_2_c0.99.dat",#10
        "libimseti/LST_blurmebetter_obfuscated_greedy_0.1_2_c0.99.dat",
        "libimseti/subset_400_ratings.txt",

    ]
    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            X[user_id-1, movie_id-1] = np.round(rating)
            if genders[user_id-1] == "F":
                T[user_id-1] = 1
            else:
                T[user_id-1] = 0

    if small:
        X = X[:,valid_movies]
        #movie_ids = movie_index2id[valid_movie_ind]
        return X, T, 0#movie_ids
    else:
        return X, T, 0


#print(load_libimseti_data()[0].shape)
#export_subset(nr_users=400)
#print(load_libimseti_data_subset())