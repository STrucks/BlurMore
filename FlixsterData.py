import numpy as np


def load_flixster_data(max_user=400, max_item=66730):
    user_id2index, user_index2id = load_user_id_index_dict()
    movie_id2index, movie_index2id = load_movie_id_index_dict()
    user_id2gender = load_user_gender()
    X = np.zeros(shape=(max_user, max_item))
    T = np.zeros(shape=(max_user,))
    with open("Flixster/ratings.txt", 'r', encoding='utf-16') as f:
        for line in f.readlines()[1:]:
            if len(line)<3:
                continue
            else:
                user_id, movie_id, rating, _ = line.split("\t")
                #print(user_id, )
                # if the user is not in the id_index dict, it means that it does not have a valid gender or age
                if user_id in user_id2index.keys():
                    # check if the user and movie is in the subset:
                    if user_id2index[user_id] < max_user and movie_id2index[movie_id]< max_item:
                        #print(user_id2index[user_id], movie_id2index[movie_id])
                        X[user_id2index[user_id], movie_id2index[movie_id]] = np.round(float(rating))
                        if user_id2gender[user_id] == "Female":
                            T[user_id2index[user_id]] = 1
                        else:
                            T[user_id2index[user_id]] = 0
    return X, T


def load_user_id_index_dict():
    """
    loads a dictionary for user id <-> index for users having a valig gender, age and ratings
    :return: id2index, index2id dictionary
    """
    # first we have to find out which users (id) have rated movies
    valid_user_ids = []
    with open("Flixster/ratings.txt", 'r', encoding='utf-16') as f:
        for line in f.readlines()[1:]:
            if len(line) < 3:
                continue
            else:
                user_id, movie_id, rating, _ = line.split("\t")
                #if user_id not in valid_user_ids:
                valid_user_ids.append(user_id)
    valid_user_ids = set(valid_user_ids)
    id2index = {}
    index2id = {}
    index = 0
    with open("Flixster/profile.txt", 'r') as f:
        for line in f.readlines()[1:]:
            user_id, gender, _, _, _, _, age = line.replace("\n","").split(",")
            if user_id in valid_user_ids:
                if gender == "Female" or gender == "Male":
                    if len(age) > 0:
                        id2index[user_id] = index
                        index2id[index] = user_id
                        index+=1
    return id2index, index2id


def load_user_gender():
    profiles = {}
    with open("Flixster/profile.txt", 'r') as f:
        for index, line in enumerate(f.readlines()[1:]):
            userid, gender, _, _, _, _, age = line.split(",")
            if gender == "Female" or gender == "Male":
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
    with open("Flixster/movie.txt", 'r') as f:
        for index, line in enumerate(f.readlines()[1:]):
            line = line.replace("\n", "")
            begin = line.rfind(",")
            _, movie_id = line[0:begin], line[begin + 1:]

            id2index[movie_id] = index
            index2id[index] = movie_id
    return id2index, index2id


def export_subset(nr_users=1000):
    X, T = load_flixster_data(max_user=nr_users)
    user_id2index, user_index2id = load_user_id_index_dict()
    movie_id2index, movie_index2id = load_movie_id_index_dict()
    with open("Flixster/subset_" + str(nr_users) + ".txt", 'w') as f:
        for user_index, user in enumerate(X[0:nr_users, :]):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(user_index2id[user_index]) + "::" + str(movie_index2id[index_movie]) + "::" + str(rating) + "::000000\n")


def load_flixster_data_subset(file="Flixster/subset_400.txt", small=False):
    user_id2index, user_index2id = load_user_id_index_dict()
    movie_id2index, movie_index2id = load_movie_id_index_dict()
    user_id2gender = load_user_gender()
    X = np.zeros(shape=(400, 66730))
    T = np.zeros(shape=(400,))
    with open(file, 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            X[user_id2index[user_id], movie_id2index[movie_id]] = np.round(float(rating))
            if user_id2gender[user_id] == "Female":
                T[user_id2index[user_id]] = 1
            else:
                T[user_id2index[user_id]] = 0
    if small:
        valid_movie_ind = np.argwhere(np.sum(X, axis=0) != 0)[:,0]
        X = X[:,valid_movie_ind]
        #movie_ids = movie_index2id[valid_movie_ind]
        return X, T, valid_movie_ind
    else:
        return X, T, 0


def load_flixster_data_subset_masked(file_index=-1, small=False, valid_movies=[]):
    user_id2index, user_index2id = load_user_id_index_dict()
    movie_id2index, movie_index2id = load_movie_id_index_dict()
    user_id2gender = load_user_gender()
    X = np.zeros(shape=(400, 66730))
    T = np.zeros(shape=(400,))

    files = [
        "Flixster/FX_blurme_obfuscated_0.01_greedy_avg_top-1.dat",
        "Flixster/FX_blurme_obfuscated_0.05_greedy_avg_top-1.dat",
        "Flixster/FX_blurme_obfuscated_0.1_greedy_avg_top-1.dat",
        "Flixster/FX_blurmepp_obfuscated_greedy_0.01_2.dat",
        "Flixster/FX_blurmepp_obfuscated_greedy_0.05_2.dat",
        "Flixster/FX_blurmepp_obfuscated_greedy_0.1_2.dat",#5
        "Flixster/FX_blurme_obfuscated_0.01_random_avg_top-1.dat",
        "Flixster/FX_blurme_obfuscated_0.05_random_avg_top-1.dat",
        "Flixster/FX_blurme_obfuscated_0.1_random_avg_top-1.dat",
        "Flixster/FX_blurmepp_obfuscated_random_0.01_2.dat",
        "Flixster/FX_blurmepp_obfuscated_random_0.05_2.dat", # 10
        "Flixster/FX_blurmepp_obfuscated_random_0.1_2.dat",
        "Flixster/subset_400.txt",
        "Flixster/FX_blurmebetter_obfuscated_greedy_0.01_2_c0.99.dat",
        "Flixster/FX_blurmebetter_obfuscated_greedy_0.05_2_c0.99.dat",
        "Flixster/FX_blurmebetter_obfuscated_greedy_0.1_2_c0.99.dat", #15
    ]
    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            X[user_id2index[user_id], movie_id2index[movie_id]] = np.round(float(rating))
            if user_id2gender[user_id] == "Female":
                T[user_id2index[user_id]] = 1
            else:
                T[user_id2index[user_id]] = 0

    if small:
        X = X[:,valid_movies]
        #movie_ids = movie_index2id[valid_movie_ind]
        return X, T, 0#movie_ids
    else:
        return X, T, 0


#print(list(reversed(sorted(load_user_id_index_dict()[1])))[-50:])
#load_flixster_data()
#export_subset(nr_users=400)
#load_flixster_data_subset()