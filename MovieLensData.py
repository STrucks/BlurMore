import numpy as np
from Utils import save_object, load_object


def movie_id_index_20m():
    """
    This function creates a dictionary that assigns each movie id a unique identifier. Since the movie ids in the 100k
    data range up to 131262, but only ~1600 movies are used, the dictionary gives you a the index in the user item
    matrix for a given movie_id
    :return: said dictionary
    """
    dict = {}
    index = 0
    with open("ml-20m/movies.csv", 'r', encoding='UTF-8') as f:
        for line in f.readlines()[1:]:
            movieId  = line[:line.find(",")]
            movieId = int(movieId)
            if movieId not in dict:
                dict[movieId] = index
                index += 1
    return dict


def load_user_item_matrix_100k(max_user=943, max_item=1682):
    """
        this function loads the user x items matrix from the **old** movie lens data set.
        Both input parameter represent a threshold for the maximum user id or maximum item id
        The highest user id is  and the highest movie id is  for the original data set, however, the masked data
        set contains only 943 users and  items
        :return: user-item matrix
        """
    import os.path
    #if os.path.isfile("objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item)):
    #    load_object("objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item))
    df = np.zeros(shape=(max_user, max_item))
    with open("ml-100k/u.data", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split()
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            df[user_id-1, movie_id-1] = rating

    #save_object(df, "objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item))
    return df


def load_user_item_matrix_100k_masked(max_user=943, max_item=1682, file_index=-1):
    """
        this function loads the user x items matrix from the **old** movie lens data set.
        Both input parameter represent a threshold for the maximum user id or maximum item id
        The highest user id is 138493 and the highest movie id is 27278 for the original data set, however, the masked data
        set contains only 943 users and 1330 items
        :return: user-item matrix
        """
    import os.path
    #if os.path.isfile("objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item)):
    #    load_object("objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item))

    df = np.zeros(shape=(max_user, max_item))
    masked_files = ["ml-100k/global_random_ml.csv",
                    "ml-100k/ml_shuffle_0.5_k40.csv",
                    "ml-100k/shuffle_remove_features.csv",
                    "ml-100k/shuffle_remove_features_277.csv",
                    "ml-100k/shuffle_feauture_111.csv",
                    "ml-100k/shuffle_feauture_1140.csv",#5
                    "ml-100k/shuffle_within_1153.csv",
                    "ml-100k/blurme_obfuscated_0.1.csv",
                    "ml-100k/blurme_obfuscated_0.05.csv",
                    "ml-100k/blurme_obfuscated_0.01.csv",
                    "ml-100k/blurme_obfuscated_0.1_greedy_avg.csv",#10

                    ]
    with open(masked_files[file_index], 'r') as f:
        for line in f.readlines()[1:]:
            user_id, movie_id, rating = line.split(",")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating #np.random.randint(1, 6)
    #df = np.random.randint(0, 6, size=(max_user, max_item))
    #save_object(df, "objs/user-item_Matrix_" + str(max_user) + "_" + str(max_item))
    return df


def load_user_item_matrix_1m(max_user=6040, max_item=3952):
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    #import os.path
    #if os.path.isfile("objs/user-item_Matrix_1m_" + str(max_user) + "_" + str(max_item)):
    #    load_object("objs/user-item_Matrix_1m_" + str(max_user) + "_" + str(max_item))

    #id_index, _ = load_movie_id_index_dict()
    df = np.zeros(shape=(max_user, max_item))
    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    #save_object(df, "objs/user-item_Matrix_1m_" + str(max_user) + "_" + str(max_item))
    return df


def load_user_item_matrix_1m_binary():
    X = load_user_item_matrix_1m()
    ratings = np.argwhere(X>0)
    print(ratings)
    X = np.zeros(shape=X.shape)
    for x, y in ratings:
        X[x, y] = 1
    return X


def load_user_item_matrix_1m_masked(max_user=6040, max_item=3952, file_index=-1):

    files = ["ml-1m/blurme_obfuscated_0.1_random_highest.dat",
             "ml-1m/blurme_obfuscated_0.05_random_highest.dat",
             "ml-1m/blurme_obfuscated_0.01_random_highest.dat",
             "ml-1m/blurme_obfuscated_0.05_sampled_highest.dat",
             "ml-1m/blurme_obfuscated_0.01_sampled_highest.dat",
             "ml-1m/blurme_obfuscated_0.01_greedy_highest.dat",  #5
             "ml-1m/blurme_obfuscated_0.01_greedy_avg.dat",
             "ml-1m/blurme_obfuscated_0.01_sampled_avg.dat",
             "ml-1m/blurme_obfuscated_0.01_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.05_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.1_random_avg.dat",  #10
             "ml-1m/blurme_obfuscated_0.1_sampled_avg.dat",
             "ml-1m/blurme_obfuscated_0.1_greedy_avg.dat",
             "ml-1m/rebalanced_(100,1500).dat",
             "ml-1m/rebalanced_(100,1000).dat",
             "ml-1m/rebalanced_(100,700).dat",  #15
             "ml-1m/blurme_obfuscated_0.1_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.05_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.01_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.2_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.2_greedy_avg.dat",  #20
             "ml-1m/blurme_obfuscated_0.05_greedy_avg.dat",
             "ml-1m/blurme_obfuscated_Normalized_0.1_greedy_avg.dat",
             "ml-1m/blurme_obfuscated_0.01_random_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.05_random_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.1_random_avg_top100.dat",  #25
             "ml-1m/blurme_obfuscated_0.2_random_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.01_greedy_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.05_greedy_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.1_greedy_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.2_greedy_avg_top100.dat",  #30
             "ml-1m/blurme_obfuscated_0.01_sampled_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.05_sampled_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.1_sampled_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.2_sampled_avg_top100.dat",
             "ml-1m/blurme_obfuscated_0.2_random_avg_top1000.dat",  #35
             "ml-1m/blurme_obfuscated_0.1_random_avg_top3950.dat",
             "ml-1m/blurme_obfuscated_0.1_sampled_avg_top3950.dat",
             "ml-1m/blurme_obfuscated_0.1_sampled_avg_top2000.dat",
             "ml-1m/blurme_obfuscated_0.1_sampled_avg_top1000.dat",
             "ml-1m/blurme_obfuscated_0.1_sampled_avg_top500.dat",  #40
             "ml-1m/blurme_obfuscated_0.01_random_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.05_random_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.1_random_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.2_random_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.01_sampled_avg_top500.dat",  #45
             "ml-1m/blurme_obfuscated_0.05_sampled_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.1_sampled_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.2_sampled_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.01_greedy_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.05_greedy_avg_top500.dat",  #50
             "ml-1m/blurme_obfuscated_0.1_greedy_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.2_greedy_avg_top500.dat",
             "ml-1m/blurmepp_obfuscated_greedy_0.01_2.dat",
             "ml-1m/blurmepp_obfuscated_greedy_0.05_2.dat",
             "ml-1m/blurmepp_obfuscated_greedy_0.1_2.dat",  #55
             "ml-1m/blurmepp_obfuscated_0.2_2.dat",
             "ml-1m/blurme_obfuscated_normalized_0.1_greedy_avg_top500.dat",
             "ml-1m/blurme_obfuscated_0.1_random_avg_top2000.dat",
             "ml-1m/blurme_obfuscated_0.1_random_avg_top1000.dat",
             "ml-1m/blurme_obfuscated_0.05_random_avg_top2000.dat",  #60
             "ml-1m/blurme_obfuscated_0.01_random_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.05_random_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.1_random_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.2_random_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.01_sampled_avg_top-1.dat",  #65
             "ml-1m/blurme_obfuscated_0.05_sampled_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.1_sampled_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.2_sampled_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.01_greedy_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.05_greedy_avg_top-1.dat",  #70
             "ml-1m/blurme_obfuscated_0.1_greedy_avg_top-1.dat",
             "ml-1m/blurme_obfuscated_0.2_greedy_avg_top-1.dat",
             "ml-1m/blurmepp_obfuscated_random_0.01_2.dat",
             "ml-1m/blurmepp_obfuscated_random_0.05_2.dat",
             "ml-1m/blurmepp_obfuscated_random_0.1_2.dat",  #75
             "ml-1m/blurmepp_x2_obfuscated_greedy_0.01_2.dat",
             "ml-1m/blurmepp_x2_obfuscated_greedy_0.05_2.dat",
             "ml-1m/blurmepp_x2_obfuscated_greedy_0.1_2.dat",
             "ml-1m/blurme_x2_obfuscated_0.01_greedy.dat",
             "ml-1m/blurme_x2_obfuscated_0.05_greedy.dat",  #80
             "ml-1m/blurme_x2_obfuscated_0.1_greedy.dat",
             "ml-1m/blurmebetter_obfuscated_random_0.01_2.dat",
             "ml-1m/blurmebetter_obfuscated_random_0.05_2.dat",
             "ml-1m/blurmebetter_obfuscated_random_0.1_2.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.01_2_c0.5.dat",  #85
             "ml-1m/blurmebetter_obfuscated_greedy_0.05_2_c0.5.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.1_2_c0.5.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.01_2_c0.8.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.05_2_c0.8.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.1_2_c0.8.dat",  #90
             "ml-1m/blurmebetter_obfuscated_greedy_0.01_2_c0.95.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.05_2_c0.95.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.1_2_c0.95.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.01_2_c0.99.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.05_2_c0.99.dat",  #95
             "ml-1m/blurmebetter_obfuscated_greedy_0.1_2_c0.99.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.01_2_c0.999999.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.05_2_c0.999999.dat",
             "ml-1m/blurmebetter_obfuscated_greedy_0.1_2_c0.999999.dat",
             "ml-1m/blurmepp_x2_obfuscated_random_0.01_2.dat", #100
             "ml-1m/blurmepp_x2_obfuscated_random_0.05_2.dat",
             "ml-1m/blurmepp_x2_obfuscated_random_0.1_2.dat",
             "ml-1m/blurme_x2_obfuscated_0.01_random.dat",
             "ml-1m/blurme_x2_obfuscated_0.05_random.dat",
             "ml-1m/blurme_x2_obfuscated_0.1_random.dat",#105

             ]
    #id_index, _ = load_movie_id_index_dict()
    df = np.zeros(shape=(max_user, max_item))

    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df


def load_movie_id_index_dict():
    id_index = {}
    index_id = {}
    with open("ml-1m/movies.dat", 'r') as f:
        for index, line in enumerate(f.readlines()):
            id, name, genres = line.split("::")
            id_index[int(id)] = index
            index_id[index] = int(id)
    return id_index, index_id


def load_user_item_matrix_1m_limited_ratings(limit=20):
    user_item = load_user_item_matrix_1m()
    user_item_limited = np.zeros(shape=user_item.shape)
    for user_index, user in enumerate(user_item):
        # filter rating indices
        rating_index = np.argwhere(user > 0).reshape(1, -1)[0]
        # shuffle them
        np.random.shuffle(rating_index)
        for i in rating_index[:limit]:
            user_item_limited[user_index, i] = user[i]
    #print(np.sum(user_item_limited, axis=1))
    return user_item_limited


def load_user_genre_matrix_1m(one_hot=False, top=5):
    user_item = load_user_item_matrix_1m()
    movie_genre = load_movie_genre_matrix_1m(combine=True)
    print(user_item.shape, movie_genre.shape)
    user_genre = np.zeros(shape=(user_item.shape[0], movie_genre.shape[1]))
    for user_index, user in enumerate(user_item):
        for movie_index, rating in enumerate(user):
            if rating > 0:
                user_genre[user_index, :] += movie_genre[movie_index, :]
    if one_hot:
        u_g_one = np.zeros(shape=user_genre.shape)
        for user_index, user in enumerate(user_genre):
            top_5_index = user.argsort()[-top:][::-1]
            for index in top_5_index:
                u_g_one[user_index, index] = 1

        user_genre = u_g_one

    return user_genre


def load_user_genre_matrix_100k(one_hot=False, top=5):
    user_item = load_user_item_matrix_100k()
    movie_genre = load_movie_genre_matrix_100k(combine=False)
    print(user_item.shape, movie_genre.shape)
    user_genre = np.zeros(shape=(user_item.shape[0], movie_genre.shape[1]))
    for user_index, user in enumerate(user_item):
        for movie_index, rating in enumerate(user):
            if rating > 0:
                user_genre[user_index, :] += movie_genre[movie_index, :]
    if one_hot:
        u_g_one = np.zeros(shape=user_genre.shape)
        for user_index, user in enumerate(user_genre):
            top_5_index = user.argsort()[-top:][::-1]
            for index in top_5_index:
                u_g_one[user_index, index] = 1

        user_genre = u_g_one

    return user_genre


def load_user_genre_matrix_100k_obfuscated(one_hot=False, top=5):
    user_item = load_user_item_matrix_100k_masked()
    movie_genre = load_movie_genre_matrix_100k(combine=False)
    print(user_item.shape, movie_genre.shape)
    user_genre = np.zeros(shape=(user_item.shape[0], movie_genre.shape[1]))
    for user_index, user in enumerate(user_item):
        for movie_index, rating in enumerate(user):
            if rating > 4:
                user_genre[user_index, :] += movie_genre[movie_index, :]
    if one_hot:
        u_g_one = np.zeros(shape=user_genre.shape)
        for user_index, user in enumerate(user_genre):
            top_5_index = user.argsort()[-top:][::-1]
            for index in top_5_index:
                u_g_one[user_index, index] = 1

        user_genre = u_g_one

    return user_genre


def load_gender_vector_1m(max_user=6040):
    """
        this function loads and returns the gender for all users with an id smaller than max_user
        :param max_user: the highest user id to be retrieved
        :return: the gender vector
        """
    gender_vec = []
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines()[:max_user]:
            user_id, gender, age, occ, postcode = line.split("::")
            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)

    return np.asarray(gender_vec)


def load_gender_vector_100k(max_user=943):
    """
    this function loads and returns the gender for all users with an id smaller than max_user
    :param max_user: the highest user id to be retrieved
    :return: the gender vector
    """
    gender_vec = []
    with open("ml-100k/userObs.csv", 'r') as f:
        for line in f.readlines()[1:]:
            if len(line) < 2:
                continue
            else:
                userid, age, gender, occupation, zipcode = line.split(", ")
                if gender == "M":
                    gender_vec.append(0)
                else:
                    gender_vec.append(1)
    return np.asarray(gender_vec)


def load_occupation_vector_1m(max_user=6040):
    """
    this function loads and returns the occupation for all users with an id smaller than max_user
    :param max_user: the highest user id to be retrieved
    :return: the occupation vector
    """
    occ_vec = []
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines()[:max_user]:
            user_id, gender, age, occ, postcode = line.split("::")
            occ_vec.append(int(occ))
    return np.asarray(occ_vec)


def load_occupation_vector_100k(max_user=943):
    occ_labels = {}
    with open("ml-100k/occupationLabels.csv", 'r') as f:
        for line in f.readlines():
            occ, label = line.replace("\n", "").split(",")
            occ_labels[occ] = int(label)
    #print(occ_labels)
    occ_vector = []
    with open("ml-100k/userObs.csv", 'r') as f:
        for line in f.readlines()[1:]:
            if len(line) < 2:
                continue
            else:
                userid, age, gender, occupation, zipcode = line.split(", ")
                occ_vector.append(occ_labels[occupation])
    #print(occ_vector)
    return np.asarray(occ_vector)


def load_age_vector_1m(border=30):
    age_vector = []
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines():
            userid, gender, age, occupation, zipcode = line.split("::")
            if int(age) < border:
                age_vector.append(0)
            else:
                age_vector.append(1)
    return np.asarray(age_vector)


def load_age_vector_100k(border=30):
    age_vector = []
    with open("ml-100k/userObs.csv", 'r') as f:
        for line in f.readlines()[1:]:
            if len(line) < 2:
                continue
            else:
                userid, age, gender, occupation, zipcode = line.split(", ")
                if int(age) < border:
                    age_vector.append(0)
                else:
                    age_vector.append(1)
    return np.asarray(age_vector)


def data_exploration():
    import pandas as pd
    df = pd.read_csv("ml-100k/userObs.csv")
    from matplotlib import pyplot as plt
    import collections
    a = df[' age']
    counter = collections.Counter(a)
    plt.bar(counter.keys(), counter.values())
    plt.show()


def gender_user_dictionary_1m():
    gender_dict = {}
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines():
            userid, gender, age, occupation, zipcode = line.split("::")
            if userid not in gender_dict:
                gender_dict[int(userid)-1] = gender
    return gender_dict


def load_movie_genre_matrix_1m(combine=False):
    """
    This function loads the movie genre matrix for ML 1m. Said matrix is MxG, where M denotes the movie_id and G the
    genre id.
    :return: said matrix
    """
    if combine:
        # Since Drama co-occurs so frequently with Romance and Comedy, we will consider movies that are romantic
        # dramas just as dramas. Same for Drama & Comedy and Romantic & Comedy
        genres = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama",
                  "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                  "Western"]
        matrix = np.zeros(shape=(3952, len(genres)))

        with open("ml-1m/movies.dat", 'r') as f:
            for line in f.readlines():
                id, name, genre = line.replace("\n", "").split("::")
                genre = genre.split("|")
                if "Drama" in genre and "Romance" in genre:
                    genre.remove("Romance")
                if "Drama" in genre and "Comedy" in genre:
                    genre.remove("Comedy")
                if "Romance" in genre and "Comedy" in genre:
                    genre.remove("Romance")

                for g in genre:
                    matrix[int(id) - 1, genres.index(g)] = 1
    else:
        genres = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        matrix = np.zeros(shape=(3952, len(genres)))

        with open("ml-1m/movies.dat", 'r') as f:
            for line in f.readlines():
                id, name, genre = line.replace("\n", "").split("::")
                genre = genre.split("|")
                for g in genre:
                    matrix[int(id)-1, genres.index(g)] = 1
    return matrix


def load_movie_genre_matrix_100k(combine=False):
    """
    This function loads the movie genre matrix for ML 100k. Said matrix is MxG, where M denotes the movie_id and G the
    genre id.
    :return: said matrix
    """
    if combine:
        # Since Drama co-occurs so frequently with Romance and Comedy, we will consider movies that are romantic
        # dramas just as dramas. Same for Drama & Comedy and Romantic & Comedy
        print("not implemented yet")
    else:
        genres = ["unknown", "Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        matrix = np.zeros(shape=(3952, len(genres)))

        with open("ml-100k/u.item", 'r') as f:
            for id, line in enumerate(f.readlines()):
                genre_vector = line.replace("\n", "")[-37:]
                genre_vector = genre_vector.split("|")
                genre_vector = np.asarray([int(x) for x in genre_vector])
                matrix[int(id)-1, :] += genre_vector
    return matrix


def load_movie_id_dictionary_1m():
    dict = {}
    with open("ml-1m/movies.dat", 'r') as f:
        for line in f.readlines():
            id, name, genres = line.split("::")
            dict[int(id)] = name
    return dict


def load_movie_id_dictionary_100k():
    dict = {}
    with open("ml-100k/u.item", 'r') as f:
        for line in f.readlines():
            start = line.find("|")
            end = line.find("|", start+1)
            id = line[0:start]
            name = line[start+1:end]
            dict[int(id)] = name
    return dict


def balance_dataset():
    print("hi")