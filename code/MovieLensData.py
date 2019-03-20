import numpy as np
from Utils import save_object, load_object


def load_user_item_matrix_100k(max_user=943, max_item=1330):
    """
        this function loads the user x items matrix from the **old** movie lens data set.
        Both input parameter represent a threshold for the maximum user id or maximum item id
        The highest user id is 138493 and the highest movie id is 27278 for the original data set, however, the masked data
        set contains only 943 users and 1330 items
        :return: user-item matrix
        """
    import os.path
    if os.path.isfile("objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item)):
        load_object("objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item))

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-100k/u.data", 'r') as f:
        for line in f.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split()
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id < max_user and movie_id < max_item:
                df[user_id, movie_id] = rating + 1

    save_object(df, "objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item))
    return df


def load_user_item_matrix(max_user=943, max_item=1330):
    """
    this function loads the user x items matrix from the movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 138493 and the highest movie id is 27278 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    import os.path
    if os.path.isfile("objs/user-item_Matrix_" + str(max_user) + "_" + str(max_item)):
        load_object("objs/user-item_Matrix_" + str(max_user) + "_" + str(max_item))

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-20m/ml_shuffle_0.5_k40.csv", 'r') as f:
        for line in f.readlines()[1:]:
            user_id, movie_id, rating = line.split(",")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id < max_user and movie_id < max_item:
                df[user_id, movie_id] = rating+1

    save_object(df, "objs/user-item_Matrix_" + str(max_user) + "_" + str(max_item))
    return df


def load_gender_vector(max_user=943):
    """
    this function loads and returns the gender for all users with an id smaller than max_user
    :param max_user: the highest user id to be retrieved
    :return: the gender vector
    """
    import pandas as pd
    df = pd.read_csv("ml-20m/userObs.csv")
    gender_vec = np.zeros(shape=(max_user,))
    for i in range(max_user):
        if 'F' in df[' gender'][i]:
            gender_vec[i] = 1
    return gender_vec


def load_occupation_vector(max_user=943):
    """
    this function loads and returns the occupation for all users with an id smaller than max_user
    :param max_user: the highest user id to be retrieved
    :return: the occupation vector
    """
    import pandas as pd
    df = pd.read_csv("ml-20m/userObs.csv")
    occ_vec = np.zeros(shape=(max_user,))
    import collections

    counter = collections.Counter(df[' occupation'])
    keys = list(counter.keys())
    for i in range(max_user):
        occ_vec[i] = keys.index(df[' occupation'][i])
    return occ_vec


def data_exploration():
    import pandas as pd
    df = pd.read_csv("ml-20m/userObs.csv")
    from matplotlib import pyplot as plt
    import collections
    a = df[' age']
    counter = collections.Counter(a)
    plt.bar(counter.keys(), counter.values())
    plt.show()


#print(load_gender_vector(max_user=100))
print(load_user_item_matrix())
