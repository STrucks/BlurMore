import numpy as np
from Utils import save_object, load_object


def load_user_item_matrix_100k(max_user=943, max_item=1682):
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
                df[user_id-1, movie_id] = rating

    save_object(df, "objs/user-item_Matrix_old_" + str(max_user) + "_" + str(max_item))
    return df


def load_user_item_matrix_100k_masked(max_user=943, max_item=1682):
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
    with open("ml-20m/ml_shuffle_0.5_k40.csv", 'r') as f:
        for line in f.readlines()[1:]:
            user_id, movie_id, rating = line.split(",")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id < max_user and movie_id < max_item:
                df[user_id-1, movie_id] = rating

    save_object(df, "objs/user-item_Matrix_" + str(max_user) + "_" + str(max_item))
    return df


def load_user_item_matrix_1m(max_user=6040, max_item=3952):
    """
        this function loads the user x items matrix from the  movie lens data set.
        Both input parameter represent a threshold for the maximum user id or maximum item id
        The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
        set contains only 943 users and 1330 items
        :return: user-item matrix
        """
    import os.path
    #if os.path.isfile("objs/user-item_Matrix_1m_" + str(max_user) + "_" + str(max_item)):
    #    load_object("objs/user-item_Matrix_1m_" + str(max_user) + "_" + str(max_item))

    df = np.zeros(shape=(max_user, max_item))
    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id < max_user and movie_id < max_item:
                df[user_id-1, movie_id-1] = rating

    save_object(df, "objs/user-item_Matrix_1m_" + str(max_user) + "_" + str(max_item))
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
    with open("ml-20m/occupationLabels.csv", 'r') as f:
        for line in f.readlines():
            occ, label = line.replace("\n", "").split(",")
            occ_labels[occ] = int(label)
    #print(occ_labels)
    occ_vector = []
    with open("ml-20m/userObs.csv", 'r') as f:
        for line in f.readlines()[1:]:
            if len(line) < 2:
                continue
            else:
                userid, age, gender, occupation, zipcode = line.split(", ")
                occ_vector.append(occ_labels[occupation])
    #print(occ_vector)
    return np.asarray(occ_vector)


def load_age_vector_1m(boarder=30):
    age_vector = []
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines():
            userid, gender, age, occupation, zipcode = line.split("::")
            if int(age) < boarder:
                age_vector.append(0)
            else:
                age_vector.append(1)
    return np.asarray(age_vector)


def load_age_vector_100k(boarder=30):
    age_vector = []
    with open("ml-20m/userObs.csv", 'r') as f:
        for line in f.readlines()[1:]:
            if len(line) < 2:
                continue
            else:
                userid, age, gender, occupation, zipcode = line.split(", ")
                if int(age) < boarder:
                    age_vector.append(0)
                else:
                    age_vector.append(1)
    return np.asarray(age_vector)


def data_exploration():
    import pandas as pd
    df = pd.read_csv("ml-20m/userObs.csv")
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


#print(load_gender_vector(max_user=100))
#print(load_user_item_matrix())
