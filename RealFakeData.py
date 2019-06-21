import numpy as np


def simulate_data(shape, distr=[]):
    distr = [1486126 / 1586126, 6110 / 1586126, 11370 / 1586126, 27145 / 1586126, 34174 / 1586126, 21201 / 1586126]
    print(distr)
    fake_data = np.zeros(shape=shape)
    for user_index in range(shape[0]):
        for item_index in range(shape[1]):
            fake_data[user_index, item_index] = np.random.choice(np.arange(6), p=distr)

    return fake_data


def load_real_fake_data_ML_100k():
    from MovieLensData import load_user_item_matrix_100k, load_user_item_matrix_100k_masked
    data = []
    real = load_user_item_matrix_100k()
    #fake = load_user_item_matrix_100k()
    #fake = simulate_data(real.shape)
    fake = load_user_item_matrix_100k_masked()
    #fake = np.random.randint(5, size=real.shape)
    print(fake)
    data = np.zeros(shape=(len(real)+len(fake), len(real[0])))
    labels = np.zeros(shape=(len(real)+len(fake),))
    for user_index, user in enumerate(real):
        data[user_index,:] = user
        labels[user_index] = 1
    for user_index, user in enumerate(fake):
        data[len(real) + user_index,:] = user
        labels[len(real)+user_index] = 0

    from Utils import shuffle_two_arrays
    data, labels = shuffle_two_arrays(data, labels)
    return data, labels

def load_user_item_matrix_1m_masked(max_user=6040, max_item=3952, file_index=-1):

    df = np.zeros(shape=(max_user, max_item))
    files = ["ml-1m/blurme_obfuscated_0.1_random_highest.dat",
             "ml-1m/blurme_obfuscated_0.05_random_highest.dat",
             "ml-1m/blurme_obfuscated_0.01_random_highest.dat",
             "ml-1m/blurme_obfuscated_0.05_sampled_highest.dat",
             "ml-1m/blurme_obfuscated_0.01_sampled_highest.dat",
             "ml-1m/blurme_obfuscated_0.01_greedy_highest.dat",#5
             "ml-1m/blurme_obfuscated_0.01_greedy_avg.dat",
             "ml-1m/blurme_obfuscated_0.01_sampled_avg.dat",
             "ml-1m/blurme_obfuscated_0.01_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.05_random_avg.dat",
             "ml-1m/blurme_obfuscated_0.1_random_avg.dat",#10
             "ml-1m/blurme_obfuscated_0.1_sampled_avg.dat",
             "ml-1m/blurme_obfuscated_0.1_greedy_avg.dat",
             "ml-1m/rebalanced_(100,1500).dat",
             "ml-1m/rebalanced_(100,1000).dat",
             "ml-1m/rebalanced_(100,700).dat",#15
             "ml-1m/blurme_obfuscated_0.1_greedy_avg_top500.dat",
             "ml-1m/blurmepp_obfuscated_0.1_2.dat",

             ]
    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df


def load_real_fake_data_ML_1m():
    from MovieLensData import load_user_item_matrix_1m
    data = []
    real = load_user_item_matrix_1m()
    real = real[0:int(real.shape[0]/2), :]
    #fake = load_user_item_matrix_100k()
    #fake = simulate_data(real.shape)
    fake = load_user_item_matrix_1m_masked(file_index=-1)
    fake = fake[0:int(fake.shape[0]/2), :]
    #fake = np.random.randint(5, size=real.shape)
    print(fake)
    data = np.zeros(shape=(real.shape[0]+fake.shape[0], real[0].shape[0]))
    labels = np.zeros(shape=(real.shape[0]+fake.shape[0],))
    for user_index, user in enumerate(real):
        data[user_index,:] = user
        labels[user_index] = 1
    for user_index, user in enumerate(fake):
        data[len(real) + user_index,:] = user
        labels[len(real)+user_index] = 0

    from Utils import shuffle_two_arrays
    data, labels = shuffle_two_arrays(data, labels)
    return data, labels

