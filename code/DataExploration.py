from MovieLensData import load_user_item_matrix_100k, load_user_item_matrix
from matplotlib import pyplot as plt
import numpy as np


def histogram():
    df = {
        'user_id': [],
        'age': [],
        'gender': [],
        'occupation': [],
        'postcode': [],
    }
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines():
            id, gender, age, occ, post = line.replace("\n", "").split("::")
            df['user_id'].append(id)
            df['age'].append(age)
            df['gender'].append(gender)
            df['occupation'].append(int(occ))
            df['postcode'].append(post)
    import collections
    key = 'age'
    a = df[key]
    counter = collections.Counter(a)
    plt.bar(counter.keys(), counter.values())
    plt.xlabel(key)
    plt.ylabel('frequency')
    plt.show()


def show_user_item_matrix():
    X = load_user_item_matrix()
    X_ = load_user_item_matrix_100k()
    plt.subplot(3,1,1)
    plt.imshow(X_)
    plt.title("original user item matrix")
    plt.subplot(3,1,2)
    plt.imshow(X)
    plt.title("masked user item matrix")
    X_diff = X_ - X
    plt.subplot(3,1,3)
    plt.imshow(X_diff)
    plt.title("difference between these two")
    plt.show()


def gender_rating_distr():
    frequencies_male = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }
    frequencies_female = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }
    user_gender = {}
    with open("ml-1m/users.dat", 'r') as f:
        for line in f.readlines():
            user_id, gender, _, _, _ = line.split("::")
            user_gender[user_id] = gender

    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            rating = int(rating)
            if user_gender[user_id] == 'M':
                frequencies_male[rating] += 1
            elif user_gender[user_id] == 'F':
                frequencies_female[rating] += 1
    print(frequencies_male)
    sum_male = sum(frequencies_male.values())
    propotion_male = [value / sum_male for value in frequencies_male.values()]
    sum_female = sum(frequencies_female.values())
    propotion_female = [value / sum_female for value in frequencies_female.values()]

    from matplotlib import pyplot as plt
    plt.plot(propotion_male, label="propotion male")
    plt.plot(propotion_female, label="propotion female")
    plt.xticks([0,1,2,3,4], [1,2,3,4,5])
    plt.xlabel("rating")
    plt.ylabel("Percent of ratings")
    plt.legend()
    plt.show()


def show_avg_rating_gender_per_movie(movie_id = 1):
    import MovieLensData as MD
    gender_dict = MD.gender_user_dictionary_1m()
    user_item = MD.load_user_item_matrix_1m()
    ratings = user_item[:, movie_id]
    male_ratings = []
    female_ratings = []
    for user_id, rating in enumerate(ratings):
        if rating > 0:
            if gender_dict[user_id] == 'M':
                male_ratings.append(rating)
            else:
                female_ratings.append(rating)

    plt.bar(["male", "female"], [np.average(male_ratings), np.average(female_ratings)])
    plt.show()


def test_avg_rating_gender_per_movie():
    import MovieLensData as MD
    from scipy.stats import ttest_ind
    gender_dict = MD.gender_user_dictionary_1m()
    user_item = MD.load_user_item_matrix_1m()
    counter = 0
    for movie_id in range(1, 3953):
        ratings = user_item[:, movie_id-1]
        male_ratings = []
        female_ratings = []
        for user_id, rating in enumerate(ratings):
            if rating > 0:
                if gender_dict[user_id] == 'M':
                    male_ratings.append(rating)
                else:
                    female_ratings.append(rating)

        _, p_value = ttest_ind(male_ratings, female_ratings)
        if p_value < 0.05/6040:
            counter += 1
            #plt.bar(["male", "female"], [np.average(male_ratings), np.average(female_ratings)])
            #plt.show()


    print(counter)

histogram()
#test_avg_rating_gender_per_movie()