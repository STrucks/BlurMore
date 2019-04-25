from MovieLensData import load_user_item_matrix_100k, load_user_item_matrix_100k
from matplotlib import pyplot as plt
import numpy as np
import MovieLensData as MD


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
    key = 'occupation'
    a = df[key]
    counter = collections.Counter(a)
    plt.bar(counter.keys(), counter.values())
    plt.xlabel(key)
    plt.ylabel('frequency')
    plt.show()


def rating_exploration_100k():
    X = load_user_item_matrix_100k()
    rating_distr = {}
    for index, user in enumerate(X):
        nr_ratings = 0
        for rating in user:
            if rating > 0:
                nr_ratings += 1
        if nr_ratings == 0:
            print(index, user)
        if nr_ratings in rating_distr:
            rating_distr[nr_ratings] += 1
        else:
            rating_distr[nr_ratings] = 1
    print(rating_distr)
    plt.bar(rating_distr.keys(), rating_distr.values())
    plt.show()


def show_user_item_matrix():
    X = load_user_item_matrix_100k()
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
    print(len(user_item[0]))
    for movie_id in range(len(user_item[0])):
        ratings = user_item[:, movie_id]
        male_ratings = []
        female_ratings = []
        for user_id, rating in enumerate(ratings):
            if rating > 0:
                if gender_dict[user_id] == 'M':
                    male_ratings.append(rating)
                else:
                    female_ratings.append(rating)

        _, p_value = ttest_ind(male_ratings, female_ratings)
        if p_value < 0.05/len(user_item[0]):
            counter += 1
            #plt.bar(["male", "female"], [np.average(male_ratings), np.average(female_ratings)])
            #plt.show()


    print(counter)


def genre_exploration_1m():
    genres = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    import MovieLensData as MD
    import matplotlib.pyplot as plt
    movie_genre = MD.load_movie_genre_matrix_1m()
    # plot genre frequencies:
    genre_frequency = np.sum(movie_genre, axis=0)
    plt.bar(genres, genre_frequency)
    plt.show()
    print(genre_frequency)

    # number of genres per movie:
    genre_count = np.sum(movie_genre, axis=1)
    #for index, count in enumerate(genre_count):
    #    if count == 0:
    #        print(index)
    import collections
    counter = collections.Counter(genre_count)
    print(counter)
    plt.bar(counter.keys(), counter.values())
    plt.xlabel("#genres")
    plt.ylabel('frequency')
    plt.show()


def loyal_vs_diverse():
    #X = MD.load_user_item_matrix_1m()
    #T = MD.load_gender_vector_1m()
    genres = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movie_genre = MD.load_movie_genre_matrix_1m()
    user_genre_distr = np.zeros(shape=(6040, movie_genre.shape[1]))
    print(user_genre_distr.shape)
    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            movie_id = int(movie_id)-1
            user_id = int(user_id)-1

            user_genre_distr[user_id,:] += movie_genre[movie_id,:]
    loyal_percents = [0.3, 0.4, 0.5, 0.6]
    for i, loyal_percent in enumerate(loyal_percents):
        loyal_count = 0
        for user in user_genre_distr:
            if max(user)/sum(user) > loyal_percent:
                loyal_count+=1
        print("For threshold", loyal_percent, ",", loyal_count, "users are considered loyal")


def show_correlation_genre():
    genres = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movie_genre = MD.load_movie_genre_matrix_1m()
    print(movie_genre.shape)
    cooc = np.zeros(shape=(movie_genre.shape[1], movie_genre.shape[1]))
    # show the simple co-occurrence matrix:
    for movie in movie_genre:
        pairs = []
        for index1 in range(len(movie)):
            if movie[index1] == 1:
                for index2 in range(index1+1, len(movie)):
                    if movie[index2] == 1:
                        pairs.append([index1, index2])
        for one, two in pairs:
            cooc[one, two] += 1
            cooc[two, one] += 1

    import seaborn as sb
    fig, ax = plt.subplots()
    ax = sb.heatmap(cooc, linewidths=0.5)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(genres)))
    ax.set_yticks(np.arange(len(genres)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(genres)
    ax.set_yticklabels(genres)
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), ha="left")
    plt.title("Co-occurrence of movie genres in ML 1m")
    plt.show()



#histogram()
test_avg_rating_gender_per_movie()
#loyal_vs_diverse()
#genre_exploration_1m()
#rating_exploration_100k()
#show_correlation_genre()