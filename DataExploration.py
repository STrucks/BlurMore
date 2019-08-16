from MovieLensData import load_user_item_matrix_100k, load_user_item_matrix_100k
from matplotlib import pyplot as plt
import matplotlib
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




def rating_exploration():
    X = MD.load_user_item_matrix_1m()
    rating_distr = {}
    for index, user in enumerate(X):
        nr_ratings = 0
        for rating in user:
            if rating > 0:
                nr_ratings += 1
        if nr_ratings == 0:
            print(index, user)
        if nr_ratings > 200:
            continue
        if nr_ratings in rating_distr:
            rating_distr[nr_ratings] += 1
        else:
            rating_distr[nr_ratings] = 1
    print(rating_distr)
    plt.rcParams.update({'font.size': 22})
    plt.bar(rating_distr.keys(), rating_distr.values())
    plt.xlabel("#ratings per user")
    plt.ylabel("frequency")
    plt.show()
    rating_distr = {}
    for index, user in enumerate(X):
        for rating in user:
            if rating not in rating_distr:
                rating_distr[rating] = 1
            else:
                rating_distr[rating] += 1
    print(rating_distr)
    print(X.shape[0]*X.shape[1])
    plt.bar(rating_distr.keys(), rating_distr.values())
    plt.show()

    rating_distr = {}
    X = np.transpose(X)
    for index, item in enumerate(X):
        nr_ratings = 0
        for rating in item:
            if rating > 0:
                nr_ratings += 1
        if nr_ratings == 0:
            print(index, item)
            continue
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


def show_avg_rating_gender_per_movie(movie_id=1):
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


def test_avg_rating_gender_per_movie_1m():
    import MovieLensData as MD
    from scipy.stats import ttest_ind, mannwhitneyu
    gender_dict = MD.gender_user_dictionary_1m()
    user_item = MD.load_user_item_matrix_1m()

    movies = {}
    with open("ml-1m/movies.dat", 'r') as f:
        for line in f.readlines():
            id, name, genre = line.replace("\n", "").split("::")
            movies[int(id)] = name + "::" + genre
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

        try:
            _, p_value = mannwhitneyu(male_ratings, female_ratings)

            if p_value < 0.05/len(user_item[0]):
                #print(movie_id+1, "%.2f" % np.average(male_ratings), len(male_ratings), "%.2f" % np.average(female_ratings), len(female_ratings), p_value)
                counter += 1
                #plt.bar(["male", "female"], [np.average(male_ratings), np.average(female_ratings)])
                #plt.show()
                if np.average(male_ratings) > np.average(female_ratings):
                    print(str(movie_id + 1) + "::" + movies[movie_id + 1] + "::M")
                if np.average(male_ratings) < np.average(female_ratings):
                    print(str(movie_id + 1) + "::" + movies[movie_id + 1] + "::F")
        except:
            print("Testing failed for", movie_id)

    print(str(1 + 1) + "::" + movies[1])
    print(counter)


def test_avg_rating_gender_per_movie_100k():
    import MovieLensData as MD
    from scipy.stats import ttest_ind, mannwhitneyu
    gender_vec = MD.load_gender_vector_100k()
    user_item = MD.load_user_item_matrix_100k()

    movies = {}
    with open("ml-100k/u.item", 'r') as f:
        for line in f.readlines():
            i1 = line.find("|")
            id = line[:i1]
            i2 = line.find("|", i1+1)
            name = line[i1+1:i2]
            movies[int(id)] = name
    counter = 0
    print(len(user_item[0]))
    for movie_id in range(len(user_item[0])):
        ratings = user_item[:, movie_id]
        male_ratings = []
        female_ratings = []
        for user_id, rating in enumerate(ratings):
            if rating > 0:
                if gender_vec[user_id] == 0:
                    male_ratings.append(rating)
                else:
                    female_ratings.append(rating)
        try:

            if len(male_ratings) == 0:
                male_ratings = np.array([0])
            if len(female_ratings) == 0:
                female_ratings = np.array([0])
            if np.average(male_ratings) == np.average(female_ratings):
                continue

            _, p_value = ttest_ind(male_ratings, female_ratings)
            #print(p_value)
            if p_value < (0.05/len(user_item[0])):
                #print(movie_id+1, "%.2f" % np.average(male_ratings), len(male_ratings), "%.2f" % np.average(female_ratings), len(female_ratings), p_value)
                counter += 1
                #print(np.average(male_ratings) , np.average(female_ratings))
                #print(male_ratings, female_ratings)
                #plt.bar(["male", "female"], [np.average(male_ratings), np.average(female_ratings)])
                #plt.show()
                if np.average(male_ratings) > np.average(female_ratings):
                    print(str(movie_id + 1) + "::" + movies[movie_id + 1] + "::M")
                if np.average(male_ratings) < np.average(female_ratings):
                    print(str(movie_id + 1) + "::" + movies[movie_id + 1] + "::F")
        except:
            print(male_ratings, female_ratings)
            print("Testing failed for", movie_id)
            continue

    print("counter", counter)


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
    movie_genre = MD.load_movie_genre_matrix_1m(combine=True)
    user_genre_distr = np.zeros(shape=(6040, movie_genre.shape[1]))
    user_gender_dict = MD.gender_user_dictionary_1m()
    print(user_genre_distr.shape)
    with open("ml-1m/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            movie_id = int(movie_id)-1
            user_id = int(user_id)-1

            user_genre_distr[user_id,:] += movie_genre[movie_id,:]
    loyal_percents = [0.5, 0.6, 0.7]
    for i, loyal_percent in enumerate(loyal_percents):
        loyal_count = 0
        for user_index, user in enumerate(user_genre_distr):
            if max(user)/sum(user) > loyal_percent:
                if True:
                    #print the user:
                    print(user_gender_dict[user_index])
                    top_5_index = user.argsort()[-5:][::-1]
                    for index in top_5_index:
                        print(genres[index], user[index])

                loyal_count += 1
        print("For threshold", loyal_percent, ",", loyal_count, "users are considered loyal")

    if True:
        user_loyalty_male = []
        user_loyalty_female = []
        for user_index, user in enumerate(user_genre_distr):
            loyalty = max(user) / sum(user)
            if user_gender_dict[user_index] == 'M':
                user_loyalty_male.append(loyalty)
                plt.scatter(user_index, loyalty, c='b')
            else:
                user_loyalty_female.append(loyalty)
                plt.scatter(user_index, loyalty, c='r')
        print(np.average(user_loyalty_male))
        print(np.average(user_loyalty_female))
        #plt.show()


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
    plt.rcParams.update({'font.size': 22})

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
    plt.setp(ax.get_xticklabels(), rotation=-30,  ha="left", rotation_mode="anchor")
    plt.title("Co-occurrence of movie genres in ML 1m")
    plt.show()


def plot_genre_1m():
    genres = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
              "Western"]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    movie_genre = MD.load_movie_genre_matrix_1m(combine=False)
    ax1.bar(genres, np.sum(movie_genre, axis=0))
    ax1.set_title("genre distribution in ML 1m")
    plt.setp(ax1.get_xticklabels(), rotation=-45, ha="left")

    movie_genre = MD.load_movie_genre_matrix_1m(combine=True)
    ax2.bar(genres, np.sum(movie_genre, axis=0))
    ax2.set_title("genre distribution in ML 1m, Drama&Romance are combined to Drama ect.")
    plt.setp(ax2.get_xticklabels(), rotation=-45, ha="left")
    plt.show()


def show_gender_genre_comparison():
    plt.rcParams.update({'font.size': 28})
    # This plot shows the
    genres = ["Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
              "Western"]

    movie_genre = MD.load_movie_genre_matrix_1m()
    male_genre = np.zeros(shape=(len(genres, )))
    female_genre = np.zeros(shape=(len(genres, )))
    user_gender_dict = MD.gender_user_dictionary_1m()
    user_genre = MD.load_user_genre_matrix_1m()
    for user_index, user in enumerate(user_genre):
        if user_gender_dict[user_index] == "M":
            male_genre += user
        else:
            female_genre += user
    print(male_genre)
    print(female_genre)
    x = np.arange(len(genres))
    ax = plt.subplot(111)
    ax.bar(x-0.2, male_genre/750000, width=0.4, label='male')
    ax.bar(x+0.2, female_genre/250000, width=0.4, label='female')
    plt.xticks(x, ("Action", "Adventure", "Animation", "Children\'s", "Comedy", "Crime", "Documentary", "Drama",
              "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
              "Western"))
    plt.legend()
    plt.tight_layout()
    plt.setp(ax.get_xticklabels(), rotation=40)
    plt.ylabel("Normalized rating count")
    plt.xlabel("Genres")
    plt.show()


def PCA_100k():
    X = MD.load_user_item_matrix_100k()
    T = MD.load_gender_vector_100k()
    males = X[np.argwhere(T==0)[:,0]]
    females = X[np.argwhere(T==1)[:,0]]
    print(females.shape)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    PC = pca.fit_transform(males)
    for x, y in PC:
        plt.scatter(x, y, c='b')
    PC = pca.fit_transform(females)
    for x, y in PC:
        plt.scatter(x, y, c='r')

    plt.show()

    for index, (x,y) in enumerate(PC):
        if T[index] == 0:
            plt.scatter(x, y, c='b')
        else:
            plt.scatter(x, y, c='r')
    plt.show()
    print(PC)


def feature_importance_100k():
    from sklearn.ensemble import ExtraTreesClassifier
    X = MD.load_user_item_matrix_100k()
    T = MD.load_gender_vector_100k()
    importance = np.zeros(shape=(X.shape[1],))
    for i in range(100):
        model = ExtraTreesClassifier()
        model.fit(X, T)
        importance += model.feature_importances_
    importance /= 100
    plt.plot(range(len(importance)), importance)
    plt.xlabel("movie index")
    plt.ylabel("importance")
    plt.show()
    counter = 0
    for movie, score in enumerate(importance):
        if score >= 0.002:
            print(movie + 1, end=",")
            counter += 1
    print()
    print(counter)


def feature_importance_1m():
    plt.rcParams.update({'font.size': 18})

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import seaborn as sns
    X = MD.load_user_item_matrix_1m()
    T = MD.load_gender_vector_1m()
    importance = np.zeros(shape=(X.shape[1],))
    importance2 = np.zeros(shape=(X.shape[1],))

    for i in range(10):
        model = LogisticRegression()
        model2 = RandomForestClassifier()
        model.fit(X, T)
        model2.fit(X,T)
        importance += model.coef_[0]
        importance2 += model2.feature_importances_
    importance /= 10
    importance2 /= 10

    #plt.bar(range(1,len(importance[0:30])+1), importance[0:30])
    #plt.xlabel("movie index")
    #plt.ylabel("importance")
    #plt.show()
    #sns.distplot(importance, kde=False)
    plt.hist(importance2, bins=np.linspace(0,0.001,50))
    #sns.kdeplot(importance,shade=True,cut=0)
    #sns.rugplot(importance)
    plt.xlabel("importance")
    plt.ylabel("frequency")
    plt.title("Importance of movies distribution")
    plt.show()
    importance_id = zip(importance, range(1,len(importance)+1))
    importance_id = list(reversed(sorted(importance_id)))
    importance_id2 = zip(importance, range(1, len(importance) + 1))
    importance_id2 = list(sorted(importance_id2))
    importance_id3 = zip(importance2, range(1, len(importance2) + 1))
    importance_id3 = list(reversed(sorted(importance_id3)))
    set1 = set()
    set2 = set()
    set3 = set()

    names = MD.load_movie_id_dictionary_1m()
    top = 100
    for (_, id), (_,id2), (_,id3) in zip(importance_id[0:top], importance_id2[0:top], importance_id3[0:top]):
        print(names[id], "|", names[id2], "|", names[id3])
        set1.add(names[id])
        set2.add(names[id2])
        set3.add(names[id3])

    #print(set3)

    print(set3.intersection(set2.union(set1)))

    #print(importance_id)
    """
    counter = 0
    for movie, score in enumerate(importance):
        if score >= 0.002:
            print(movie + 1, end=",")
            counter += 1
    print()
    print(counter)
    nr_ratings = np.zeros(shape=(X.shape[1],))
    for index, movie in enumerate(np.transpose(X)):
        counter = 0
        for rating in movie:
            if rating > 0:
                counter += 1
        nr_ratings[index] = counter

    avg_nr_per_importance = {}
    nr_ratings_importance = []
    for nr, imp in zip(nr_ratings, importance):
        if imp in avg_nr_per_importance:
            avg_nr_per_importance[imp].append(nr)
        else:
            avg_nr_per_importance[imp] = [nr]
        nr_ratings_importance.append([nr, imp])


    #for key in avg_nr_per_importance.keys():
    #    avg_nr_per_importance[key] = np.average(avg_nr_per_importance[key])
    #print(avg_nr_per_importance)
    plt.subplot(1,2,1)
    for nr, imp in nr_ratings_importance:
        plt.scatter(nr, imp)
    plt.xlabel("#ratings")
    plt.ylabel("importance")

    plt.subplot(1,2,2)
    for nr, imp in nr_ratings_importance:
        if nr < 100:
            plt.scatter(nr, imp)
    plt.xlabel("#ratings")
    plt.ylabel("importance")

    plt.show()
    """


def find_good_threshold():
    plt.rcParams.update({'font.size': 18})
    import Classifiers
    import Utils
    max_user = 6040
    max_item = 3952
    # X = MD.load_user_item_matrix_1m_limited_ratings(limit=200)  # max_user=max_user, max_item=max_item)
    X = MD.load_user_item_matrix_1m()
    T = MD.load_gender_vector_1m()  # max_user=max_user)

    X_train, T_train = X[0:int(0.8 * len(X))], T[0:int(0.8 * len(X))]
    X_test, T_test = X[int(0.8 * len(X)):], T[int(0.8 * len(X)):]

    # print(X)
    # X = Utils.remove_significant_features(X, T)
    # X = feature_selection(X, T, Utils.select_male_female_different)
    # X = Utils.normalize(X)
    # X = Utils.standardize(X)
    # X = chi2_selection(X, T)

    precision = 20
    begin, end = 0, 0.001
    auc_rel = []
    auc_irrel = []
    std_rel = []
    std_irrel = []
    size_r = []
    size_i = []
    for t in np.linspace(begin, end, precision):
        print(X_train.shape)
        X_train_important, X_train_compl = Utils.random_forest_selection(X_train, T_train, threshold=t)
        print(X_train_important.shape)
        size_r.append(X_train_important.shape[1])
        size_i.append(X_train_compl.shape[1])

        mean_auc_r, std_auc_r = Classifiers.log_reg(X_train_important, T_train, show_plot=False)
        mean_auc_i, std_auc_i = Classifiers.log_reg(X_train_compl, T_train, show_plot=False)
        auc_rel.append(mean_auc_r)
        auc_irrel.append(mean_auc_i)
        std_rel.append(std_auc_r)
        std_irrel.append(std_auc_i)

    auc_rel, auc_irrel, std_rel, std_irrel = np.asarray(auc_rel), np.asarray(auc_irrel), np.asarray(
        std_rel), np.asarray(std_irrel)
    plt.subplot(1, 2, 1)

    plt.plot(np.linspace(begin, end, precision), auc_rel, c='b', label='AUC of important features')
    auc_upper = np.minimum(auc_rel + std_rel, 1)
    auc_lower = np.maximum(auc_rel - std_rel, 0)
    plt.fill_between(np.linspace(begin, end, precision), auc_lower, auc_upper, color='grey', alpha=.2)

    plt.plot(np.linspace(begin, end, precision), auc_irrel, c='r', label='AUC of not important features')
    auc_upper = np.minimum(auc_irrel + std_irrel, 1)
    auc_lower = np.maximum(auc_irrel - std_irrel, 0)
    plt.fill_between(np.linspace(begin, end, precision), auc_lower, auc_upper, color='grey', alpha=.2)

    plt.xlabel("Threshold")
    plt.ylabel("Mean AUC")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(begin, end, precision), size_r, c='b', label='number of important movies')
    plt.plot(np.linspace(begin, end, precision), size_i, c='r', label='number of irrelevant movies')
    plt.xlabel("Threshold")
    plt.ylabel("#samples in data")
    plt.legend()
    plt.show()


def comp_BM_and_BMpp():
    plt.rcParams.update({'font.size': 28})
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    interval_start, interval_end = 0, 50
    X = MD.load_user_item_matrix_1m()
    X_movie_count = [sum([1 if x > 0 else 0 for x in movie]) for movie in np.transpose(X)[interval_start:interval_end]]
    movie_count_o = np.asarray(X_movie_count)
    ax1.bar(range(interval_start,interval_end), X_movie_count)
    ax1.set_title("(A)\nOriginal data")
    ax1.set_xlabel("movie ID")
    ax1.set_ylabel("#ratings")
    #ax1.set_xticks(range(1,6), [1,2,3,4,5])
    print("Original Data:", sum(X_movie_count))

    X = MD.load_user_item_matrix_1m_masked(file_index=63)
    X_movie_count = [sum([1 if x > 0 else 0 for x in movie]) for movie in np.transpose(X)[interval_start:interval_end]]
    masked_count = np.asarray(X_movie_count)
    ax2.bar(range(interval_start, interval_end), X_movie_count)
    ax2.set_title("(B)\nBlurMe data")
    ax2.set_xlabel("movie ID")
    ax2.set_ylabel("#ratings")
    print("BlurMe Data:", sum(X_movie_count))

    X = MD.load_user_item_matrix_1m_masked(file_index=75)
    X_movie_count = [sum([1 if x > 0 else 0 for x in movie]) for movie in np.transpose(X)[interval_start:interval_end]]
    masked_count2 = np.asarray(X_movie_count)
    ax3.bar(range(interval_start, interval_end), X_movie_count)
    ax3.set_title("(C)\nBlurM(or)e data")
    ax3.set_xlabel("movie ID")
    ax3.set_ylabel("#ratings")
    print("BlurMe++ Data:", sum(X_movie_count))
    print(movie_count_o-masked_count)
    print(masked_count-masked_count2)
    plt.show()


def avg_rating_diff():
    X = MD.load_user_item_matrix_100k()
    T = MD.load_gender_vector_100k()
    name_dict = MD.load_movie_id_dictionary_100k()
    males_indecies = np.argwhere(T==0)[:,0]
    females_indecies = np.argwhere(T == 1)[:, 0]
    differences = np.zeros(shape=X.shape[1],)
    for movie_index, movie in enumerate(np.transpose(X)):
        avg_m = []
        male_ratings = movie[males_indecies]
        avg_f = []
        female_ratings = movie[females_indecies]
        for m_r, f_r in zip(male_ratings, female_ratings):
            if m_r > 0:
                avg_m.append(m_r)
            if f_r > 0:
                avg_f.append(f_r)
        avg_m = np.average(avg_m)
        avg_f = np.average(avg_f)
        if not (np.isnan(avg_m) or np.isnan(avg_f)):
            differences[movie_index] = avg_m-avg_f
    differences = [[differences[index], index] for index in range(differences.shape[0])]
    differences = np.asarray(list(reversed(sorted(differences))))
    print(differences[0:20,1])
    names = [name_dict[index+1] for index in np.concatenate((differences[0:20,1], differences[-20:,1]))]
    print(names)
    fig, ax = plt.subplots()
    ax.barh(range(40), np.concatenate((differences[0:20,0], differences[-20:,0]), axis=0), align='center')
    ax.set_yticks(range(40))
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Difference')
    ax.set_title('Rating difference between males and females')

    plt.show()


def rating_distr():
    T = MD.load_gender_vector_1m()
    X = MD.load_user_item_matrix_1m()
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    frequencies = np.zeros(shape=(6,))
    ratings = []
    movie_ids = []
    for user in X:
        for index, rating in enumerate(user):
            frequencies[int(rating)] += 1
            if rating > 0:
                movie_ids.append(index+1)
                ratings.append(rating)
    print(frequencies)
    print(sum(frequencies[1:]), np.mean(frequencies[1:]), np.var(frequencies[1:]))
    print("mean:", np.dot(np.arange(0, 6), frequencies) / sum(frequencies), "without 0:",
          np.dot(np.arange(1, 6), frequencies[1:]) / sum(frequencies[1:]))
    print("Avg", np.average(ratings), "var", np.var(ratings))
    print(len(set(movie_ids)))
    ax1.bar(range(5), frequencies[1:])
    ax1.set_xlabel("Original")
    X = MD.load_user_item_matrix_1m_masked(file_index=71)# greedy 10%
    frequencies = np.zeros(shape=(6,))
    ratings = []

    for user in X:
        for rating in user:
            frequencies[int(rating)] += 1
            if rating > 0:
                ratings.append(rating)

    print(frequencies)
    print(sum(frequencies[1:]), np.mean(frequencies[1:]), np.var(frequencies[1:]))

    print("mean:", np.dot(np.arange(0, 6), frequencies) / sum(frequencies), "without 0:",
          np.dot(np.arange(1, 6), frequencies[1:]) / sum(frequencies[1:]))
    print("Avg", np.average(ratings), "var", np.var(ratings))

    ax2.bar(range(5), frequencies[1:])
    ax2.set_xlabel("BlurMe")
    X = MD.load_user_item_matrix_1m_masked(file_index=55)# BlurMe++ 10%, fac=2
    frequencies = np.zeros(shape=(6,))
    ratings = []
    for user in X:
        for rating in user:
            frequencies[int(rating)] += 1
            if rating > 0:
                ratings.append(rating)
    print(frequencies)
    print(sum(frequencies[1:]), np.mean(frequencies[1:]), np.var(frequencies[1:]))
    print("mean:", np.dot(np.arange(0, 6), frequencies) / sum(frequencies), "without 0:",
          np.dot(np.arange(1, 6), frequencies[1:]) / sum(frequencies[1:]))
    print("Avg", np.average(ratings), "var", np.var(ratings))

    ax3.bar(range(5), frequencies[1:])
    ax3.set_xlabel("BlurMe++")
    plt.show()


def movie_ratings():
    X = MD.load_user_item_matrix_1m()
    freqs = []
    for movie in np.transpose(X):
        freq = 0
        for rating in movie:
            if rating > 0:
                freq+=1
        freqs.append(freq)
    print(list(sorted(freqs)))
    print(set(freqs))
    #plt.bar(range(len(freqs)), freqs)
    #plt.show()


def RMSE_plot():
    original = [0.8766,0,0,0]
    blurme = [0,0.8686,0.8553,0.8385]
    blurmepp = [0,0.8711,0.8640,0.8468]

    ind = np.arange(len(original))  # the x locations for the groups
    width = 0.35  # the width of the bars
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, original, width, label='Original')
    rects2 = ax.bar(ind-width/2, blurme, width, label='BlurMe')
    rects2 = ax.bar(ind+width/2, blurmepp, width, label='BlurMe++')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for the different data sets')
    ax.set_xticks(ind)
    ax.set_xticklabels(('None', '1', '5', '10'))
    ax.set_xlabel("Percent ratings added")
    ax.legend()
    fig.tight_layout()

    plt.show()


def compare_real_fake():
    import RealFakeData as RFD
    real = MD.load_user_item_matrix_1m()
    real = real[0:40, 0:40]
    # fake = load_user_item_matrix_100k()
    # fake = simulate_data(real.shape)
    fake_bm = RFD.load_user_item_matrix_1m_masked(file_index=12)
    fake_bm = fake_bm[0:40, 0:40]

    fake_bmpp = RFD.load_user_item_matrix_1m_masked(file_index=17)
    fake_bmpp = fake_bmpp[00:40, 0:40]
    print(fake_bmpp.shape)

    plt.subplot(3,3,1)
    plt.imshow(real)
    plt.title("real")

    plt.subplot(3,3,4)
    plt.imshow(fake_bm)
    plt.title("fake_bm")

    plt.subplot(3, 3, 7)
    plt.imshow(fake_bmpp)
    plt.title("fake_bmpp")

    plt.subplot(3, 3, 5)
    plt.imshow(real-fake_bm)
    plt.title("real-fake_bm")

    plt.subplot(3, 3, 8)
    plt.imshow(real - fake_bmpp)
    plt.title("real-fake_bmpp")

    plt.show()


def stats_of_flixster():
    import FlixsterData as FD
    id2index, index2id = FD.load_user_id_index_dict()
    #print("number of users in data set:", len(index2id))
    user_gender = FD.load_user_gender()
    gender_vector = [user_gender[key] for key in user_gender]
    #print("possible values for gender", set(gender_vector))
    import collections
    counter = collections.Counter(gender_vector)
    #print("Distribution of values", counter)
    #print("making", counter['Male']+counter['Female'], "valid users")
    print("valid user profiles:", len(id2index))
    movies = FD.load_movies()
    print("number of movies", len(movies.keys()))

    interactions = 0
    nr_ratings = np.zeros(shape=(935267,))
    ratings = []
    with open("Flixster/ratings.txt", 'r', encoding='utf-16') as f:
        for line in f.readlines()[1:]:
            if len(line) < 2:
                continue
            else:
                user_id, _, rating, _ = line.split("\t")
                if user_id in id2index:
                    #if id2index[user_id] < 100:
                    nr_ratings[id2index[user_id]] += 1
                    ratings.append(float(rating))
                interactions += 1
    print("number of interactions", interactions)
    print("ratings avg", np.average(ratings), np.var(ratings))
    counter = 0
    for rating in nr_ratings:
        if rating > 0:
            counter += 1
    print(counter)

    print("on subset of 400 users:")
    X, T = FD.load_flixster_data()
    interactions = 0
    for user in X:
        for rating in user:
            if rating > 0:
                interactions += 1
    print("interactions:", interactions)
    counter = collections.Counter(T)
    print(counter)


def stats_libimseti():
    ratings = []
    with open("libimseti/ratings.dat", 'r') as f:
        for line in f.readlines():
            user_id, _, rating = line.split(",")
            ratings.append(float(rating))

    print("ratings avg", np.average(ratings), np.var(ratings))
    plt.hist(ratings)
    plt.show()


def global_stats_ML_obf():
    plt.rcParams.update({'font.size': 28})
    original = MD.load_user_item_matrix_1m()
    blurme = MD.load_user_item_matrix_1m_masked(file_index=71)
    blurmore = MD.load_user_item_matrix_1m_masked(file_index=55)
    ratings = [1, 2, 3, 4, 5]
    rating_distribution = np.zeros(shape=(3,5))
    nr_ratings = np.zeros(shape=(3, blurme.shape[0]))
    rating_distribution_new = np.zeros(shape=(5,))

    """"""
    for user_index in range(blurme.shape[0]):
        for movie_index in range(blurme.shape[1]):
            #if original[user_index, movie_index] != blurme[user_index, movie_index]:
            #    rating_distribution_new[int(blurme[user_index, movie_index]) - 1] += 1

            if original[user_index, movie_index] in ratings:
                nr_ratings[0, user_index] += 1
                rating_distribution[0, int(original[user_index, movie_index])-1] += 1
            if blurme[user_index, movie_index] in ratings:
                nr_ratings[1, user_index] += 1
                rating_distribution[1, int(blurme[user_index, movie_index])-1] += 1
            if blurmore[user_index, movie_index] in ratings:
                nr_ratings[2, user_index] += 1
                rating_distribution[2, int(blurmore[user_index, movie_index])-1] += 1
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.bar(range(1, 6), rating_distribution[1,:] - rating_distribution[0, :])
    ax2.bar(range(1, 6), rating_distribution[2, :] - rating_distribution[0, :])
    plt.show()
    print("new ratings:", rating_distribution)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    rating_distribution[0, :] /= sum(rating_distribution[0,:])
    rating_distribution[1, :] /= sum(rating_distribution[1,:])
    rating_distribution[2, :] /= sum(rating_distribution[2,:])
    print(rating_distribution)
    nr_ratings[0, :] = list(reversed(sorted(nr_ratings[0, :])))
    nr_ratings[1, :] = list(reversed(sorted(nr_ratings[1, :])))
    nr_ratings[2, :] = list(reversed(sorted(nr_ratings[2, :])))
    print(nr_ratings)
    ax1.bar(range(1, 6), rating_distribution[0, :])
    ax1.set_xlabel("Rating")
    ax1.set_ylabel("Rating Frequency in %")
    ax1.set_title("Rating frequency \noriginal ML")
    ax2.bar(range(1, 6), rating_distribution[1, :])
    ax2.set_xlabel("Rating")
    #ax2.set_ylabel("Rating Frequency")
    ax2.set_title("Rating frequency \nBlurMe")
    ax3.bar(range(1, 6), rating_distribution[2, :])
    ax3.set_xlabel("Rating")
    #ax3.set_ylabel("Rating Frequency")
    ax3.set_title("Rating frequency \nBlurM(or)e")
    plt.show()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.bar(range(100), nr_ratings[0, 0:100])
    ax2.bar(range(100), nr_ratings[1, 0:100])
    ax3.bar(range(100), nr_ratings[2, 0:100])
    plt.show()

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.plot(range(300), nr_ratings[0, -300:])
    ax2.plot(range(300), nr_ratings[1, -300:])
    ax3.plot(range(300), nr_ratings[2, -300:])
    plt.show()


#histogram()
#test_avg_rating_gender_per_movie_1m()
#test_avg_rating_gender_per_movie_100k()
#loyal_vs_diverse()
#genre_exploration_1m()
#rating_exploration_100k()
#rating_exploration()
#show_correlation_genre()
#plot_genre_1m()
#show_gender_genre_comparison()
#PCA_100k()
#feature_importance_1m()
find_good_threshold()
#comp_BM_and_BMpp()
#avg_rating_diff()
#gender_rating_distr()
#rating_distr()
#movie_ratings()
#RMSE_plot()
#compare_real_fake()
#stats_of_flixster()
#stats_libimseti()
#global_stats_ML_obf()