from MovieLensData import load_user_item_matrix_100k, load_user_item_matrix
from matplotlib import pyplot as plt


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
            id, age, gender, occ, post = line.replace("\n", "").split("::")
            df['user_id'].append(id)
            df['age'].append(age)
            df['gender'].append(gender)
            df['occupation'].append(occ)
            df['postcode'].append(post)
    import collections
    a = df['age']
    counter = collections.Counter(a)
    plt.bar(counter.keys(), counter.values())
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

histogram()
#show_user_item_matrix()
