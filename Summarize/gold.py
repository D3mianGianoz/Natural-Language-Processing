import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def speed_test(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        for x in range(5000):
            results = func(*args, **kwargs)
        t2 = time.time()
        print('%s took %0.3f ms' % (func.__name__, (t2 - t1) * 1000.0))
        return results

    return wrapper


@speed_test
def compare_bitwise(x, y):
    set_x = frozenset(x)
    set_y = frozenset(y)
    return set_x & set_y


@speed_test
def compare_listcomp(x, y):
    return [i for i, j in zip(x, y) if i == j]


@speed_test
def compare_intersect(x, y):
    return frozenset(x).intersection(y)


def similarity(selected_paragraphs, gold):
    """
    Percentage similarity of lists
    using "|" operator + compare_intersect()
    Args:
        selected_paragraphs: selected paragraphs
        gold: gold standard choice
    Returns:
        percentage of similarity
    """
    #
    total_length = len(set(selected_paragraphs) | set(gold))
    res = len(compare_intersect(selected_paragraphs, gold)) / float(total_length) * 100

    # printing result
    print("Percentage similarity among lists is : " + str(res) + "\n")
    return res


def tf_idf_sum(corpus, rate):
    """Given a corpus, it learns vocabulary and idf
    Args:
        rate: the compression rate
        corpus: list of paragraphs of the document
    Returns:
        Returns the sum of tf-idf for each paragraph.
    """
    # scikit learn calls
    tf_idf_vectorizer = TfidfVectorizer(use_idf=True, analyzer='word', stop_words='english')
    tf_idf = tf_idf_vectorizer.fit_transform(corpus)

    # create the dict
    i = 0
    sum_tf = {}
    while i < len(corpus):
        values = tf_idf[i].T.todense()
        df = pd.DataFrame(values, index=tf_idf_vectorizer.get_feature_names(), columns=["TF-IDF"])
        sum_tf[i] = (sum(df.iloc[:, 0]))
        i += 1

    # sort it by score
    sum_tf = sorted(sum_tf.items(), key=lambda x: x[1], reverse=False)

    # delete the values
    n_deleted_values: int = int(rate / 100 * len(sum_tf))
    del sum_tf[:n_deleted_values]

    # sort them again
    sum_tf = sorted(sum_tf, key=lambda x: x[0], reverse=False)
    # unzip using zip and (*) operator
    res = list(zip(*sum_tf))

    return res[0]
