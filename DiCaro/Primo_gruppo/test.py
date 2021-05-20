import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


from nltk.corpus import brown
from matplotlib import pyplot
import nltk

data_list = []
y_list = []

tt = nltk.TextTilingTokenizer(demo_mode=True)
text = brown.raw()[:4000]
if text is None:
    text = brown.raw()[:10000]
s, ss, d, b = tt.tokenize(text)
pyplot.xlabel("Sentence Gap index")
pyplot.ylabel("Gap Scores")
pyplot.plot(range(len(s)), s, label="Gap Scores")
pyplot.plot(range(len(ss)), ss, label="Smoothed Gap scores")
pyplot.plot(range(len(d)), d, label="Depth scores")
pyplot.stem(range(len(b)), b)
pyplot.legend()
pyplot.show()

# for i in range(67):
#     X, Y = make_blobs(n_samples=300, n_features=1,
#                       centers=8, cluster_std=1.8, random_state=101 + i)
#
#     x_new = list(itertools.chain.from_iterable(X))
#
#     data_list.append(x_new)
#     y_list.append(Y)
#
# fig = plt.figure(figsize=(10, 10))
#
# # kmeans = KMeans(n_clusters=6, random_state=0)
# for n_clusters in [4, 5, 6, 7]:
#     fig = plt.figure(figsize=(10, 10))
#     kmeans = KMeans(n_clusters=n_clusters, random_state=10, algorithm="elkan")
#     test = [np.mean(pt) for pt in data_list]
#
#     y = kmeans.fit(data_list)
#     centroids = kmeans.cluster_centers_
#
#     for o in range(len(data_list[0])):
#         plt.scatter(test, [pt for pt in kmeans.labels_], c=kmeans.labels_, cmap='brg')
#     fig.show()
