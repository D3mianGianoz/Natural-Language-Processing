import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

data_list = []
y_list = []

for i in range(67):
    X, Y = make_blobs(n_samples=300, n_features=1,
                      centers=6, cluster_std=1.8, random_state=101+i)

    x_new = list(itertools.chain.from_iterable(X))

    data_list.append(x_new)
    y_list.append(Y)

fig = plt.figure(figsize=(10, 10))
kmeans = KMeans(n_clusters=6, random_state=0)

test = [np.mean(pt) for pt in data_list]

y = kmeans.fit(data_list)
centroids = kmeans.cluster_centers_
for o in range(len(data_list[0])):
    plt.scatter(test, [pt for pt in kmeans.labels_], c=kmeans.labels_, cmap='brg')

fig.show()
