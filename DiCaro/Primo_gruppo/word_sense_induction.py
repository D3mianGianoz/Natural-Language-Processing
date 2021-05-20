# Word sense induction TLN_dicaro_1.3
# Warning: this method does not work particularly well as is, there are better implementation
import nltk
from pathlib import Path
from datetime import datetime

import pandas as pd
# We download a pre-computed space
import gensim.downloader as gensim_api

from DiCaro.Primo_gruppo.utils import clean_up, lesk, get_cxt_vectors

if __name__ == '__main__':
    space = gensim_api.load("glove-wiki-gigaword-300")
    # This gives us an embedding for a word, across all of its context.
    # How can I get an embedding for a single context?
    # The simplest option:
    # Get a context embedding by averaging over the word embeddings in the context.

    # let's get a list of medium-frequency content words
    nltk.download('brown')

    fd = nltk.FreqDist(nltk.corpus.brown.words())

    # "some medium-frequency words in the Brown corpus")
    # for example:
    # fall, sweet, empty, side, show, character, box, window, feet, plants, fire

    # 1 part
    # test our methodology
    word_of_interest = input("Choose a word you wanna induce the meaning\n word_of_interest: ")
    sentences_of_interest = [s for s in nltk.corpus.brown.sents() if word_of_interest in s]
    sent0 = sentences_of_interest[0]

    print("The first sentence with the word", word_of_interest, "is:", sent0)

    sent0_cleaned = clean_up(sentence=sent0, word_of_interest=word_of_interest, space=space, frequency_d=fd)

    print("Our cleaned-up first context of the word", word_of_interest, "is:", sent0_cleaned)

    # Here is how we make an embedding for this context of the word of interest
    words_with_embeddings = [w for w in sent0_cleaned if w in space]
    embedding0 = sum(space[w] for w in words_with_embeddings) / len(words_with_embeddings)

    # what does this tell us?
    # since this has the same dimensionality as word vectors, we can check
    # which word vectors are close to this
    print("The embedding of this sentence is somewhat similar to:\n", space.similar_by_vector(embedding0))

    # 2 part
    # We make embeddings for all the sentences with the word of interest
    ctx_vectors, sentences_with_vectors = get_cxt_vectors(word=word_of_interest, sentences=sentences_of_interest,
                                                          space=space, fd=fd)

    print("We got embeddings for", len(ctx_vectors), "out of", len(sentences_of_interest), "contexts.")

    # 3 part
    # We have to decide how many clusters to make.
    # Let's try 3.
    from sklearn.cluster import KMeans
    from collections import Counter
    # Plot the cluster
    import matplotlib.pyplot as plt
    import numpy as np

    nclusters = 3
    kmeans_obj = KMeans(n_clusters=nclusters)
    kmeans_obj.fit(ctx_vectors)
    label_list = kmeans_obj.labels_

    path = Path(".") / "output" / "WSI"
    title = word_of_interest

    # create a figure
    fig = plt.figure(figsize=(12, 12))

    # Let's print the sentences that got clustered together and write them to a file
    with open(path / f'{word_of_interest}_verbal_cluster.txt', "w", encoding="utf-8-sig") as f:
        for clusternumber in range(nclusters):
            in_cluster = "\n\nSentences in cluster # " + str(clusternumber)
            print(in_cluster)
            f.write(in_cluster + "\n")

            # for cluster plot, reset every time
            sentences = []
            synsets = []
            # define subplots, and their positions in figure
            plt.subplot(221 + clusternumber)

            for index, sent in enumerate(sentences_with_vectors):
                if label_list[index] == clusternumber:
                    sent_print = " ".join(sent)
                    print(sent_print)
                    sentences.append(" ".join(sent))
                    f.write(sent_print + "\n")

            for i in sentences:
                synsets.append(lesk(word_of_interest, i))

            D = Counter(synsets)
            pd.DataFrame(D, index=['quantity']).plot(kind='bar', ax=plt.gca())
            # plt1.bar(range(len(D)), D.values(), tick_label=list(D.keys()), width=0.8, color=['r', 'g', 'b'])
            plt.title(f"Cluster n°: {str(clusternumber)} of {title}")
            now = datetime.now().strftime(f"{title} - %d.%m.%Y-%H:%M:%S")  # dd/mm/YY-H:M:S
            plt.savefig(path / f'{now}-synset.png')

    # adjusting space between subplots
    fig.subplots_adjust(hspace=.5, wspace=0.5)
    # show subplot?
    plt.show()

    # setting a style to use
    plt.style.use('fivethirtyeight')

    # Plot all the embedding of the clusters
    plt.figure(figsize=(10, 10))
    # for elem in contextvectors[0]:
    plt.scatter([np.mean(pt) for pt in ctx_vectors],
                [label for label in label_list],
                c=label_list,
                cmap='brg')

    plt.title("Word of interest is " + title)
    plt.xlabel("Sentences")
    plt.ylabel("Num of cluster")
    # saving plot in output folder
    now = datetime.now().strftime(f"{title} - %d.%m.%Y-%H:%M:%S")  # dd/mm/YY-H:M:S
    plt.savefig(path / f'{now}-context_plot.png')
    plt.show()
    print(f"\n{title}'s plot saved in output folder.")
