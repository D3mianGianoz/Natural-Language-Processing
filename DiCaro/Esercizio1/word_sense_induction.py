# Word sense induction Task 1.3
# Warning: this method does not work particularly well as is, there are better implementation
import string
import nltk
from pathlib import Path
from datetime import datetime

# We download a pre-computed space
import gensim.downloader as gensim_api


def clean_up(sentence, word_of_interest, space):
    stopwords = nltk.corpus.stopwords.words("english")
    sent_cleaned = []
    for w in sentence:
        if w.lower() == word_of_interest:
            # don't keep the target word
            pass
        elif w.lower() in stopwords or w.strip(string.punctuation) == "":
            # drop this
            pass
        elif w not in fd or fd[w] <= 20:
            # drop this
            pass
        elif w in space:
            sent_cleaned.append(w.lower())
        else:
            pass
    return sent_cleaned


def make_context_vector(sent, word_of_interest, space):
    sent_cleaned = clean_up(sentence=sent, word_of_interest=word_of_interest, space=space)

    if len(sent_cleaned) > 0:
        return sum(space[w] for w in sent_cleaned) / len(sent_cleaned)
    else:
        return None


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
    word_of_interest = "special"
    sentences_of_interest = [s for s in nltk.corpus.brown.sents() if word_of_interest in s]
    sent0 = sentences_of_interest[0]

    print("The first sentence with the word", word_of_interest, "is:", sent0)

    sent0_cleaned = clean_up(sentence=sent0, word_of_interest=word_of_interest, space=space)

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
    contextvectors = []
    sentences_with_vectors = []
    for sent in sentences_of_interest:
        embedding = make_context_vector(sent, word_of_interest, space)
        if embedding is not None:
            sentences_with_vectors.append(sent)
            contextvectors.append(embedding)

    print("We got embeddings for", len(contextvectors), "out of", len(sentences_of_interest), "contexts.")

    # 3 part
    # We have to decide how many clusters to make.
    # Let's try 4.

    from sklearn.cluster import KMeans

    nclusters = 4
    kmeans_obj = KMeans(n_clusters=nclusters)
    kmeans_obj.fit(contextvectors)
    label_list = kmeans_obj.labels_

    path = Path(".") / "output" / "WSI"
    # Let's print the sentences that got clustered together and write them to a file
    with open(path / f'{word_of_interest}.txt', "w", encoding="utf-8-sig") as f:
        for clusternumber in range(nclusters):
            in_cluster = "\n\nSentences in cluster # " + str(clusternumber)
            print(in_cluster)
            f.write(in_cluster + "\n")
            for index, sent in enumerate(sentences_with_vectors):
                if label_list[index] == clusternumber:
                    sent_print = " ".join(sent)
                    print(sent_print)
                    f.write(sent_print+"\n")

    # Plot the cluster
    import matplotlib.pyplot as plt
    import numpy as np

    title = word_of_interest
    plt.figure(figsize=(10, 10))
    # for elem in contextvectors[0]:
    plt.scatter([np.mean(pt) for pt in contextvectors],
                [label for label in label_list],
                c=label_list,
                cmap='brg')

    plt.title("Word of interest is " + title)
    plt.xlabel("Sentences")
    plt.ylabel("Num of cluster")
    # saving plot in output folder
    now = datetime.now().strftime(f"{title} - %d.%m.%Y-%H:%M:%S")  # dd/mm/YY-H:M:S
    plt.savefig(path / f'{now}.png')
    plt.show()
    print(f"\n{title}'s plot saved in output folder.")
