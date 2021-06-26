import nltk
# We download a pre-computed space
import gensim.downloader as gensim_api
from sklearn.cluster import KMeans

from DiCaro.Primo_gruppo.utils import get_cxt_vectors

if __name__ == '__main__':
    space_bis = gensim_api.load("glove-wiki-gigaword-300")
    fd = nltk.FreqDist(nltk.corpus.brown.words())

    w1 = "bar"
    w2 = "special"

    w1 = input("Choose a word you wanna induce the meaning\n word_of_interest: ")
    w2 = input("Choose another word, not important\nw2: ")

    pseudo_word = w1 + w2

    w1_list = []
    w2_list = []
    sentences_reformulated = []

    for s in nltk.corpus.brown.sents():
        if w1 in s:
            w1_list.append(s)
        if w2 in s:
            w2_list.append(s)

    for s in nltk.corpus.brown.sents():
        if (w1 in s) or (w2 in s):
            sentences_reformulated.append(s)

    for w in sentences_reformulated:
        if w1 in w:
            print("The first sentence with the word", w1, "is:", w)
            break
    for w in sentences_reformulated:
        if w2 in w:
            print("The first sentence with the word", w2, "is:", w)
            break

    for w in sentences_reformulated:
        for n, a in enumerate(w):
            if (a == w1) or (a == w2):
                w[n] = pseudo_word

    for w in w1_list:
        for n, a in enumerate(w):
            if a == w1:
                w[n] = pseudo_word

    for w in w2_list:
        for n, a in enumerate(w):
            if a == w2:
                w[n] = pseudo_word

    ctx_vectors_bis, sentences_with_vectors = get_cxt_vectors(word=pseudo_word, sentences=sentences_reformulated,
                                                          space=space_bis, fd=fd)

    print("We got embeddings for", len(ctx_vectors_bis), "out of", len(sentences_reformulated), "contexts.")

    numclusters = 3
    kmeans_obj = KMeans(n_clusters=numclusters)
    kmeans_obj.fit(ctx_vectors_bis)
    label_list = kmeans_obj.labels_

    # Let's print the sentences that got clustered together.
    for clusternumber in range(numclusters):
        print("\n\n")
        print("Sentences in cluster", clusternumber)
        c_bar = 0
        c_special = 0
        for index, sent in enumerate(sentences_with_vectors):
            if label_list[index] == clusternumber:
                # print(" ".join(sent))
                if sent in w1_list:
                    c_bar += 1
                elif sent in w2_list:
                    c_special += 1

        print(f"Cluster {clusternumber} have {c_bar} matching in {w1} and {c_special} matching in {w2}")

        precision = round(c_special / (c_bar + c_special), 2)
        print("Precision: " + str(precision))
        recall = round(c_special / len(w2_list), 2)
        print("Recall: " + str(recall))
        F1 = round(2 * ((precision * recall) / (precision + recall)), 2)
        print("F1 score: : " + str(F1))
