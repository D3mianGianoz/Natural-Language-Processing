import string
import nltk
from nltk.corpus import wordnet as wn
from Radicioni.Wsd.utilities import bag_of_word

"""
Word Sense Induction methods
"""


def clean_up(sentence, word_of_interest, space, frequency_d):
    stopwords = nltk.corpus.stopwords.words("english")
    sent_cleaned = []
    for w in sentence:
        w = w.lower()
        if w == word_of_interest:
            # don't keep the target word
            pass
        elif w in stopwords or w.strip(string.punctuation) == "":
            # drop this
            pass
        elif w not in frequency_d or frequency_d[w] <= 20:
            # drop this
            pass
        elif w in space:
            sent_cleaned.append(w)
    return sent_cleaned


def make_context_vector(sent, word_of_interest, space, fd):
    sent_cleaned = clean_up(sentence=sent, word_of_interest=word_of_interest, space=space, frequency_d=fd)

    if len(sent_cleaned) > 0:
        return sum(space[w] for w in sent_cleaned) / len(sent_cleaned)
    else:
        return None


def get_cxt_vectors(word, sentences, space, fd, ):
    context_vectors = []
    sentences_with_vectors = []
    for sent in sentences:
        embedding = make_context_vector(sent, word, space, fd)
        if embedding is not None:
            sentences_with_vectors.append(sent)
            context_vectors.append(embedding)

    return context_vectors, sentences_with_vectors


"""
Lesk DeMaria Implementation
"""


def max_freq(word):
    synsets = wn.synsets(word)
    sense2freq = None
    freq_max = 0

    for s in synsets:
        freq = 0
        for lemma in s.lemmas():
            freq += lemma.count()
            if freq > freq_max:
                freq_max = freq
                sense2freq = s
    return sense2freq


def lesk(word, sentence):
    # inizializzazione
    max_overlap = 0
    best_sense = max_freq(word)

    # If I choose the bag of words approach
    context = bag_of_word(sentence)
    signature = []

    for ss in wn.synsets(word):
        signature += ss.definition().split()
        signature += ss.lemma_names()

        overlap = set(signature).intersection(context)
        signature.clear()

        if len(overlap) > max_overlap:
            best_sense = ss
            max_overlap = len(overlap)

    return best_sense
