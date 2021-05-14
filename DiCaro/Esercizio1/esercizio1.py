## TLN_dicaro_1.1 & TLN_dicaro_1.2

"""
CONSEGNA:
* Date delle definizioni per quattro concetti (due concreti e due astratti),
calcolare la similarità fra di esse.

* Aggregare anche le definizioni secondo le dimensioni di concretezza e
specificità e ri-calcolare i punteggi.

* Effettuare del pre-processing se necessario prima del calcolo.
Analizzare i risultati.

SVOLGIMENTO:
* I termini sono:
                Generico    Specifico
            ==========================
Concreto    |   Paper     Sharpener
Astratto    |   Courage   Apprehension

* Si è scelto di filtrare le stopwords come fase di pre-processing, per
concentrarsi sui termini salienti.

* Si nota come, nel caso di termini concreti, la similarità sia
significativamente più elevata di quanto non acccada per i termini astratti.
Questo è probabilmente dovuto alla possibilità di utilizzare degli attributi
visivi per descrivere il termine.

* Nel caso dei termini astratti, invece, la mancanza di questi attributi
concreti porta a definizioni meno simili fra di loro.

* Abbiamo provato, oltre la baseline, due diverse misure di similarità

"""

# handling files
import csv
from pathlib import Path

# numeric staples
import numpy as np
import pandas as pd

# graph for the result
import matplotlib.pyplot as plt
from datetime import datetime

# for preprocessing
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

# Scikit learn stuff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SET = 0
LIST = 1


def load_data(path):
    """ It reads che definition's CSV
    Args:
        path: csv path
    Returns:
         four list containing the read definitions.
    """
    with open(path, "r", encoding="utf-8") as definitions:
        reader = csv.reader(definitions, delimiter=',')

        def_abstract_generic = []
        def_concrete_generic = []
        def_abstract_specific = []
        def_concrete_specific = []

        first = True
        for line in reader:
            if not first:
                def_abstract_generic.append(line[0])
                def_concrete_generic.append(line[1])
                def_abstract_specific.append(line[2])
                def_concrete_specific.append(line[3])
            else:
                first = False

        # Courage,Paper,Apprehension,Sharpener
        return def_abstract_generic, def_concrete_generic, def_abstract_specific, def_concrete_specific


def preprocess(definition, mode):
    """ It does some preprocess: removes the stopwords, punctuation and does the
    lemmatization of the tokens inside the sentence.
    Args:
        mode: enum for defining the kind of return data structure
        definition: a string representing a definition
    Returns:
        - a set of string which contains the preprocessed string tokens.
        OR
        - original array preprocessed without eliminating duplicates
    """

    lemmatized_tokens = None
    # Removing stopwords
    definition = definition.lower()
    stop_words = set(stopwords.words('english'))
    punctuation = {',', ';', '(', ')', '{', '}', ':', '?', '!', '.'}
    wnl = nltk.WordNetLemmatizer()
    tokens = nltk.word_tokenize(definition)
    tokens = list(
        filter(lambda x: x not in stop_words and x not in punctuation, tokens))

    # Lemmatization
    if mode == SET:
        lemmatized_tokens = set(wnl.lemmatize(t) for t in tokens)
    elif mode == LIST:
        lemmatized_tokens = ' '.join(list(wnl.lemmatize(t) for t in tokens))

    return lemmatized_tokens


def compute_overlap_terms(definitions):
    """ It computes the overlap between the two set of the preprocessed terms
    Args:
        definitions: a list of definitions (strings)
    Returns:
        a list of length pow(len(definition)) containing the similarity
    score of each definition.
    """

    # list of similarity score for each type of definition (length: pow(len(definition)))
    results = []

    i = 0
    while i < len(definitions):
        a = preprocess(definitions[i], SET)  # set of terms of the first definition
        j = i + 1
        while j < len(definitions) - 1:
            # print(i,j)  # DEBUG
            b = preprocess(definitions[j], SET)  # set of terms of the second definition
            # Computing similarity between definitions
            t = len(a & b) / min(len(a), len(b))
            results.append(t)
            j = j + 1

        i = i + 1

    return results


def compute_overlap_pos(definitions):
    """ It computes the overlap between the two set of the preprocessed definitions 
    converted in POS tagging.
    Args:
        definitions: a list of definitions (strings)
    Returns:
        a list of length |definitions| containing the maximum similarity
    score of each definition.
    """

    results = []

    i = 0
    while i < len(definitions):
        text1 = word_tokenize(definitions[i])
        temp_a = nltk.pos_tag(text1)
        a = set(x[1] for x in temp_a)

        j = i + 1

        # Performed (n(n+1))/2 times
        while j < len(definitions) - 1:
            # print(i,j)  # DEBUG
            text2 = word_tokenize(definitions[j])
            temp_b = nltk.pos_tag(text2)
            b = set(y[1] for y in temp_b)

            # Computing similarity between definitions
            # intersec = [z for z in a if z in b]
            t = len(a & b) / min(len(a), len(b))  # normalization step
            results.append(t)
            j = j + 1

        i = i + 1

    return results


def compute_overlap_cosine(definitions):
    """ It computes the overlap between the two set of the preprocessed definitions 
    using cosine similarity.
    Args:
        definitions: a list of definitions (strings)
    Returns:
        a list of length |definitions| containing the maximum similarity
    score of each definition.
    """

    # Preprocess step.
    clean_defs = []
    for de in definitions:
        clean_defs.append(preprocess(de, LIST))

    '''
    CountVectorizer will create k vectors in n-dimensional space, where:
    - k is the number of sentences,
    - n is the number of unique words in all sentences combined.
    If a sentence contains a certain word, the value will be 1 and 0 otherwise
    '''
    vectorizer = CountVectorizer().fit_transform(clean_defs)
    vectors = vectorizer.toarray()

    results = []
    i = 0

    # Performed (n(n+1))/2 times
    while i < len(vectors):
        a = vectors[i]
        j = i + 1
        while j < len(definitions) - 1:
            # print(i,j)  # DEBUG
            b = vectors[j]

            # Computing cosine similarity between definitions.
            # cosine_similarity() expect 2D arrays, and the input vectors are
            # 1D arrays, so we need reshaping.
            a = a.reshape(1, -1)
            b = b.reshape(1, -1)
            res = cosine_similarity(a, b)[0][0]

            results.append(res)
            j = j + 1

        i = i + 1

    return results


def plot_panda_frame(title, my_percentage):
    """
    Plot the matrix using mathplotlib and saves it in the output folder
    Args:
        title: name of the plot
        my_percentage: matrix containing the proper percentage
    """
    print_val = [[my_percentage["generic_abstract"], my_percentage["generic_concrete"]],
                 [my_percentage["specific_abstract"], my_percentage["specific_concrete"]]]
    df1 = pd.DataFrame(print_val, columns=["Abstract", "Concrete"],
                       index=["Generic", "Specific"])
    df1.plot.bar()
    plt.xticks(rotation=30, horizontalalignment="center")
    plt.title(title)
    plt.xlabel("Concepts")
    plt.ylabel("Similarity (higher is better)")
    # saving plot in output folder
    now = datetime.now().strftime(f"{title} - %d.%m.%Y-%H:%M:%S")  # dd/mm/YY-H:M:S
    plt.savefig('output/{}.png'.format(now))
    plt.show()
    print(f"\n{title}'s plot saved in output folder.")


if __name__ == '__main__':
    path = Path('.') / 'input' / '1-1_defs.csv'
    defs = load_data(path)  # Loading the definition file

    count = 0
    first_row = []  # generic abstract, concrete
    second_row = []  # specific abstract, concrete
    third_row = []  # generic 2 abstract, concrete
    fourth_row = []  # specific 2 abstract, concrete
    fifth_row = []  # generic 3 abstract, concrete
    sixth_row = []  # specific 3 abstract, concrete

    percentage1 = {
        "generic_abstract": 0,
        "generic_concrete": 0,
        "specific_abstract": 0,
        "specific_concrete": 0
    }

    percentage2 = {
        "generic_abstract": 0,
        "generic_concrete": 0,
        "specific_abstract": 0,
        "specific_concrete": 0
    }

    percentage3 = {
        "generic_abstract": 0,
        "generic_concrete": 0,
        "specific_abstract": 0,
        "specific_concrete": 0
    }

    for d in defs:
        # computing the mean of the overlap of the definitions
        overlap_terms = compute_overlap_terms(d)
        mean_terms = np.mean(overlap_terms)

        overlap_pos = compute_overlap_pos(d)
        mean_pos = np.mean(overlap_pos)

        overlap_cosine = compute_overlap_cosine(d)
        mean_cosine = np.mean(overlap_cosine)

        # filling the rows
        if count == 0:
            first_row.append('{:.0%}'.format(mean_terms))
            percentage1["generic_abstract"] = mean_terms
            third_row.append('{:.0%}'.format(mean_pos))
            percentage2["generic_abstract"] = mean_pos
            fifth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["generic_abstract"] = mean_cosine
        elif count == 1:
            first_row.append('{:.0%}'.format(mean_terms))
            percentage1["generic_concrete"] = mean_terms
            third_row.append('{:.0%}'.format(mean_pos))
            percentage2["generic_concrete"] = mean_pos
            fifth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["generic_concrete"] = mean_cosine
        elif count == 2:
            second_row.append('{:.0%}'.format(mean_terms))
            percentage1["specific_abstract"] = mean_terms
            fourth_row.append('{:.0%}'.format(mean_pos))
            percentage2["specific_abstract"] = mean_pos
            sixth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["specific_abstract"] = mean_cosine
        else:
            second_row.append('{:.0%}'.format(mean_terms))
            percentage1["specific_concrete"] = mean_terms
            fourth_row.append('{:.0%}'.format(mean_pos))
            percentage2["specific_concrete"] = mean_pos
            sixth_row.append('{:.0%}'.format(mean_cosine))
            percentage3["specific_concrete"] = mean_cosine

        count += 1

    # build and print dataframe
    df_baseline = pd.DataFrame([first_row, second_row], columns=["Abstract", "Concrete"],
                               index=["Generic", "Specific"])
    df_pos = pd.DataFrame([third_row, fourth_row], columns=["Abstract", "Concrete"],
                          index=["Generic", "Specific"])
    df_cosine = pd.DataFrame([fifth_row, sixth_row], columns=["Abstract", "Concrete"],
                             index=["Generic", "Specific"])

    # Pandas Print -------------------------------------------------------------

    print("\n\nBaseline:\n")
    print(df_baseline)
    print("\nPOS Experiment:\n")
    print(df_pos)
    print("\nCosine Similarity Experiment:\n")
    print(df_cosine)

    # Pandas Plot -------------------------------------------------------------

    # Baseline
    plot_panda_frame(title="Baseline", my_percentage=percentage1)

    # POS Experiment
    plot_panda_frame(title="POS Experiment", my_percentage=percentage2)

    # Cosine Similarity Experiment
    plot_panda_frame(title="Cosine Similarity Experiment", my_percentage=percentage3)
