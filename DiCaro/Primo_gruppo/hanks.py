# TLN_dicaro_1.4
from pathlib import Path
import nltk
import re
import pandas as pd
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from collections import Counter
'''
CONSEGNA:
* Implementare un sistema basato sulla teoria di Hanks per la costruzione del
significato.

* Scelto un verbo transitivo (quindi valenza >= 2), recuperare da un corpus
delle istanze in cui viene usato.

* Effettuare il parsing di queste frasi per identificare i supersensi di
WordNet associati agli argomenti del verbo (subject e object).

* Calcolare le frequenze di questi supersensi per i due ruoli e stampare le
possibili combinazioni.

SVOLGIMENTO:
* Si è scelto il verbo 'break', in particolare il presente terza persona singolare.

* Il corpus utilizzato è Wikipedia, da cui sono state estratte 3000 frasi, usando sketch engine

* Per il parsing a dipendenze si è usata la libreria spaCy.

* Sono state scartate quelle frasi in cui il verbo non presenta entrambi i ruoli
richiesti.

* I termini che svolgono i ruoli vengono lemmatizzati e si va poi a calcolare
il loro synset migliore tramite WSD (algoritmo di Lesk). Nel caso il soggetto
sia 'he', è necessario forzare il suo synset a 'person.n.01' per evitare che
venga erroneamente riconosciuto come 'elio'.

* Con questi synset si individua il relativo supersenso, andando a calcolare
poi frequenze e combinazioni possibili.

* Si verifica anche cosa accade raggruppando le combinazioni con ordine inverso,
poiché probabilmente rappresentanti un uso attivo e passivo del verbo.
'''
from DiCaro.Primo_gruppo.word_sense_induction import lesk


def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def skip_synset_search(pos_tag):
    """
    Helper function for forcing certainty synset to some
    :param pos_tag:
    :return: boolean value, Synset (or None)
    """
    # pronouns and proper nouns
    if pos_tag == "PRP" or pos_tag == 'NNP':
        return True, wordnet.synset('person.n.01')
    # Which / That
    # elif pos_tag == "WDT" or pos_tag == "IN":
    #     return True, wordnet.synset('artifact.n.01')
    else:
        return False, None


def iequal(a, b):
    """
    Helper method for comparing tuples with string element
    :param a: tuples
    :param b: string
    :return: True if string in tuples
    """
    for value in a:
        try:
            sub = re.sub("\'", "", str(value))
            if b == sub:
                return True
        except AttributeError:
            return value == b


def plot_hanks(data, max_v, title):
    """
    Create a Data Frame only for plotting
    :param data: dictionary to bar plot
    :param max_v: maximum value of labels
    :param title: name of the plot to show
    :return: ax to plot from
    """
    plot_df = pd.DataFrame(data, index=['quantity'])
    print(plot_df.head(5))

    # Sort the value
    plot_df = plot_df.sort_values(by=['quantity'], axis=1, ascending=False)

    ax = plot_df.plot(kind='barh', title=title, width=8)
    colors, labels = ax.get_legend_handles_labels()

    # envelop in counter
    data = Counter(data)

    # sort the most coomon labels keep only max_v
    best_legend, order = zip(*data.most_common(max_v))
    zipped = zip(colors, labels)
    colors, labels = zip(*[(color, label) for (color, label) in zipped if iequal(best_legend, label)])

    ax.legend(colors, labels, loc='best')
    return ax


def update_occurrences(ss1, ss2, dictionary):
    """
    Update the dictionary {(s1:s2): counter}
    :param ss1: synset1
    :param ss2: synset2
    :param dictionary: counter dictionary
    """
    if ss1 is not None and ss2 is not None:
        # Getting supersenses
        t = (ss1.lexname(), ss2.lexname())

        # Getting frequency
        if t in dictionary.keys():
            dictionary[t] = dictionary[t] + 1
        else:
            dictionary[t] = 1


def fancy_pint(title, dictionary):
    """
    Helper function for evaluating performance
    :param title: string regarding the strategy
    :param dictionary: the semantic strategy
    """
    print('\n%s:\n\tFinding Semantic Clusters (percentage, count of instances, semantic cluster):' % title)
    for key, value in sorted(dictionary.items(), key=lambda x: x[1]):
        to_print = str(round((value / f_length) * 100, 2))
        print("\t[{}%] - {} - {}".format(to_print, value, key))


if __name__ == '__main__':
    # 1 I choose said, that have a good context in brown
    verb_of_interest = ["breaks"]

    # 2. let's get a list of sentences with our verb_of_interest in from https://app.sketchengine.eu/
    # already pre-processed
    # path = Path('.') / 'input' / 'concordance_brown_family.csv'
    path = Path('.') / 'input' / 'concordance_enwiki.csv'

    brown_says = pd.read_csv(path)

    print('[1] - Extracting sentences...')
    sentences = brown_says["Sentence"]
    head_test = sentences.head(1001)
    head_test = head_test.apply(lambda x: x.strip("<s> \'\' \" </"), lambda x: re.sub('</"', "", x))

    import spacy
    nlp = spacy.load("en_core_web_sm")
    # Construction via add_pipe with default model
    sents = nlp(u'A woman is walking through the door.')

    for token in sents:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_)

    print('\n[2] - Extracting fillers...')
    # parsing di tutte le frasi
    head_test_v = head_test.apply(lambda x: nlp(x)).values

    fillers = []  # [(subj, obj, sentence)]
    for i in range(len(head_test_v)):
        sentence = head_test_v[i]
        obj = None
        subj = None
        for token in sentence:
            # Se il verbo reggente è il verbo di nostro interesse
            if token.head.text in verb_of_interest:
                # oggetto
                if token.dep == 416:
                    obj = (token.text, token.tag_)
                # soggetto
                elif token.dep == 429:
                    subj = (token.text, token.tag_)
        if obj and subj:
            fillers.append((subj, obj, sentence.text))

    f_length = len(fillers)
    print(f"\n[3] - Total of {f_length} Fillers")

    our_lesk_semantic_types = {}  # {(s1, s2): count}
    nltk_lesk_semantic_types = {}  # {(s1, s2): count}
    for f in fillers:
        print("\t{}".format(f))

        subj_simple = f[0][0]
        subj_tag = f[0][1]
        obj_simple = f[1][0]
        obj_tag = f[1][1]

        # Filtering pronouns
        skip_subj, skipped_s1 = skip_synset_search(pos_tag=subj_tag)
        skip_obj, skipped_s2 = skip_synset_search(pos_tag=obj_tag)

        # Vado a chiamare lesk sulla coppia della parte prima della mia frase e complemento oggetto dopo il verbo?
        if not skip_subj:
            # Our Lesk
            s1 = lesk(subj_simple, f[2])
            # Wordnet Lesk
            s3 = nltk.wsd.lesk(f[2], subj_simple)
        else:
            s1 = s3 = skipped_s1

        if not skip_obj:
            s2 = lesk(obj_simple, f[2])
            s4 = nltk.wsd.lesk(f[2], obj_simple)
        else:
            s2 = s4 = skipped_s1

        # Our
        update_occurrences(s1, s2, our_lesk_semantic_types)

        # Library
        update_occurrences(s3, s4, nltk_lesk_semantic_types)

    our_lesk_ = '[4.1] - "Our Lesk"'
    fancy_pint(our_lesk_, our_lesk_semantic_types)

    nltk_lesk_ = '[4.2] - "NLTK Lesk"'
    fancy_pint(nltk_lesk_, nltk_lesk_semantic_types)

    # Plot
    ax1 = plot_hanks(our_lesk_semantic_types, 10, f"Hanks cluster with {our_lesk_} for {verb_of_interest[0]}")
    plt.show()
    ax2 = plot_hanks(nltk_lesk_semantic_types, 10, f"Hanks cluster with {nltk_lesk_} for {verb_of_interest[0]}")
    plt.show()

    # Alternative parsing
    # from nltk.parse.corenlp import CoreNLPServer, CoreNLPParser, CoreNLPDependencyParser
    #
    # # The server needs to know the location of the following files:
    # #   - stanford-corenlp-X.X.X.jar
    # #   - stanford-corenlp-X.X.X-models.jar
    # STANFORD = os.path.join("/home/damians/Scripts/GitHub/", "stanford-corenlp-4.2.1")
    #
    # # Create the server
    # jars = (
    #     os.path.join(STANFORD, "stanford-corenlp-4.2.1.jar"),
    #     os.path.join(STANFORD, "stanford-corenlp-4.2.1-models.jar"),
    # )
    #
    # # Start the server in the background
    # with CoreNLPServer(*jars):
    #     # test of parsing
    #     parser = CoreNLPDependencyParser()
    #     parsed_sentence = next(parser.raw_parse("I put the book in the box on the table."))
    #     # parsed_sentence.pretty_print()
    #
    #     lemma_df = head_test.apply(lambda x: next(parser.parse(x)))
    #     # print([[(governor, dep, dependent) for governor, dep, dependent in parse.triples()] for parse in lemma])
    #     # lemma_col = lemma.to_conll(4)
