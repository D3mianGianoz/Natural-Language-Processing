import operator
import pyconll
from collections import Counter
from math import log

"""
Questa fase è resa necessaria dal fatto che è ovviamente possibile, in fase di test, trovare delle parole
che non sono presenti nel training set.
Per questa motivazione è necessario taggare con una certa probabilità anche codeste parole
sconosciute. Per farlo sono state create nuove probabilità di emissione.
"""


def basic_smooth(path, emission, tags) -> (dict, dict, dict):
    """
    First 3 basic smoothing assumption.

    - P(unk|NOUN) = 1 Ad una parola sconosciuta si attribuisce il tag NOUN con una probabilità massima.
    - P(unk|NOUN)=P(unk|VERB)=0.5 Ad una parola sconosciuta si
      attribuisce il tag NOUN e VERB con il 50% di probabilità per ognuno.
    - P(unk|ti) = 1/#(PoS_TAGs) Ad una parola sconosciuta viene associato ogni tag con probabilità
      uniforme ovvero ogni possibile tag ha la stessa probabilità di essere associato a quella parola.
    :param path: path to file
    :param emission: probability of word given tag
    :param tags: lista dei possibili tag
    :return: em_noun, em_nn_vb, em_uniform
    """
    em_noun = dict()
    em_nn_vb = dict()
    em_uniform = dict()
    with open(path, 'r') as file:
        for line in file:
            if line.split('\t')[0].isdigit():
                splitlines = line.split('\t')
                word = splitlines[1]  # word form
                if word not in emission:
                    em_noun[word], em_nn_vb[word], em_uniform[word] = dict(), dict(), dict()

                    # P(unk|NOUN) = 1
                    em_noun[word]['NOUN'] = log(1)

                    # P(unk|NOUN)=P(unk|VERB)=0.5
                    em_nn_vb[word]['NOUN'] = log(0.5)
                    em_nn_vb[word]['VERB'] = log(0.5)

                    # P(unk|ti) = 1/#(PoS_TAGs)
                    for tag in tags:
                        if tag != 'START':
                            em_uniform[word][tag] = log(1 / (len(tags) - 1))

    return em_noun, em_nn_vb, em_uniform


# Statistical smoothing assumption on DEV file
def statistical_smooth(path_dev, path_test, emission, tags):
    """
    Smoothing statistico su set di validazione (Dev set).

    Si prendono in considerazione le parole che compaiono solamente una volta (unique-word).
    Per ogni tag si conta quante volte è associato ad una unique-word.

    Dunque, per ogni parola sconosciuta viene calcolata la probabilità per ogni tag come
    il rapporto tra il numero totale di occorrenze di tale tag per le unique-word e il numero di quest’ultime.

    P(unk|ti) = #(PoS_TAGs_UNIQUE)ti/#(UNIQUE) )

    :param path_dev: percorso per il dev treebank
    :param path_test: percorso per il dev treebank
    :param emission: dizionario multidimensionale contenente le probabilità di emissione
    :param tags: dizionario multidimensionale contenente le probabilità di transizione
    :return:
    """
    word_count = Counter()
    tag_count = Counter()
    tag_count_by_word = dict()

    num_dev_unk_words = 0
    dev_smooth = dict()

    with open(path_dev, 'r') as dev:
        for line in dev:
            if line.split('\t')[0].isdigit():
                splitline = line.split('\t')
                word = splitline[1]
                tag = splitline[3]

                if word not in word_count:
                    tag_count_by_word[word] = Counter()
                word_count.update({word: 1})
                tag_count_by_word[word].update({tag: 1})

        for word in word_count.keys():
            for tag in tag_count_by_word[word].keys():
                # se la parola è univoca
                if word_count[word] == 1:
                    tag_count.update({tag: tag_count_by_word[word][tag]})

            if word_count[word] == 1:
                num_dev_unk_words += 1

    test_phrases, _ = load_treebank_values(path_test)

    for phrase in test_phrases:
        for word in phrase:
            if word not in emission:
                dev_smooth[word] = dict()
                for tag in tags:
                    if tag != 'START':
                        if tag_count[tag] > 0:
                            dev_smooth[word][tag] = log(tag_count[tag] / num_dev_unk_words)
    return dev_smooth


def baseline(phrase, n_tags_given_word) -> dict:
    """
    Questa funzione rappresenta una possibile implementazione del semplice
    approccio statistico basato sulle frequenze.

    - Se la parola è conosciuta prendiamo il suo tag più frequente;
    - se è sconosciuta assumiamo sia un nome.
    
    Baseline is performance of stupidest possible method. Tag every word with its most frequent tag.
    Tag unknown words as nouns.
    Partly easy because:

    - Many words are unambiguous
    - You get points for them (the, a, etc.) and for punctuation marks!

    :param phrase: frase da analizzare
    :param n_tags_given_word: Counter realizzato per contare quali tag sono associati ad una parola
    :return: res: dict di pos tagging, uno per ogni parola
    """
    res = dict()

    for word in phrase:
        if word in n_tags_given_word:
            res[word] = max(n_tags_given_word[word].items(), key=operator.itemgetter(1))[0]
        else:
            res[word] = 'NOUN'
    return res


def load_treebank_values(path_to_file):
    """
    Retrive from .pyconll files the test set and parse it in appropriate data structure
    :param path_to_file: position on the disk
    :return: test_set: list of phrase, test_pos: list of pos tagging (gold)
    """
    test = pyconll.load_from_file(path_to_file)
    test_set = []
    test_pos = []
    counter = -1
    for sentence in test:
        counter += 1
        test_set.append([])
        test_pos.append([])
        for token in sentence:
            test_set[counter].append(token.form)
            test_pos[counter].append(token.upos)

    return test_set, test_pos
