import operator
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
                word = splitlines[1] #word form
                if word not in emission:
                    em_noun[word], em_nn_vb[word], em_uniform[word] = dict(), dict(), dict()

                    # P(unk|NOUN) = 1
                    em_noun[word]['NOUN'] = log(1)

                    # P(unk|NOUN)=P(unk|VERB)=0.5
                    em_nn_vb[word]['NOUN'] = log(0.5)
                    em_nn_vb[word]['VERB'] = log(0.5)

                    # P(unk|ti) = 1/#(PoS_TAGs)
                    for tag in tags:
                        if tag != 'BLANK':
                            em_uniform[word][tag] = log(1 / (len(tags) - 1))

    return em_noun, em_nn_vb, em_uniform


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
