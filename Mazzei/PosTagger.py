import math
import pyconll
import pandas as pd
from collections import Counter

from Mazzei.Smoothing import basic_smooth, baseline

ALMOST_ZERO_P = -99999
LATIN = 42
GREEK = 17

LATIN_DIC = {
    "dev": "corpus/latin-dev.conllu",
    "test": "corpus/latin-test.conllu",
    "train": "corpus/latin-train.conllu"
}

GREEK_DIC = {
    "dev": "corpus/greek-dev.conllu",
    "test": "corpus/greek-test.conllu",
    "train": "corpus/greek-train.conllu"
}


def num_occurrence(dictionary):
    total_sum = 0
    for elem in dictionary:
        total_sum += dictionary[elem]
    return total_sum


# Debugging printing function
def pretty_print(d, indent=0):
    i = 1
    for key, value in d.items():
        i += 1
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print(value, indent + 2)
        else:
            print('\t' * (indent + 2) + str(value))


class PosTagger:
    def __init__(self, language: int):
        self.n_tags_given_word = dict()
        self.n_tags_given_tag = dict()

        if language == LATIN:
            self.lang = LATIN_DIC
        elif language == GREEK:
            self.lang = GREEK_DIC
        else:
            raise ValueError('bad constructor parameter!', language)

        # 1. Contatori
        self.create_counters(path=self.lang["train"])
        # 2. Computing probabilities
        self.prob_word_given_tag, self.prob_tag_given_pred_tag = self.calculate_prob()
        # 3. Smoothing/testing
        smoothing_tuple = basic_smooth(path=self.lang["test"],
                                       emission=self.prob_word_given_tag, tags=self.prob_tag_given_pred_tag)

        self.rating_metrics = {
            "max_noun smoothing": smoothing_tuple[0],
            "NN_and_VB smoothing": smoothing_tuple[1],
            "uniform smoothing": smoothing_tuple[2]
        }

        # 4. Decoding and rating performance
        self.rate(path_to_test=self.lang["test"])

    def create_counters(self, path: str):
        """
        Parsing corpus and counting tags given word and tags given tag adding special tag 'START' and 'END'.
        We are using collections.Counter() as inner data structure

        Il primo ('START') è un tag usato per calcolare la probabilità che una parola inizi una frase,
        e che quindi sia la prima. 'END' analogamente serve per calcolare la probabilità che una parola
        non sia seguita da nient’altro,ovvero sia alla fine della frase.

        1. Calculating C(ti, wi), n_tags_given_word: counts of tags given specific words
        2. Calculating C(ti-1, ti), n_tags_given_tag: counts of tags given tag
        :param path: path of the corpus
        """
        with open(path, 'r') as corpus:
            tag = ''
            prev_tag = ''
            for line in corpus:
                if line[0] == '#':
                    pass
                # Handle ending of phrases
                if line == '\n':
                    self.n_tags_given_tag[tag].update({'END': 1})

                if line.split('\t')[0].isdigit():
                    split_line = line.split('\t')
                    word = split_line[2]
                    tag = split_line[3]

                    # Calculating C(ti, wi): counts of tags given specific words
                    if word not in self.n_tags_given_word:
                        self.n_tags_given_word[word] = Counter()
                    self.n_tags_given_word[word].update({tag: 1})

                    # Calculating C(ti-1, ti): counts of tags given tag
                    if int(line.split('\t')[0]) > 1:
                        if prev_tag not in self.n_tags_given_tag:
                            self.n_tags_given_tag[prev_tag] = Counter()
                        self.n_tags_given_tag[prev_tag].update({tag: 1})

                    # Handling start of phrase
                    if int(line.split('\t')[0]) == 1:
                        if 'START' not in self.n_tags_given_tag:
                            self.n_tags_given_tag['START'] = Counter()
                        self.n_tags_given_tag['START'].update({tag: 1})

                    prev_tag = tag

    def calculate_prob(self):
        """
        Creating the statistical model and calculate P(ti | ti-1) and P(wi | ti)
        P(ti | ti-1): probabilità che compaia un tag t dato un tag precedente ti
        P(wi | ti): probabilità che un tag t sia attribuito ad una parola w
        :return:
        """
        prob_word_given_tag = self.aux_calculate_prob(self.n_tags_given_word, self.n_tags_given_tag)
        prob_tag_given_pred_tag = self.aux_calculate_prob(self.n_tags_given_tag)

        return prob_word_given_tag, prob_tag_given_pred_tag

    @staticmethod
    def aux_calculate_prob(n_tags_giv: dict, given_tag=None) -> dict:
        """
        Iterating through the multi dimension dict we can calculate the natural log of probability.
        La prob è salvata in maniera logaritmica in modo tale da non avere valori troppo piccoli in fase di decoding
        evitando fenomeno di underflow.
        :param given_tag: optional parameter n_tags_given_tag needed when computing emission probability
        :param n_tags_giv: dictionary representing n°tags given either t_1:tag or w:word
        :return: result:dict containing all probabilities (either a_ij o b_ij)
        """
        result = dict()
        for element in n_tags_giv:
            for n_tag in n_tags_giv[element]:
                if element not in result:
                    result[element] = dict()
                if given_tag is not None:
                    # Calculate P(wi | ti)
                    p = math.log(n_tags_giv[element][n_tag] / num_occurrence(given_tag[n_tag]))
                else:
                    # Calculate P(ti | ti-1)
                    p = math.log(n_tags_giv[element][n_tag] / num_occurrence(n_tags_giv[element]))
                result[element][n_tag] = p
        return result

    @staticmethod
    def get_test_values(path_to_test):
        """
        Retrive from .pyconll files the test set and parse it in appropriate data structure
        :param path_to_test: position on the disk
        :return: test_set: list of phrase, test_pos: list of pos tagging (gold)
        """
        test = pyconll.load_from_file(path_to_test)
        test_set = []
        test_pos = []
        counter = -1
        for sentence in test:
            counter += 1
            test_set.append([])
            test_pos.append([])
            for token in sentence:
                test_set[counter].append(token.lemma)
                test_pos[counter].append(token.upos)

        return test_set, test_pos

    def rate(self, path_to_test: str) -> None:
        """
        Method to evaluate the performance of our system, final step in the initialization of our class
        :param path_to_test: path of the test set used for evaluation
        """
        phrases, correct_tags = self.get_test_values(path_to_test)

        self.rating_metrics["baseline"] = dict()

        for key, metric in self.rating_metrics.items():
            # Reset for every metrics
            n_correct_pos = 0
            total_n_words = 0

            for j, phrase in enumerate(phrases):
                # TODO statistical smoothing
                total_n_words += len(phrase)
                if key != "baseline":
                    pos_backpointer, viterbi = self.viterbi_algo(phrase, metric)
                    # pretty_v = pd.DataFrame(viterbi)
                    # print(pretty_v.head())
                    # pretty_print(viterbi, 2)
                else:
                    pos_backpointer = baseline(phrase, self.n_tags_given_word)

                # print(pos_backpointer)
                # pretty_print(pos_backpointer, 2)

                for k, token in enumerate(phrase):
                    if pos_backpointer[token] == correct_tags[j][k]:
                        n_correct_pos += 1

            print(f"Accuracy of technique {key} is: {round((n_correct_pos * 100) / total_n_words, 3)} %")

    def viterbi_algo(self, phrase: list, smoothed_p: dict):
        """
        In una fase preliminare viene inizializzata la prima colonna, per ogni tag, con la somma dei logaritmi
        della probabilità che quel tag sia preceduto dal tag START (probabilità di transizione) e della probabilità
        che alla prima parola della frase sia associato un tale tag (probabilità di emissione). Nel caso non
        esistesse una entry nella probabilità di transizione (default e smoothing) per un tag la cella nella matrice
        corrispondente, allora viene messa a probabilità nulla (ALMOST_ZERO_P).

        Segue poi una fase ricorsiva dove per ogni parola, per ogni possibile tag, vengono calcolate tutte le
        somme in logaritmo delle probabilità riferite alla colonna precedente più la somma di quelle riferite alla
        parola attuale; nella colonna attuale viene salvato solo il valore massimo tra tutte quelle appena calcolate.
        In parallelo viene salvato anche il tag con la probabilità più alta di essere riferito alla parola precedente.
        In questo modo viene tenuta traccia del miglior tag da attribuire ad ogni parola.

        L’ultimo passaggio è analogo al primo, ma fa riferimento al tag LAST andando a completare l’ultima
        colonna della matrice. A causa dell' utilizzo dei log una probabilità nulla non sarà espressa con 0 ma con un
        numero molto piccolo, un questo caso si è scelto ALMOST_ZERO_P, usandolo nella matrice risultante
        Alla fine dell’algoritmo si ottiene, grazie alla matrice di Viterbi, la più probabile taggatura (POS tagging)
        per la frase in input.

        :param phrase: frase da analizzare
        :param smoothed_p: dizionario extra per migliorare le performance
        :return: pos_back_pointer, viterbi
        """
        # class variables with more clear and short names
        trans_prob: dict = self.prob_tag_given_pred_tag
        emission_prob: dict = self.prob_word_given_tag

        # 0. local variables
        back_pointer = dict()
        pos_back_pointer = dict()
        viterbi = dict()
        maximum_tag: str = '_'  # init as empty space

        # 1. Initialization step
        hmm_state: str

        for hmm_state in trans_prob.keys():
            if hmm_state != 'START':
                # create the rows of the matrix
                viterbi[hmm_state] = list()
                back_pointer[hmm_state] = list()

                # In case we don't find we assign 0 probability
                probability: float = ALMOST_ZERO_P

                if hmm_state in trans_prob['START']:
                    first_word = phrase[0]
                    if first_word in smoothed_p:
                        if hmm_state in smoothed_p[first_word]:
                            probability = trans_prob['START'][hmm_state] + smoothed_p[first_word][hmm_state]
                    else:
                        if hmm_state in emission_prob[first_word]:
                            probability = trans_prob['START'][hmm_state] + emission_prob[first_word][hmm_state]

                # Update the dictionaries key = pos:probability
                viterbi[hmm_state].append(probability)
                back_pointer[hmm_state].append(maximum_tag)

        # 2. Recursion step
        last_t = 0
        for t in range(len(phrase)):
            if t != 0:
                for state in trans_prob.keys():
                    if state != 'START':
                        maximum_tag, bpointer, m_vit = self.maximum(vit=viterbi, word=phrase[t],
                                                                    index=t - 1, m_tag=maximum_tag,
                                                                    current_state=state, smoothed=smoothed_p)

                        #  nella colonna attuale viene salvato solo il valore massimo tra tutte quelle appena calcolate.
                        viterbi[state].append(round(m_vit, 3))
                        back_pointer[state].append(round(bpointer, 3))

                    # il tag con la probabilità più alta di essere riferito alla parola precedente
                    pos_back_pointer[phrase[t - 1]] = maximum_tag
            # salvo l' indice finale
            if t == len(phrase) - 1:
                last_t = t

        # 3. Termination step
        m_path: float = ALMOST_ZERO_P
        last_tag: str = '_'
        for s in trans_prob.keys():
            if s != 'START' and 'END' in trans_prob[s]:
                temp = viterbi[s][last_t] + trans_prob[s]['END']
                if temp > m_path:
                    m_path = temp
                    last_tag = s

        # Update dictionary
        viterbi['END'] = m_path
        pos_back_pointer[phrase[last_t]] = last_tag

        return pos_back_pointer, viterbi

    # Maximum function for viterbi  algorithm
    def maximum(self, vit: dict, word: str, index: int, m_tag: str, current_state: str, smoothed: dict):
        """
        Helper function of the viterbi algorithm, it handle the task of
        computing max and argmax for the recursion step

        :param vit: viterbi matrix
        :param word: token that we are currently analizing
        :param index: index of the previous word/column (t-1) important for retrieving probabilities
        :param m_tag: maximum tag founded till now
        :param current_state: HMM state that we are analyzing
        :param smoothed: auxiliary dict for word not present in the training set
        :return: m_tag: max tag founded for probability,
                bpointer: back pointer for reconstructing the path, m_vit: updated viterbi matrix
        """

        m_vit, bpointer = ALMOST_ZERO_P, ALMOST_ZERO_P
        # clear and short names
        trans_prob: dict = self.prob_tag_given_pred_tag
        emission_prob: dict = self.prob_word_given_tag

        for state in trans_prob.keys():
            if state != 'START':

                if word in smoothed:
                    # We are gonna use the smoothed probabilities in case of proper nouns
                    temp_dict = smoothed
                else:
                    temp_dict = emission_prob

                if current_state in temp_dict[word] and current_state in trans_prob[state]:
                    temp1 = vit[state][index] + trans_prob[state][current_state] + temp_dict[word][current_state]
                    if temp1 > m_vit:
                        m_vit = temp1

                    temp2 = vit[state][index] + trans_prob[state][current_state]
                    if temp2 > bpointer:
                        bpointer = temp2
                        m_tag = state

        return m_tag, bpointer, m_vit


if __name__ == '__main__':
    print("\nPOS tagging Latin LLCT\n")
    pos_latin = PosTagger(LATIN)
    print("\nPOS tagging Ancient Greek Perseus \n")
    pos_greek = PosTagger(GREEK)