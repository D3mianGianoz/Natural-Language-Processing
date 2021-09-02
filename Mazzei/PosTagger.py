import math
import pyconll
from collections import Counter


ALMOST_ZERO_P = -99999
LATIN = 42
GREEK = 17
LATIN_DEV = "corpus/latin-dev.conllu"
LATIN_TEST = "corpus/latin-test.conllu"
LATIN_TRAIN = "corpus/latin-train.conllu"
GREEK_DEV = "corpus/greek-dev.conllu"
GREEK_TEST = "corpus/greek-test.conllu"
GREEK_TRAIN = "corpus/greek-train.conllu"


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
            # 1. Contatori
            self.create_counters(LATIN_TRAIN)
            # 2.
            self.prob_word_given_tag, self.prob_tag_given_pred_tag = self.calculate_prob()
            # 3.
            self.rate(LATIN_TEST)

    def create_counters(self, path: str):
        """
        Parsing corpus and counting tags given word and tags given tag adding special tag 'START' and 'END'.

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
            for i, line in enumerate(corpus):
                if line[0] == '#':
                    pass
                # Handle ending of phrases
                if line == '\n':
                    self.n_tags_given_tag[tag].update({'END': 1})

                if line.split('\t')[0].isdigit():
                    split_line = line.split('\t')
                    tag = split_line[3]
                    word = split_line[1]

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
        # Creating the statistical model and calculate P(ti | ti-1) and P(wi | ti)
        P(ti | ti-1): probabilità che compaia un tag t dato un tag precedente ti
        P(wi | ti): probabilità che un tag t sia attribuito ad una parola w
        :return:
        """
        prob_tag_given_pred_tag = self.aux_calculate_prob(self.n_tags_given_tag)
        prob_word_given_tag = self.aux_calculate_prob(self.n_tags_given_word, self.n_tags_given_tag)

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
                    r = math.log(n_tags_giv[element][n_tag] / num_occurrence(given_tag[n_tag]))
                else:
                    # Calculate P(ti | ti-1)
                    r = math.log(n_tags_giv[element][n_tag] / num_occurrence(n_tags_giv[element]))
                result[element][n_tag] = r
        return result

    def rate(self, path_to_test: str):
        phrases, correct_tags = self.get_test_values(path_to_test)
        
        # TODO smoothing technique
        for phrase in phrases[:3]:
            pos_backpointer, viterbi = self.viterbi_algo(phrase, dict())
            print("Che bello questo: ", pos_backpointer)
            pretty_print(viterbi, 2)

            # decoding = self.viterbi(phrase, self.prob_tag_given_pred_tag, self.prob_word_given_tag, dict())

    @staticmethod
    def get_test_values(path_to_test):
        """
        Retrive from .pyconll files the test set
        :param path_to_test:
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

    def viterbi_algo(self, phrase: list, smoothed_p: dict):
        """
        In una fase preliminare viene inizializzata la prima colonna, per ogni tag, con la somma dei logaritmi
        della probabilità che quel tag sia preceduto dal tag START (probabilità di transizione) e della probabilità
        che alla prima parola della frase sia associato un tale tag (probabilità di emissione). Nel caso non
        esistesse una entry nella probabilità di transizione (default e smoothing) per un tag la cella nella matrice
        corrispondente, allora viene messa a probabilità nulla.

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
        for hmm_state in trans_prob.keys():
            if hmm_state != 'START':
                viterbi[hmm_state] = list()
                back_pointer[hmm_state] = list()
                f_token: str = phrase[0]
                probability: float = ALMOST_ZERO_P

                if f_token in smoothed_p:
                    # TODO
                    pass
                else:
                    # computing the probability
                    try:
                        if hmm_state in emission_prob[f_token] and hmm_state in trans_prob['START']:
                            # instead of multiplying we sum
                            probability = trans_prob['START'][hmm_state] + emission_prob[f_token][hmm_state]
                    except KeyError:
                        # word not in the dict
                        print("Token not present")

                # Update the dictionaries
                viterbi[hmm_state].append(probability)
                back_pointer[hmm_state].append(maximum_tag)

        # 2. Recursion step
        for t in range(len(phrase)):
            if t != 0:
                for state in trans_prob.keys():
                    if state != 'START':
                        maximum_tag, bpointer, m_vit = self.maximum(viterbi, phrase[t], t-1, maximum_tag, state)

                        #  nella colonna attuale viene salvato solo il valore massimo tra tutte quelle appena calcolate.
                        viterbi[state].append(round(m_vit, 3))
                        back_pointer[state].append(round(bpointer, 3))

                    # il tag con la probabilità più alta di essere riferito alla parola precedente
                    pos_back_pointer[phrase[t - 1]] = maximum_tag

        # 3. Termination step
        m_path = ALMOST_ZERO_P
        last_tag = ''
        for s in trans_prob.keys():
            if s != 'START':
                if 'END' in trans_prob[s]:
                    temp = viterbi[s][t] + trans_prob[s]['END']
                    if temp > m_path:
                        m_path = temp
                        last_tag = s

        # Update dictionary
        viterbi['END'] = m_path
        pos_back_pointer[phrase[t]] = last_tag

        return pos_back_pointer, viterbi

    # Maximum function for viterbi  algorithm
    def maximum(self, vit, word, index, m_tag, current_state, smoothed=None):
        """
        Helper function of the viterbi algorithm, it handle the task of
        computing max and argmax for the recursion step
        :param vit: viterbi matrix
        :param word: token that we are currently analizing
        :param index: index of the previous word/column (t-1) important for retrieving probabilities
        :param m_tag:
        :param current_state: HMM state that we are analyzing
        :param smoothed: auxiliary dict for word not present in the training set
        :return:
        """
        if smoothed is None:
            smoothed = dict()

        m_vit, bpointer = ALMOST_ZERO_P, ALMOST_ZERO_P
        # clear and short names
        trans_prob: dict = self.prob_tag_given_pred_tag

        for state in trans_prob.keys():
            if state != 'START':
                if word in smoothed:
                    emission_prob = smoothed
                else:
                    emission_prob: dict = self.prob_word_given_tag

                try:
                    if current_state in emission_prob[word] and current_state in trans_prob[state]:
                        temp1 = vit[state][index] + trans_prob[state][current_state] + emission_prob[word][current_state]
                        if temp1 > m_vit:
                            m_vit = temp1

                        temp2 = vit[state][index] + trans_prob[state][current_state]
                        if temp2 > bpointer:
                            bpointer = temp2
                            m_tag = state
                except KeyError:
                    pass

        return m_tag, bpointer, m_vit


if __name__ == '__main__':
    pos = PosTagger(LATIN)
