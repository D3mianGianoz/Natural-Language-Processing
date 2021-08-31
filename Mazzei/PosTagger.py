import math
from collections import Counter

LATIN = 42
GREEK = 17
LATIN_DEV = "corpus/latin-dev.conllu"
LATIN_TEST = "corpus/latin-test.conllu"
LATIN_TRAIN = "corpus/latin-train.conllu"
GREEK_DEV = "corpus/greek-dev.conllu"
GREEK_TEST = "corpus/greek-test.conllu"
GREEK_TRAIN = "corpus/greek-train.conllu"


def num_occurrence(d):
    total_sum = 0
    for elem in d:
        total_sum += d[elem]
    return total_sum


class PosTagger:
    def __init__(self, language: int):
        self.n_tags_given_word = dict()
        self.n_tags_given_tag = dict()

        # 1. Contatori
        if language == LATIN:
            self.create_counters(LATIN_TEST)
        # 2.
        self.prob_word_given_tag, self.prob_tag_given_pred_tag = self.calculate_prob()

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
        # Creating the statistical model and Calculate P(ti | ti-1) and P(wi | ti)
        P(ti | ti-1): probabilità che compaia un tag t dato un tag precedente ti
        P(wi | ti): probabilità che un tag t sia attribuito ad una parola w
        :return:
        """
        prob_tag_given_pred_tag = self.aux_calculate_prob(self.n_tags_given_tag)
        prob_word_given_tag = self.aux_calculate_prob(self.n_tags_given_word)

        return prob_word_given_tag, prob_tag_given_pred_tag

    @staticmethod
    def aux_calculate_prob(dict1: dict) -> dict:
        """
        Iterating through the multi dimension dict we can calculate the natural log of probability.
        La prob è salvata in maniera logaritmica in modo tale da non avere valori troppo piccoli in fase di decoding
        evitando fenomeno di underflow.
        :param dict1: dictionary representing n°tags given either t_1:tag or w:word
        :return: result:dict containing all probabilities (either a_ij o b_ij)
        """
        result = dict()
        for element in dict1:
            for n_tag in dict1[element]:
                if element not in result:
                    result[element] = dict()
                result[element][n_tag] = math.log(dict1[element][n_tag] / num_occurrence(dict1[element]))
        return result


if __name__ == '__main__':
    pos = PosTagger(LATIN)





