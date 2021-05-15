# TLN_dicaro_2.1

"""
CONSEGNA:
* Ispirandosi al text-tiling, implementare un algoritmo di segmentazione del
testo.

* Sfruttare informazioni come le frequenze e le co-occorrenze e eventuale
pre-processing del testo.

SVOLGIMENTO:
* Si è scelto di utilizzare un testo su Trump-wall già usato nella seconda
parte del corso.

"""

from pathlib import Path
from DiCaro.Primo_gruppo.aggregate_concepts import preprocess

ITERATIONS = 3

# Cerco solo i salti -negativi- maggiori  una volta e mezza la media
SCALE_FACTOR = 1.5


def read_text(path):
    """
    :param path: absolute Path of the file
    :return: list of sentences of the textual file
    """
    with open(path, "r", encoding="utf8") as f:
        content = f.read()
        lines = content.split(". ")
        returned_list = []
        for line in lines:
            line = line.replace("\n", "")
            returned_list.append(line)
        return returned_list


def calculate_frequencies(sentences):
    """
     Calculate the frequency of a word (non-stop-words) in a sentence, for all sentences
    :param sentences: lines of text to be processed
    :return: dictionary containing the frequencies
    """
    terms = {}
    pos = 0
    for sentence in sentences:
        words = preprocess(sentence, 0)
        for word in words:
            if word in terms:
                (terms[word])[pos] += 1
            else:
                terms[word] = [0] * len(sentences)
                (terms[word])[pos] = 1
        pos += 1
    return terms


def generate_columns(length_sentences, terms):
    """
    Generate a vector containing the frequencies of each word, for each sentence
    :param length_sentences: n° of sentences
    :param terms: dict of the frequency
    :return: list with frequencies of each word, for each sentence
    """

    columns = list(range(length_sentences))

    for key in terms.keys():
        for i in range(length_sentences):
            freqs = terms[key]
            column = columns[i]
            column.append(freqs[i])
    return columns


if __name__ == '__main__':
    trump_path: Path = Path(".") / "input" / "Trump-wall.txt"
    ss = read_text(trump_path)
    ss_length = len(ss)

    frequency_dict = calculate_frequencies(sentences=ss)
    position_breakpoints = [0]

    col = generate_columns(ss_length, frequency_dict)






