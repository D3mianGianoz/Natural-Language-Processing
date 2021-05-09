import csv
import re


def read_from_annotation(path):
    """
    it parses the annotated words's csv.
    Args:
        path: path to the annotated word's csv.
    Returns:
         list of annotated terms. Format: [((w1, w2), value)]
    """

    annotation_list = []
    with open(path, 'r', encoding="utf-8-sig") as file:
        reader_results = csv.reader(file, delimiter=',')
        next(reader_results)
        for line in reader_results:
            copule_words = (line[0].lower(), line[1].lower())
            annotation_list.append((copule_words, float(line[2])))
    return annotation_list


def parse_nasari_dictionary(path):
    """It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.
    Returns:
         a dictionary representing the Nasari input file.
    Format: {word: {term:score}}
    """

    nasari_dict = {}

    with open(path, 'r', encoding="utf8") as file:
        for line in file.readlines():
            splits = line.split("\t")
            vector_nasari = []

            for term in splits[1:]:
                vector_nasari.append(re.sub('\n', '', term))

            for key in splits[:1]:
                new_key = key.split("__")

            nasari_dict[new_key[0]] = vector_nasari

    return nasari_dict


def babel_key(path):
    """It parse the SemEval17_IT_senses2synsets.txt file to extract keys as words and values as
    Babel synsets
    Params:
        path
    Returns:
        dict
    """
    key = ''
    synsets_dict = {}
    synsets = []

    with open(path, 'r', encoding="utf8") as file:
        for line in file.readlines():
            if '#' in line:
                synsets = []
                key = re.sub('#', '', line)
            else:
                synsets.append(re.sub('\n', '', line))

            synsets_dict[re.sub('\n', '', key)] = synsets

    return synsets_dict


def get_nasari_vectors(path, word, nasari_dict):
    """Given a path for the babel key function, a word and nasari dict
    it returns a list of dense nasari vectors associated to that word
    Params:
        path: path o
        word:
        Nasari dict:
    Returns:
        Nasari vectors in a dense form of a word
    """

    babel_codes = babel_key(path)[str(word)]

    nasari = []

    for code in babel_codes:
        if code in nasari_dict.keys():
            nasari.append(nasari_dict[str(code)])

    return nasari
