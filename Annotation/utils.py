import re
import requests


def parse_nasari_dictionary(path):
    """It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.
    Returns:
        First: a dictionary in which each BabelID is associated with the corresponding NASARI's vector.
        Second: a lexical dictionary that associate to each BabelID the corresponding english term.
    Format: {babelID: {term:score}}, {babelID: word_en}
    """

    nasari_dict = {}
    babel_word_nasari = {}

    with open(path, 'r', encoding="utf8") as file:
        for line in file.readlines():
            splits = line.split("\t")
            vector_nasari = []

            for term in splits[1:]:
                vector_nasari.append(re.sub('\n', '', term))

            for key in splits[:1]:
                new_key = key.split("__")

            nasari_dict[new_key[0]] = vector_nasari
            babel_word_nasari[new_key[0]] = new_key[1]

    return nasari_dict, babel_word_nasari


def babel_key(path):
    """It parse the SemEval17_IT_senses2synsets.txt file to extract keys as words and values as
    Babel synsets
    Args:
        path: absolute path of text file
    Returns:
        synsets_dict: a dictionary containing the italian word follower by the list of its BabelID. Format: {word_it: [BabelID]}
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


def get_nasari_vectors(sense2s_dic, word, nasari_dict):
    """Given a dic for the babel key, a word and nasari dict
    it returns a list of dense nasari vectors associated to that word
    Args:
        sense2s_dic: dic for the babel key
        word: what we are looking for
        Nasari dict:
    Returns:
        Nasari vectors in a dense form of a word
    """

    babel_codes = sense2s_dic[word]
    nasari = []
    matching_bbl = []

    for code in babel_codes:
        if code in nasari_dict.keys():
            nasari.append(nasari_dict[code])
            matching_bbl.append(code)

    return nasari, matching_bbl


def get_synset_terms(sense):
    """It uses the BabelNet HTTP API for getting the first three Lemmas of the word
    associated to the given Babel Synset ID.
    Args:
        sense: sense's BabelID
    Returns:
         the first three lemmas of the given sense.
          An error string if there are none
    """

    url = "https://babelnet.io/v5/getSynset"
    params = {
        "id": sense,
        "key": "67adc825-bbcc-4cd4-8d6c-71ecdb875e7c",  # my API key | backup key 036afd10-7afe-4064-8fe1-1b27515fb8f4
        "targetLang": "IT"  # Important: we are searching results in italian
    }

    req = requests.get(url=url, params=params)
    data = req.json()

    synset_terms = []

    i = 0  # used to loop over the first three terms
    j = 0  # used to loop over all the senses
    while j < len(data["senses"]) and i < 3:
        term = data["senses"][j]["properties"]["fullLemma"]

        # added some preprocess
        term = re.sub('\_', ' ', term).lower()

        if term not in synset_terms:
            synset_terms.append(term)
            i += 1

        j += 1

    if len(synset_terms) == 0:
        return "Empty synset terms"
    else:
        return synset_terms
