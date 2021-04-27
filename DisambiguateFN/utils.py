import hashlib
from random import randint
from random import seed

from nltk.corpus import framenet as fn
from nltk.corpus import wordnet as wn


def read_correct_synsets(path):
    """
    Support function, it parse the txt file. Each line is compose by
    a couple of terms and their annotation.
    Params:
        path: input path of the CSV file
    Returns:
         a list, representation of the input file. Its format will be [(syn)]
    """
    correct_synset = []
    with open(path, "r") as f:
        syn = f.read()
        synset = syn.split("\n")
        for syn in synset:
            if "#" not in syn:
                correct_synset.append(syn)
    return correct_synset


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def get_wordnet_context(word):
    """
       Params:
            word: word for which we need to find meaning
       Returns:
            a dictionary of Synset associated to the given word
    """
    synsets = wn.synsets(word)
    ret = {}  # return variable

    for syns in synsets:
        if syns.examples():
            t = [syns.lemma_names()[0], syns.examples()[0]]
        else:
            t = [syns.lemma_names()[0], []]

            # 3 è una costante fornita dalla consegna,
            # non si considerano path più lunghi
            i = 0
            for hypo in syns.hyponyms():
                if i == 3:
                    break
                if hypo.lemma_names():
                    t.append(hypo.lemma_names()[0])
                if hypo.examples():
                    t.append(hypo.examples()[0])
                i += 1

            i = 0
            for hyper in syns.hypernyms():
                if i == 3:
                    break
                if hyper.lemma_names():
                    t.append(hyper.lemma_names()[0])
                if hyper.examples():
                    t.append(hyper.examples()[0])
                i += 1

        ret[syns.name()] = t

    return ret


"""
    The functions below these lines were made by professor Radicioni
"""


def get_frame_set_for_student(surname, list_len=5):
    ids_list = []
    nof_frames = len(fn.frames())
    base_idx = (abs(int(hashlib.sha512(surname.encode('utf-8')).hexdigest(), 16)) % nof_frames)
    print('\nstudent: ' + surname)
    framenet_IDs = get_frames_ids()
    i = 0
    offset = 0
    seed(1)
    while i < list_len:
        fID = framenet_IDs[(base_idx + offset) % nof_frames]
        f = fn.frame(fID)
        ids_list.append(fID)
        fNAME = f.name
        print('\tID: {a:4d}\tframe: {framename}'.format(a=fID, framename=fNAME))
        offset = randint(0, nof_frames)
        i += 1
    return ids_list


def print_frames_with_ids():
    for x in fn.frames():
        print('{}\t{}'.format(x.ID, x.name))


def get_frames_ids():
    return [f.ID for f in fn.frames()]
