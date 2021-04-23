import hashlib
from random import randint
from random import seed

from nltk.corpus import framenet as fn


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


def get_wordnet_context(word):
    """
       Params:
            word: word for which we need to find meaning
       Returns:
            a dictionary of Synset associated to the given word
    """
    pass


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
        fID = framenet_IDs[(base_idx+offset)%nof_frames]
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
