import re
from pathlib import Path

import nltk
from nltk.corpus import framenet as fn

from DisambiguateFN.utils import read_correct_synsets, get_frame_set_for_student

surnames = ['gianotti', 'Demaria']
"""
In this file, we executes Task 2 (FrameNet Disambiguation)
"""


def evaluate_performance():
    pass


def get_main_clause(frame_name):
    """Get of the main clause from the frame name ("reggente").
    Params:
        frame_name: the name of the frame
    Returns:
         the main clause inside the frame name
    """
    tokens = nltk.word_tokenize(re.sub('\_', ' ', frame_name))
    tokens = nltk.pos_tag(tokens)

    for elem in reversed(tokens):
        if elem[1] == "NN" or elem[1] == "NNS":
            return elem[0]


def populate_contexts(f, mode: str):
    context_w = []  # the Framenet context
    context_s = {}  # the Wordnet context

    if mode == "Frame name":
        main_clause = get_main_clause(f.name)
        context_w = [main_clause, f.definition]

        context_s = {"test": main_clause}

    elif mode == "FEs":
        # Populating ctx_w for FEs
        pass
    elif mode == "LUs":
        # Populating ctx_w for LUs
        pass

    return context_w, context_s


def bag_of_words(ctx_fn, ctx_wn):
    return True


if __name__ == "__main__":
    correct_synsets_path = Path('.') / 'datasets' / 'AnnotationsFN'
    output_path = Path('.') / 'output'

    with open(output_path / 'results.csv', "w", encoding="utf-8") as out:

        print("Assigning Synsets...")

        for surname in surnames:
            frame_ids = get_frame_set_for_student(surname)

            surname_path = correct_synsets_path / (surname + '.txt')
            read_correct_synsets(surname_path)
            for fID in frame_ids:
                frame = fn.frame_by_id(fID)
                # calculate context of FN: ctx_w and WN: ctx_s
                ctx_w, ctx_s = populate_contexts(frame, "Frame name")
                sense_name = bag_of_words(ctx_fn=ctx_w, ctx_wn=ctx_s)

            out.write("Frame name, {0}, Wordnet Synset, {1}\n".format(frame.name, sense_name))

    evaluate_performance()
