import csv
import re
from pathlib import Path

import nltk
from nltk.corpus import framenet as fn
from nltk.corpus import stopwords

from DisambiguateFN.utils import read_correct_synsets, get_frame_set_for_student, get_wordnet_context, find_between

"""
In this file, we executes Task 2 (FrameNet Disambiguation)
"""

stop_words = set(stopwords.words('english'))
wnl = nltk.WordNetLemmatizer()


# The idea is to remove punctuation and stop words
def clean_word(word):
    word = re.sub('[^A-Za-z0-9 ]+', '', word)
    # Returns the input word unchanged if it cannot be found in WordNet.
    if word not in stop_words:
        return wnl.lemmatize(word)
    return ""


def evaluate_performance(gold_path, out_path, none=True):
    """
    Doing output evaluation and print the result on the console.
    If none = False it is not considering none while counting matching
    Parameters
    ----------
    gold_path
    out_path
    none
    """
    total_none = 0
    test = 0
    with open(out_path, "r", encoding="utf-8") as results:
        reader_golden = read_correct_synsets(gold_path)
        reader_results = csv.reader(results, delimiter=',')

        items_in_results = []  # list of items in results
        items_in_golden = []  # list of items in gold

        for line_out in reader_results:
            items_in_results.append(line_out[-1])

        for elem in reader_golden:
            # Unwrapping the synset
            start = 'Synset(\''
            end = '\')'

            if elem == 'None':
                items_in_golden.append(elem)
                #pass
            else:
                items_in_golden.append(find_between(elem, start, end))

        # counting equal elements
        index = 0
        contatore = 0

        while index < len(items_in_golden):
            # Item either not found in wordnet or with no correct sense in the resource
            if (items_in_results[index] == 'None' or items_in_golden[index] == 'None') and none is False:
                total_none += 1
            else:
                contatore += 1
                result = items_in_results[index]
                gold = items_in_golden[index]
            # print(f'risultato:{result} ---- golden:{gold}')
            if result == gold:
                test += 1
            index += 1

        total_len = contatore

    print("\nPrecision: {0} / {1} Synsets -> {2:.2f} % Number of None: {3}".format(test, total_len, (test / total_len) * 100, total_none))

def get_main_clause(frame_name):
    """Get of the main clause from the frame name ("reggente").
    Params:
        frame_name: the name of the frame
    Returns:
         the main clause inside the frame name
    """
    tokens = nltk.word_tokenize(re.sub('_', ' ', frame_name))
    tokens = nltk.pos_tag(tokens)

    for elem in reversed(tokens):
        if elem[1] == "NN" or elem[1] == "NNS":
            return elem[0]


def populate_contexts(f, mode: str):
    """It populates 2 disambiguation context (one for Framenet and one for Wordnet)
    given a frame name.

    Params:
        frame: the frame name.
        mode: a string indicating the way to create context the possibility are: "Frame name", "FEs" and "LUs".
    Returns:
         two list (context_w, context_s) representing the populated contexts.
    """
    context_w = []  # the Framenet context
    context_s = []  # the Wordnet context

    if mode == "Frame name":
        if "Motion" in f.name:
            main_clause = "Motion"
        elif "_" in f.name:
            main_clause = get_main_clause(f.name)
        else:
            main_clause = f.name
        # The context in this case contains the frame name and his definition.
        context_w = [main_clause, f.definition]

        # Here, the context is a list of synset associated to the frame name.
        # In each synset are usually present word, glosses and examples.
        context_s = get_wordnet_context(main_clause)

    elif mode == "FEs":
        # Populating ctx_w for FEs
        for key in sorted(f.FE):
            if "_" in key:
                main_clause = get_main_clause(key)
            else:
                main_clause = key
            fe = f.FE[key]
            context_w.append([main_clause, fe.definition])

            # appending the value to ctx_s list.
            context_s.append(get_wordnet_context(main_clause))

    elif mode == "LUs":
        # Populating ctx_w for LUs
        for key in sorted(f.lexUnit):
            lu_key = re.sub('\.[a-z]+', '', key)
            context_w.append(lu_key)

            # appending the value to ctx_s list.
            context_s.append(get_wordnet_context(lu_key))

    return context_w, context_s


def bag_of_words(ctx_fn, ctx_wn):
    """ Given two disambiguation context, it returns the best sense using the
     bag of words mapping between the input arguments.
    Params:
        sent: sentence
    Returns:
        the best sense
    """
    sentences_fn = set()  # set of all Framenet FEs and their descriptions
    sentences_wn = {}  # dictionary of all Wordnet synset, glosses and examples.
    temp_max = -1
    ret = False

    for sentence in ctx_fn:
        for word in sentence.split():
            word_clean = clean_word(word)
            sentences_fn.add(word_clean)

    # transform the ctx_w dictionary into a set, in order to compute
    # intersection.
    for key in ctx_wn:  # for each WN synset
        temp_set = set()
        for sentence in ctx_wn[key]:  # for each sentence inside WN synset
            if sentence:
                for word in sentence.split():
                    # print(f"Parola da pulire {word}")
                    word = clean_word(word)
                    # print(f"Parola da pulita {word}")
                    temp_set.add(word)  # add words to temp_set

        # computing intersection between temp_set and sentences_fn.
        # Putting the result inside sentences_wn[key].
        # Each entry in sentences_wn will have the cardinality of the
        # intersection as his "score" at the first position.
        sentences_wn[key] = (len(temp_set & sentences_fn), temp_set)

        # update max score and save the associated sentence.
        if temp_max < sentences_wn[key][0]:
            temp_max = sentences_wn[key][0]
            ret = (key, sentences_wn[key])

    # return the best sense maximizing the score
    if ret:
        return ret[0]
    else:
        # couldn't find the word on Wordnet ex: Cognizer
        # Warning("No bag of words created")
        return None


if __name__ == "__main__":
    correct_synsets_path = Path('.') / 'datasets' / 'AnnotationsFN'
    output_path = Path('.') / 'output'
    surnames = ['Demaria', 'gianotti']
    fn_modes = ["Frame name", "FEs", "LUs"]

    print("Assigning Synsets...")

    for surname in surnames:
        path = output_path / f'task2_results_{surname}.csv'
        with open(path, "w", encoding="utf-8") as out:

            frame_ids = get_frame_set_for_student(surname)
            # Retrive gold annotation
            surname_path = correct_synsets_path / (surname + '.txt')

            for fID in frame_ids:
                frame = fn.frame_by_id(fID)
                # calculate context of FN: ctx_w and WN: ctx_s
                ctx_w, ctx_s = populate_contexts(f=frame, mode=fn_modes[0])
                sense_name = bag_of_words(ctx_fn=ctx_w, ctx_wn=ctx_s)
                # Write it to file
                out.write("{0}, {1},Wordnet Synset,{2}\n".format(fn_modes[0], frame.name, sense_name))

                # Implemented (Demaria)
                ctx_w, ctx_s = populate_contexts(f=frame, mode=fn_modes[1])
                for (i, j) in zip(ctx_w, ctx_s):
                    sense_name = bag_of_words(ctx_fn=i, ctx_wn=j)
                    out.write("{0}, {1},Wordnet Synset,{2}\n".format(fn_modes[1][:-1], i[0], sense_name))

                # Implemented (Gianotti)
                ctx_w_LU, ctx_s_LU = populate_contexts(f=frame, mode=fn_modes[2])
                for (i, j) in zip(ctx_w_LU, ctx_s_LU):
                    sense_name = bag_of_words(ctx_fn=i, ctx_wn=j)
                    out.write("{0}, {1},Wordnet Synset,{2}\n".format(fn_modes[2][:-1], i, sense_name))

        evaluate_performance(gold_path=surname_path, out_path=path)
        evaluate_performance(gold_path=surname_path, out_path=path, none=False)
