import re
from pathlib import Path

import nltk
from nltk.corpus import framenet as fn
from nltk.corpus import stopwords

from DisambiguateFN.utils import read_correct_synsets, get_frame_set_for_student, get_wordnet_context

"""
In this file, we executes Task 2 (FrameNet Disambiguation)
"""

# TODO Da controllare se faccia più danni oppure fa il suo lavoro
# L'idea è quella di pulire le parole prima di unire i due contesti
# Roberto guarda se preferisci i risultati di no_stop o semplici (no punteggiatura)
def clean_word(word):
    stop_words = set(stopwords.words('english'))
    word = re.sub('[^A-Za-z0-9 ]+', '', word)
    # Returns the input word unchanged if it cannot be found in WordNet.
    wnl = nltk.WordNetLemmatizer()
    if word not in stop_words:
        return wnl.lemmatize(word)
    return ""


def evaluate_performance():
    pass


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

        print(context_w)
        # Here, the context is a list of synset associated to the frame name.
        # In each synset are usually present word, glosses and examples.
        context_s = get_wordnet_context(main_clause)

    elif mode == "FEs":
        # Populating ctx_w for FEs
        for i in sorted(f.FE):
            if "_" in i:
                main_clause = get_main_clause(i)
            else:
                main_clause = i
        # The context in this case contains the frame name and his definition.
            fe = f.FE[i]
            context_w.append([main_clause, fe.definition])
    
        # Here, the context is a list of synset associated to the frame name.
        # In each synset are usually present word, glosses and examples.
            context_s.append(get_wordnet_context(main_clause))
        
    elif mode == "LUs":
        # Populating ctx_w for LUs
        pass

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
    sentences_wn = {}  # dictionary of all Wordnet sysnset, glosses and examples.
    temp_max = 0
    ret = None

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
                    print(f"Parola da pulire {word}")
                    word = clean_word(word)
                    print(f"Parola da pulita {word}")
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
        return Warning("No bag of words created")


if __name__ == "__main__":
    correct_synsets_path = Path('.') / 'datasets' / 'AnnotationsFN'
    output_path = Path('.') / 'output'
    surnames = ['gianotti', 'Demaria']
    fn_modes = ["Frame name", "FEs", "LUs"]

    with open(output_path / 'task2_results_no_stop.csv', "w", encoding="utf-8") as out:

        print("Assigning Synsets...")

        for surname in surnames:
            frame_ids = get_frame_set_for_student(surname)
            out.write("Student {0},\n".format(surname))
            # Retrive gold annotation
            surname_path = correct_synsets_path / (surname + '.txt')
            read_correct_synsets(surname_path)

            for fID in frame_ids:
                frame = fn.frame_by_id(fID)
                # calculate context of FN: ctx_w and WN: ctx_s
                ctx_w, ctx_s = populate_contexts(f=frame, mode=fn_modes[0])
                sense_name = bag_of_words(ctx_fn=ctx_w, ctx_wn=ctx_s)
                # Write it to file
                out.write("{0}, {1}, Wordnet Synset, {2}\n".format(fn_modes[0], frame.name, sense_name))

                # TODO
                ctx_w_FE, ctx_s_FE = populate_contexts(f=frame, mode=fn_modes[1])

                ctx_w_LU, ctx_s_LU = populate_contexts(f=frame, mode=fn_modes[2])

    evaluate_performance()
