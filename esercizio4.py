from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from Radicioni.Annotation.utils import get_nasari_vectors, parse_nasari_dictionary, babel_key, get_synset_terms

END = 1
TAB = 0


def mean(n1, n2):
    return (n1 + n2) / 2


def max_cos_sim_couple(babel_word1, babel_word2, dict_n):
    """Given 2 words in input it calculates the maximum of the cosin similarity
    between the two terms
    Args:
        babel_word1: first word
        babel_word2: the second word
        dict_n: NASARI dictionary
    Returns:
        1st: max of the cosine similarity
        2nd: the couple of senses (their BabelID) that maximise the score
    """
    maximum = 0
    senses = (None, None)
    # retrive nasari vector and list of bbl
    vec1, bbl_ids1 = get_nasari_vectors(sense2syns, str(babel_word1), dict_n)
    vec2, bbl_ids2 = get_nasari_vectors(sense2syns, str(babel_word2), dict_n)

    for bid1, v1 in zip(bbl_ids1, vec1):
        for bid2, v2 in zip(bbl_ids2, vec2):

            # Computing and storing the cosine similarity.
            val = cosine_similarity([v1], [v2])

            if val > maximum:
                maximum = val
                senses = (bid1, bid2)
    return 4 * maximum, senses


def compute_gold_and_bl1_bl2():
    """Helper function for the main
    Calculate the similarity scores of all word pairs (if possible)
    Returns:
        gold: list of gold scores
        best_synset_bbl_ids: list of golden babels ID
    """
    gold = []
    best_synset_bbl_ids = []
    for (i, j) in zip(df_media['parola1'], df_media['parola2']):
        max_score, (bl1, bl2) = max_cos_sim_couple(str(i), str(j), dict_nasari)
        best_synset_bbl_ids.append((bl1, bl2))
        if max_score == 0:
            gold.append('na')
        else:
            gold.append(max_score[0][0])
    return gold, best_synset_bbl_ids


def write_terms(out_terms, o, value):
    """Helper function of task4.2 that write to the given output the terms
    Args:
        out_terms: synset recovered from babelnet
        o: output file
        value: mode for correct formatting of the file
    Returns:
        nasari_terms: a string containing all the terms concatenated
    """
    nasari_terms = ""
    for term in out_terms:
        if term != out_terms[len(out_terms) - 1]:
            o.write(term + ",")  # if not the last term, put a ","
            nasari_terms += term + ","
        else:
            if value:
                o.write(term + "\n")  # put a newline if last elem
            else:
                o.write(term + "\t")  # otherwise, put a separator
            nasari_terms += term
    return nasari_terms


if __name__ == '__main__':
    base_path = Path('.') / 'datasets'
    annotation_path = base_path / 'AnnotationSemEval'
    nasari_path = base_path / 'NASARI_vectors' / 'mini_NASARI.tsv'
    SemEval_path = base_path / 'SemEval17_IT_senses2synsets.txt'
    output_path = Path('.') / 'output'

    # Task 1: Semantic Similarity
    #
    # 1. annotate by hand the couple of words in [0,4] range
    # 2. compute inter-rate agreement with Spearman and Pearson indexes
    # 3. Compute the cosine similarity between the hand-annotated scores and
    #    Nasari best score given the two terms
    # 4. Evaluate the total quality using again the Spearman and Pearson
    #    indexes between the human annotation scores and the Nasari scores.

    print('Task 1: Semantic Similarity')

    df_demaria = pd.read_csv(annotation_path / 'Demaria.csv')
    df_gianotti = pd.read_csv(annotation_path / 'Gianotti.csv')

    demaria_score = np.empty(0)
    gianotti_score = np.empty(0)
    mean_score = []

    # 1. Annotation's scores, used for evaluation and computing mean value for each couple of annotation score
    for (avg_s, go) in zip(df_demaria['score'], df_gianotti['score']):
        demaria_score = np.append(demaria_score, avg_s)
        gianotti_score = np.append(gianotti_score, go)
        mean_score.append(mean(avg_s, go))

    df_media = df_demaria.copy()
    df_media['score'] = mean_score
    # I create and save the file with the annotated average values
    df_media.to_csv(annotation_path / 'means.csv', index=False, header=True)

    # 2. Computing the inter-rate agreement. This express if the two annotations are consistent
    inter_rate_pearson = scipy.stats.pearsonr(demaria_score, gianotti_score)
    inter_rate_spearman = scipy.stats.spearmanr(demaria_score, gianotti_score)

    # Read and store ONCE the SemEval17.txt
    sense2syns = babel_key(SemEval_path)

    # Retrieve nasari and babel word mapping vectors (for future use)
    dict_nasari, babel_word_nasari = parse_nasari_dictionary(nasari_path)

    # 3. Computing the cosine similarity between the hand-annotated scores and the best synset (for later use)
    df_media['gold'], best_syn_ids = compute_gold_and_bl1_bl2()

    # I extract the pairs by evaluation, excluding the values not found (na)
    our_value = np.empty(0)
    gold_value = np.empty(0)
    for (avg_s, go) in zip(mean_score, df_media['gold']):
        if go != "na":
            our_value = np.append(our_value, avg_s)
            gold_value = np.append(gold_value, go)

    # 4. Evaluate the total quality using again the Spearman and Pearson
    # indexes between the human annotation scores and the Nasari scores.
    pearson = scipy.stats.pearsonr(our_value, gold_value)
    spearman = scipy.stats.spearmanr(our_value, gold_value)

    from contextlib import redirect_stdout

    with open(output_path / 'task4.1_output.txt', 'w') as f:
        with redirect_stdout(f):
            print('Task 1: Semantic Similarity')
            print("Correlation between different group members annotation: a.k.a. inter-rate agreement")
            print('\tInter-rate agreement - Pearson: {0}, {1}'.format(inter_rate_pearson, inter_rate_spearman))

            print("Correlation between different our average value and gold value, computed with nasari vectors")
            print('\tEvaluation - Person: {0}, {1}'.format(pearson, spearman))

    # Task 2: Sense Identification
    #
    # 1. annotate by hand the couple of words in the format specified
    # 2. compute inter-rate agreement with the Cohen's Kappa score
    # 3. Compute the cosine similarity between the hand-annotated scores and
    #    Nasari best score given the two terms
    # 4. Evaluate the total quality using the argmax function. Evaluate both
    #    the single sense and both the senses in the couple.
    print("\nTask 2: Sense Identification.")
    # 1. Already done in the Task 1.1

    # 2. Computing the inter-rate agreement. This express if the two score are consistent
    demaria_ann = []
    gianotti_ann = []

    df_bab_demaria = pd.read_csv('Demaria_sense_bblID.csv')
    df_bab_gianotti = pd.read_csv('Gianotti_sense_bblID.csv')
    
    for (n_dem1, n_dem2) in zip(df_bab_demaria['babelID1'], df_bab_demaria['babelID2']):
        demaria_ann = np.append(demaria_ann, [n_dem1,n_dem2])
    for (n_gia1, n_gia2) in zip(df_bab_gianotti['babelID1'], df_bab_gianotti['babelID2']):
        gianotti_ann = np.append(gianotti_ann, [n_gia1,n_gia2])

    k = cohen_kappa_score(demaria_ann, gianotti_ann)
    print('\tInter-rate agreement - Cohen Kappa: {0}'.format(k)) 

    # Retrive annotation from file
    df_sense = pd.read_csv(annotation_path / 'input_sense_bblID.csv')

    # TODO remove this testing limitation
    df_sense = df_sense.head(25)

    with open(output_path / 'task4.2_results.tsv', "w", encoding="utf-8") as out:
        # used for print progress bar
        percentage = 0
        first_print = True

        # used for final comparison. It is an in-memory copy
        nasari_out = []

        # first row are labels
        out.write("{}\t{}\t{}\t{}\t{}\t{}\n".format('parola1', 'parola2', 'babelID1', 'babelID2', 'synset1', 'synset2'))

        # 3. we already did most of the heavy-lifting in the first part
        for index, row in df_media.iterrows():
            w1 = row['parola1']
            w2 = row['parola2']
            # re-using the result from compute_gold_and_s1_s2() function for retrieving the best couple
            (s1, s2) = best_syn_ids[index]
            # if both Babel Synset exists and are not None
            if s1 is not None and s2 is not None:
                out.write("{}\t{}\t{}\t{}\t".format(w1, w2, s1, s2))
                out_terms_1 = get_synset_terms(s1)
                out_terms_2 = get_synset_terms(s2)

                nasari_terms1 = write_terms(out_terms_1, out, TAB)
                nasari_terms2 = write_terms(out_terms_2, out, END)
            else:
                out.write("{}\t{}\tNone\tNone\tNone\tNone\n".format(row[0], row[1]))

            # populate the nasari_out list.
            nasari_out.append((w1, w2, s1, s2, nasari_terms1, nasari_terms2))

            # updating percentage
            percentage += 2

            if first_print:
                print('\tDownloading terms from BabelNet.')
                print('\t#', end="")
                first_print = False
            if percentage % 10 == 0:
                print('#', end="")
            else:
                print('-', end="")

    # create a DataFrame for easier comparison
    df_nas_out = pd.DataFrame(nasari_out, columns=['parola1', 'parola2', 'babelID1', 'babelID2', 'synset1', 'synset2'])

    # 4. Evaluate both the single sense and both the senses in the couple.
    count_single = 0
    count_couple = 0
    for index, row in df_sense.iterrows():
        bblID1 = row['babelID1']
        bblID2 = row['babelID2']
        # retrive computed babelIDs from pandas frame
        computed_bl1 = df_nas_out.iloc[index, 2]
        computed_bl2 = df_nas_out.iloc[index, 3]
        # calculate bool values
        arg0 = computed_bl1 == bblID1
        arg1 = computed_bl2 == bblID2
        if arg0:
            count_single += 1
        if arg1:
            count_single += 1
        if arg0 and arg1:
            count_couple += 1
    print("\n\tSingle: {0} / 100 ({0}%) - Couple: {1} / 50 ({2:.0f}%)"
          .format(count_single, count_couple, (count_couple * 100 / 50)))
