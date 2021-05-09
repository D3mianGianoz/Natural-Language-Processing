from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from Radicioni.Annotation.utils import get_nasari_vectors, parse_nasari_dictionary, babel_key


def media(n1, n2):
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


def compute_gold_and_s1_s2():
    # Calcolo i punteggi di similarità di tutte le coppie di parole (se possibile)
    gold = []
    best_synset = []
    for (i, j) in zip(df_media['parola1'], df_media['parola2']):
        max_score, (s1, s2) = max_cos_sim_couple(str(i), str(j), dict_nasari)
        if max_score == 0:
            gold.append('na')
        else:
            gold.append(max_score[0][0])
            best_synset.append((s1, s2))
    return gold, best_synset


if __name__ == '__main__':
    base_path = Path('.') / 'datasets'
    annotation_path = base_path / 'AnnotationSemEval'
    nasari_path = base_path / 'NASARI_vectors' / 'mini_NASARI.tsv'
    SemEval_path = base_path / 'SemEval17_IT_senses2synsets.txt'
    output_path = Path('.') / 'output'

    babel_API_key = "036afd10-7afe-4064-8fe1-1b27515fb8f4"

    # Task 1: Semantic Similarity
    #
    # 1. annotate by hand the couple of words in [0,4] range
    # 2. compute inter-rate agreement with Spearman and Pearson indexes
    # 3. Compute the cosine similarity between the hand-annotated scores and
    # Nasari best score given the two terms
    # 4. Evaluate the total quality using again the Spearman and Pearson
    # indexes between the human annotation scores and the Nasari scores.

    df_demaria = pd.read_csv(annotation_path / 'Demaria.csv')
    df_gianotti = pd.read_csv(annotation_path / 'Gianotti.csv')

    demaria_score = np.empty(0)
    gianotti_score = np.empty(0)
    mean_score = []

    # 1. Annotation's scores, used for evaluation and computing mean value for each couple of annotation score
    for (avg_s, go) in zip(df_demaria['score'], df_gianotti['score']):
        demaria_score = np.append(demaria_score, avg_s)
        gianotti_score = np.append(gianotti_score, go)
        mean_score.append(media(avg_s, go))

    df_media = df_demaria.copy()
    df_media['score'] = mean_score
    # Creo e salvo il file con i valori medi annotati
    df_media.to_csv(annotation_path / 'means.csv', index=False, header=True)

    # 2. Computing the inter-rate agreement. This express if the two annotations are consistent
    inter_rate_pearson = scipy.stats.pearsonr(demaria_score, gianotti_score)
    inter_rate_spearman = scipy.stats.spearmanr(demaria_score, gianotti_score)

    # Read and store ONCE the SemEval..txt
    sense2syns = babel_key(SemEval_path)

    # Recupero i vettori di nasari e babel word mapping (for future use)
    dict_nasari, babel_word_nasari = parse_nasari_dictionary(nasari_path)

    # 3. Computing the cosine similarity between the hand-annotated scores and the best synset (for later use)
    df_media['gold'], best_syn = compute_gold_and_s1_s2()

    # estraggo le coppie per valutazione, escludendo i valori non trovati (na)
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
    # Nasari best score given the two terms
    # 4. Evaluate the total quality using the argmax function. Evaluate both
    # the single sense and both the senses in the couple.

    # 1. Already done in the Task 1.1

    # 2. Computing the inter-rate agreement. This express if the two score are consistent
    k = cohen_kappa_score(demaria_score, gianotti_score)

    # Retrive annotation from file
    df_sense = pd.read_csv(annotation_path / 'input_sense_bblID.csv')

    # TODO remove this testing limitation
    df_sense = df_sense.head(12)

    # re-using the result from compute_gold_and_s1_s2() function for retrieving the best couple
    for (s1, s2) in best_syn:
        # if both Babel Synset exists and are not None
        if s1 is not None and s2 is not None:
            print("{}\t{}\t".format(s1, s2))

    for bblID1, bblID2 in zip(df_sense['babelID1'], df_sense['babelID1']):
        pass

