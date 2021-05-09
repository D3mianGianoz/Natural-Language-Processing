from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

from Radicioni.Annotation.utils import get_nasari_vectors, parse_nasari_dictionary


def media(n1, n2):
    return (n1 + n2) / 2


def max_cos_sim(w1, w2):
    """Given 2 words in imput it calculates the maximum of the cosin similarity
    between the two terms
    Params:
        word 1
        word 2
    Returns:
        max of the cosine similarity
    """
    i = 0
    j = 0
    maximum = 0
    vec1 = get_nasari_vectors(SemEval_path, str(w1), dict_na)
    vec2 = get_nasari_vectors(SemEval_path, str(w2), dict_na)
    for v1 in vec1:
        x = v1
        i += 1
        for v2 in vec2:
            y = v2
            val = cosine_similarity([x], [y])
            j += 1
            if val > maximum:
                maximum = val
    return 4 * maximum


def compute_gold():
    # Calcolo i punteggi di similarità di tutte le coppie di parole (se possibile)
    gold = []
    for (i, j) in zip(df_media['parola1'], df_media['parola2']):
        elem = max_cos_sim(str(i), str(j))
        if elem == 0:
            gold.append('na')
        else:
            gold.append(elem[0][0])
    return gold


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

    # Recupero i vettori di nasari
    dict_na = parse_nasari_dictionary(nasari_path)

    # 3. Computing the cosine similarity between the hand-annotated scores and
    df_media['gold'] = compute_gold()

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



