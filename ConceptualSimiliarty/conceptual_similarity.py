import time as t

from ConceptualSimiliarty.MetricsSimilarity import SimilarityMetrics
from ConceptualSimiliarty.WordNetAPIClient import WordNetAPIClient
from ConceptualSimiliarty.correlation_indexes import pearson_index, spearman_index

"""
Compute Concept Similarity.
Using three metrics, compute conceptual similarity on 353 couple of terms.
At the end, also computes the Pearson and Spearman correlation indexes.
"""


def parse_word_sim_353(path):
    """
    Support function, it parse the WordSim353 CSV file. Each line is compose by
    a couple of terms and their annotation.
    Params:
        path: input path of the CSV file
    Returns:
         a list, representation of the input file. Its format will be [(w1, w2, gold_annotation)]
    """
    word_sim_array = []
    with open(path + "WordSim353.csv", "r") as fileWord:
        next(fileWord)
        for line in fileWord:
            (word1, word2, human) = line.rstrip("\n").split(",")
            word_sim_array.append([word1, word2, float(human)])

    return word_sim_array


def conceptual_similarity(options):
    """Computes the conceptual similarity and writes the results in two CSV file
    in the output folder.
    Params:
        options: a dictionary that contains the input and output paths.
    Format:
        { "input": "...", "output": "..." }
    """
    ws353 = parse_word_sim_353(options["input"])
    print("[cs] - WordSim353.csv parsed.")

    client = WordNetAPIClient()

    # matrix of similarity list, one list
    similarities = []
    metric_obj = SimilarityMetrics(client)
    time_start = t.time()

    # A list of 2 tuples: the first containing the reference to the metrics implementation
    # the second containing his name (a tuple of 3 strings).
    metrics = list(zip(*metric_obj.get_all()))

    to_remove = []
    count_total_senses = 0  # to count the senses total

    # Looping over the list of all the three metrics.
    for metric in metrics[0]:
        sim_metric = []  # similarity list for this metric

        index = 0
        for couple_terms in ws353:

            synset1 = WordNetAPIClient.get_synsets(couple_terms[0])
            synset2 = WordNetAPIClient.get_synsets(couple_terms[1])

            max_senses = []  # list of senses similarity
            for sense1 in synset1:
                for sense2 in synset2:
                    count_total_senses += 1
                    max_senses.append(metric(sense1, sense2))
            if len(max_senses) == 0:  # word without senses (ex.: proper nouns)
                max_senses = [-1]
                to_remove.append(index)
            sim_metric.append(max(max_senses))
            index += 1
        similarities.append(sim_metric)

    time_end = t.time()
    print("[cs] - Total senses similarity: {}".format(count_total_senses))
    print("[cs] - Time elapsed: {0:0.2f} seconds".format(time_end - time_start))

    # Removing word without senses
    for index in range(len(ws353)):
        if index in to_remove:
            del ws353[index]
            for s in range(len(similarities)):
                del similarities[s][index]

    golden = [row[2] for row in ws353]  # the list of golden annotations

    pearson_list = []
    spearman_list = []

    for i in range(len(metrics[1])):
        yy = similarities[i]
        pearson_list.append(pearson_index(golden, yy))
        spearman_list.append(spearman_index(golden, yy))
        # spearman_list.append(scipy.stats.spearmanr(golden, yy))

    with open(options["output"] + 'task1_results.csv', "w") as out:
        out.write("word1, word2, {}, {}, {}, gold\n"
                  .format(metrics[1][0], metrics[1][1], metrics[1][2]))
        for index in range(len(ws353)):
            out.write("{0}, {1}, {2:.2f}, {3:.2f}, {4:.2f}, {5}\n"
                      .format(ws353[index][0], ws353[index][1], similarities[0][index],
                              similarities[1][index], similarities[2][index], ws353[index][2], )
                      )

    with open(options["output"] + 'task1_indices.csv', "w") as out:
        out.write(" , Pearson, Spearman\n")
        for index in range(len(pearson_list)):
            out.write("{}, {}, {}\n".format(metrics[1][index], str(pearson_list[index]),
                                            spearman_list[index]))
