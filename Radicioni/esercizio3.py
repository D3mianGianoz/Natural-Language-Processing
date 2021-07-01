import sys
from pathlib import Path
from tqdm import tqdm

from Radicioni.Summarize.gold import tf_idf_sum, similarity
from Radicioni.Summarize.utils import weighted_overlap, read_from_file, parse_nasari_dictionary
from Radicioni.Wsd.utilities import bag_of_word


def get_nasari_vectors(title, nasari_dict):
    """Given a sentence, it creates a bag of words of this sentence
    and return the Nasari nasari for the words in the sentence
    Args:
        title:
        nasari_dict: dictionary containing the proper words
    Returns:
        list of Nasari vectors of words in title
    """

    # create bag of words
    bag = bag_of_word(title)

    nasari = {}
    # store the best scoring
    test = {}

    for word in bag:
        if word in nasari_dict.keys():
            array = nasari_dict[str(word)]
            test[word] = list(filter(lambda x: x[1] > 500, array))
            nasari[word] = array

    for values in test.values():
        for couple in values:
            first = couple[0]
            if first not in bag and first in nasari_dict.keys():
                nas = nasari_dict[first]
                # penalize the newcomers
                nasari[first] = list(map(lambda x: [first, round(x[1] * 0.7, 2)], nas))

    return nasari


def create_context(titles, nas_dict):
    """Creates the context
    Args:
        titles:
        nas_dict: the dict containing nasari
    Returns:
        A unified dict of the context

    """
    context = {}

    for t in titles:
        x = get_nasari_vectors(t, nas_dict)
        context.update(x)

    return context


def summarization():
    """ Applies summarization to the given document, with the given percentage.
    Args:
    Return:
         the summarization of the given document.
    """

    weighted_paragraphs = []
    par = 0

    # compute nasari
    nasari_vectors = create_context(selected, nasari_dict)

    # Stampa delle dimensioni del topic (numero di vettori presenti) e del
    # numero di paragrafi.
    print("Numero Topic:" + str(len(nasari_vectors)))
    print("Numero paragrafi:" + str(len(paragraphs)))

    for paragraph in paragraphs:
        # Weighted Overlap average inside the paragraph.
        par_wo = 0
        par_context = get_nasari_vectors(paragraph, nasari_dict)

        for word in par_context:
            topic_wo = 0
            for vector in nasari_vectors:
                topic_wo = topic_wo + weighted_overlap(par_context[word], nasari_vectors[
                    vector])
            if topic_wo != 0:
                topic_wo = topic_wo / len(nasari_vectors)

            # Sum all words WO in the paragraph's WO
            par_wo += topic_wo
            # print(f'score {topic_wo} e paragrafo {par_wo}') debug print

        if len(par_context) > 0:
            par_wo = par_wo / len(par_context)
            weighted_paragraphs.append((paragraph, par_wo))
            # print(str(par) + ": " + str(par_wo))
            par += 1

    n_paragraphs_to_delete = int(len(weighted_paragraphs) * compression_rate / 100)
    weighted_paragraphs_ordered = sorted(weighted_paragraphs, key=lambda tup: tup[1], reverse=False)

    # store deleted paragraphs
    paragraphs_to_delete = weighted_paragraphs_ordered[:n_paragraphs_to_delete]
    del weighted_paragraphs_ordered[:n_paragraphs_to_delete]

    # sort them again
    weighted_paragraphs_ordered = [i[0] for i in weighted_paragraphs_ordered]
    paragraphs_to_delete_ordered = [i[0] for i in paragraphs_to_delete]

    i = 0
    summary_list = []
    selected_paragraphs_list = []
    deleted_paragraphs_list = []
    for paragraph in paragraphs:
        if paragraph in weighted_paragraphs_ordered:
            summary_list.append(paragraph)
            selected_paragraphs_list.append(i)
        if paragraph in paragraphs_to_delete_ordered:
            deleted_paragraphs_list.append(i)
        i += 1

    return summary_list, selected_paragraphs_list, deleted_paragraphs_list


def handle_reader_and_gold(file_path, mode):
    print("Path:" + str(file_path))
    if mode == 'titles':
        parag, select = read_from_file(f_path, mode='titles')
    elif mode == 'frequencies':
        parag, select = read_from_file(f_path, mode='frequencies')
    else:
        raise ValueError

    return parag, select, tf_idf_sum(parag, compression_rate)


if __name__ == "__main__":
    nasari_path = Path('.') / 'datasets' / 'NASARI_vectors' / 'dd-small-nasari-15.txt'
    path_output = Path('.') / 'output' / 'task3'
    path = Path('.') / 'datasets' / 'text-documents'
    file_paths = [path / 'Ebola-virus-disease.txt',
                  path / 'Andy-Warhol.txt',
                  path / 'Life-indoors.txt',
                  path / 'Napoleon-wiki.txt',
                  path / 'Trump-wall.txt']

    try:
        valid = "Please choose and type a valid"
        compression_rate = int(input((valid + " compression rate(%): 10, 20, 30\n Rate: ")))
        if compression_rate not in [10, 20, 30]:
            raise ValueError

        mode_int = int(input("%s topic-picker technique: \n1 - titles, \n2 - frequencies\n technique: " % valid))
        if mode_int == 1:
            mode = "titles"
        elif mode_int == 2:
            mode = "frequencies"
        else:
            raise ValueError

    # If something else that is not a valid input
    # the ValueError exception will be called.
    except ValueError:
        # The cycle will go on until validation
        print("Error! This is not a valid number, using default value")
        compression_rate: int = 10
        mode = "titles"

    print("\nSummarization.\nReduction percentage: {} | topic-picker technique: {}".format(compression_rate, mode))

    nasari_dict = parse_nasari_dictionary(nasari_path)
    # showing progress bar
    progress_bar = tqdm(desc="Percentage", total=5, file=sys.stdout)
    print("\n------------------------------------------------------")

    for f_path in file_paths:
        name_file = 'summary_' + str(f_path.name) + ".txt"
        path_write = path_output / name_file

        # read files and create gold
        paragraphs, selected, (gold, del_gold) = handle_reader_and_gold(f_path, mode)

        with open(path_write, "w") as write:
            summary, selected_paragraphs, deleted_paragraphs = summarization()
            print("Selected paragraphs:" + str(selected_paragraphs))
            print("Gold paragraphs:" + str(gold))
            accuracy_selected = similarity(selected_paragraphs, gold)
            print("###############################################")
            print("Deleted paragraphs:" + str(deleted_paragraphs))
            print("Deleted Gold paragraphs:" + str(del_gold))
            accuracy_deleted = similarity(deleted_paragraphs, del_gold)

            # write to file
            for par in summary:
                # print(par + "\n")
                write.write(par + "\n\n")

        progress_bar.update(1)
