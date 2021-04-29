import sys
from pathlib import Path
from tqdm import tqdm
from Summarize.utils import read_from_file, parse_nasari_dictionary
from Wsd.utilities import bag_of_word


def get_nasari_vectors(title, nasari_dict):
    """Given a sentence, it creates a bag of words of this sentence
    and return the Nasari nasari for the words in the sentence
    Params:
        title:
        Nasari vector+
    Returns:
        Nasari nasari of words in title
    """

    # create bag of words
    bag = bag_of_word(title)

    nasari = {}
    # store the best scoring ones
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
                nasari[first] = list(map(lambda x: round(x[1] * 0.7, 2), nas))

    return nasari


def create_context(titles, nas_dict):
    """Creates the context
    Params:
        titles
    Returns:
        A unified dict of the context
    """
    context = {}

    for t in titles:
        x = get_nasari_vectors(t, nas_dict)
        context.update(x)

    return context


def summarization(file_path, nasari_dict, percentage):
    """ Applies summarization to the given document, with the given percentage.
    Params:
        file_path: path of the input document
        nasari_dict: Nasari dictionary
        percentage: reduction percentage
    Return:
         the summarization of the given document.
    """

    paragraphs, titles = read_from_file(f_path)
    weighted_paragraphs = []
    print("Path:" + str(file_path))

    # compute nasari
    nasari_vectors = create_context(titles, nasari_dict)

    # Stampa delle dimensioni del topic (numero di vettori presenti) e del
    # numero di paragrafi.
    print("Numero Topic:" + str(len(nasari_vectors)))
    print("Numero paragrafi:" + str(len(paragraphs)))

    for paragraph in paragraphs:
        # Weighted Overlap average inside the paragraph.
        score_wo = 0
        par_context = create_context(paragraph, nasari_dict)

        for word in par_context:
            # TODO
            pass

    to_keep = len(paragraphs) - int(round((percentage / 100) * len(paragraphs), 0))

    # Sort by highest score and keeps all the important entries. From first to "to_keep"
    new_document = None

    return new_document


def weighted_overlap_demaria(v1, v2):
    """Weighted Overlap between two nasari vectors v1 and v2 extracted from keys
    Params: 
        v1: first nasari vector extracter from a key
        v2: second nasari vector extracter from a key
    Returns:
        weighted overlap between v1 and v2
    """
    wo = 0
    dim_overlap = 0
    numerator = 0
    denominator = 0
    counter_v1 = 0
    counter_v2 = 0
    for i in v1:
        counter_v2 = 0
        counter_v1 += 1
        for j in v2:
            counter_v2 += 1
            if i[0] == j[0]:
                print(i, j)
                numerator += 1/(counter_v1+counter_v2)
                dim_overlap += 1
                denominator += 1/(2*dim_overlap)
                wo = numerator/denominator
    return wo


if __name__ == "__main__":
    nasari_path = Path('.') / 'datasets' / 'NASARI_vectors' / 'dd-small-nasari-15.txt'

    path = Path('.') / 'datasets' / 'text-documents'
    file_paths = [path / 'Ebola-virus-disease.txt',
                  path / 'Andy-Warhol.txt',
                  path / 'Life-indoors.txt',
                  path / 'Napoleon-wiki.txt',
                  path / 'Trump-wall.txt']

    compression_rate: int = 10
    print("Summarization.\nReduction percentage: {}".format(compression_rate))

    nasari_dict = parse_nasari_dictionary(nasari_path)
    # showing progress bar
    progress_bar = tqdm(desc="Percentage", total=5, file=sys.stdout)
    print("\n----------------------------")

    f_path = file_paths[0]
    summarization(f_path, nasari_dict, compression_rate)

    progress_bar.update(1)
