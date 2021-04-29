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
        Nasari vector
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
                nasari[first] = nas

    return nasari


def create_context(titles, dict_n):
    """Creates the context
    Params:
        titles
    Returns:
        A unified dict of the context
    """
    context = {}

    for t in titles:
        x = get_nasari_vectors(t, dict_n)
        context.update(x)

    return context

def weighted_overlap_demaria(v1, v2):
    """Weighted Overlap between two nazari vectors v1 and v2 extracted from keys
    Params: 
        v1: first nasari vector extracter from a key
        v2: second nasari vector extracter from a key
    Returns:
        weighted overlap between v1 and v2
    """
    WO = 0
    dim_overlap = 0
    numeratore = 0
    denominatore = 0
    contatorev1 = 0
    contatorev2 = 0
    for i in v1:
        contatorev2 = 0
        contatorev1 += 1
        for j in v2:
            contatorev2 += 1
            if i[0] == j[0]:
                print(i,j)
                numeratore += 1/(contatorev1+contatorev2)
                dim_overlap += 1
                denominatore += 1/(2*dim_overlap)
                WO = numeratore/denominatore
    return WO


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
    print(f'La bellezza del file {f_path}')
    paragraphs, titles = read_from_file(f_path)

    vect1 = create_context(titles, nasari_dict)
    for title in vect1.values():
        print(title)

    progress_bar.update(1)
