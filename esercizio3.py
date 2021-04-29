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
