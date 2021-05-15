# TLN_dicaro_2.1

"""
CONSEGNA:
* Ispirandosi al text-tiling, implementare un algoritmo di segmentazione del
testo.

* Sfruttare informazioni come le frequenze e le co-occorrenze e eventuale
pre-processing del testo.

SVOLGIMENTO:
* Breve lettura iniziale di
https://www.sciencedirect.com/science/article/pii/S0031320316303399#bib12

* Si è scelto di utilizzare un testo su Trump-wall già usato nella seconda
parte del corso.

* Si crea per prima cosa il dizionario del testo, annotando le frequenze di
ogni parola (previo filtering e lemmatizzazione).

* Si creano poi i vettori che contengono i termini presenti in ogni frase.

* Si calcola la cosine similarity fra tutti questi vettori colonna.

* Si itera su queste cosine similarities per identificare i punti in cui il
valore scende al di sotto della loro media di una certa percentuale: in questi
punti viene inserito un punto di cambio di discorso.

* L'operazione viene ripetuta per un certo numero di iterazioni, usando come
confronto la media di quel preciso segmento invece che la media complessiva.

* Come confronto si usa implementazione di nltk disponibile a
 https://www.nltk.org/_modules/nltk/tokenize/texttiling.html

"""

from pathlib import Path
import matplotlib.pyplot as plt
import nltk
from scipy import spatial

from DiCaro.Primo_gruppo.aggregate_concepts import preprocess


# I only look for jumps -negative- greater than one, and a half times the average
SCALE_FACTOR = 1.5


def read_text(path):
    """
    Read the given .txt file
    :param path: absolute Path of the file
    :return: list of sentences of the textual file
    """
    with open(path, "r", encoding="utf8") as f:
        content = f.read()
        lines = content.split(". ")
        returned_list = []
        for line in lines:
            line = line.strip("\n")
            returned_list.append(line)
        return returned_list


def calculate_frequencies(sentences):
    """
     Calculate the frequency of a word (non-stop-words) in a sentence, for all sentences
    :param sentences: lines of text to be processed
    :return: dictionary containing the frequencies
    """
    terms = {}
    pos = 0
    for sentence in sentences:
        words = preprocess(sentence, 0)
        for word in words:
            if word in terms:
                (terms[word])[pos] += 1
            else:
                terms[word] = [0] * len(sentences)
                (terms[word])[pos] = 1
        pos += 1
    return terms


def generate_columns(length_sentences, terms):
    """
    Generate a vector containing the frequencies of each word, for each sentence
    :param length_sentences: n° of sentences
    :param terms: dict of the frequency
    :return: list with frequencies of each word, for each sentence
    """

    columns = [[] for _ in range(length_sentences)]

    for key in terms.keys():
        for i in range(length_sentences):
            freqs = terms[key]
            column = columns[i]
            column.append(freqs[i])
    return columns


def compute_cosine(columns):
    """
    Calculates the cosine similarity between all sentences, based on word frequencies
    using scipy library, spatial module
    :param columns: list of list
    :return: list of measurement, one for each column
    """
    cosine_similarities = []
    for i in range(1, len(columns)):
        cosine_similarity = 1 - spatial.distance.cosine(columns[i - 1], columns[i])
        cosine_similarities.append(cosine_similarity)
    return cosine_similarities


def calculate_avg_drop(block):
    """
    It calculates the average of a descent of cosine in block
    :param block: slice to consider
    :return: average value if found, 0 otherwise
    """
    sum_drops = 0
    num_drops = 0
    for j in range(1, len(block)):
        if block[j - 1] - block[j] > 0:
            sum_drops += block[j - 1] - block[j]
            num_drops += 1
    if num_drops == 0:
        return 0
    else:
        return sum_drops / num_drops


def plot_sim(cos, b_points):
    first = True
    # first plot
    plt.plot(cos, label="Cosine similarity")
    # second plot (each point drawn individually)
    for point in b_points:
        if first:
            first = False  # add label only once
            plt.axvline(point, color='r', label="Breakpoints")
        else:
            plt.axvline(point, color='r')

    plt.title(title_file)
    plt.xlabel("Number of sentences")
    plt.ylabel("Cos Score")
    plt.legend()
    plt.savefig('output/{}.png'.format(title_file))
    plt.show()
    print(f"\n{title_file}'s plot saved in output folder.")


def plot_text_tiling(result1, title):
    s, sgs, d, b = result1
    plt.title(str(file_path.stem))
    plt.xlabel("Sentence Gap index")
    plt.ylabel("Gap Scores")
    plt.plot(range(len(s)), s, label="Gap Scores")
    plt.plot(range(len(sgs)), sgs, label="Smoothed Gap scores")
    plt.plot(range(len(d)), d, label="Depth scores")
    plt.stem(range(len(b)), b)
    plt.legend()
    plt.savefig('output/{}.png'.format(title))
    plt.show()
    print(f"\n{title}'s plot saved in output folder.")


if __name__ == '__main__':
    file_path: Path = Path(".") / "input" / "Trump-wall.txt"
    title_file = str(file_path.stem)

    ss = read_text(file_path)
    ss_length = len(ss)

    frequency_dict = calculate_frequencies(sentences=ss)
    position_breakpoints = [0]

    col = generate_columns(ss_length, frequency_dict)
    cos_similarities = compute_cosine(col)

    iterations: int = 3
    try:
        valid = "Please choose and type a valid"
        iterations = int(input((valid + " n° of iterations or press enter for default (3)\n iterations: ")))

    # If something else that is not a valid input,
    # then ValueError exception will be called.
    except ValueError:
        # The cycle will go on until validation
        print("Error! This is not a valid number, using default value")

    for iterations in range(iterations):
        index = 0
        while index < len(position_breakpoints):
            break_p_l_updated = len(position_breakpoints)
            left_index = position_breakpoints[index]
            if index == break_p_l_updated - 1:
                right_index = ss_length - 1
            else:
                right_index = position_breakpoints[index + 1]

            # Debugging
            print(left_index, right_index)

            avg = calculate_avg_drop(cos_similarities[left_index:right_index])
            avg = avg * SCALE_FACTOR

            for b_point in range(left_index + 1, right_index):
                """
                If the jump is greater than the average, a new Breakpoint is created
                """
                if (cos_similarities[b_point - 1] - cos_similarities[b_point]) > avg:
                    position_breakpoints.insert(index + 1, b_point)
                    break

            # Update the index
            index += 1

    print(f'Position breakpoints: {position_breakpoints}')
    plot_sim(cos=cos_similarities, b_points=position_breakpoints)

    # For comparison again nltk text tiling implementation
    # Demo mode will also plot a nice graph
    tt = nltk.TextTilingTokenizer(demo_mode=True)
    # We need to process again the file and have it raw
    path_str = str(file_path.resolve())
    file_raw = nltk.corpus.gutenberg.raw(path_str)
    result = tt.tokenize(file_raw)
    plot_text_tiling(result, f"TexTiling of {title_file}")
