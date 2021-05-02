def read_from_file(path):
    """It parse the given document give input
     Params:
        path: input document path
     Returns:
         a list of all document's paragraph and a list of all titles
     """

    titles = []
    paragraphs = []
    with open(path, "r", encoding="utf8") as f:
        file = f.read()
        text = file.split("\n")
        for paragraph in text:
            if paragraph != "\n" and paragraph != "":
                word_of_paragraph = paragraph.split(" ")
                if len(word_of_paragraph) > 13:
                    paragraphs.append(paragraph)
                else:
                    titles.append(paragraph)
        # Removes the link to the source of the article, placed on the top
        titles.pop(0)
    return paragraphs, titles


def parse_nasari_dictionary(path):
    """It parse the Nasari input file, and it converts into a more convenient
    Python dictionary.
    Returns:
         a dictionary representing the Nasari input file.
    Format: {word: {term:score}}
    """

    nasari_dict = {}
    with open(path, 'r', encoding="utf8") as file:
        for line in file.readlines():
            splits = line.split(";")
            vector_nasari = []

            for term in splits[2:]:
                k = term.split("_")
                if len(k) > 1:
                    vector_nasari.append([k[0], float(k[1])])

            nasari_dict[splits[1].lower()] = vector_nasari

    return nasari_dict


def compute_overlap(topic, paragraph):
    """ Support function used in Weighted Overlap's function below.
    Params:
        topic: Vector representation of the topic
        paragraph: Vector representation of the paragraph
    Return:
        intersection between the given parameters
    """
    return topic & paragraph


def rank(vector, nasari_vector):
    """ Computes the rank of the given vector.
    Params:
        vector: input vector
        nasari_vector: input Nasari vector
    Return:
         vector's rank (position inside the nasari_vector)
    """

    for i in range(len(nasari_vector)):
        if nasari_vector[i] == vector:
            return i + 1


def weighted_overlap(topic_nasari_vector, paragraph_nasari_vector):
    """ Implementation of the Weighted Overlap metrics (Pilehvar et al.)
    Params:
        topic_nasari_vector: Nasari vector representing the topic
        paragraph_nasari_vector: Nasari vector representing the paragraph
    Return:
        square-rooted Weighted Overlap if exist, 0 otherwise.
    """

    # transform from array-list to dict for easier comparison
    topic_nasari_vector = {item[0]: item[1] for item in topic_nasari_vector}
    paragraph_nasari_vector = {item[0]: item[1] for item in paragraph_nasari_vector}

    overlap_keys = compute_overlap(topic_nasari_vector.keys(),
                                   paragraph_nasari_vector.keys())

    if len(overlap_keys) > 0:
        overlaps = list(overlap_keys)

        # sum 1/(rank() + rank())
        den = sum(1 / (rank(q, list(topic_nasari_vector)) +
                       rank(q, list(paragraph_nasari_vector))) for q in overlaps)

        # sum 1/(2*i)
        num = sum(list(map(lambda x: 1 / (2 * x),
                           list(range(1, len(overlaps) + 1)))))

        return den / num

    return 0


def weighted_overlap_demaria(v1, v2):
    """Weighted Overlap between two nasari vectors v1 and v2 extracted from keys
    Params:
        v1: first nasari vector extracted from a key
        v2: second nasari vector extracted from a key
    Returns:
        weighted overlap between v1 and v2
    """
    wo = 0
    dim_overlap = 0
    numerator = 0
    denominator = 0
    counter_v1 = 0
    for i in v1:
        counter_v2 = 0
        counter_v1 += 1
        for j in v2:
            counter_v2 += 1
            if i[0] == j[0]:
                print(i, j)
                numerator += 1 / (counter_v1 + counter_v2)
                dim_overlap += 1
                denominator += 1 / (2 * dim_overlap)
                wo = numerator / denominator
    return wo

