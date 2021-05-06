import csv


def read_from_annotation(path):
    """
    it parses the annotated words's csv.
    Args:
        path: path to the annotated word's csv.
    Returns:
         list of annotated terms. Format: [((w1, w2), value)]
    """

    annotation_list = []
    with open(path, 'r', encoding="utf-8-sig") as file:
        reader_results = csv.reader(file, delimiter=',')
        next(reader_results)
        for line in reader_results:
            copule_words = (line[0].lower(), line[1].lower())
            annotation_list.append((copule_words, float(line[2])))
    return annotation_list
