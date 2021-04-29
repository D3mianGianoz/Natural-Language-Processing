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
