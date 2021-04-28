def read_from_file(path):
    """
     It parse the given document give input
     Params:
        path: input document path
     Returns:
         a list of all document's paragraph and a list of all titles
     """
    titles = []
    paragraphs = []
    f = open(path, "r")
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
