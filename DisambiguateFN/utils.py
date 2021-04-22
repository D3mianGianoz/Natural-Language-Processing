
def read_correct_synsets(path):
    """
    Support function, it parse the txt file. Each line is compose by
    a couple of terms and their annotation.
    Params:
        path: input path of the CSV file
    Returns:
         a list, representation of the input file. Its format will be [(syn)]
    """
    correct_synset = []
    with open(path, "r") as f:
        syn = f.read()
        synset = syn.split("\n")
        for syn in synset:
            if "#" not in syn:
                correct_synset.append(syn)
    return correct_synset
