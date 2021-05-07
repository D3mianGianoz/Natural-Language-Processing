from pathlib import Path
import re

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
            splits = line.split("\t")
            vector_nasari = []
            
            for term in splits[1:]:
                vector_nasari.append(re.sub('\n', '', term))
                
            for key in splits[:1]:
                new_key = key.split("n__")

            nasari_dict[splits[0]] = vector_nasari

    return nasari_dict

from Radicioni.Annotation.parsing import read_from_annotation

if __name__ == '__main__':
    base_path = Path('.') / 'datasets'
    annotation_path = base_path / 'AnnotationSemEval'
    nasari_path = base_path / 'mini_NASARI.tsv'

    annotation_gianotti = read_from_annotation(annotation_path / 'Gianotti.csv')
    print(annotation_gianotti)

