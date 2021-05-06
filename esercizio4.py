from pathlib import Path

from Radicioni.Annotation.parsing import read_from_annotation

if __name__ == '__main__':
    base_path = Path('.') / 'datasets'
    annotation_path = base_path / 'AnnotationSemEval'
    nasari_path = base_path / 'mini_NASARI.tsv'

    annotation_gianotti = read_from_annotation(annotation_path / 'Gianotti.csv')
    print(annotation_gianotti)

