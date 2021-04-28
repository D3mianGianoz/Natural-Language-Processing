from pathlib import Path

from Summarize.utils import read_from_file

if __name__ == "__main__":
    nasari_path = Path('.') / 'datasets' / 'NASARI_vectors' / 'dd-small-nasari-15.txt'

    path = Path('.') / 'datasets' / 'text-documents'
    file_paths = [path / 'Ebola-virus-disease.txt',
                  path / 'Andy-Warhol.txt',
                  path / 'Life-indoors.txt',
                  path / 'Napoleon-wiki.txt',
                  path / 'Trump-wall.txt']

    compression_rate: int = 10

    paragraphs, titles = read_from_file(file_paths[0])

    print(paragraphs)
    print(titles)
