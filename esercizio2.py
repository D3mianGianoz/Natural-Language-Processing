from pathlib import Path

from DisambiguateFN.utils import read_correct_synsets, get_frame_set_for_student

surnames = ['gianotti', 'Demaria']
"""
In this file, we executes Task 2 (FrameNet Disambiguation)
"""

if __name__ == "__main__":
    correct_synsets_path = Path('.') / 'datasets' / 'AnnotationsFN'
    output_path = Path('.') / 'output'

    with open(output_path / 'results.csv', "w", encoding="utf-8") as out:
        for surname in surnames:
            fids = get_frame_set_for_student(surname)

            surname_path = correct_synsets_path / (surname + '.txt')
            read_correct_synsets(surname_path)



