from pathlib import Path

from nltk.corpus import framenet as fn

from DisambiguateFN.utils import read_correct_synsets, get_frame_set_for_student

surnames = ['gianotti', 'Demaria']
"""
In this file, we executes Task 2 (FrameNet Disambiguation)
"""

if __name__ == "__main__":
    correct_synsets_path = Path('.') / 'datasets' / 'AnnotationsFN'
    output_path = Path('.') / 'output'

    with open(output_path / 'results.csv', "w", encoding="utf-8") as out:

        print("Assigning Synsets...")

        for surname in surnames:
            frame_ids = get_frame_set_for_student(surname)

            surname_path = correct_synsets_path / (surname + '.txt')
            read_correct_synsets(surname_path)
            for fID in frame_ids:
                frame = fn.frame_by_id(fID)

                # calculate context ctx_f and ctx_w


