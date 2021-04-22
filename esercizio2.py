from pathlib import Path

from DisambiguateFN.utils import read_correct_synsets

"""
In this file, we executes Task 2 (FrameNet Disambiguation)
"""

if __name__ == "__main__":
    surname = ['gianotti']
    correctSynsetsPath = Path('.') / 'datasets' / 'AnnotationsFN' / (surname[0] + '.txt')
    test = read_correct_synsets(correctSynsetsPath)
