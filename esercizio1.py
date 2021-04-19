from pathlib import Path

from ConceptualSimiliarty.conceptual_similarity import conceptual_similarity
from Wsd.word_sense_disambiguation import word_sense_disambiguation

"""
In this file, we executes both Task 1.1 (conceptual similarity) and Task 1.2 (word 
sense disambiguation).
"""

if __name__ == "__main__":
    word_sim = Path('.') / "datasets" / "WordSim353.csv"
    sem_core = Path('.') / "datasets" / "br-a01"
    output = Path('.') / "output"
    options_conceptual_similarity = {
        "input": word_sim,
        "output": output
    }
    options_word_sense_disambiguation = {
        "input": sem_core,
        "output": output
    }

    print("Task 1.1: Conceptual Similarity")
    conceptual_similarity(options_conceptual_similarity)

    print("\nTask 1.2: Word Sense Disambiguation")
    print("[Lesk] - Running Lesk algorithm...")
    word_sense_disambiguation(options_word_sense_disambiguation)
    print("\n[Lesk] - Done.")
