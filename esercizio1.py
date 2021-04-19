from ConceptualSimiliarty.conceptual_similarity import conceptual_similarity

"""
In this file, we executes both Task 1 (conceptual similarity) and Task 2 (word 
sense disambiguation).
"""

if __name__ == "__main__":
    print("Task 1: Conceptual Similarity")
    options_conceptual_similarity = {
        "input": "/home/damian/Documents/Magistrale/Radicioni/datasets/",
        "output": "/home/damian/Documents/Magistrale/Radicioni/output/",
    }
    conceptual_similarity(options_conceptual_similarity)
