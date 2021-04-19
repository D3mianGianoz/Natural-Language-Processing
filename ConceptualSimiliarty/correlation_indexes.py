import numpy as np
"""
This module contains the correlation indices implementations.
Alternatively one could you use scypy implementation
"""


def pearson_index(x, y):
    """Implementation of the Pearson index.
    Args:
         x: golden value
         y: similarity list
    Returns:
        Pearson correlation index
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    std_dev_x = np.std(x)
    std_dev_y = np.std(y)

    modified__x = [elem - mu_x for elem in x]
    modified__y = [elem - mu_y for elem in y]

    num = np.mean(np.multiply(modified__x, modified__y))
    numeric = std_dev_x * std_dev_y

    return num / numeric


def spearman_index(x, y):
    """Implementation of the Spearman index.
    Args:
        x: golden value
        y: similarity list
    Returns:
         Spearman correlation index
    """
    rank__x = define_rank(x)
    rank__y = define_rank(y)

    return pearson_index(rank__x, rank__y)


def define_rank(vector):
    """
    Args:
        vector: numeric vector
    Returns:
        ranks list, sorted as the input order
    """
    x_couple = [(vector[i], i) for i in range(len(vector))]
    x_couple_sorted = sorted(x_couple, key=lambda x: x[0])
    
    return [y for (x, y) in x_couple_sorted]
