# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 2022 19:00

@author: Svein Jakob Kristoffersen (sveinjakobkristoffersen@gmail.com)

Functions used for machine learning setup.

"""

import numpy as np


def make_x_y(list_of_lists):
    """
    Makes X and y matrices of list of lists for use in machine learning.
    Each list in list of lists correspond to a class.
    """
    y = []
    X = []

    i = 0
    for single_list in list_of_lists:
        y.extend(i for _ in range(len(single_list)))
        i += 1
        X.extend(single_list)
    return np.array(X), np.array(y)


def undersample_list(majority_list, new_size=None):
    """
    Performs random removal of elements in a list, until it reaches wanted
    size
    """
    indices = np.random.choice(majority_list.shape[0], new_size)
    return majority_list[indices]

