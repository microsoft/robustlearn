# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .algs.diversify import Diversify

ALGORITHMS = [
    'diversify'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in ALGORITHMS:
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return Diversify
