# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from alg.algs.SDMix import SDMix

ALGORITHMS = [
    'SDMix'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
