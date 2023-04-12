# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def get_params():
    paramname = {'diversify': [' --latent_domain_num ',
                               ' --alpha1 ',  ' --alpha ', ' --lam ']}
    paramlist = {
        'diversify': [[2, 3, 5, 10, 20], [0.1, 0.5, 1], [0.1, 1, 10], [0], [[1, 150], [3, 50], [5, 30], [10, 15], [30, 5]]]
    }
    return paramname, paramlist
