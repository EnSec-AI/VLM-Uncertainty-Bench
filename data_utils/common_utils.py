import json
import math
import random
import numpy as np

OPTIONS = ['A', 'B', 'C', 'D']
ALL_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F']


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    #if type(value) is str and value.lower() == 'none':
    #    return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def open_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data