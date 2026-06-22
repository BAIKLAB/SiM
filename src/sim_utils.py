import random

import numpy as np
import torch


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_dict_to_vector(state_dict, remove_keys=None):
    if remove_keys is None:
        remove_keys = []

    flat_params = []
    for key, value in state_dict.items():
        if key in remove_keys:
            continue
        flat_params.append(value.reshape(-1))
    return torch.cat(flat_params)


def vector_to_state_dict(vector, reference_state_dict, remove_keys=None):
    if remove_keys is None:
        remove_keys = []

    new_state_dict = {}
    idx = 0
    for key, value in reference_state_dict.items():
        if key in remove_keys:
            new_state_dict[key] = value.clone()
            continue

        numel = value.numel()
        new_state_dict[key] = vector[idx: idx + numel].reshape_as(value).clone()
        idx += numel

    return new_state_dict
