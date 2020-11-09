import string

import numpy as np


_CHR_IDX = string.ascii_lowercase


class GaussianOrthogonalRandomMatrix:

    def __init__(self, rows, columns, scaling=0):
        self.rows = rows
        self.columns = columns
        self.scaling = scaling
        assert self.scaling in [0, 1], 'Scaling must be one of {0, 1}'

    def get_2d_array(self):
        nb_full_blocks = int(self.rows / self.columns)
        block_list = []

        square_size = (self.columns, self.columns)
        for _ in range(nb_full_blocks):
            unstructured_block = np.random.normal(size=square_size)
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q)

        remaining_rows = self.rows - nb_full_blocks * self.columns

        if remaining_rows > 0:
            unstructured_block = np.random.normal(size=square_size)
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = np.vstack(block_list)

        multiplier = self._get_multiplier()
        return np.matmul(np.diag(multiplier), final_matrix)

    def _get_multiplier(self):
        if self.scaling == 0:
            size = (self.rows, self.columns)
            multiplier = np.linalg.norm(np.random.normal(size=size), axis=1)
        elif self.scaling == 1:
            multiplier = np.sqrt(self.columns) * np.ones(self.rows)
        return multiplier


def kernel_feature_creator(data, projection_matrix, attention_dims):
    data_normalizer = 1.0 / (np.sqrt(np.sqrt(data.shape[-1])))
    ratio = 1.0 / np.sqrt(projection_matrix.shape[0])
    data_mod_shape = data.shape[0:2] + projection_matrix.shape
    random_matrix = np.zeros(data_mod_shape) + projection_matrix

    normalised_data = data_normalizer * data
    equation = _get_einsum_equation(len(data.shape))
    data_dash = np.einsum(equation, normalised_data, random_matrix)
    data_dash_cos = ratio * np.cos(data_dash)
    data_dash_sin = ratio * np.sin(data_dash)
    data_dash = np.concatenate((data_dash_cos, data_dash_sin), axis=-1)

    # Constructing D_data and data^{'}
    diag_data = np.square(data)
    diag_data = np.sum(diag_data, axis=data.ndim - 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = np.expand_dims(diag_data, axis=data.ndim - 1)

    # Additional renormalization for numerical stability
    data_renormalizer = np.max(diag_data, attention_dims, keepdims=True)
    diag_data -= data_renormalizer
    diag_data = np.exp(diag_data)
    data_prime = data_dash * diag_data
    return data_prime


def _get_einsum_equation(rank):
    strings = _CHR_IDX[:rank+1]
    source1 = strings[:-1]
    source2 = strings[:2] + strings[-1] + strings[-2]
    combine_equation = f"{source1},{source2}->{source1[:-1]+strings[-1]}"
    return combine_equation
