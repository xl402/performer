import string

import numpy as np
import tensorflow as tf

from networks.build_attention import build_kernel_equation


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
            unstructured_block = tf.random.normal(shape=square_size)
            q, _ = tf.linalg.qr(unstructured_block)
            q = tf.transpose(q)
            block_list.append(q)

        remaining_rows = self.rows - nb_full_blocks * self.columns
        if remaining_rows > 0:
            unstructured_block = tf.random.normal(shape=square_size)
            q, _ = tf.linalg.qr(unstructured_block)
            q = tf.transpose(q)
            block_list.append(q[:remaining_rows])
        final_matrix = tf.concat(block_list, axis=0)

        multiplier = self._get_multiplier()
        out = tf.matmul(tf.linalg.diag(multiplier), final_matrix)
        tf.stop_gradient(out)
        return out

    def _get_multiplier(self):
        if self.scaling == 0:
            shape = (self.rows, self.columns)
            multiplier = tf.linalg.norm(tf.random.normal(shape=shape), axis=1)
        elif self.scaling == 1:
            columns = tf.constant(self.columns, dtype=tf.dtypes.float32)
            rows = tf.constant(self.rows)
            multiplier = tf.math.sqrt(columns) * tf.ones(rows)
        return multiplier

    def __repr__(self):
        out = "GaussianOrthogonalRandomMatrix(rows={}, columns={}, scaling={})"
        out = out.format(self.rows, self.columns, self.scaling)
        return out


def kernel_feature_creator(data, projection_matrix, is_query):
    head_dim = tf.constant(data.shape[-1], dtype=tf.dtypes.float32)
    support_dim = tf.constant(projection_matrix.shape[0], dtype=tf.dtypes.float32)
    data_normalizer = 1.0 / (tf.math.sqrt(tf.math.sqrt(head_dim)))
    ratio = 1.0 / tf.math.sqrt(support_dim)
    data_mod_shape = tf.concat([tf.shape(data)[0:2], tf.shape(projection_matrix)], axis=0)
    random_matrix = tf.zeros(data_mod_shape) + projection_matrix

    normalised_data = data_normalizer * data
    dot_product_equation = build_kernel_equation(len(data.shape))
    data_dash = tf.einsum(dot_product_equation, normalised_data, random_matrix)

    diag_data = tf.math.square(data)
    diag_data = tf.math.reduce_sum(diag_data, axis=- 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = tf.expand_dims(diag_data, axis=-1)

    if is_query:
        last_dims_t = len(data_dash.shape) - 1
        data_dash = ratio * (
                  tf.math.exp(data_dash - diag_data -
                  tf.math.reduce_max(data_dash, axis=last_dims_t, keepdims=True)) + 1e-4)
    else:
        data_dash = ratio * (tf.math.exp(data_dash - diag_data - tf.math.reduce_max(data_dash)) + 1e-4)
    return data_dash
