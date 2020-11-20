import string
from functools import partial

import numpy as np
import tensorflow as tf

from performer.networks.build_attention import build_kernel_equation


_CHR_IDX = string.ascii_lowercase


class GaussianOrthogonalRandomMatrix:
    def __init__(self, rows, columns, scaling=0):
        self.rows = rows
        self.columns = columns
        self.scaling = scaling
        assert self.scaling in [0, 1], 'Scaling must be one of {0, 1}'

    def sample(self):
        shape = (self.rows, self.columns)
        unstructured_block = tf.random.normal(shape=shape)
        q, r = tf.linalg.qr(unstructured_block)
        final_matrix = q if self.rows >= self.columns else r

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
    data_hat = tf.einsum(dot_product_equation, normalised_data, random_matrix)

    diag_data = tf.math.square(data)
    diag_data = tf.math.reduce_sum(diag_data, axis=- 1)
    diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
    diag_data = tf.expand_dims(diag_data, axis=-1)

    if is_query:
        last_dims_t = len(data_hat.shape) - 1
        func = partial(tf.math.reduce_max, axis=last_dims_t, keepdims=True)
    else:
        func = tf.math.reduce_max
    out = ratio * (tf.math.exp(data_hat - diag_data - func(data_hat)) + 1e-4)
    return out
