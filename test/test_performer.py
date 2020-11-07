from itertools import product

import tensorflow as tf
import numpy as np
import pytest

from random_matrix_sampler import GaussianOrthogonalRandomMatrix


def _test_performer_scaled_dot_product_attention_approximates_keras():
    batch_size, seq_len, dim = 5, 20, 3
    input_shape = (batch_size, seq_len, dim)
    query = tf.random.uniform(input_shape)
    key = tf.random.uniform(input_shape)
    value = tf.random.uniform(input_shape)
    expected = tf.keras.layers.Attention()([query, key, value])
    assert np.allclose(expected, result)


@pytest.mark.parametrize('rows, columns', product([1, 10, 20], [1, 10, 20]))
def test_gaussian_orthogonal_random_matrix_has_correct_shape(rows, columns):
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling=0)
    out = sampler.get_2d_array()
    assert out.shape == (rows, columns)


@pytest.mark.parametrize('shape, scaling', product([2, 4, 100], [0, 0.1, 2]))
def test_gaussian_orthogonal_random_matrix_off_diags_are_zeros(shape, scaling):
    rows, columns, scaling = shape, shape, 0
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling)
    out = sampler.get_2d_array()
    out = out @ out.T
    out = out - np.diag(np.diag(out))
    assert np.allclose(out, np.zeros(out.shape))
