import tensorflow as tf
import numpy as np

from random_matrix_sampler import GaussianOrthogonalRandomMatrix


def _test_performer_scaled_dot_product_attention_approximates_keras():
    batch_size, seq_len, dim = 5, 20, 3
    input_shape = (batch_size, seq_len, dim)
    query = tf.random.uniform(input_shape)
    key = tf.random.uniform(input_shape)
    value = tf.random.uniform(input_shape)
    expected = tf.keras.layers.Attention()([query, key, value])
    pass


def test_gaussian_orthogonal_random_matrix_has_correct_shape():
    nb_rows, nb_columns, scaling = 100, 20, 0
    sampler = GaussianOrthogonalRandomMatrix(nb_rows, nb_columns, scaling)
    out = sampler.get_2d_array()
    assert out.shape == (nb_rows, nb_columns)


def test_gaussian_orthogonal_random_matrix_off_diags_are_zeros():
    nb_rows, nb_columns, scaling = 40, 40, 0
    sampler = GaussianOrthogonalRandomMatrix(nb_rows, nb_columns, scaling)
    out = sampler.get_2d_array()
    out = out @ out.T
    out = out - np.diag(np.diag(out))
    assert np.allclose(out, np.zeros(out.shape))
