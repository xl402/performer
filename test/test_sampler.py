from itertools import product

import numpy as np
import pickle
import pytest
import tensorflow as tf

from performer.networks.random_matrix_sampler import GaussianOrthogonalRandomMatrix
from performer.networks.random_matrix_sampler import kernel_feature_creator


# data_shape, projection_matrix_shape, expected_shape
KERNEL_DATA = [((4, 1, 2, 3, 5, 8), (100, 8), (4, 1, 2, 3, 5, 100)),
               ((3, 2, 4, 7, 8, 9), (1, 9), (3, 2, 4, 7, 8, 1)),
               ((1, 1, 2, 3), (10, 3), (1, 1, 2, 10))]


@pytest.mark.parametrize('rows, columns', product([1, 10, 20], [1, 10, 20]))
def test_gaussian_orthogonal_random_matrix_has_correct_shape(rows, columns):
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling=0)
    out = sampler.sample()
    assert out.shape == (rows, columns)


def test_gaussian_orthogonal_random_matix_repr_is_readable():
    sampler = GaussianOrthogonalRandomMatrix(100, 10, scaling=1)
    out = str(sampler)
    expected = 'GaussianOrthogonalRandomMatrix(rows=100, columns=10, scaling=1)'
    assert out == expected


@pytest.mark.parametrize('shape, scaling', product([2, 4, 100], [0, 1]))
def test_gaussian_orthogonal_random_matrix_off_diags_are_zeros(shape, scaling):
    rows, columns, scaling = shape, shape, scaling
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling)
    out = sampler.sample().numpy()
    out = out @ out.T
    out = out - np.diag(np.diag(out))
    assert np.allclose(out, np.zeros(out.shape), atol=1e-4)


def test_gaussian_orthogonal_random_matrix_raises_on_invalid_scaling_factor():
    with pytest.raises(AssertionError) as e:
        GaussianOrthogonalRandomMatrix(10, 10, scaling=0.1)
    assert "Scaling must be one of {0, 1}" in str(e)


@pytest.mark.parametrize('kernel_data', KERNEL_DATA)
def test_kernel_feature_creator_returns_correct_shape(kernel_data):
    data_shape, proj_shape, expected_shape = kernel_data
    data = tf.random.uniform(shape=data_shape)
    projection_matrix = tf.random.uniform(shape=proj_shape)
    result = kernel_feature_creator(data, projection_matrix, True)
    assert result.shape == expected_shape


def test_kernel_feature_creator_approximates_attention():
    Q = tf.random.uniform(shape=(1, 2, 3, 4))
    K = tf.random.uniform(shape=(1, 2, 3, 4))
    P = GaussianOrthogonalRandomMatrix(2000, 4).sample()
    Q_hat = kernel_feature_creator(Q, P, is_query=True)
    K_hat = kernel_feature_creator(K, P, is_query=False)
    A = _attention(Q, K, 'nonlinear')
    A_hat = _attention(Q_hat, K_hat, 'linear')
    assert np.allclose(A, A_hat, atol=0.5)


def _attention(Q, K, method='linear'):
    A = np.einsum("abcd,abed->abce", Q, K)
    if method == 'nonlinear':
        A = np.exp(A / np.sqrt(Q.shape[-1]))
    shape = A.shape
    ones = np.ones(shape=shape[:-1])
    D = np.einsum("abce,abc->abc", A, ones)
    D = 1 / D
    DA = np.einsum("abc,abce->abce", D, A)
    return DA
