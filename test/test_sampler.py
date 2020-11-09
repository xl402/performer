from itertools import product

import numpy as np
import pytest
import tensorflow as tf

from random_matrix_sampler import GaussianOrthogonalRandomMatrix
from random_matrix_sampler import kernel_feature_creator, _get_einsum_equation


EINSUM_EQUATION = [(6, 'abcdef,abgf->abcdeg'),
                   (3, 'abc,abdc->abd'),
                   (7, 'abcdefg,abhg->abcdefh')]

KERNEL_DATA = [((4, 1, 2, 3, 5, 8), (2, 3, 4), (100, 8), (4, 1, 2, 3, 5, 200)),
               ((3, 2, 4, 7, 8, 9), (2, 4), (1, 9), (3, 2, 4, 7, 8, 2)),
               ((1, 1, 2, 3), (2,), (10, 3), (1, 1, 2, 20))]

@pytest.mark.parametrize('rows, columns', product([1, 10, 20], [1, 10, 20]))
def test_gaussian_orthogonal_random_matrix_has_correct_shape(rows, columns):
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling=0)
    out = sampler.get_2d_array()
    assert out.shape == (rows, columns)


@pytest.mark.parametrize('shape, scaling', product([2, 4, 100], [0, 1]))
def test_gaussian_orthogonal_random_matrix_off_diags_are_zeros(shape, scaling):
    rows, columns, scaling = shape, shape, scaling
    sampler = GaussianOrthogonalRandomMatrix(rows, columns, scaling)
    out = sampler.get_2d_array()
    out = out @ out.T
    out = out - np.diag(np.diag(out))
    assert np.allclose(out, np.zeros(out.shape))


def test_gaussian_orthogonal_random_matrix_raises_on_invalid_scaling_factor():
    with pytest.raises(AssertionError) as e:
        GaussianOrthogonalRandomMatrix(10, 10, scaling=0.1)
    assert "Scaling must be one of {0, 1}" in str(e)


@pytest.mark.parametrize('kernel_data', KERNEL_DATA)
def test_kernel_feature_creator_returns_correct_shape(kernel_data):
    data_shape, attn_axes, proj_shape, expected_shape = kernel_data
    data = np.random.uniform(size=data_shape)
    projection_matrix = np.random.uniform(size=proj_shape)
    result = kernel_feature_creator(data, projection_matrix, attn_axes)
    assert result.shape == expected_shape


@pytest.mark.parametrize('rank, expected', EINSUM_EQUATION)
def test_get_einsum_equation(rank, expected):
    result = _get_einsum_equation(rank)
    assert result == expected
