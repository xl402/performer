import numpy as np
import pytest
import tensorflow as tf

from performer.networks.build_attention import build_linear_attention_equation
from performer.networks.build_attention import build_quadratic_attention_equation
from performer.networks.build_attention import build_normalisation_equation
from performer.networks.build_attention import build_kernel_equation


quadratic_attention = [(4, (1, ), ('aecd,abcd->acbe', 'acbe,aecd->abcd', 4)),
                       (4, (1, 2), ('aefd,abcd->abcef', 'abcef,aefd->abcd', 5)),
                       (3, (1, 0), ('dec,abc->baed', 'baed,dec->abc', 4))]

linear_attention = [(4, (1, 2), ('aefr,aefd->ard', 'abcr,ard->abcd', 3)),
                    (4, (1, ), ('aecr,aecd->acrd', 'abcr,acrd->abcd', 4))]

normalisation_equation = [(4, (1, ), ('abcd,abc->acd', 'abcd,acd->abc', 'abc,abcd->abcd')),
                          (4, (1, 2), ('abcd,abc->ad', 'abcd,ad->abc', 'abc,abcd->abcd'))]

kernel_equation = [(6, 'abcdef,abgf->abcdeg'),
                   (3, 'abc,abdc->abd'),
                   (7, 'abcdefg,abhg->abcdefh')]


@pytest.mark.parametrize('rank, attn_axes, expected', quadratic_attention)
def test_build_quadratic_attention(rank, attn_axes, expected):
    result = build_quadratic_attention_equation(rank, attn_axes)
    dot_product_equation, combine_equation, attn_scores_rank = result
    assert dot_product_equation == expected[0]
    assert combine_equation == expected[1]
    assert attn_scores_rank == expected[2]


@pytest.mark.parametrize('rank, attn_axes, expected', linear_attention)
def test_build_linear_attention(rank, attn_axes, expected):
    result = build_linear_attention_equation(rank, attn_axes)
    assert np.all(result==expected)


@pytest.mark.parametrize('rank, attn_axes, expected', normalisation_equation)
def test_build_normalisation_equation(rank, attn_axes, expected):
    result = build_normalisation_equation(rank, attn_axes)
    assert np.all(result==expected)


@pytest.mark.parametrize('rank, expected', kernel_equation)
def test_get_einsum_equation(rank, expected):
    result = build_kernel_equation(rank)
    assert result == expected
