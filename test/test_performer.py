from itertools import product

import numpy as np
import pytest
import tensorflow as tf

from performer import Performer

INPUT_SHAPES = [([1, 18, 16], [1, 4, 12], [1, 4, 12], (1,)),
                ([1, 5, 3, 16], [1, 5, 4, 16], [1, 5, 4, 12], (1, 2)),
                ([1, 5, 3, 10], [1, 5, 4, 10], [1, 5, 4, 2], (2, ))]
ATTN_METHODS = ['linear', 'quadratic']


@pytest.mark.parametrize('inputs, attn_method', product(INPUT_SHAPES, ATTN_METHODS))
def test_performer_output_is_same_shape_as_query(inputs, attn_method):
    q_shape, k_shape, v_shape, attn_axis = inputs
    layer = Performer(attention_method='linear', num_heads=3, key_dim=2,
                      attention_axes=attn_axis)

    query = tf.random.uniform(shape=q_shape, dtype='float32')
    value = tf.random.uniform(shape=v_shape, dtype='float32')
    key = tf.random.uniform(shape=k_shape, dtype='float32')
    output_tensor = layer(query, value, key)
    assert all(np.array(output_tensor.shape) == q_shape)


@pytest.mark.parametrize('inputs', INPUT_SHAPES)
def test_performer_linear_attention_approximates_quadratic_attention(inputs):
    q_shape, k_shape, v_shape, attn_axis = inputs
    initializer = tf.keras.initializers.RandomNormal(seed=0)
    kwargs = {'num_heads': 3, 'key_dim': 2, 'attention_axes': attn_axis,
              'kernel_initializer': initializer, 'bias_initializer': 'zeros'}
    approx_layer = Performer(attention_method='linear', supports=1000, **kwargs)
    exact_layer = Performer(attention_method='quadratic', **kwargs)

    query = tf.random.uniform(shape=q_shape, dtype='float32')
    value = tf.random.uniform(shape=v_shape, dtype='float32')
    key = tf.random.uniform(shape=k_shape, dtype='float32')

    approx_output = approx_layer(query, value, key)
    exact_output = exact_layer(query, value, key)
    assert np.allclose(approx_output, exact_output, atol=1e-3)
