from itertools import product

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.models import load_model

from performer.networks.linear_attention import Performer

INPUT_SHAPES = [([1, 18, 16], [1, 4, 12], [1, 4, 12], (1,)),
                ([1, 5, 3, 16], [1, 5, 4, 16], [1, 5, 4, 12], (1, 2)),
                ([1, 5, 3, 10], [1, 5, 4, 10], [1, 5, 4, 2], (2, ))]
ATTN_METHODS = ['linear', 'quadratic']
NUM_HEADS = [1, 3, 5]
KEY_DIMS = [1, 5]


@pytest.mark.parametrize('attn_method', ATTN_METHODS)
def test_can_save_in_h5_format(tmpdir, attn_method):
    kwargs = {'num_heads': 2, 'key_dim': 20,
              'attention_method': attn_method, 'supports':2}
    model, (x, y) = get_fitted_model(**kwargs)
    model.save(tmpdir.join('model.h5'))
    reconstructed_model = load_model(tmpdir.join('model.h5'),
                                     custom_objects={'Performer': Performer})
    reconstructed_model.layers[1] = model.layers[1]
    result = model.predict(x)
    reconstructed_result = reconstructed_model.predict(x)
    assert np.allclose(result, reconstructed_result)


@pytest.mark.parametrize('attn_method', ATTN_METHODS)
def test_can_save_as_tensorflow_model(tmpdir, attn_method):
    kwargs = {'num_heads': 2, 'key_dim': 20,
              'attention_method': attn_method, 'supports':2}
    model, (x, y) = get_fitted_model(**kwargs)
    model.save(tmpdir.join('model'))
    reconstructed_model = load_model(tmpdir.join('model'))
    result = model.predict(x)
    reconstructed_result = reconstructed_model.predict(x)
    assert np.allclose(result, reconstructed_result)


@pytest.mark.parametrize('inputs, attn_method', product(INPUT_SHAPES, ATTN_METHODS))
def test_performer_output_is_same_shape_as_query(inputs, attn_method):
    q_shape, k_shape, v_shape, attn_axis = inputs
    layer = Performer(attention_method='linear', num_heads=3, key_dim=2,
                      attention_axes=attn_axis, supports=10)

    query = tf.random.uniform(shape=q_shape, dtype='float32')
    value = tf.random.uniform(shape=v_shape, dtype='float32')
    key = tf.random.uniform(shape=k_shape, dtype='float32')
    output_tensor = layer([query, key, value])
    assert all(np.array(output_tensor.shape) == q_shape)


@pytest.mark.parametrize('inputs, num_heads, key_dim', product(INPUT_SHAPES, NUM_HEADS, KEY_DIMS))
def test_performer_linear_attention_approximates_quadratic_attention(inputs, num_heads, key_dim):
    q_shape, k_shape, v_shape, attn_axis = inputs
    initializer = tf.keras.initializers.RandomNormal(seed=0)
    kwargs = {'num_heads': num_heads, 'key_dim': key_dim, 'attention_axes': attn_axis,
              'kernel_initializer': initializer, 'bias_initializer': 'zeros'}
    approx_layer = Performer(attention_method='linear', supports=1000, **kwargs)
    exact_layer = Performer(attention_method='quadratic', **kwargs)

    tf.random.set_seed(42)
    query = tf.random.uniform(shape=q_shape, dtype='float32')
    value = tf.random.uniform(shape=v_shape, dtype='float32')
    key = tf.random.uniform(shape=k_shape, dtype='float32')

    approx_output = approx_layer([query, key, value])
    exact_output = exact_layer([query, key, value])
    assert np.allclose(approx_output, exact_output, atol=1e-3)


@pytest.mark.parametrize('attn_method', ATTN_METHODS)
def test_performer_is_compatible_with_keras_input_layer(attn_method):
    layer = Performer(num_heads=2, key_dim=20, attention_method=attn_method, supports=1)
    query = tf.keras.layers.Input(shape=[4, 3])
    out = layer([query, query])
    np.testing.assert_array_equal(out.shape, [None, 4, 3])


def test_performer_raises_on_linear_attention_without_supports():
    with pytest.raises(RuntimeError) as e:
        Performer(num_heads=2, key_dim=20, attention_method='linear')
    assert 'must have numbers of supports specified' in str(e)


@pytest.mark.parametrize('attn_method', ATTN_METHODS)
def test_performer_freezes_during_inference_time(attn_method):
    kwargs = {'num_heads': 2, 'key_dim': 20,
              'attention_method': attn_method, 'supports':2}
    model, (x, y) = get_fitted_model(**kwargs)
    y1 = model.predict(x)
    y2 = model.predict(x)
    assert np.allclose(y1, y2)


def get_fitted_model(**kwargs):
    layer = Performer(**kwargs)
    x = tf.random.uniform(shape=(2, 4, 3))
    y = tf.random.uniform(shape=(2, 4, 3))
    inputs = tf.keras.layers.Input(shape=[4, 3])
    outputs = layer([inputs, inputs])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile("adam", "mean_squared_error")
    model.fit(x, y, epochs=1)
    fitting_data = (x, y)
    return model, fitting_data
