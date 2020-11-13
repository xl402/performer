import numpy as np
import pytest
import tensorflow as tf

from performer import Performer


def test_performer_compute_attention_gets_correct_output_shape():
    layer = Performer(attention_method='linear', num_heads=3, key_dim=2)
    query = tf.random.uniform(shape=[1, 18, 16], dtype='float32')
    value = tf.random.uniform(shape=[1, 4, 16], dtype='float32')
    output_tensor, weights = layer(query, value, return_attention_scores=True)
    assert all(np.array(output_tensor.shape) == [1, 18, 16])
    # assert all(np.array(weights.shape) == [1, 3, 18, 4])
