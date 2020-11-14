import math
import logging

from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.layers import core
from tensorflow import multiply, einsum
import tensorflow as tf

from networks.random_matrix_sampler import GaussianOrthogonalRandomMatrix as GOR
from networks.random_matrix_sampler import kernel_feature_creator
from networks.build_attention import build_linear_attention_equation
from networks.build_attention import build_quadratic_attention_equation
from networks.build_attention import build_normalisation_equation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Performer(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        self.attention_method = kwargs.pop('attention_method', 'quadratic')
        message = 'invalid attention method'
        assert self.attention_method in ['linear', 'quadratic'], message
        if self.attention_method == 'quadratic':
            self._compute_attention = self.quadratic_attention
            self._build_attention_equation = build_quadratic_attention_equation
        else:
            self.scaling = kwargs.pop('scaling', 1)
            self.supports = kwargs.pop('supports', 200)
            self.sampler = GOR(self.supports, kwargs['key_dim'], self.scaling)
            self._compute_attention = self.linear_attention
            self._build_attention_equation = build_linear_attention_equation
            self._build_normalisation_equation = build_normalisation_equation

        super().__init__(*args, **kwargs)

    def _build_attention(self, rank):
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        self._dot_product_equation, self._combine_equation, attn_scores_rank = (
             self._build_attention_equation(rank, attn_axes=self._attention_axes))
        norm_axes = tuple(range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._softmax = advanced_activations.Softmax(axis=norm_axes)
        self._dropout_layer = core.Dropout(rate=self._dropout)
        if hasattr(self, '_build_normalisation_equation'):
            self._normalisation_equations = self._build_normalisation_equation(rank, self._attention_axes)


    def quadratic_attention(self, query, key, value, attention_mask=None, training=None):
        query = multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        attention_scores = einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training)

        attention_output = einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

    def linear_attention(self, query, key, value, attention_mask=None, training=None):
        random_features = self.sampler.get_2d_array()
        lifted_query = kernel_feature_creator(query, random_features, True)
        lifted_key = kernel_feature_creator(key, random_features, False)

        kv = einsum(self._dot_product_equation, lifted_key, value)
        qkv = einsum(self._combine_equation, lifted_query, kv)
        shape = tf.compat.v1.placeholder(tf.float32, shape=lifted_key.shape[:-1])
        ones = tf.ones_like(shape)

        eq1, eq2, eq3 = self._normalisation_equations
        k_ones = einsum(eq1, lifted_key, ones)
        D = einsum(eq2, lifted_query, k_ones)
        D = 1 / (D + 1e-6)
        out = einsum(eq3, D, qkv)
        return out, None


if __name__ == '__main__':
    initializer = tf.keras.initializers.RandomNormal(seed=0)
    layer = Performer(num_heads=2, key_dim=20, attention_method='quadratic',
                      kernel_initializer=initializer, bias_initializer='zeros')
    linear_layer = Performer(num_heads=2, key_dim=20, attention_method='linear', supports=1000,
                      kernel_initializer=initializer, bias_initializer='zeros')

    query = tf.keras.layers.Input(shape=[4, 3])

    exact = layer(query, query)
    approx = linear_layer(query, query)
