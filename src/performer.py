import math
import logging

from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.keras.layers import core
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
import tensorflow as tf

from random_matrix_sampler import GaussianOrthogonalRandomMatrix as GOR
from build_attention import build_linear_attention_equation
from build_attention import build_quadratic_attention_equation

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Performer(MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        attention_method = kwargs.pop('attention_method', 'quadratic')
        assert attention_method in ['linear', 'quadratic'], 'invalid attention method'
        if attention_method == 'quadratic':
            self._compute_attention = self.quadratic_attention
            self._build_attention_equation = build_quadratic_attention_equation
        else:
            self._compute_attention = self.linear_attention
            self._build_attention_equation = build_linear_attention_equation

        supports = kwargs.pop('supports', 100)
        self.supports = supports

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

    def quadratic_attention(self, query, key, value,
                           attention_mask=None, training=None):
        logger.debug(f"\ndot product equation: {self._dot_product_equation}")
        logger.debug(f"\ncombine equation: {self._combine_equation}")

        query = math_ops.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        attention_scores = special_math_ops.einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training)

        attention_output = special_math_ops.einsum(self._combine_equation, attention_scores_dropout, value)

        return attention_output, attention_scores

    def linear_attention(self, query, key, value, attention_mask=None, training=None):
        raise(NotImplementedError)


if __name__ == '__main__':
    layer = Performer(num_heads=2, key_dim=2, attention_method='quadratic')
    query = tf.keras.Input(shape=[18, 16])
    key = tf.keras.Input(shape=[4, 16])
    value = tf.keras.Input(shape=[4, 3])
    output_tensor, weights = layer(query, key, value, return_attention_scores=True)
