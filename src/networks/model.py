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
        self.scaling = kwargs.pop('scaling', 0)
        self.supports = kwargs.pop('supports', None)
        self._check_attention_method_is_valid()
        if self.attention_method == 'quadratic':
            self._compute_attention = self.quadratic_attention
            self._build_attention_equation = build_quadratic_attention_equation
        else:
            self._compute_attention = self.linear_attention
            self._build_attention_equation = build_linear_attention_equation
            self._check_supports_is_not_none()
            self.sampler = GOR(self.supports, kwargs['key_dim'], self.scaling)
            self._random_features = self.sampler.get_2d_array()
            self._build_normalisation_equation = build_normalisation_equation
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super(Performer, self).get_config()
        config.update({'scaling': self.scaling, 'supports': self.scaling, 'attention_method': self.attention_method})
        return config

    def _check_supports_is_not_none(self):
        if self.supports is None:
            raise(RuntimeError('must have numbers of supports specified'))

    def _check_attention_method_is_valid(self):
        message = 'invalid attention method'
        assert self.attention_method in ['linear', 'quadratic'], message

    def _build_attention(self, rank):
        if self._attention_axes is None:
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)
        self._add_attention_equation(rank)
        self._add_soft_max_and_dropout_layers()
        if hasattr(self, '_build_normalisation_equation'):
            self._add_normalisation_equation(rank)

    def _add_attention_equation(self, rank):
        result = self._build_attention_equation(rank, self._attention_axes)
        self._dot_product_equation, self._combine_equation, attn_scores_rank = result
        norm_axes = tuple(range(attn_scores_rank - len(self._attention_axes), attn_scores_rank))
        self._norm_axes = norm_axes

    def _add_soft_max_and_dropout_layers(self):
        self._softmax = advanced_activations.Softmax(axis=self._norm_axes)
        self._dropout_layer = core.Dropout(rate=self._dropout)

    def _add_normalisation_equation(self, rank):
        result = self._build_normalisation_equation(rank, self._attention_axes)
        self._k1_equation, self._q_k1_equation, self._qk1_q_equation = result

    def quadratic_attention(self, query, key, value, attention_mask=None, training=None):
        query = multiply(query, 1. / math.sqrt(float(self._key_dim)))
        attention_scores = einsum(self._dot_product_equation, key, query)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
        attention_output = einsum(self._combine_equation, attention_scores_dropout, value)
        return attention_output, attention_scores

    def linear_attention(self, query, key, value, attention_mask=None, training=None):
        if attention_mask is not None:
            raise(NotImplementedError('masked linear attention not implemented'))
        random_features = self._get_random_features(training)
        lifted_query = kernel_feature_creator(query, random_features, True)
        lifted_key = kernel_feature_creator(key, random_features, False)
        kv = einsum(self._dot_product_equation, lifted_key, value)
        qkv = einsum(self._combine_equation, lifted_query, kv)
        normalised_qkv = self._normalise(lifted_key, lifted_query, qkv)
        return normalised_qkv, None

    @tf.function
    def _get_random_features(self, train):
        out = self.sampler.get_2d_array() if train is None else self._random_features
        return out

    def _normalise(self, lifted_key, lifted_query, qkv):
        ones = tf.ones_like(lifted_key[..., 0])
        k_ones = einsum(self._k1_equation, lifted_key, ones)
        D = einsum(self._q_k1_equation, lifted_query, k_ones)
        D = 1. / (D + 1e-6)
        normalised_qkv = einsum(self._qk1_q_equation, D, qkv)
        return normalised_qkv
