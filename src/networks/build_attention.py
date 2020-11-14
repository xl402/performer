import string

import numpy as np


_CHR_IDX = string.ascii_lowercase


def build_quadratic_attention_equation(rank, attn_axes):
    # HERE:
    # Source = Values (and Key dimensions)
    # Product = intermediate
    # Target = Query/Output
    # Slight issue that the attention is not interpretable as A^{tilde} cannot be interpreted as attention

    # what the final output should be
    target_notation = _CHR_IDX[:rank]
    # tuple of batch dimensions (ALL) - attn_axes, penultimate (heads dim)
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank
    source_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            source_notation += target_notation[i]
        else:
            source_notation += _CHR_IDX[letter_offset]
            letter_offset += 1
    # 1. First group of axis are all batches (SAME), then multiply (target * source) attn_axes
    product_notation = "".join([target_notation[i] for i in batch_dims] +
                               [target_notation[i] for i in attn_axes] +
                               [source_notation[i] for i in attn_axes])
    dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
                                          product_notation)
    attn_scores_rank = len(product_notation)
    combine_equation = "%s,%s->%s" % (product_notation, source_notation,
                                      target_notation)
    return dot_product_equation, combine_equation, attn_scores_rank


def build_linear_attention_equation(rank, attn_axes):
    target_notation = _CHR_IDX[:rank]
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    letter_offset = rank

    # construct V notation
    v_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            v_notation += target_notation[i]
        else:
            v_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    # construct K_tilde notation
    k_notation = list(v_notation)
    k_notation[-1] = 'r'
    k_notation = ''.join(k_notation)

    # construct K_tilde @ V notation
    k_v_notation = "".join([target_notation[i] for i in batch_dims]) + 'r' + v_notation[-1]

    # construct Q_tilde notation
    q_notation = target_notation[:-1] + 'r'

    k_v_product_equation = "%s,%s->%s" % (k_notation, v_notation,
                                          k_v_notation)
    attn_scores_rank = len(k_v_notation)
    q_kv_product_equation = "%s,%s->%s" % (q_notation, k_v_notation,
                                           target_notation)
    return k_v_product_equation, q_kv_product_equation, attn_scores_rank


def build_normalisation_equation(rank, attn_axes):
    _CHR_IDX = string.ascii_lowercase
    source = _CHR_IDX[:rank]
    target = "".join(np.delete(list(source), attn_axes, 0))
    eq1 = f"{source},{source[:-1]}->{target}"
    eq2 = f"{source},{target}->{source[:-1]}"
    eq3 = f"{source[:-1]},{source}->{source}"
    return eq1, eq2, eq3


def build_kernel_equation(rank):
    strings = _CHR_IDX[:rank+1]
    source1 = strings[:-1]
    source2 = strings[:2] + strings[-1] + strings[-2]
    combine_equation = f"{source1},{source2}->{source1[:-1]+strings[-1]}"
    return combine_equation
