import string

import numpy as np


def build_quadratic_attention_equation(rank, attn_axes):
    batch_dims = _get_index_of_batch_dims(rank, attn_axes)
    query_notation, value_notation = _get_query_and_value_notation(rank, batch_dims)
    key_notation = value_notation
    qk_product_notation = "".join([query_notation[i] for i in batch_dims] +
                                  [query_notation[i] for i in attn_axes] +
                                  [key_notation[i] for i in attn_axes])
    kq_product_equation = f"{key_notation},{query_notation}->{qk_product_notation}"
    qk_v_product_equation = f"{qk_product_notation},{value_notation}->{query_notation}"
    attn_scores_rank = len(qk_product_notation)
    return kq_product_equation, qk_v_product_equation, attn_scores_rank


def build_linear_attention_equation(rank, attn_axes):
    batch_dims = _get_index_of_batch_dims(rank, attn_axes)
    query_notation, value_notation = _get_query_and_value_notation(rank, batch_dims)

    query_hat_notation = query_notation[:-1] + 'r'
    key_hat_notation = value_notation[:-1] + 'r'
    key_value_notation = "".join([key_hat_notation[i] for i in batch_dims] +
                                 list('r' + value_notation[-1]))
    kv_product_equation = f"{key_hat_notation},{value_notation}->{key_value_notation}"
    q_kv_product_equation = f"{query_hat_notation},{key_value_notation}->{query_notation}"
    attn_scores_rank = len(key_value_notation)
    return kv_product_equation, q_kv_product_equation, attn_scores_rank


def _get_index_of_batch_dims(rank, attn_axes):
    batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
    return batch_dims


def _get_query_and_value_notation(rank, batch_dims):
    chr_idx = string.ascii_lowercase
    query_notation = chr_idx[:rank]
    letter_offset = rank
    value_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            value_notation += query_notation[i]
        else:
            value_notation += chr_idx[letter_offset]
            letter_offset += 1
    return query_notation, value_notation


def build_normalisation_equation(rank, attn_axes):
    chr_idx = string.ascii_lowercase
    key_hat_notation = chr_idx[:rank]
    query_hat_notation = key_hat_notation
    ones_array_notation = key_hat_notation[:-1]
    key_ones_notation = "".join(np.delete(list(key_hat_notation), attn_axes, 0))
    k1_equation = f"{key_hat_notation},{ones_array_notation}->{key_ones_notation}"
    q_k1_equation = f"{query_hat_notation},{key_ones_notation}->{ones_array_notation}"
    qk1_q_equation = f"{ones_array_notation},{key_hat_notation}->{query_hat_notation}"
    return k1_equation, q_k1_equation, qk1_q_equation


def build_kernel_equation(rank):
    chr_idx = string.ascii_lowercase
    strings = chr_idx[:rank+1]
    data_notation = strings[:-1]
    sampling_matrix_notation = strings[:2] + strings[-1] + strings[-2]
    product = data_notation[:-1]+strings[-1]
    combine_equation = f"{data_notation},{sampling_matrix_notation}->{product}"
    return combine_equation
