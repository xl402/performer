import numpy as np


class GaussianOrthogonalRandomMatrix:

    def __init__(self, nb_rows, nb_columns, scaling=0):
        self.nb_rows = nb_rows
        self.nb_columns = nb_columns
        self.scaling = scaling

    def get_2d_array(self):

        nb_full_blocks = int(self.nb_rows / self.nb_columns)
        block_list = []

        for _ in range(nb_full_blocks):
            unstructured_block = np.random.normal(size=(self.nb_columns, self.nb_columns))
            # computes qr factorisation
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q)
        remaining_rows = self.nb_rows - nb_full_blocks * self.nb_columns
        if remaining_rows > 0:
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = np.vstack(block_list)

        if self.scaling == 0:
            # seemingly a weird way of generating a random vector??
            multiplier = np.linalg.norm(np.random.normal(size=(self.nb_rows, self.nb_columns)), axis=1)
        elif self.scaling == 1:
            # scale by sqrt of the number of columns
            multiplier = np.sqrt(float(self.nb_columns)) * np.ones(shape=self.nb_rows)
        else:
            raise ValueError('Scaling must be one of {0, 1}. Was %s' % self.scaling)

        return np.matmul(np.diag(multiplier), final_matrix)
