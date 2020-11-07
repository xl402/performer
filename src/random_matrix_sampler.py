import numpy as np


class GaussianOrthogonalRandomMatrix:

    def __init__(self, rows, columns, scaling=0):
        self.rows = rows
        self.columns = columns
        self.scaling = scaling
        assert self.scaling in [0, 1], 'Scaling must be one of {0, 1}'

    def get_2d_array(self):
        nb_full_blocks = int(self.rows / self.columns)
        block_list = []

        square_size = (self.columns, self.columns)
        for _ in range(nb_full_blocks):
            unstructured_block = np.random.normal(size=square_size)
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q)

        remaining_rows = self.rows - nb_full_blocks * self.columns

        if remaining_rows > 0:
            unstructured_block = np.random.normal(size=square_size)
            q, _ = np.linalg.qr(unstructured_block)
            q = np.transpose(q)
            block_list.append(q[0:remaining_rows])
        final_matrix = np.vstack(block_list)

        multiplier = self._get_multiplier()
        return np.matmul(np.diag(multiplier), final_matrix)

    def _get_multiplier(self):
        if self.scaling == 0:
            # weird way of generating a random vector??
            size = (self.rows, self.columns)
            multiplier = np.linalg.norm(np.random.normal(size=size), axis=1)
        elif self.scaling == 1:
            multiplier = np.sqrt(self.columns) * np.ones(self.rows)
        return multiplier
