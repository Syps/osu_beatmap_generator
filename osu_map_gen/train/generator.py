import numpy as np


class HDF5DataGenerator:
    def __init__(self, batch_size, dim_x=69, dim_y=41, dim_z=1, first=None,
                 last=None, n_difficulties=1):

        if first and last:
            raise ValueError("""
            Can only generate from either first x percent 
            or last x percent of input model_data
            """)

        self.first = first
        self.last = last
        self.batch_size = batch_size
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.n_difficulties = n_difficulties

    def generate(self, X, y):
        while True:
            indexes = np.arange(0, X.shape[0], dtype=np.int)

            if self.first:
                end = int(X.shape[0] * self.first)
                indexes = indexes[:end]
            elif self.last:
                start = int(X.shape[0] * (1 - self.last))
                indexes = indexes[start:]

            np.random.shuffle(indexes)
            num_batches = np.arange(0, indexes.shape[0] // self.batch_size,
                                    dtype=np.int)

            for batch_number in num_batches:
                batch_start = batch_number * self.batch_size
                batch_end = (batch_number + 1) * self.batch_size

                _X = np.empty(
                    (self.batch_size, self.dim_x, self.dim_y, self.dim_z))
                _y = np.empty((self.batch_size, 2 * self.n_difficulties))

                for i, index in enumerate(indexes[batch_start: batch_end]):
                    _X[i, :, :, 0] = X[index]
                    _y[i] = y[index]

                yield _X, _y
