import random
from error_transformer import ErrorTransformer
import numpy as np


class MissingValuesTransformer(ErrorTransformer):
    def __init__(self, fraction, column):
        ErrorTransformer.__init__(self, fraction, column)

    def fit(self, data):
        # set the length of the dataset and calculate the number of corruptions that need to be done
        num_rows = len(data)
        self.num_affected_rows = int(num_rows * self.fraction)

        # randomly select indices of the rows that are going to be transformed
        data_indices = list(data.index)
        self.affected_rows_indices = random.sample(data_indices, self.num_affected_rows)

        self.is_fitted_ = True

    def transform(self, data):
        if self.is_fitted_:
            df_copy = data.copy(deep=True)

            # For the rows that should contain missing values, set the value in the given column to NaN
            for index in self.affected_rows_indices:
                df_copy.at[index, self.column] = np.nan

            return df_copy
