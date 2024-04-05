import random
from error_transformer import ErrorTransformer


class SwappedValuesTransformer(ErrorTransformer):
    def __init__(self, fraction, column, second_column):
        ErrorTransformer.__init__(self, fraction, column)
        self.second_column = second_column

    def fit(self, data, y=None):
        # set the length of the dataset and calculate the number of corruptions that need to be done
        num_rows = len(data)
        self.num_affected_rows = int(num_rows * self.fraction)

        # randomly select indices of the rows that are going to be transformed
        data_indices = list(data.index)
        self.affected_rows_indices = random.sample(data_indices, self.num_affected_rows)

        self.is_fitted_ = True
        return self

    def transform(self, data):
        if self.is_fitted_:
            # create a deep copy of the data so the original dataframe remains the same
            df_copy = data.copy(deep=True)

            # for each row that should contain a swapped value error, swap the values of the two given columns
            for index in self.affected_rows_indices:
                col1_val = df_copy.at[index, self.column]
                df_copy.at[index, self.column] = df_copy.at[index, self.second_column]
                df_copy.at[index, self.second_column] = col1_val

            return df_copy
