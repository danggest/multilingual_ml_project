import pandas as pd
import numpy as np
import random
from error_transformer import ErrorTransformer


class CategoricalLabelTransformer(ErrorTransformer):
    def __init__(self, fraction, column, categories=None):
        ErrorTransformer.__init__(self, fraction, column)
        self.categories = categories

    # FIX: maybe add the option to give a list of categories that haven't been seen in the dataset

    def fit(self, data, y=None):
        # set the length of the dataset and calculate the number of corruptions that need to be done
        num_rows = len(data)
        self.num_affected_rows = int(num_rows * self.fraction)
        affected_column = data[self.column]

        # if no list with categories was supplied, the categories from which the corruptions will be chosen are the ones that are present in
        # the original dataset
        if self.categories == None:
            self.categories = list(affected_column.unique())

        # randomly select indices of the rows that are going to be transformed
        data_indices = list(data.index)
        self.affected_rows_indices = random.sample(data_indices, self.num_affected_rows)

        self.is_fitted_ = True

        return self

    def transform(self, data):
        if self.is_fitted_:
            # create a deep copy of the data so the original dataframe remains the same
            df_copy = data.copy(deep=True)

            # for each row that should have a corrupted value, we select a random category(not the current value) and set it as the value.
            for index in self.affected_rows_indices:
                current_val = df_copy.at[index, self.column]

                if isinstance(self.categories, dict):
                    df_copy.at[index, self.column] = self.categories[current_val]

                else:
                    # create a copy of the categories because normal assignment would mean that removing a value from categories_min_current
                    # would also delete the value from self.categories
                    categories_min_current = self.categories.copy()

                    # if the current value is in the list we remove it so we can't randomly select the same value
                    if current_val in self.categories:
                        categories_min_current.remove(current_val)

                    random_category = random.choice(categories_min_current)
                    df_copy.at[index, self.column] = random_category

            return df_copy
