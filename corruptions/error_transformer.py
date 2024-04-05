import random


class ErrorTransformer:
    def __init__(self, fraction, column, seed=42):
        random.seed(seed)
        self.fraction = fraction
        self.column = column
        self.is_fitted_ = False
        self.num_affected_rows = 0
        self.affected_rows_indices = []
