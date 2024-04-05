from sklearn.base import TransformerMixin, BaseEstimator


# HandleSwappedColumns tries to detect swapped values and correct them
class HandleSwappedColumns(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.is_fitted = False
        self.swapped_indices = None
        self.stars_correct = None
        self.supported_lan = ["en", "tr", "es", "english", "turkish", "spanish"]

    def fit(self, data, y=None):
        # star rating column is the only column that should contain integers so if the column only contains int, it means there are now swapped values in this column
        if data["stars"].dtype == "int64":
            self.stars_correct = True
        else:
            self.stars_correct = False

        self.is_fitted = True

        return self

    def correct_column_swaps(self, row):
        if row["review_body"] in self.supported_lan:
            lan = row["review_body"]
            row["review_body"] = row["language"]
            row["language"] = lan
        if not self.stars_correct:
            if type(row["review_body"]) == int:
                star = row["review_body"]
                row["review_body"] = row["stars"]
                row["stars"] = star
            if type(row["language"]) == int:
                star = row["language"]
                row["language"] = row["stars"]
                row["stars"] = star
        return row

    def transform(self, data):
        if self.is_fitted:
            df_copy = data.copy(deep=True)

            df_copy = df_copy.apply(self.correct_column_swaps, axis=1)

            return df_copy
