from sklearn.base import TransformerMixin, BaseEstimator


# HandleMissingValuesTransformer drops rows with missing values in the review_body column since there is now reasonable way of predicting the star rating without it
# Missing values in the language are not problematic as the language will be predicted by the HandleLanguageErrors class
class HandleMissingValuesTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, column):
        self.is_fitted = False
        self.column = column

    def fit(self, data, y=None):
        self.is_fitted = True

        return self

    def transform(self, data):
        if self.is_fitted:
            # only remove rows with missing values in "review_body" because the values in "language" will be predicted
            if self.column == "review_body":
                df_copy = data.copy(deep=True)
                df_copy = df_copy.dropna(subset=[self.column])

                return df_copy
