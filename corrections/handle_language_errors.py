import transformers
import random
from sklearn.base import TransformerMixin, BaseEstimator
from langdetect import detect_langs


# HandleLanguageErrors predicts the value that should be in the language column
class HandleLanguageErrors(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.is_fitted = False
        self.supported_lan = ["es", "en", "tr"]

    def fit(self, data, y=None):
        self.is_fitted = True

        return self

    # Predicts the language of the review_body text, if the predicted language is not in the supported languages 'tr', 'es' or 'en' then a random one of the supported languages
    # will be chosen
    def predict_lan(self, row):
        pred = detect_langs(row["review_body"])
        pred_lan = None
        for lan_score in pred:
            if lan_score.lang in self.supported_lan:
                pred_lan = lan_score.lang
                break
        if pred_lan == None:
            pred_lan = random.choice(self.supported_lan)

        return pred_lan

    def transform(self, data):
        if self.is_fitted:
            df_copy = data.copy(deep=True)

            df_copy["language"] = df_copy.apply(self.predict_lan, axis=1)

            return df_copy
