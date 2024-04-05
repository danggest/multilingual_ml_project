import transformers
from sklearn.base import TransformerMixin, BaseEstimator


# The HandleTypoTransformer based on the language stated in the language column, corrects errors in the column specified by the user
class HandleTypoTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, column):
        print("HandleTypoTransformer started")
        self.is_fitted = False
        self.column = column
        self.en_spell_check = transformers.pipeline(
            "text2text-generation", model="oliverguhr/spelling-correction-english-base"
        )
        self.es_spell_check = transformers.pipeline(
            "text2text-generation",
            model="jorgeortizfuentes/spanish-spellchecker-mbart-large-cc25_3e",
        )
        self.tr_spell_check = transformers.pipeline(
            "text2text-generation", model="trkdncer/bart_base_tr_spelling"
        )
        self.corrected_data = None

    def fit(self, data, y=None):
        self.is_fitted = True

        return self

    # using language specific models to correct typos in the text
    def spell_check(self, row):
        print("Spell check started")
        if row["language"] == "en":
            corrected_text = self.en_spell_check(row[self.column], max_length=128)
            return corrected_text[0]["generated_text"]
        elif row["language"] == "tr":
            corrected_text = self.tr_spell_check(row[self.column], max_length=128)
            return corrected_text[0]["generated_text"]
        else:
            corrected_text = self.es_spell_check(row[self.column], max_length=128)
            return corrected_text[0]["generated_text"]

    def transform(self, data):
        print("HandleTypoTransformer transform started")
        if self.is_fitted:
            df_copy = data.copy(deep=True)

            df_copy[self.column] = df_copy.apply(self.spell_check, axis=1)

            print("HandleTypoTransformer transform finished")
            self.corrected_data = df_copy
            return df_copy
