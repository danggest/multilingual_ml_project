import random
import spacy
import textacy
from corrections.swap_verb import swap_words, split_into_sentences
from .error_transformer import ErrorTransformer


class SentenceStructureTransformer(ErrorTransformer):
    # idea: use pos-tagging on tokenized text and let the user decide what kind of words he wants to swap
    # and if one of the desired pos-tags isn't available use
    def __init__(self, fraction, column, type="verb_object"):
        ErrorTransformer.__init__(self, fraction, column)
        self.type = type

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
            nlp_english = spacy.load("en_core_web_sm")
            nlp_turkish = spacy.blank("tr")
            nlp_spanish = spacy.blank("es")

            df_copy = data.copy(deep=True)

            # if the type is 'rand'(random), the text of each row gets tokenized and two random tokens will be swapped
            if self.type == "rand":
                for index in self.affected_rows_indices:
                    text = df_copy.at[index, self.column]
                    tokens_doc = ""
                    if df_copy.at[index, "language"] == "en":
                        tokens_doc = nlp_english(text)
                    elif df_copy.at[index, "language"] == "tr":
                        tokens_doc = nlp_turkish(text)
                    else:
                        tokens_doc = nlp_spanish(text)

                    tokens = [token.text for token in tokens_doc]

                    if len(tokens) > 1:
                        first_index = random.randint(0, len(tokens) - 1)
                        second_index = random.randint(0, len(tokens) - 1)

                        first_token = tokens[first_index]
                        tokens[first_index] = tokens[second_index]
                        tokens[second_index] = first_token
                    else:
                        # might want to check for a way that we can pick a different row(or maybe even before we enter the for loop check if len>1)
                        pass

                    unstructured_sentence = " ".join(tokens)
                    df_copy.at[index, self.column] = unstructured_sentence
            # swap verb and object to mimic errors caused by non native speakers (subject-verb-object language and subject-object-verb languages)
            # if verb or object is not present we randomly swap two words
            elif self.type == "verb_object":
                for index in self.affected_rows_indices:
                    text = df_copy.at[index, self.column]
                    if df_copy.at[index, "language"] == "en":
                        result = []
                        sentences = split_into_sentences(text)
                        for sentence in sentences:
                            tokens = nlp_english(sentence)
                            text_ext = textacy.extract.subject_verb_object_triples(
                                tokens
                            )
                            if next(text_ext, -1) == -1:
                                result.append(sentence)
                            for triple in text_ext:
                                swapped_review = []
                                sent_object = triple.object[0].orth_
                                verb = triple.verb[0].orth_
                                swapped_sentence = swap_words(
                                    sentence, verb, sent_object
                                )
                                swapped_review.append(swapped_sentence)
                                result.append(" ".join(swapped_review))
                                corrupted_review = " ".join(result)
                                df_copy.at[index, self.column] = corrupted_review
                """for index in self.affected_rows_indices:
                    text = df_copy.at[index, self.column]
                    tokens_doc = ""
                    elif df_copy.at[index, 'language'] == "tr":
                        tokens_doc = nlp_turkish(text)
                    elif df_copy.at[index, 'language'] == "fr":
                        tokens_doc = nlp_french(text)
                    elif df_copy.at[index, 'language'] == "de":
                        tokens_doc = nlp_german(text)                                   
                    else:
                        tokens_doc = nlp_spanish(text)
                    
                    tokens_pos_tags = [token.pos_ for token in tokens_doc]
                    if 'VERB' in tokens_pos_tags and  in tokens_pos_tags:"""
                # pass

            return df_copy
