import math
import random
import string
import re
from .error_transformer import ErrorTransformer


class TypoTransformer(ErrorTransformer):
    def __init__(self, fraction, column, fraction_typos, typo_mode="random"):
        ErrorTransformer.__init__(self, fraction, column)
        self.fraction_typos = fraction_typos
        self.typo_mode = typo_mode
        self.corrupted_data = None

    def fit(self, data, y=None):
        # set the length of the dataset and calculate the number of corruptions that need to be done
        num_rows = len(data)
        self.num_affected_rows = int(num_rows * self.fraction)

        # randomly select indices of the rows that are going to be transformed
        data_indices = list(data.index)
        self.affected_rows_indices = random.sample(data_indices, self.num_affected_rows)

        self.is_fitted_ = True
        return self

    def __random_char_language_based(self, language):
        # returns a random character, the list(or alphabet) from which this character is chosen is language dependent
        random_char = ""
        if language == "tr":
            turkish_with_ascii = "çÇğĞıİöÖşŞüÜ" + string.ascii_letters
            turkish_chars = [
                char for char in turkish_with_ascii if char not in "qwxQWX"
            ]
            random_char = random.choice(turkish_chars)
        elif language == "es":
            spanish_chars = "ÁáÉéÍíÓóÚúÑñÜü" + string.ascii_letters
            random_char = random.choice(spanish_chars)
        else:
            random_char = random.choice(string.ascii_letters)

        return random_char

    def transform(self, data):
        # todos:
        # maybe detect language so that it can be used for other textual datasets that dont contain the language column(not necessary if we split on whitespace)
        # maybe for the adding and changing add map with for each character the characters that are close to it(mimic proximity)
        if self.is_fitted_:
            # create deep copy of the original dataset to ensure the original is untouched
            df_copy = data.copy(deep=True)

            # specify the different typo categories
            typo_cat = ["missing", "adding", "changing"]

            # for each row that should be affected by typos we tokenize the text
            for index in self.affected_rows_indices:
                text = df_copy.at[index, self.column]
                language = df_copy.at[index, "language"]

                # use a regular expression to split the text in tokens. Tokens are either words or punctuation. The apostrophe is treated as
                # an alphanumeric character so a word like "doesn't" is one token. This is desirable for reconstructing the sentence
                tokens = re.findall(r"\'*\w+\'*\w*|[^\w\s]", text)

                # find the indices of punctuation in tokens and the indices of the tokens representing words
                punctuation_indices = [
                    index
                    for index, token in enumerate(tokens)
                    if re.match(r"^[^\w\s]$", token)
                ]
                possible_typo_indices = [
                    index
                    for index in range(len(tokens))
                    if index not in punctuation_indices
                ]

                # we calculate the number of words that should have typos (rounded up) and randomly decide which words should
                # contain the typos
                num_typos = math.ceil(len(possible_typo_indices) * self.fraction_typos)
                typo_tokens_indices = random.sample(possible_typo_indices, num_typos)

                # for each word that should contain a typo we check what kind of typo the word should get. If typo_mode is random
                # we randomly select a typo category for each word
                for typo_index in typo_tokens_indices:
                    if self.typo_mode == "random":
                        type_typo = random.choice(typo_cat)
                    else:
                        type_typo = self.typo_mode

                    correct_word = tokens[typo_index]
                    length_correct_word = len(correct_word)
                    # for the missing typo we randomly remove a character from the word
                    if type_typo == "missing":
                        char_index = random.randint(0, length_correct_word - 1)
                        typo_word = (
                            correct_word[:char_index] + correct_word[char_index + 1 :]
                        )
                        tokens[typo_index] = typo_word
                    # for the adding typo we randomly add a character to the word. the character is randomly generated from a language specific list.
                    # this list isn't 100% accurate since some of the characters can also be found in english but this is highly unlikely for reviews
                    elif type_typo == "adding":
                        char_index = random.randint(-1, length_correct_word)
                        random_char = self.__random_char_language_based(language)
                        typo_word = []
                        if char_index == -1:
                            typo_word = random_char + correct_word
                        elif char_index == length_correct_word:
                            typo_word = correct_word + random_char
                        else:
                            typo_word = (
                                correct_word[:char_index]
                                + random_char
                                + correct_word[char_index:]
                            )
                        tokens[typo_index] = typo_word
                    # for the changing typo we randomly change a character of the word into a randomly generated character also language based
                    else:
                        char_index = random.randint(0, length_correct_word - 1)
                        random_char = self.__random_char_language_based(language)
                        typo_word = (
                            correct_word[:char_index]
                            + random_char
                            + correct_word[char_index + 1 :]
                        )
                        tokens[typo_index] = typo_word

                # reconstruct the text with the typos and set it in the copy of the dataframe. We reconstruct the text by adding a space
                # between non-punctuation tokens. For punctuation tokens there will only be a space after. This is not guaranteed to restore
                # the exact structure but in the most cases it will
                text_with_typos = ""
                for token_index, token in enumerate(tokens):
                    if token_index in punctuation_indices:
                        text_with_typos = text_with_typos + token
                    else:
                        text_with_typos = text_with_typos + " " + token
                df_copy.at[index, self.column] = text_with_typos
            print("TypoTransformer finished")
            self.corrupted_data = df_copy
            return df_copy
