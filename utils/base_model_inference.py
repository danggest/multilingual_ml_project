import sparknlp
import pandas as pd
import pyspark
from sparknlp.base import *
from sparknlp.annotator import *


class BaseModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.spark = sparknlp.start(gpu=True, apple_silicon=True)
        self.is_fitted = False

    def fit(self, data, y=None):
        print("BaseModelInference fit started")
        if self.is_fitted == False:
            document_assembler = (
                DocumentAssembler().setInputCol("review_body").setOutputCol("document")
            )

            tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

            normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalized")

            stopwords_cleaner = (
                StopWordsCleaner()
                .setInputCols("normalized")
                .setOutputCol("cleanTokens")
                .setCaseSensitive(False)
            )

            lemma = (
                LemmatizerModel.pretrained("lemma_antbnc")
                .setInputCols(["cleanTokens"])
                .setOutputCol("lemma")
            )

            glove_embeddings = (
                WordEmbeddingsModel()
                .pretrained()
                .setInputCols(["document", "lemma"])
                .setOutputCol("embeddings")
                .setCaseSensitive(False)
            )

            embeddingsSentence = (
                SentenceEmbeddings()
                .setInputCols(["document", "embeddings"])
                .setOutputCol("sentence_embeddings")
                .setPoolingStrategy("AVERAGE")
            )

            classsifierdl = (
                ClassifierDLModel.load(self.model_path)
                .setInputCols(["sentence_embeddings"])
                .setOutputCol("class")
            )

            self.pipeline = pyspark.ml.Pipeline(
                stages=[
                    document_assembler,
                    tokenizer,
                    normalizer,
                    stopwords_cleaner,
                    lemma,
                    glove_embeddings,
                    embeddingsSentence,
                    classsifierdl,
                ]
            )
            self.is_fitted = True
        return self

    def transform(self, dataset):
        print("BaseModelInference transform started")
        if self.is_fitted == True:
            # if pandas dataframe convert to spark dataframe
            if isinstance(dataset, pd.DataFrame):
                dataset = self.spark.createDataFrame(dataset)

            preds = self.pipeline.fit(dataset).transform(dataset)
            preds_df = preds.select("stars", "review_body", "class.result").toPandas()
            preds_df["result"] = preds_df["result"].apply(lambda x: int(x[0]))

            return preds_df["result"]
        else:
            return None
