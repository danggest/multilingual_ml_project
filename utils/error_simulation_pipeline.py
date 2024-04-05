from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


class ErrorSimulationPipeline:
    def __init__(self, steps, data):
        self.steps = steps
        self.data = data
        self.pipeline = None
        self.predictions = None
        self.corrupted_data = None

    def exec(self):
        self.pipeline = Pipeline(self.steps)
        self.predictions = self.pipeline.fit_transform(self.data)

    def report(self):
        print(classification_report(self.data["stars"], self.predictions))
