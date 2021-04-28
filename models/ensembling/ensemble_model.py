import pandas as pd

from data.dataset import RecommendationDataset
from models.model import Model
from sklearn.linear_model import Lasso, LinearRegression

class EnsembleModel(Model):
    def __init__(self, models, filter_unnecessary=True):
        super().__init__()
        self.models = models
        self.ensemble_model = Lasso(normalize=True) if filter_unnecessary else LinearRegression(normalize=True)
        self.filter_unnecessary = filter_unnecessary

    def get_name(self) -> str:
        return "Ensemble"

    def train(self, dataset: RecommendationDataset) -> None:
        self._train_once(dataset)
        if self.filter_unnecessary:
            new_models = []
            for i in range(len(self.models)):
                if self.ensemble_model.coef_[i] > 1e4:
                    new_models.append(self.models[i])
            self.models = new_models
            self._train_once(dataset)

    def _train_once(self, dataset: RecommendationDataset) -> None:
        df = dataset.data[[dataset.user_col, dataset.item_col, dataset.score_col]].copy()
        for model in self.models:
            scores = model.predict_scores(dataset)
            df[model.get_name()] = scores[model.prediction_col]
        df = df.fillna(0.)
        self.ensemble_model.fit(df[[model.get_name() for model in self.models]], df[dataset.score_col])

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        df = dataset.data[[dataset.user_col, dataset.item_col]].copy()
        for model in self.models:
            scores = model.predict_scores(dataset)
            df[model.get_name()] = scores[model.prediction_col]
        df = df.fillna(0.)
        df[self.prediction_col] = self.ensemble_model.predict(df[[model.get_name() for model in self.models]])
        return df[[dataset.user_col, dataset.item_col, self.prediction_col]]