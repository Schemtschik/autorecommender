import pandas as pd
import numpy as np

from data.dataset import RecommendationDataset
from models.model import Model
import cornac

from recommenders.reco_utils.recommender.cornac.cornac_utils import predict, predict_ranking

class BPRModel(Model):
    def __init__(self, factors=200, epochs=100):
        super().__init__()
        self.model = None
        self.factors = factors
        self.epochs = epochs

    def get_name(self) -> str:
        return "BPR"

    def get_params(self):
        return f"factors={self.factors}, epochs={self.epochs}"

    def is_cold_start_appliable(self) -> bool:
        return False

    def train(self, dataset: RecommendationDataset) -> None:
        self.model = cornac.models.BPR(
            k=self.factors,
            max_iter=self.epochs,
            learning_rate=0.01,
            lambda_reg=0.001,
            verbose=True,
            seed=42
        )
        self.model.fit(self._wrap_dataset(dataset))

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        result = predict(self.model, dataset.data, dataset.user_col, dataset.item_col, self.prediction_col)
        result.prediction = result.prediction.astype(np.float64)
        return result

    def _wrap_dataset(self, dataset: RecommendationDataset) -> cornac.data.Dataset:
        return cornac.data.Dataset.from_uir(
            dataset.data[[dataset.user_col, dataset.item_col, dataset.score_col]]
                .itertuples(index=False), seed=42
        )