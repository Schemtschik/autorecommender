import pandas as pd
import torch
import numpy as np

from data.dataset import RecommendationDataset
from models.model import Model
import cornac

from recommenders.reco_utils.recommender.cornac.cornac_utils import predict, predict_ranking

class BiVAEModel(Model):
    def __init__(self, factors=50, epochs=500):
        super().__init__()
        self.model = None
        self.factors = factors
        self.epochs = epochs

    def get_name(self) -> str:
        return "BiVAE"

    def train(self, dataset: RecommendationDataset) -> None:
        self.model = cornac.models.BiVAECF(
            k=self.factors,
            encoder_structure=[100],
            act_fn="tanh",
            likelihood="pois",
            n_epochs=self.epochs,
            batch_size=128,
            learning_rate=0.001,
            seed=42,
            use_gpu=torch.cuda.is_available(),
            verbose=True
        )
        self.model.fit(self._wrap_dataset(dataset))

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        result = predict(self.model, dataset.data, dataset.user_col, dataset.item_col, "prediction")
        result.prediction = result.prediction.astype(np.float64)
        return result

    def _wrap_dataset(self, dataset: RecommendationDataset) -> cornac.data.Dataset:
        return cornac.data.Dataset.from_uir(
            dataset.data[[dataset.user_col, dataset.item_col, dataset.score_col]]
                .itertuples(index=False), seed=42
        )