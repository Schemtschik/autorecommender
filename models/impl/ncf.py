import pandas as pd

from data.dataset import RecommendationDataset
from models.model import Model
from recommenders.reco_utils.recommender.ncf.ncf_singlenode import NCF
from recommenders.reco_utils.recommender.ncf.dataset import Dataset as NCFDataset



class NCFModel(Model):
    def __init__(self, epochs=100):
        super().__init__()
        self.model = None
        self.epochs = epochs

    def get_name(self) -> str:
        return "NCF"

    def train(self, dataset: RecommendationDataset) -> None:
        self.model = NCF(
            n_users=dataset.data[dataset.user_col].nunique(),
            n_items=dataset.data[dataset.item_col].nunique(),
            model_type="NeuMF",
            n_factors=4,
            layer_sizes=[16,8,4],
            n_epochs=self.epochs,
            batch_size=256,
            learning_rate=1e-3,
            verbose=5,
            seed=42
        )
        self.model.fit(self._wrap_dataset(dataset))

    def predict_scores(self, dataset: RecommendationDataset) -> pd:
        prediction = self.model.predict(dataset.data[dataset.user_col], dataset.data[dataset.item_col], True)
        result = dataset.data.copy()
        result[self.prediction_col] = prediction
        return result

    def _wrap_dataset(self, dataset: RecommendationDataset) -> NCFDataset:
        return NCFDataset(
            train=dataset.data,
            col_user=dataset.user_col,
            col_item=dataset.item_col,
            col_rating=dataset.score_col,
            col_timestamp=dataset.timestamp_col
        )