import pandas as pd

from data.dataset import RecommendationDataset
from models.model import Model

from reco_utils.recommender.deeprec.models.graphrec.lightgcn import LightGCN
from reco_utils.recommender.deeprec.DataModel.ImplicitCF import ImplicitCF
from reco_utils.recommender.deeprec.deeprec_utils import prepare_hparams


class LightGCNModel(Model):
    def __init__(self, top_size, epochs=50):
        super().__init__()
        self.model = None
        self.top_size = top_size
        self.epochs = epochs

    def get_name(self) -> str:
        return "LightGCN"

    def train(self, dataset: RecommendationDataset) -> None:
        hparams = prepare_hparams("./recommenders/reco_utils/recommender/deeprec/config/lightgcn.yaml",
                                  n_layers=3,
                                  batch_size=1024,
                                  epochs=self.epochs,
                                  learning_rate=0.005,
                                  top_k=self.top_size,
        )
        self.model = LightGCN(hparams, self._wrap_dataset(dataset), seed=42)
        self.model.fit()

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        return self.model.recommend_k_items(dataset.data, top_k=dataset.data[dataset.user_col].nunique(), remove_seen=True)

    def predict_k(self, dataset: RecommendationDataset, k: int) -> pd.DataFrame:
        return self.model.recommend_k_items(dataset.data, top_k=k, remove_seen=True)

    def _wrap_dataset(self, dataset: RecommendationDataset) -> ImplicitCF:
        return ImplicitCF(
            train=dataset.data, seed=42, col_user=dataset.user_col, col_item=dataset.item_col,
            col_rating=dataset.score_col, col_prediction="prediction"
        )