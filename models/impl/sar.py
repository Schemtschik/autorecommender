import pandas as pd

from data.dataset import RecommendationDataset
from models.model import Model
from recommenders.reco_utils.recommender.sar.sar_singlenode import SARSingleNode


class SarModel(Model):
    def __init__(self):
        super().__init__()
        self.model = None

    def get_name(self) -> str:
        return "SAR"

    def get_params(self):
        return ""

    def train(self, dataset: RecommendationDataset) -> None:
        self.model = SARSingleNode(
            col_user=dataset.user_col,
            col_item=dataset.item_col,
            col_rating=dataset.score_col,
            col_timestamp=dataset.timestamp_col,
            similarity_type="jaccard",
            time_decay_coefficient=30,
            timedecay_formula=True,
            normalize=True
        )
        self.model.fit(dataset.data)

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        return self.model.recommend_k_items(dataset.data, remove_seen=True)