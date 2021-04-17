import pandas as pd
import surprise

from recommenders.reco_utils.recommender.surprise.surprise_utils import predict, compute_ranking_predictions

from data.dataset import RecommendationDataset
from models.model import Model


class SvdModel(Model):
    def __init__(self, epochs=30):
        super().__init__()
        self.svd = None
        self.epochs = epochs

    def get_name(self) -> str:
        return "SVD"

    def train(self, dataset: RecommendationDataset) -> None:
        train_set = surprise.Dataset.load_from_df(dataset.data[[
            dataset.user_col, dataset.user_col, dataset.score_col
        ]], reader=surprise.Reader()).build_full_trainset()
        self.svd = surprise.SVD(random_state=0, n_factors=200, n_epochs=self.epochs, verbose=True)
        self.svd.fit(train_set)

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        return predict(self.svd, dataset.data, usercol=dataset.user_col, itemcol=dataset.item_col)

    def predict_k(self, dataset: RecommendationDataset, k: int) -> pd.DataFrame:
        return compute_ranking_predictions(self.svd, dataset.data, usercol=dataset.user_col, itemcol=dataset.item_col, remove_seen=True)