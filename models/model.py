import pandas as pd
from reco_utils.recommender.fastai.fastai_utils import cartesian_product

from data.dataset import RecommendationDataset


class Model:
    def __init__(self, prediction_col: str = "prediction"):
        self.prediction_col = prediction_col

    def get_name(self) -> str:
        raise NotImplementedError

    def is_cold_start_appliable(self) -> bool:
        # TODO: override
        return False

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def train(self, dataset: RecommendationDataset) -> None:
        raise NotImplementedError

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        raise NotImplementedError

    def predict_k(self, dataset: RecommendationDataset, k: int) -> pd.DataFrame:
        total_users = dataset.data[dataset.user_col].unique()
        total_items = dataset.data[dataset.item_col].unique()

        users_items = cartesian_product(total_users, total_items)
        users_items = pd.DataFrame(users_items, columns=[dataset.user_col, dataset.item_col])

        training_removed = pd.merge(users_items, dataset.data,
                                    on=[dataset.user_col, dataset.item_col], how='left')
        training_removed = training_removed[training_removed[dataset.score_col].isna()][[dataset.user_col, dataset.item_col]]
        return self.predict_scores(dataset._wrap_data(training_removed))


