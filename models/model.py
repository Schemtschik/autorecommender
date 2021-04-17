import pandas as pd

from data.dataset import RecommendationDataset


class Model:
    def __init__(self, prediction_col: str = "prediction"):
        self.prediction_col = prediction_col

    def get_name(self) -> str:
        raise NotImplementedError

    def is_top_predicted_by_train(self) -> bool:
        return True

    def on_start(self) -> None:
        pass

    def on_stop(self) -> None:
        pass

    def train(self, dataset: RecommendationDataset) -> None:
        raise NotImplementedError

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        raise NotImplementedError

    def predict_k(self, dataset: RecommendationDataset, k: int) -> pd.DataFrame:
        raise NotImplementedError
