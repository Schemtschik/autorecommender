from typing import Tuple
from pandas import DataFrame
import numpy as np

from recommenders.reco_utils.dataset import movielens

class RecommendationDataset:
    def __init__(
            self,
            variation: str = '100k',
            user_col: str = 'UserId',
            item_col: str = 'MovieId',
            score_col: str = 'Rating',
            timestamp_col: str = 'Timestamp',
            data: DataFrame = None
    ):
        self.variation = variation
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.timestamp_col = timestamp_col
        self.data = data

    def load(self) -> None:
        self.data = movielens.load_pandas_df(
            size=self.variation,
            header=[self.user_col, self.item_col, self.score_col, self.timestamp_col]
        )

    def _wrap_data(self, data):
        return RecommendationDataset(
            variation=self.variation,
            user_col=self.user_col,
            item_col=self.item_col,
            score_col=self.score_col,
            timestamp_col=self.timestamp_col,
            data=data
        )


def split_dataset(
        dataset: RecommendationDataset,
        ratio: float
) -> Tuple[RecommendationDataset, RecommendationDataset]:
    mask = np.random.rand(len(dataset.data)) < ratio
    return dataset._wrap_data(dataset.data[mask]), dataset._wrap_data(dataset.data[~mask])
