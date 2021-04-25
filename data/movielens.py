import pandas as pd
from reco_utils.dataset import movielens

from data.dataset import RecommendationDataset


class MovielensDataset(RecommendationDataset):
    def __init__(
            self,
            variation: str = '100k',
            user_col: str = 'UserId',
            item_col: str = 'MovieId',
            score_col: str = 'Rating',
            timestamp_col: str = 'Timestamp',
            data: pd.DataFrame = None
    ):
        super().__init__(user_col, item_col, score_col, timestamp_col, None, data)
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

    def wrap_data(self, data):
        return MovielensDataset(
            variation=self.variation,
            user_col=self.user_col,
            item_col=self.item_col,
            score_col=self.score_col,
            timestamp_col=self.timestamp_col,
            data=data
        )
