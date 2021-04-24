import os

import pandas as pd

from data.dataset import RecommendationDataset
from reco_utils.dataset.amazon_reviews import download_and_extract, data_preprocessing
from reco_utils.dataset.download_utils import maybe_download

class AmazonReviewsDataset(RecommendationDataset):
    def __init__(
            self,
            data_path = 'tmp/amazon',
            user_col: str = 'UserId',
            item_col: str = 'MovieId',
            score_col: str = 'Rating',
            timestamp_col: str = 'Timestamp',
            data: pd.DataFrame = None
    ):
        super().__init__(user_col, item_col, score_col, timestamp_col, data)
        self.data_path = data_path
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.timestamp_col = timestamp_col
        self.data = data

    def load(self) -> None:
        reviews_name = 'reviews_Movies_and_TV_5.json'
        meta_name = 'meta_Movies_and_TV.json'
        reviews_file = os.path.join(self.data_path, reviews_name)
        meta_file = os.path.join(self.data_path, meta_name)
        if not os.path.exists(reviews_file):
            download_and_extract(reviews_name, reviews_file)
        if not os.path.exists(meta_file):
            download_and_extract(meta_name, meta_file)

        self.data = None

    def wrap_data(self, data):
        return AmazonReviewsDataset(
            user_col=self.user_col,
            item_col=self.item_col,
            score_col=self.score_col,
            timestamp_col=self.timestamp_col,
            data=data
        )
