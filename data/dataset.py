from typing import Tuple

import numpy as np
import pandas as pd


class RecommendationDataset:
    def __init__(
            self,
            user_col: str,
            item_col: str,
            score_col: str,
            timestamp_col: str = None,
            data: pd.DataFrame = None
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.score_col = score_col
        self.timestamp_col = timestamp_col
        self.data = data

    def print_stats(self):
        print("Pairs: ", len(self.data))
        print("Users: ", self.data[self.user_col].nunique())
        print("Items: ", self.data[self.item_col].nunique())

    def load(self) -> None:
        pass

    def wrap_data(self, data):
        return RecommendationDataset(self.user_col, self.item_col, self.score_col, self.timestamp_col, data)

def split_randomly(
        dataset: RecommendationDataset,
        ratio: float
) -> Tuple[RecommendationDataset, RecommendationDataset]:
    mask = np.random.rand(len(dataset.data)) < ratio
    return dataset.wrap_data(dataset.data[mask]), dataset.wrap_data(dataset.data[~mask])


def split_without_cold_start(
        dataset: RecommendationDataset,
        ratio: float
) -> Tuple[RecommendationDataset, RecommendationDataset]:
    df = dataset.data.copy()
    df = df.sample(n=len(df)).reset_index(drop=True)
    user_pairs = df.groupby(dataset.user_col).first().reset_index()
    item_pairs = df.groupby(dataset.item_col).first().reset_index()
    train_pairs = pd.concat([user_pairs, item_pairs])[[dataset.user_col, dataset.item_col]]
    train_pairs['train_prob'] = 1
    df = pd.merge(df, train_pairs, on=[dataset.user_col, dataset.item_col], how='left')
    df = df.drop_duplicates(subset=[dataset.user_col, dataset.item_col])
    valid_ratio = (1 - ratio) * len(df) / len(df[df.train_prob != 1])
    assert valid_ratio < 1, "Too high validation ratio for 'without cold start' split"
    df.loc[df.train_prob != 1, 'train_prob'] = np.random.rand(len(df[df.train_prob != 1]))
    valid_mask = df.train_prob < valid_ratio
    train, valid = df[~valid_mask], df[valid_mask]
    assert set(train[dataset.user_col].unique()) == set(dataset.data[dataset.user_col].unique())
    assert set(train[dataset.item_col].unique()) == set(dataset.data[dataset.item_col].unique())
    assert len(pd.merge(train, valid, on=[dataset.user_col, dataset.item_col], how='inner')) == 0
    # assert len(train) + len(valid) == len(dataset.data), f"train_len={len(train)}, valid_len={len(valid)}, total_len={len(dataset.data)}"
    return dataset.wrap_data(train), dataset.wrap_data(valid)
