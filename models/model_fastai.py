import pandas as pd
import numpy as np
import torch
from fastai.collab import CollabDataLoaders, collab_learner

from recommenders.reco_utils.recommender.fastai.fastai_utils import cartesian_product

from data.dataset import RecommendationDataset
from models.model import Model

N_FACTORS = 40


class FastaiModel(Model):
    def __init__(self, epochs=5):
        super().__init__()
        self.epochs = epochs
        self.data_pd = None
        self.data = None
        self.learner = None

    def get_name(self) -> str:
        return "FastAI"

    def is_top_predicted_by_train(self) -> bool:
        return False

    def train(self, dataset: RecommendationDataset) -> None:
        self.data_pd = dataset.data
        self.data = CollabDataLoaders.from_df(
            dataset.data,
            user_name=dataset.user_col,
            item_name=dataset.item_col,
            rating_name=dataset.score_col,
            valid_pct=0)
        self.learner = collab_learner(self.data, n_factors=N_FACTORS, y_range=[0, 5.5], wd=1e-1)
        self.learner.fit_one_cycle(self.epochs)

    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        return score(self.learner,
                     self.data,
                     test_df=dataset.data.copy(),
                     user_col=dataset.user_col,
                     item_col=dataset.item_col,
                     prediction_col="prediction")

    def predict_k(self, dataset: RecommendationDataset, k: int) -> pd.DataFrame:
        total_users, total_items = self.data.classes.values()
        total_items = total_items[1:]
        total_users = total_users[1:]

        test_users = dataset.data[dataset.user_col].unique()
        test_users = np.intersect1d(test_users, total_users)

        users_items = cartesian_product(np.array(test_users), np.array(total_items))
        users_items = pd.DataFrame(users_items, columns=[dataset.user_col, dataset.item_col])

        training_removed = pd.merge(users_items, self.data_pd,
                                    on=[dataset.user_col, dataset.item_col], how='left')
        training_removed = training_removed[training_removed[dataset.score_col].isna()][[dataset.user_col, dataset.item_col]]

        return score(self.learner,
                     self.data,
                     test_df=training_removed,
                     user_col=dataset.user_col,
                     item_col=dataset.item_col,
                     prediction_col=self.prediction_col)


def score(
        learner,
        data,
        test_df,
        user_col,
        item_col,
        prediction_col,
        top_k=None,
):
    """Score all users+items provided and reduce to top_k items per user if top_k>0

    Args:
        learner (obj): Model.
        test_df (pd.DataFrame): Test dataframe.
        user_col (str): User column name.
        item_col (str): Item column name.
        prediction_col (str): Prediction column name.
        top_k (int): Number of top items to recommend.

    Returns:
        pd.DataFrame: Result of recommendation
    """
    # replace values not known to the model with NaN
    total_users, total_items = data.classes.values()
    test_df.loc[~test_df[user_col].isin(total_users), user_col] = np.nan
    test_df.loc[~test_df[item_col].isin(total_items), item_col] = np.nan

    # map ids to embedding ids
    u = learner._get_idx(test_df[user_col], is_item=False)
    m = learner._get_idx(test_df[item_col], is_item=True)

    um = torch.tensor([[u[i], m[i]] for i in range(len(u))])

    pred = learner.model.forward(um).detach().numpy()
    scores = pd.DataFrame(
        {user_col: test_df[user_col].astype(np.int64), item_col: test_df[item_col].astype(np.int64), prediction_col: pred}
    )
    scores = scores.sort_values([user_col, prediction_col], ascending=[True, False])
    if top_k is not None:
        top_scores = scores.groupby(user_col).head(top_k).reset_index(drop=True)
    else:
        top_scores = scores
    return top_scores
