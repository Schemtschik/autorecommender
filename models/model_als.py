import pandas as pd
from pyspark.ml.recommendation import ALS
import pyspark

from data.dataset import RecommendationDataset
from models.model import Model
from recommenders.reco_utils.common.spark_utils import start_or_get_spark


class AlsModel(Model):
    def __init__(self, top_size: int = 10, max_iter: int = 15, seed: int = 42):
        super().__init__()
        self.top_size = top_size
        self.max_iter = max_iter
        self.seed = seed

        self.spark_session = None
        self.model = None

    def on_start(self) -> None:
        self.spark_session = start_or_get_spark("ALS PySpark", memory="16g")

    def on_stop(self) -> None:
        self.spark_session.stop()

    def get_name(self) -> str:
        return "ALS"

    def _to_spark(self, df: pd.DataFrame) -> pyspark.sql.DataFrame:
        return self.spark_session.createDataFrame(df)

    def train(self, train_ds: RecommendationDataset) -> None:
        self.model = ALS(
            rank=self.top_size,
            maxIter=self.max_iter,
            implicitPrefs=False,
            regParam=0.05,
            coldStartStrategy='drop',
            nonnegative=False,
            seed=self.seed,
            userCol=train_ds.user_col,
            itemCol=train_ds.item_col,
            ratingCol=train_ds.score_col
        ).fit(self._to_spark(train_ds.data))

    def predict_scores(self, test_ds: RecommendationDataset) -> pd.DataFrame:
        return self.model.transform(self._to_spark(test_ds.data)).cache().toPandas()

    def predict_k(self, train_ds: RecommendationDataset, k: int) -> pd.DataFrame:
        data = self._to_spark(train_ds.data)
        users = data.select(train_ds.user_col).distinct()
        items = data.select(train_ds.item_col).distinct()
        user_item = users.crossJoin(items)
        dfs_pred = self.model.transform(user_item)
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            data.alias("train"),
            (dfs_pred[train_ds.user_col] == data[train_ds.user_col])
                & (dfs_pred[train_ds.item_col] == data[train_ds.item_col]),
            how='outer'
        )
        top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull()) \
            .select('pred.' + train_ds.user_col, 'pred.' + train_ds.item_col, 'pred.' + self.prediction_col)
        top_all.cache().count()
        return top_all.toPandas()
