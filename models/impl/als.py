import pandas as pd
from pyspark.ml.recommendation import ALS
import pyspark

from data.dataset import RecommendationDataset
from models.model import Model
from recommenders.reco_utils.common.spark_utils import start_or_get_spark


class AlsModel(Model):
    def __init__(self, top_size: int = 10, epochs: int = 15, seed: int = 42):
        super().__init__()
        self.top_size = top_size
        self.epochs = epochs
        self.seed = seed

        self.spark_session = None
        self.model = None

    def on_start(self) -> None:
        self.spark_session = start_or_get_spark("ALS PySpark", memory="16g")
        self.spark_session.sparkContext.setLogLevel("ERROR")

    def on_stop(self) -> None:
        self.spark_session.stop()

    def get_name(self) -> str:
        return "ALS"

    def get_params(self):
        return f"epochs={self.epochs}"

    def _to_spark(self, df: pd.DataFrame) -> pyspark.sql.DataFrame:
        return self.spark_session.createDataFrame(df)

    def train(self, train_ds: RecommendationDataset) -> None:
        self.model = ALS(
            rank=self.top_size,
            maxIter=self.epochs,
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
