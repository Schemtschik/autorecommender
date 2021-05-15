import numpy as np
import pandas as pd

from data.dataset import RecommendationDataset
from trainer import Trainer

if __name__ == "__main__":
    df = pd.read_csv("tmp/kkbox-music-recommendation-challenge/train")[["msno", "song_id", "target"]]
    df = df.sample(frac=0.05)
    df.target = df.target.astype(np.float64)
    df['timestamp'] = np.zeros(len(df)).astype(np.int64)
    dataset = RecommendationDataset(
        user_col="msno",
        item_col="song_id",
        score_col="target",
        timestamp_col="timestamp",
        data=df
    )
    dataset.load()

    trainer = Trainer(
        evaluate_top_metrics=False,
        parallel=False,
        single_model_timeout=600, # s
        ensembling_enabled = False,
        exclude_models={'BiVAE'}
    )
    trainer.train(dataset)