import numpy as np
import pandas as pd

from data.dataset import RecommendationDataset
from trainer import Trainer

df = pd.read_csv("tmp/event-recommendation-engine-challenge/train")[["user", "event", "interested", "timestamp"]]
df.interested = df.interested.astype(np.float64)
df.timestamp = np.zeros(len(df.timestamp))
df.timestamp = df.timestamp.astype(np.int64)
dataset = RecommendationDataset(
    user_col="user",
    item_col="event",
    score_col="interested",
    timestamp_col="timestamp",
    data=df
)
dataset.load()

trainer = Trainer()
trainer.train(dataset)