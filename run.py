from data.impl.movielens import MovielensDataset
from trainer import Trainer

dataset = MovielensDataset()
dataset.load()

trainer = Trainer()
trainer.train(dataset)