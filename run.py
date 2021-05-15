from data.impl.movielens import MovielensDataset
from trainer import Trainer

if __name__ == "__main__":
    dataset = MovielensDataset()
    dataset.load()

    trainer = Trainer(
        exclude_models={"NCF"},
        parallel=False,
        single_model_timeout=60, # s
    )
    trainer.train(dataset)
