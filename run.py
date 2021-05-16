from data.impl.movielens import MovielensDataset
from trainer import Trainer

if __name__ == "__main__":
    dataset = MovielensDataset()
    dataset.load()

    trainer = Trainer(
        exclude_models={"NCF"},
        parallel=False,
        single_model_timeout=600, # s
        train_without_cold_start=False,
    )
    trainer.train(dataset)
