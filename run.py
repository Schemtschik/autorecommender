import numpy as np
import pandas as pd

from time import time

import torch

from data.dataset import RecommendationDataset, split_dataset
from evaluation.evaluation import eval_pointwise, eval_top
from models.model_sar import SarModel
from models.model_svd import SvdModel
from models.model_als import AlsModel
from models.model_fastai import FastaiModel

TOP_K = 10
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

dataset = RecommendationDataset()
dataset.load()
train, valid = split_dataset(dataset, ratio=0.75)

models = [
    # AlsModel(),
    # SvdModel(),
    # SarModel(),
    FastaiModel(epochs=1),
]

results = []

for model in models:
    print(model.get_name())
    valid_for_top = train if model.is_top_predicted_by_train() else valid
    model.on_start()
    t0 = time()
    model.train(train)
    t1 = time()
    pred_top = model.predict_k(valid_for_top, TOP_K)
    t2 = time()
    pred_scores = model.predict_scores(valid)
    t3 = time()
    results.append({
        **{
            'name': model.get_name(),
            'train_time': t1 - t0,
            'predict_top_time': t2 - t1,
            'predict_all_time': t3 - t2
        },
        **eval_pointwise(valid, pred_scores),
        **eval_top(valid, pred_top, TOP_K),
    })
    model.on_stop()

print(pd.DataFrame.from_records(results))