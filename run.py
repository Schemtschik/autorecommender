from time import time

import numpy as np
import pandas as pd

from data.dataset import RecommendationDataset, split_without_cold_start
from evaluation.evaluation import eval_pointwise, eval_top
from models.model_als import AlsModel
from models.model_bivae import BiVAEModel
from models.model_sar import SarModel
from models.model_svd import SvdModel
from models.model_fastai import FastaiModel
from models.model_ncf import NCFModel
from models.model_bpr import BPRModel

TOP_K = 10
SEED = 42

np.random.seed(SEED)

dataset = RecommendationDataset()
dataset.load()
train_hot, valid_hot = split_without_cold_start(dataset, ratio=0.75)

models = [
    AlsModel(),
    SvdModel(),
    SarModel(),
    FastaiModel(),
    NCFModel(),
    BPRModel(),
    BiVAEModel(),
]

results = []

for model in models:
    print(model.get_name())
    model.on_start()
    t0 = time()
    model.train(train_hot)
    t1 = time()
    pred_top = model.predict_k(valid_hot, TOP_K)
    t2 = time()
    pred_scores = model.predict_scores(valid_hot)
    t3 = time()
    results.append({
        **{
            'name': model.get_name(),
            'train_time': t1 - t0,
            'predict_top_time': t2 - t1,
            'predict_all_time': t3 - t2
        },
        **eval_pointwise(valid_hot, pred_scores),
        **eval_top(valid_hot, pred_top, TOP_K),
    })
    model.on_stop()

print(pd.DataFrame.from_records(results))