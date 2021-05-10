from time import time
from typing import List, Set

import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from data.dataset import split_without_cold_start, RecommendationDataset
from data.encoded_dataset import EncodedRecommendationDataset
from evaluation.evaluation import eval_pointwise, eval_top
from models.ensembling.ensemble_model import EnsembleModel
from models.impl.als import AlsModel
from models.impl.cornac.bivae import BiVAEModel
from models.impl.cornac.bpr import BPRModel
from models.impl.deeprec.lightgcn import LightGCNModel
from models.impl.fastai import FastaiModel
from models.impl.ncf import NCFModel
from models.impl.sar import SarModel
from models.impl.svd import SvdModel

TOP_K = 10
SEED = 42

BASE_MODELS = [
    AlsModel(),
    BiVAEModel(),
    BPRModel(),
    FastaiModel(),
    LightGCNModel(TOP_K),
    NCFModel(),
    SarModel(),
    SvdModel(),
]

class Trainer:
    def __init__(
            self,
            evaluate_top_metrics: bool = True,
            exclude_models: Set[str] = {},
    ):
        self.evaluate_top_metrics = evaluate_top_metrics
        self.exclude_models = exclude_models
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        tf.get_logger().setLevel('ERROR')

    def train(
            self,
            dataset: RecommendationDataset
    ):
        dataset = EncodedRecommendationDataset.of(dataset)
        dataset.print_stats()
        train_hot, valid_hot = split_without_cold_start(dataset, ratio=0.75)

        models = [x for x in BASE_MODELS if x.get_name() not in self.exclude_models]

        ensemble = EnsembleModel(models, filter_unnecessary=False)

        results = []

        for model in models:
            model.on_start()

        for model in models + [ensemble]:
            print(model.get_name())
            t0 = time()
            model.train(train_hot)
            t1 = time()
            pred_scores = model.predict_scores(valid_hot)
            t2 = time()
            pred_top = model.predict_k(train_hot, TOP_K) if self.evaluate_top_metrics else None
            t3 = time()
            results.append({
                **{
                    'name': model.get_name(),
                    'train_time': t1 - t0,
                    'predict_all_time': t2 - t1,
                    'predict_top_time': t3 - t2
                },
                **eval_pointwise(valid_hot, pred_scores),
                **(eval_top(valid_hot, pred_top, TOP_K) if self.evaluate_top_metrics else {}),
            })

        for model in models:
            model.on_stop()

        results_df = pd.DataFrame.from_records(results)
        results_df['ensemble_weight'] = np.zeros(len(results_df))
        for i in range(len(ensemble.models)):
            results_df.loc[results_df.name == ensemble.models[i].get_name(), 'ensemble_weight'] = \
                ensemble.ensemble_model.coef_[i]
        results_df.ensemble_weight = results_df.ensemble_weight / results_df.ensemble_weight.sum()
        results_df.loc[results_df.name == ensemble.get_name(), 'ensemble_weight'] = 1
        results_df.to_csv('results.tsv', sep='\t', index=False)
        print(results_df)
        print("Models selected: ", [x.get_name() for x in ensemble.models])
