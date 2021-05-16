import concurrent
import os
from time import time
from typing import Set

from concurrent.futures import ThreadPoolExecutor

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
    BiVAEModel(factors=20, epochs=10),
    BiVAEModel(factors=20),
    BiVAEModel(epochs=10),
    BiVAEModel(),
    BPRModel(factors=64),
    BPRModel(factors=64, epochs=10),
    BPRModel(),
    BPRModel(epochs=10),
    FastaiModel(epochs=1),
    FastaiModel(),
    LightGCNModel(TOP_K, epochs=10),
    LightGCNModel(TOP_K),
    NCFModel(epochs=1),
    NCFModel(epochs=10),
    NCFModel(),
    SarModel(),
    SvdModel(epochs=10),
    SvdModel(),
]


class Trainer:
    def __init__(
            self,
            evaluate_top_metrics: bool = True,
            exclude_models: Set[str] = {},
            parallel: bool = True,
            single_model_timeout: float = None,
            ensembling_enabled: bool = True,
    ):
        self.evaluate_top_metrics = evaluate_top_metrics
        self.exclude_models = exclude_models
        self.parallel = parallel
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() if parallel else 1)
        self.single_model_timeout = single_model_timeout
        self.ensembling_enabled = ensembling_enabled
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

        for model in models:
            model.on_start()

        futures = [(
            model,
            self.executor.submit(self._train_model, model, train_hot, valid_hot) if self.parallel else None
        ) for model in models]
        results = []
        models_trained = []
        errors = []
        for model, future in futures:
            if not self.parallel:
                future = self.executor.submit(self._train_model, model, train_hot, valid_hot)
            try:
                result = future.result(timeout=self.single_model_timeout)
                results.append(result)
                models_trained.append(model)
            except Exception as e:
                if type(e) == concurrent.futures._base.TimeoutError:
                    if not future.done():
                        future.cancel()
                    errors.append((model.get_name_with_params(), 'Timeout'))
                    print(f"Model {model.get_name_with_params()} has timed out")
                else:
                    print(f"Model {model.get_name_with_params()} failed")
                    errors.append((model.get_name_with_params(), 'Failed'))

        if self.ensembling_enabled:
            ensemble = EnsembleModel(models_trained, filter_unnecessary=False)
            results.append(self._train_model(ensemble, train_hot, valid_hot))

        for model in models:
            model.on_stop()

        results_df = pd.DataFrame.from_records(results)
        if self.ensembling_enabled:
            results_df['ensemble_weight'] = np.zeros(len(results_df))
            for i in range(len(ensemble.models)):
                results_df.loc[results_df.name == ensemble.models[i].get_name_with_params(), 'ensemble_weight'] = \
                    ensemble.ensemble_model.coef_[i]
            results_df.ensemble_weight = results_df.ensemble_weight / results_df.ensemble_weight.sum()
            results_df.loc[results_df.name == ensemble.get_name(), 'ensemble_weight'] = 1
        results_df.to_csv('results.tsv', sep='\t', index=False)
        print("Errors: ", errors)
        print(results_df)

    def _train_model(self, model, train_hot, valid_hot):
        print(model.get_name_with_params() + ": start")
        t0 = time()
        model.train(train_hot)
        t1 = time()
        pred_scores = model.predict_scores(valid_hot)
        t2 = time()
        pred_top = model.predict_k(train_hot, TOP_K) if self.evaluate_top_metrics else None
        t3 = time()
        result = {
            **{
                'name': model.get_name_with_params(),
                'train_time': t1 - t0,
                'predict_all_time': t2 - t1,
                'predict_top_time': t3 - t2
            },
            **eval_pointwise(valid_hot, pred_scores),
            **(eval_top(valid_hot, pred_top, TOP_K) if self.evaluate_top_metrics else {}),
        }
        print(model.get_name_with_params() + ": finish")
        return result
