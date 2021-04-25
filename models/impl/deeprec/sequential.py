import os
import tempfile

from tqdm import tqdm
import pandas as pd
import numpy as np
from reco_utils.dataset.amazon_reviews import _create_vocab
from reco_utils.recommender.deeprec.io.sequential_iterator import SequentialIterator

from data.dataset import RecommendationDataset
from models.model import Model

from reco_utils.recommender.deeprec.models.sequential.sli_rec import SLI_RECModel as SeqModel
from reco_utils.recommender.deeprec.deeprec_utils import prepare_hparams


class SequentialModel(Model):
    def __init__(self, epochs=10):
        super().__init__()
        self.model = None
        self.epochs = epochs

    def get_name(self) -> str:
        return "seq"

    def is_binary(self) -> bool:
        return True

    def train(self, dataset: RecommendationDataset) -> None:
        with tempfile.TemporaryDirectory() as dir:
            train_file = os.path.join(dir, "train.tsv")
            user_vocab = os.path.join(dir, "user_vocab.pkl")
            item_vocab = os.path.join(dir, "item_vocab.pkl")
            cate_vocab = os.path.join(dir, "cate_vocab.pkl")
            yaml_file = './recommenders/reco_utils/recommender/deeprec/config/sli_rec.yaml'
            train_num_ngs = 4

            with open(train_file, 'w') as f:
                positive_mask = dataset.data[dataset.score_col] > 0.5
                df_positive, df_negative = dataset.data[positive_mask], dataset.data[~positive_mask]
                j = 0
                for i in tqdm(range(len(df_positive)), desc="preparing data"):
                    self._print_row(f, dataset, df_positive, i)
                    for _ in range(train_num_ngs):
                        self._print_row(f, dataset, df_negative, j)
                        j = (j + 1) % len(df_negative)
                    break

            os.system(f"head {train_file}")

            _create_vocab(train_file, user_vocab, item_vocab, cate_vocab)
            hparams = prepare_hparams(yaml_file,
                                      embed_l2=0.,
                                      layer_l2=0.,
                                      learning_rate=0.001,  # set to 0.01 if batch normalization is disable
                                      epochs=self.epochs,
                                      batch_size=400,
                                      show_step=20,
                                      user_vocab=user_vocab,
                                      item_vocab=item_vocab,
                                      cate_vocab=cate_vocab,
                                      need_sample=False,
                                      train_num_ngs=train_num_ngs,
            )
            input_creator = SequentialIterator
            self.model = SeqModel(hparams, input_creator, seed=42)
            self.model = self.model.fit(train_file, train_file, valid_num_ngs=train_num_ngs)

    # < label > < user_id > < item_id > < category_id > < timestamp > < history_item_ids > < history_cateory_ids > < hitory_timestamp >
    def _print_row(self, f, dataset, df, row_idx):
        f.write(str(df[dataset.score_col].iloc[row_idx]))
        f.write("\t")
        user_id = df[dataset.user_col].iloc[row_idx]
        f.write(str(user_id))
        f.write("\t")
        f.write(str(df[dataset.item_col].iloc[row_idx]))
        f.write("\t")
        f.write(str(df[dataset.category_col].iloc[row_idx]) if dataset.category_col is not None else "default_cat")
        f.write("\t")
        timestamp = df[dataset.timestamp_col].iloc[row_idx] if dataset.timestamp_col is not None else 0
        f.write(str(timestamp))
        f.write("\t")
        history_mask = df[dataset.user_col] == user_id
        if dataset.timestamp_col is not None:
            history_mask = np.logical_and(history_mask, df[dataset.timestamp_col] < timestamp)
        history = df[history_mask]
        if len(history) > 0:
            f.write(",".join([str(x) for x in history[dataset.item_col]]))
            f.write("\t")
            f.write(",".join([str(x) for x in history[dataset.category_col]] if dataset.category_col is not None else ["default_cat"] * len(history)))
            f.write("\t")
            f.write(",".join([str(x) for x in history[dataset.timestamp_col]] if dataset.timestamp_col is not None else ["0"] * len(history)))
        else:
            f.write("0\tdefault_cat\t0")
        f.write("\n")


    def predict_scores(self, dataset: RecommendationDataset) -> pd.DataFrame:
        return self.model.recommend_k_items(dataset.data, top_k=dataset.data[dataset.user_col].nunique(), remove_seen=True)

    def predict_k(self, dataset: RecommendationDataset, k: int) -> pd.DataFrame:
        return self.model.recommend_k_items(dataset.data, top_k=k, remove_seen=True)