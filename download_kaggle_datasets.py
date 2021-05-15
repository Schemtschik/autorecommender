import os
import kaggle
from pyunpack import Archive

datasets = [
    {
        "competition": "avito-context-ad-clicks",
        "train_file": "trainSearchStream.tsv.7z",
        "test_file": "testSearchStream.tsv.7z",
        "user_column": "",
        "item_column": "",
        "category_column": None,
        "timestamp_column": None,
    },
    {
        "competition": "outbrain-click-prediction",
        "train_file": "clicks_train.csv.zip",
        "test_file": "clicks_test.csv.zip",
        "user_column": "",
        "item_column": "",
        "category_column": None,
        "timestamp_column": None,
    },
    {
        "competition": "avito-context-ad-clicks",
        "train_file": "trainSearchStream.tsv.7z",
        "test_file": "testSearchStream.tsv.7z",
        "user_column": "",
        "item_column": "",
        "category_column": None,
        "timestamp_column": None,
    },
    {
        "competition": "santander-product-recommendation",
        "train_file": "train_ver2.csv.zip",
        "test_file": "test_ver2.csv.zip",
        "user_column": "",
        "item_column": "",
        "category_column": None,
        "timestamp_column": None,
    },
    {
        "competition": "kkbox-music-recommendation-challenge",
        "train_file": "train.csv.7z",
        "test_file": "test.csv.7z",
        "user_column": "",
        "item_column": "",
        "category_column": None,
        "timestamp_column": None,
    },    {
        "competition": "event-recommendation-engine-challenge",
        "train_file": "train.csv",
        "test_file": "test.csv",
        "user_column": "",
        "item_column": "",
        "category_column": None,
        "timestamp_column": None,
    },

]

for dataset in datasets:
    for i in ["train_file", "test_file"]:
        file_path = "./tmp/" + dataset["competition"] + "/" + dataset[i]
        if not os.path.exists(file_path):
            kaggle.api.competition_download_cli(
                competition=dataset["competition"],
                file_name=dataset[i],
                path="./tmp/" + dataset["competition"]
        )
        result_path = "./tmp/" + dataset["competition"] + "/" + i.split("_")[0]
        if not os.path.exists(result_path):
            if file_path.endswith(".zip") or file_path.endswith(".7z"):
                Archive(file_path).extractall("./tmp/" + dataset["competition"])
            os.rename(file_path.replace(".zip", "").replace(".7z", ""), result_path)