from data.dataset import RecommendationDataset


class EncodedRecommendationDataset(RecommendationDataset):
    def __init__(self, dataset: RecommendationDataset):
        super().__init__(dataset.user_col, dataset.item_col, dataset.score_col, dataset.timestamp_col)
        self.users = list(dataset.data[dataset.user_col].unique())
        self.items = list(dataset.data[dataset.item_col].unique())
        self.users_dict = {self.users[i]: i for i in range(len(self.users))}
        self.items_dict = {self.items[i]: i for i in range(len(self.items))}
        self.data = dataset.data.copy()
        self.data[dataset.user_col] = self.data[dataset.user_col].apply(lambda x : self.users_dict[x])
        self.data[dataset.item_col] = self.data[dataset.item_col].apply(lambda x : self.items_dict[x])

    def of(_dataset: RecommendationDataset):
        return EncodedRecommendationDataset(_dataset)
