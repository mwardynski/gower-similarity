import numpy as np
import pandas as pd

from utils import DataType


class HuangDistance:
    def __init__(
        self,
        dtypes: np.ndarray,
        gamma: np.float64 = 0.5
    ):
        self.dtypes = dtypes
        self.n_features_in_ = len(dtypes)

        self.gamma = gamma

        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL) | (
                self.dtypes == DataType.BINARY_SYMMETRIC
        )
        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE

        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)

    def fit(self, X: np.ndarray):
        pass

    def __call__(self, vector_1: np.ndarray, vector_2: np.ndarray):
        assert self.n_features_in_ == len(vector_1)
        assert self.n_features_in_ == len(vector_2)

        if self.cat_nom_num > 0:
            cat_nom_cols_1 = vector_1[self.cat_nom_idx]
            cat_nom_cols_2 = vector_2[self.cat_nom_idx]

            cat_nom_dist = 1.0 - (cat_nom_cols_1 == cat_nom_cols_2)
        else:
            cat_nom_dist = 0.0

        if self.ratio_scale_num > 0:
            ratio_scale_cols_1 = vector_1[self.ratio_scale_idx]
            ratio_scale_cols_2 = vector_2[self.ratio_scale_idx]

            ratio_scale_dist = np.linalg.norm(ratio_scale_cols_1 - ratio_scale_cols_2)
        else:
            ratio_scale_dist = 0.0

        return ratio_scale_dist + self.gamma * cat_nom_dist


class HEOM:
    def __init__(
        self,
        dtypes: np.ndarray,
    ):
        self.dtypes = dtypes

        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL) | (
                self.dtypes == DataType.BINARY_SYMMETRIC
        )
        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE

        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)

    def fit(self, X: np.ndarray):
        assert X.shape[1] == len(self.dtypes)

        self.n_features_in_ = len(self.dtypes)

        if self.ratio_scale_num > 0:
            ratio_cols = X[:, self.ratio_scale_idx]

            self.ranges_ = np.ptp(ratio_cols, axis=0)

    def __call__(self, vector_1: np.ndarray, vector_2: np.ndarray):
        assert self.n_features_in_ == len(vector_1)
        assert self.n_features_in_ == len(vector_2)

        if self.cat_nom_num > 0:
            cat_nom_cols_1 = vector_1[self.cat_nom_idx]
            cat_nom_cols_2 = vector_2[self.cat_nom_idx]

            cat_nom_dist = ...      # TODO: implement
        else:
            cat_nom_dist = 0.0

        if self.ratio_scale_num > 0:
            ratio_scale_cols_1 = vector_1[self.ratio_scale_idx]
            ratio_scale_cols_2 = vector_2[self.ratio_scale_idx]

            ratio_scale_dist = np.sum(np.abs(ratio_scale_cols_1 - ratio_scale_cols_2) / self.ranges_)
        else:
            ratio_scale_dist = 0.0

        return np.power(ratio_scale_dist + cat_nom_dist, 2)


class HVDM:
    def __init__(
        self,
        dtypes: np.ndarray,
    ):
        self.dtypes = dtypes

        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL) | (
                self.dtypes == DataType.BINARY_SYMMETRIC
        )
        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE

        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)

    def fit(self, X: np.ndarray):
        assert X.shape[1] == len(self.dtypes)

        self.n_features_in_ = len(self.dtypes)

        if self.ratio_scale_num > 0:
            ratio_cols = X[:, self.ratio_scale_idx]

            self.std_ = 4 * np.std(ratio_cols, axis=0)

    def __call__(self, vector_1: np.ndarray, vector_2: np.ndarray):
        assert self.n_features_in_ == len(vector_1)
        assert self.n_features_in_ == len(vector_2)

        if self.cat_nom_num > 0:
            cat_nom_cols_1 = vector_1[self.cat_nom_idx]
            cat_nom_cols_2 = vector_2[self.cat_nom_idx]

            cat_nom_dist = ...      # TODO: implement
        else:
            cat_nom_dist = 0.0

        if self.ratio_scale_num > 0:
            ratio_scale_cols_1 = vector_1[self.ratio_scale_idx]
            ratio_scale_cols_2 = vector_2[self.ratio_scale_idx]

            ratio_scale_dist = np.sum(np.power(np.abs(ratio_scale_cols_1 - ratio_scale_cols_2) / self.std_, 2))
        else:
            ratio_scale_dist = 0.0

        return np.sqrt(ratio_scale_dist + cat_nom_dist)


class Podani:
    def __init__(
        self,
        dtypes: np.ndarray
    ):
        self.dtypes = dtypes

        self.bin_idx = self.dtypes == DataType.BINARY_SYMMETRIC
        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL)
        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE
        self.ordinal_idx = self.dtypes == DataType.CATEGORICAL_ORDINAL

        self.bin_num = np.sum(self.bin_idx)
        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)
        self.ordinal_num = np.sum(self.ordinal_idx)

    def fit(self, X: np.ndarray):
        assert X.shape[1] == len(self.dtypes)

        self.n_features_in_ = len(self.dtypes)

        if self.ratio_scale_num > 0:
            ratio_cols = X[:, self.ratio_scale_idx]

            self.ranges_ = np.ptp(ratio_cols, axis=0)

        if self.ordinal_num > 0:
            ordinal_cols = X[:, self.ordinal_idx]

            self.rank_mappings = self.collect_rank_mappings(ordinal_cols)
            self.ordinal_cardinalities = self.collect_ordinal_cardinalities(ordinal_cols)

            self.min_ranks_ = np.array([sublist[0] for sublist in self.rank_mappings])
            self.max_ranks_ = np.array([sublist[-1] for sublist in self.rank_mappings])
            

    def collect_rank_mappings(self, data):
        rank_mappings = []

        ranks = pd.DataFrame(data).rank(method="average", axis=0).to_numpy()
        for i in range(data.shape[1]):
            unique_ranks = np.sort(np.unique(ranks[:, i]))
            rank_mappings.append(unique_ranks)

        return rank_mappings

    def collect_ordinal_cardinalities(self, data):
        ordinals_cardinality = []

        for i in range(data.shape[1]):
            unique_elements, counts = np.unique(data[:, i], return_counts=True)
            occurrences_sorted = np.array(sorted(dict(zip(unique_elements, counts)).items()))
            ordinals_cardinality.append(occurrences_sorted[:, 1])

        return ordinals_cardinality

    def __call__(self, vector_1: np.ndarray, vector_2: np.ndarray):
        assert self.n_features_in_ == len(vector_1)
        assert self.n_features_in_ == len(vector_2)

        if self.bin_num > 0:
            bin_cols_1 = vector_1[self.bin_idx]
            bin_cols_2 = vector_2[self.bin_idx]

            bin_dist = np.sum(np.power(bin_cols_1 - bin_cols_2, 2))
        else:
            bin_dist = 0.0

        if self.cat_nom_num > 0:
            cat_nom_cols_1 = vector_1[self.cat_nom_idx]
            cat_nom_cols_2 = vector_2[self.cat_nom_idx]

            #np.power(cat_nom_val, 2) not needed, because can_nom_val \in {0, 1}
            cat_nom_dist = np.sum(np.power(1.0 - (cat_nom_cols_1 == cat_nom_cols_2), 2))

        else:
            cat_nom_dist = 0.0

        if self.ratio_scale_num > 0:
            ratio_scale_cols_1 = vector_1[self.ratio_scale_idx]
            ratio_scale_cols_2 = vector_2[self.ratio_scale_idx]

            ratio_scale_dist = np.sum(np.power((ratio_scale_cols_1 - ratio_scale_cols_2) / self.ranges_, 2))
        else:
            ratio_scale_dist = 0.0

        if self.ordinal_num > 0:
            ordinal_cols_1 = vector_1[self.ordinal_idx]
            ordinal_cols_2 = vector_2[self.ordinal_idx]
            
            rank_col_1 = np.zeros(ordinal_cols_1.shape[0])
            rank_col_2 = np.zeros(ordinal_cols_2.shape[0])
            for i in range(vector_1.shape[0]):
                mapping = self.rank_mappings[i]
                rank_col_1[i] = mapping[int(ordinal_cols_1[i])]
                rank_col_2[i] = mapping[int(ordinal_cols_2[i])]

            abs_ranks_dist = np.abs(rank_col_1 - rank_col_2)
            
            first_ordinal_occur = np.zeros(self.ordinal_num)
            second_ordinal_occur = np.zeros(self.ordinal_num)
            max_ordinal_occur = np.zeros(self.ordinal_num)
            min_ordinal_occur = np.zeros(self.ordinal_num)
            for i in range(self.ordinal_num):
                first_ordinal_occur[i] = self.ordinal_cardinalities[i][ordinal_cols_1[i]]
                second_ordinal_occur[i] = self.ordinal_cardinalities[i][ordinal_cols_2[i]]
                max_ordinal_occur[i] = self.ordinal_cardinalities[i][-1]
                min_ordinal_occur[i] = self.ordinal_cardinalities[i][0]

            first_ordinal_occur = (first_ordinal_occur - 1) / 2
            second_ordinal_occur = (second_ordinal_occur - 1) / 2
            max_ordinal_occur = (max_ordinal_occur - 1) / 2
            min_ordinal_occur = (min_ordinal_occur - 1) / 2

            ordinal_dist = np.sum(np.power((abs_ranks_dist - first_ordinal_occur - second_ordinal_occur) /
                                    (self.max_ranks_ - self.min_ranks_ - max_ordinal_occur - min_ordinal_occur), 2))
        else:
            ordinal_dist = 0.0

        return np.sqrt(bin_dist + ratio_scale_dist + ordinal_dist + cat_nom_dist)


class Wishart:
    def __init__(
        self,
        dtypes: np.ndarray
    ):
        self.dtypes = dtypes

        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE
        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL) | (
            self.dtypes == DataType.BINARY_SYMMETRIC) | (self.dtypes == DataType.ORDINAL)
        self.bin_asym_idx = self.dtypes == DataType.BINARY_ASYMMETRIC

        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)
        self.bin_asym_num = np.sum(self.bin_asym_idx)

    def fit(self, X: np.ndarray):
        assert X.shape[1] == len(self.dtypes)

        self.n_features_in_ = len(self.dtypes)

        if self.ratio_scale_num > 0:
            ratio_cols = X[:, self.ratio_scale_idx]

            self.std_ = 4 * np.std(ratio_cols, axis=0)

    def __call__(self, vector_1: np.ndarray, vector_2: np.ndarray):
        assert self.n_features_in_ == len(vector_1)
        assert self.n_features_in_ == len(vector_2)

        if self.cat_nom_num > 0:
            cat_nom_cols_1 = vector_1[self.cat_nom_idx]
            cat_nom_cols_2 = vector_2[self.cat_nom_idx]

            cat_nom_dist = np.sum(np.power(cat_nom_cols_1 != cat_nom_cols_2), 2)
        else:
            cat_nom_dist = 0.0

        if self.bin_asym_num > 0:
            bin_asym_cols_1 = vector_1[self.bin_asym_idx]
            bin_asym_cols_2 = vector_2[self.bin_asym_idx]

            bin_asym_dist = np.sum(np.power((bin_asym_cols_1 == 0) & (bin_asym_cols_2 == 0), 2))
        else:
            bin_asym_dist = 0.0

        if self.ratio_scale_num > 0:
            ratio_scale_cols_1 = vector_1[self.ratio_scale_idx]
            ratio_scale_cols_2 = vector_2[self.ratio_scale_idx]

            ratio_scale_dist = np.sum(np.power((ratio_scale_cols_1 - ratio_scale_cols_2) / self.std_, 2))
        else:
            ratio_scale_dist = 0.0

        return np.sqrt(cat_nom_dist + bin_asym_dist + ratio_scale_dist)


class HarikumarPV:
    def __init__(self):
        pass



