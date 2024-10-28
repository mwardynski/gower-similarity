from typing import Optional, Union

import numpy as np
from numba import njit, prange
from scipy.stats import iqr

from utils import DataType
from weights import GowerMetricWeights


@njit
def gower_metric_call_func(vector_1: np.ndarray,
                           vector_2: np.ndarray,
                           weights: np.ndarray,
                           cat_nom_num: int,
                           bin_asym_num: int,
                           ratio_scale_num: int,
                           cat_nom_idx: np.ndarray,
                           bin_asym_idx: np.ndarray,
                           ratio_scale_idx: np.ndarray,
                           ratio_scale_normalization: str,
                           ranges_: np.ndarray,
                           h_: np.ndarray,
                           n_features_in_: int):
    assert n_features_in_ == len(vector_1)
    assert n_features_in_ == len(vector_2)

    if cat_nom_num > 0:
        cat_nom_cols_1 = vector_1[cat_nom_idx]
        cat_nom_cols_2 = vector_2[cat_nom_idx]

        cat_nom_dist = 1.0 - (cat_nom_cols_1 == cat_nom_cols_2)
        cat_nom_dist[np.isnan(cat_nom_cols_1) | np.isnan(cat_nom_cols_2)] = 1.0

        if weights is not None:
            cat_nom_dist = cat_nom_dist @ weights[cat_nom_idx]
        else:
            cat_nom_dist = cat_nom_dist.sum()
    else:
        cat_nom_dist = 0.0

    if bin_asym_num > 0:
        bin_asym_cols_1 = vector_1[bin_asym_idx]
        bin_asym_cols_2 = vector_2[bin_asym_idx]

        # 0 if x1 == x2 == 1 or x1 != x2, so it's same as 1 if x1 == x2 == 0
        bin_asym_dist = np.asarray((bin_asym_cols_1 == 0) & (bin_asym_cols_2 == 0), dtype=np.float64)
        bin_asym_dist[np.isnan(bin_asym_cols_1) | np.isnan(bin_asym_cols_2)] = 1.0

        if weights is not None:
            bin_asym_dist = bin_asym_dist @ weights[bin_asym_idx]
        else:
            bin_asym_dist = bin_asym_dist.sum()
    else:
        bin_asym_dist = 0.0

    if ratio_scale_num > 0:
        ratio_scale_cols_1 = vector_1[ratio_scale_idx]
        ratio_scale_cols_2 = vector_2[ratio_scale_idx]
        ratio_dist = np.abs(ratio_scale_cols_1 - ratio_scale_cols_2)

        if ratio_scale_normalization == "kde":
            above_threshold = ratio_dist >= ranges_
            below_threshold = ratio_dist <= h_

        ratio_dist = ratio_dist / ranges_
        ratio_dist[np.isnan(ratio_scale_cols_1) | np.isnan(ratio_scale_cols_2)] = 1.0

        if ratio_scale_normalization == "kde":
            ratio_dist[above_threshold] = 1.0
            ratio_dist[below_threshold] = 0.0

        if weights is not None:
            ratio_dist = ratio_dist @ weights[ratio_scale_idx]
        else:
            ratio_dist = ratio_dist.sum()
    else:
        ratio_dist = 0.0

    distance = cat_nom_dist + bin_asym_dist + ratio_dist

    # Normalization
    distance /= n_features_in_

    return distance


class GowerMetric:
    def __init__(
        self,
        dtypes: np.array,
        ratio_scale_normalization: str = "range",
        weights: Optional[Union[list, str, np.ndarray]] = None,
        precomputed_weights_file: Optional[str] = None,
        verbose: int = 0
    ):
        assert (
            weights is None
            or type(weights) == np.ndarray
            or type(weights) == list
            or weights == "precomputed"
            or weights == "cpcc"
        )
        assert (
            ratio_scale_normalization == "range"
            or ratio_scale_normalization == "iqr"
            or ratio_scale_normalization == "kde"
        )

        self.dtypes = dtypes  # initialize with np.array of column data types
        self.weights = weights
        self.precomputed_weights_file = precomputed_weights_file
        self.verbose = verbose
        self.ranges_: np.ndarray  # values of ranges in .ratio_scale() (iqr or traditional range)
        self.h_: np.ndarray  # h values in .ratio_scale()
        self.n_features_in_: int
        self.ratio_scale_normalization: str = ratio_scale_normalization

        # Bit masks for certain column types
        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE
        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL) | (
            self.dtypes == DataType.BINARY_SYMMETRIC
        )
        self.bin_asym_idx = self.dtypes == DataType.BINARY_ASYMMETRIC

        # Sums of columns of given type
        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)
        self.bin_asym_num = np.sum(self.bin_asym_idx)

    def fit(self, X):
        assert X.shape[1] == len(self.dtypes)

        self.n_features_in_ = X.shape[1]
        self.ranges_ = np.ndarray([])
        self.h_ = np.ndarray([])

        if self.ratio_scale_num > 0:
            ratio_cols = X[:, self.ratio_scale_idx]

            col_mean = np.nanmean(ratio_cols, axis=0)
            nan_indices = np.where(np.isnan(ratio_cols))
            ratio_cols[nan_indices] = np.take(col_mean, nan_indices[1])

            if self.ratio_scale_normalization == "range":
                self.ranges_ = np.ptp(ratio_cols, axis=0)

            elif self.ratio_scale_normalization in {"iqr", "kde"}:
                self.ranges_ = iqr(ratio_cols, axis=0)

                # Needs this check
                zero_values_mask = self.ranges_ == 0
                self.ranges_[zero_values_mask] = np.ptp(
                    ratio_cols[:, zero_values_mask], axis=0
                )

            if self.ratio_scale_normalization == "kde":
                n = X.shape[0]
                c = 1.06

                s = np.std(ratio_cols, axis=0)
                self.h_ = (
                    c / n ** (1 / 5) * np.min([s, self.ranges_ / 1.34], axis=0)
                )

        loader = GowerMetricWeights(self)
        if type(self.weights) != np.ndarray and type(self.weights) != list:
            if self.weights == "precomputed":
                loader.load_weights(self.precomputed_weights_file)
            elif type(self.weights) != np.ndarray and self.weights == "cpcc":
                self.weights = np.ones(self.n_features_in_)
                loader.select_weights(X)

        if self.weights is not None:
            loader.select_number_of_clusters(X)

    def __call__(
        self, vector_1: np.ndarray, vector_2: np.ndarray,
    ) -> np.float64:

        return gower_metric_call_func(
            vector_1,
            vector_2,
            self.weights,
            self.cat_nom_num,
            self.bin_asym_num,
            self.ratio_scale_num,
            self.cat_nom_idx,
            self.bin_asym_idx,
            self.ratio_scale_idx,
            self.ratio_scale_normalization,
            self.ranges_,
            self.h_,
            self.n_features_in_
        )


# init with k sets of weights
@njit(parallel=True)
def _init_weights(k, n_features_in_):
    return np.array(
        [
            [
                np.random.uniform(
                    1 / (3 * n_features_in_),
                    1
                    - (n_features_in_ - 1)
                    / (3 * n_features_in_),
                )
                for _ in prange(n_features_in_)
            ]
            for _ in prange(k)
        ]
    )



