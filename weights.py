import os.path
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numba import njit, prange
from scipy.cluster.hierarchy import (
    linkage,
    cophenet,
    fcluster,
)
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from scipy.stats import iqr

from utils import DataType


@njit
def gower_metric_call_func(
    vector_1: np.ndarray,
    vector_2: np.ndarray,
    weights: np.ndarray,
    cat_nom_num: int,
    bin_asym_num: int,
    ratio_scale_num: int,
    num_interval_num: int,
    cat_nom_idx: np.ndarray,
    bin_asym_idx: np.ndarray,
    ratio_scale_idx: np.ndarray,
    num_interval_idx: np.ndarray,
    ratio_scale_normalization: str,
    num_interval_normalization: str,
    ranges_: np.ndarray,
    h_: np.ndarray,
    n_features_in_: int,
):
    assert n_features_in_ == len(vector_1)
    assert n_features_in_ == len(vector_2)

    # numeric interval section
    if num_interval_num > 0:
        num_int_cols_1 = vector_1[num_interval_idx]
        num_int_cols_2 = vector_2[num_interval_idx]

        if num_interval_normalization == "range":
            num_int_dist = 1 - (np.abs(num_int_cols_1 - num_int_cols_2) / ranges_)
        elif num_interval_normalization == "iqr":
            iqr_val = ranges_
            num_int_dist = 1 - (np.abs(num_int_cols_1 - num_int_cols_2) / iqr_val)
        else:
            raise ValueError("Invalid normalization method, please choose otherwise")

        # max - min
        Rt = np.abs(num_int_cols_1 - num_int_cols_2)
        num_int_dist = 1 - (np.abs(num_int_cols_1 - num_int_cols_2) / Rt)

        # if both variables are nonmissing then 1
        num_int_dist[np.isnan(num_int_cols_1) | np.isnan(num_int_cols_2)] = 1.0
        num_int_dist[num_int_dist < 0] = 0

        if weights is not None:
            num_int_dist = num_int_dist @ weights[num_interval_idx]
        else:
            num_int_dist = num_int_dist.sum()
    else:
        num_int_dist = 0.0

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
        bin_asym_dist = np.asarray(
            (bin_asym_cols_1 == 0) & (bin_asym_cols_2 == 0), dtype=np.float64
        )
        bin_asym_dist[
            np.isnan(bin_asym_cols_1) | np.isnan(bin_asym_cols_2)
        ] = 1.0

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
        ratio_dist[
            np.isnan(ratio_scale_cols_1) | np.isnan(ratio_scale_cols_2)
        ] = 1.0

        if ratio_scale_normalization == "kde":
            ratio_dist[above_threshold] = 1.0
            ratio_dist[below_threshold] = 0.0

        if weights is not None:
            ratio_dist = ratio_dist @ weights[ratio_scale_idx]
        else:
            ratio_dist = ratio_dist.sum()
    else:
        ratio_dist = 0.0

    distance = cat_nom_dist + bin_asym_dist + ratio_dist + num_int_dist

    # Normalization
    distance /= n_features_in_

    return distance


class GowerMetricDummy:
    def __init__(
        self,
        dtypes: np.array,
        ratio_scale_normalization: str = "range",
        num_interval_normalization: str = None,
        weights: Optional[Union[list, str, np.ndarray]] = None,
        precomputed_weights_file: Optional[str] = None,
        verbose: int = 0,
    ):
        assert (
            weights is None
            or weights == "precomputed"
            or weights == "cpcc"
            or type(weights) == np.ndarray
            or type(weights) == list
        )
        assert (
            ratio_scale_normalization == "range"
            or ratio_scale_normalization == "iqr"
            or ratio_scale_normalization == "kde"
        )
        assert(
            num_interval_normalization == "range"
            or num_interval_normalization == "iqr"
        )

        self.dtypes = dtypes  # initialize with np.array of column data types
        self.weights = weights
        self.precomputed_weights_file = precomputed_weights_file
        self.verbose = verbose
        self.ranges_: np.ndarray  # values of ranges in .ratio_scale() (iqr or traditional range)
        self.h_: np.ndarray  # h values in .ratio_scale()
        self.n_features_in_: int
        self.ratio_scale_normalization: str = ratio_scale_normalization
        self.num_interval_normalization: str = num_interval_normalization

        # Bit masks for certain column types
        self.ratio_scale_idx = self.dtypes == DataType.RATIO_SCALE
        self.cat_nom_idx = (self.dtypes == DataType.CATEGORICAL_NOMINAL) | (
            self.dtypes == DataType.BINARY_SYMMETRIC
        )
        self.bin_asym_idx = self.dtypes == DataType.BINARY_ASYMMETRIC
        self.num_interval_idx = self.dtypes == DataType.NUMERIC_INTERVAL

        # Sums of columns of given type
        self.cat_nom_num = np.sum(self.cat_nom_idx)
        self.ratio_scale_num = np.sum(self.ratio_scale_idx)
        self.bin_asym_num = np.sum(self.bin_asym_idx)
        self.num_interval_num = np.sum(self.num_interval_idx)

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
        if isinstance(self.weights, str):
            if self.weights == "precomputed":
                loader.load_weights(self.precomputed_weights_file)
            elif self.weights == "cpcc":
                self.weights = np.ones(self.n_features_in_)
                loader.select_weights(X)

        if self.weights is not None:
            loader.select_number_of_clusters(X)

    def __call__(
        self,
        vector_1: np.ndarray,
        vector_2: np.ndarray,
    ) -> np.float64:

        return gower_metric_call_func(
            vector_1,
            vector_2,
            self.weights,
            self.cat_nom_num,
            self.bin_asym_num,
            self.ratio_scale_num,
            self.num_interval_num,
            self.cat_nom_idx,
            self.bin_asym_idx,
            self.ratio_scale_idx,
            self.num_interval_idx,
            self.ratio_scale_normalization,
            self.num_interval_normalization,
            self.ranges_,
            self.h_,
            self.n_features_in_,
        )


# init with k sets of weights
@njit(parallel=True)
def _init_weights(k, n_features_in_):
    return np.array(
        [
            [
                np.random.uniform(
                    1 / (3 * n_features_in_),
                    1 - (n_features_in_ - 1) / (3 * n_features_in_),
                )
                for _ in prange(n_features_in_)
            ]
            for _ in prange(k)
        ]
    )


class GowerMetricWeights:
    def __init__(
        self,
        gower,
        _cpcc_threshold=None,
        _save_computed_weights=True,
    ):
        self.gower = gower
        self._cpcc_threshold = _cpcc_threshold
        self._save_computed_weights = _save_computed_weights

        self.S: np.ndarray  # n_features_in x X.size x X.size matrix for cpcc derivative

        self._x: np.ndarray  # distance matrix for current iteration
        self._Z: np.ndarray  # linkage for current iteration

    # Distance in respect to kth column
    def _S_k(self, vector_1: np.ndarray, vector_2: np.ndarray, k: np.int64):
        bit_mask = np.zeros(vector_1.size)
        bit_mask[k] = 1

        return self.gower(vector_1 * bit_mask, vector_2 * bit_mask)

    def _cpcc(self, weights, X, t=None, pbar=None):

        self.gower.weights = weights
        self._x = pdist(X, metric=self.gower)

        self._Z = linkage(self._x, method="average", metric=self.gower)
        return -cophenet(self._Z, self._x)[0]

    def _cophenetic_dist(self, x):
        Z = linkage(x, method="average", metric=self.gower)
        return cophenet(Z, x)

    # Jacobian for cpcc
    def _cpcc_jac(self, weights, X, t):
        x_ = np.average(self._x)
        t_ = np.average(t)

        a = self._x - x_
        b = t - t_

        factor = (b - a * np.sum(a * b) / np.sum(np.power(a, 2))) / np.sqrt(
            np.sum(np.power(a, 2)) * np.sum(np.power(b, 2))
        )
        res = -np.sum(self.S * factor, axis=1)

        return res

    def select_weights(self, X):
        number_of_initial_weights = 10
        initial_weights = _init_weights(
            number_of_initial_weights, self.gower.n_features_in_
        )

        # Every set of initial weights has same ratio, but different scale,
        # so distance matrix init_S will be same for every set, and so will be cophenetic distance
        init_S = pdist(X, metric=self.gower)
        init_cophet_dist = self._cophenetic_dist(init_S)

        initial_weights_scores = [
            init_cophet_dist for _ in range(number_of_initial_weights)
        ]

        best_weights = (0.0,)

        # We calculate self.S at the beginning, because it will not change in the process
        self.S = np.array(
            [
                pdist(X, metric=self._S_k, k=k)
                for k in prange(self.gower.n_features_in_)
            ]
        )

        # 'maxiter' option doesn't work for some reason
        # Issue was addressed on GitHub, but from what I saw there's no solution.
        for i in range(number_of_initial_weights):
            print(f"{i} set of weights")
            self.gower.weights = initial_weights[i]
            opt_weights = minimize(
                self._cpcc,
                self.gower.weights,
                args=(X, initial_weights_scores[i][1]),
                method="L-BFGS-B",
                jac=self._cpcc_jac,
                options={"maxiter": 20},
                bounds=((0, 1) for _ in range(self.gower.n_features_in_)),
            )

            # if we found better set of weights
            if opt_weights.fun < best_weights[0]:
                best_weights = (opt_weights.fun, opt_weights.x)

            # check for satisfying cpcc_threshold
            if (
                self._cpcc_threshold is not None
                and best_weights[0] < -self._cpcc_threshold
            ):
                break

        self.gower.weights = best_weights[1]
        self.gower.weights /= np.sum(self.gower.weights)

        if self._save_computed_weights:
            self._save_weights()

    # TODO - to delete in final version. Keep for now for convenience.
    def load_weights(self, file_name: str):
        self.gower.weights = np.loadtxt(
            file_name, delimiter=",", dtype=np.float64
        )

    def _save_weights(self):
        weights_dir_name = (
            str(Path().absolute()) + "\\gower_metric_saved_weights"
        )
        if not os.path.exists(weights_dir_name):
            os.mkdir(weights_dir_name)

        rand_id = np.random.randint(0, 100000)
        # converted_weights = np.asarray([list(self.gower.weights)])
        dest_file_name = (
            weights_dir_name + "\\saved_weights_" + str(rand_id) + ".csv"
        )
        np.savetxt(dest_file_name, self.gower.weights, delimiter=",")
        print(f"Weights saved in {dest_file_name}")

    def select_number_of_clusters(self, X):
        Z = linkage(X, method="average", metric=self.gower)

        # So scores[i] refers to score for max i clusters
        scores = [0.0, 0.0, 0.0]

        for i in range(3, 10):
            pred_labels = fcluster(Z, t=i, criterion="maxclust")
            if len(np.unique(pred_labels)) > 1:
                scores.append(
                    silhouette_score(X, pred_labels, metric=self.gower)
                )
            else:
                scores.append(0.0)
        scores = np.array(scores)

        self.gower.number_of_clusters_ = np.argmax(scores)
