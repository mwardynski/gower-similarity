import os.path
import timeit
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import sklearn.utils.validation
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (
    linkage,
    cophenet,
    fcluster,
)
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from utils import DataType


class GowerMetric:
    def __init__(
        self,
        dtypes: np.array,
        ratio_scale_normalization: str = "range",
        weights: Optional[Union[list, str]] = None,
        precomputed_weights_file: Optional[str] = None,
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

        self.dtypes = dtypes  # initialize with np.array of column data types
        self.weights = weights
        self.precomputed_weights_file = precomputed_weights_file
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
        self.cat_nom_num = sum(self.cat_nom_idx)
        self.ratio_scale_num = sum(self.ratio_scale_idx)
        self.bin_asym_num = sum(self.bin_asym_idx)

    def fit(self, X):
        assert X.shape[1] == len(self.dtypes)

        self.ranges_ = np.empty(self.ratio_scale_num)

        if self.ratio_scale_normalization == "kde":
            self.h_ = np.empty(X.shape[1])
        self.n_features_in_ = X.shape[1]

        ratio_cols = X[:, self.ratio_scale_idx]
        if self.ratio_scale_normalization == "range":
            self.ranges_ = np.ptp(ratio_cols, axis=0)
        elif self.ratio_scale_normalization in {"iqr", "kde"}:
            q1, q3 = np.percentile(ratio_cols, [25, 75], axis=0)
            self.ranges_ = q3 - q1

            # If any iqr ends up being equal to 0, we change it to standard range
            # zero_ranges_idx = (self.ranges_ == 0) & (self.ratio_scale_idx)
            # self.ranges_[zero_ranges_idx] = np.ptp(X[:, zero_ranges_idx], axis=0)

        if self.ratio_scale_normalization == "kde":
            n = X.shape[0]
            c = 1.06

            s = np.std(ratio_cols, axis=0)
            self.h_[self.ratio_scale_idx] = (
                c / n ** (1 / 5) * np.min([s, self.ranges_ / 1.34], axis=0)
            )

        loader = GowerMetricWeights(self)
        if self.weights == "precomputed":
            loader.load_weights(self.precomputed_weights_file)
        elif self.weights == "cpcc":
            self.weights = np.ones(self.n_features_in_)
            loader.select_weights(X)

        loader.select_number_of_clusters(X)

    def __call__(
        self, vector_1: np.ndarray, vector_2: np.ndarray,
    ):
        # TODO - no check for fit
        # sklearn.utils.validation.check_is_fitted(self)
        assert self.n_features_in_ == len(vector_1)
        assert self.n_features_in_ == len(vector_2)

        # Distance for cat_nom and bin_sym columns
        if self.cat_nom_num > 0:
            cat_nom_cols_1 = vector_1[self.cat_nom_idx]
            cat_nom_cols_2 = vector_2[self.cat_nom_idx]
            cat_nom_dist = 1.0 - (cat_nom_cols_1 == cat_nom_cols_2)

            # Weighted variant - dot product with according self.weights columns
            # Unweighted - a sum of distances between all columns
            if self.weights is not None:
                cat_nom_dist = cat_nom_dist @ self.weights[self.cat_nom_idx]
            else:
                cat_nom_dist = cat_nom_dist.sum()
        else:
            cat_nom_dist = 0.0

        # Distance for bin_asym columns
        if self.bin_asym_num > 0:
            bin_asym_cols_1 = vector_1[self.bin_asym_idx]
            bin_asym_cols_2 = vector_2[self.bin_asym_idx]

            zeros = (bin_asym_cols_1 == 0) | (bin_asym_cols_2 == 0)

            bin_asym_dist = (bin_asym_cols_1 != bin_asym_cols_2) | zeros

            if self.weights is not None:
                bin_asym_dist = bin_asym_dist @ self.weights[self.bin_asym_idx]
            else:
                bin_asym_dist = bin_asym_dist.sum()
        else:
            bin_asym_dist = 0.0

        # Distance for ratio_scale columns
        if self.ratio_scale_num > 0:
            ratio_scale_cols_1 = vector_1[self.ratio_scale_idx]
            ratio_scale_cols_2 = vector_2[self.ratio_scale_idx]
            ratio_dist = np.abs(ratio_scale_cols_1 - ratio_scale_cols_2)

            above_threshold = ratio_dist >= self.ranges_
            if self.ratio_scale_normalization == "kde":
                below_threshold = ratio_dist <= self.h_[self.ratio_scale_idx]

            ratio_dist /= self.ranges_
            ratio_dist[above_threshold] = 1.0

            if self.ratio_scale_normalization == "kde":
                ratio_dist[below_threshold] = 0.0

            if self.weights is not None:
                ratio_dist = ratio_dist @ self.weights[self.ratio_scale_idx]
            else:
                ratio_dist = ratio_dist.sum()
        else:
            ratio_dist = 0.0

        # Sum of all distances
        distance = cat_nom_dist + bin_asym_dist + ratio_dist

        # Normalization
        distance /= self.n_features_in_

        return distance


class GowerMetricWeights:
    def __init__(
        self,
        gower: GowerMetric,
        _cpcc_threshold=None,
        _save_computed_weights=True,
    ):
        self.gower = gower
        self._cpcc_threshold = _cpcc_threshold
        self._save_computed_weights = _save_computed_weights
        self.iteration = 0

    def cat_nom_bin_sym(
        self, vector_1: np.ndarray, vector_2: np.ndarray
    ) -> np.ndarray:

        return vector_1 == vector_2

    def ratio_scale(
        self,
        vector_1: np.ndarray,
        vector_2: np.ndarray,
        val_range: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:

        absolute = vector_1 - vector_2
        absolute = np.abs(absolute)

        ones = absolute >= val_range
        zeros = absolute <= h

        if vector_1.size == 1:
            if ones:
                return np.array(1.0)
            elif zeros:
                return np.array(0.0)
            else:
                absolute /= val_range
        else:
            absolute /= val_range
            absolute[zeros] = 0.0
            absolute[ones] = 1.0

        return 1.0 - absolute

    def _S_k(self, vector_1: np.ndarray, vector_2: np.ndarray, k: np.int64):
        return (
            self.ratio_scale(
                np.array(vector_1[k]),
                np.array(vector_2[k]),
                np.array(self.gower.ranges_[k]),
                np.array(self.gower.h_[k]),
            )
            if self.gower.dtypes[k] == DataType.RATIO_SCALE
            else self.cat_nom_bin_sym(vector_1[k], vector_2[k])
        )

    def _cpcc(self, weights, X, t=None, pbar=None):
        self.iteration += 1
        print(f"{self.iteration} iteration")
        self.gower.weights = weights
        x = pdist(X, metric=self.gower)
        Z = linkage(x, method="average", metric=self.gower)
        return -cophenet(Z, x)[0]

    def _cophenetic_dist(self, x):
        Z = linkage(x, method="average", metric=self.gower)
        return cophenet(Z, x)

    def _cpcc_jac(self, weights, X, t):
        x = pdist(X, metric=self.gower)
        x_ = np.average(x)
        t_ = np.average(t)

        a = x - x_
        b = t - t_

        # Distance matrix calculated only for k feature
        S_i_j_k = lambda k: pdist(X, metric=self._S_k, k=k)

        # Derivative of CPCC as a function of k
        derivative_cpcc_func = lambda k: -np.sum(
            S_i_j_k(k)
            * (
                (b - a * np.sum(a * b) / np.sum(np.power(a, 2)))
                / np.sqrt(np.sum(np.power(a, 2)) * np.sum(np.power(b, 2)))
            )
        )
        return np.array(
            [
                -derivative_cpcc_func(k)
                for k in range(self.gower.n_features_in_)
            ]
        )

    def _init_weights(self, k):
        return np.array(
            [
                [
                    np.random.uniform(
                        1 / (3 * self.gower.n_features_in_),
                        1
                        - (self.gower.n_features_in_ - 1)
                        / (3 * self.gower.n_features_in_),
                    )
                    for _ in range(self.gower.n_features_in_)
                ]
                for _ in range(k)
            ]
        )

    def select_weights(self, X):
        number_of_initial_weights = 4
        initial_weights = self._init_weights(number_of_initial_weights)

        X = X.copy()
        enc = OrdinalEncoder()
        enc.set_params(encoded_missing_value=-1)

        if self.gower.cat_nom_num != 0:
            fit_X = X[:, self.gower.cat_nom_idx]

            enc.fit(fit_X)
            fit_X = enc.transform(fit_X)
            X[:, self.gower.cat_nom_idx] = fit_X

        X = np.ndarray.astype(X, dtype=np.float64)

        initial_weights_scores = []
        for i in range(number_of_initial_weights):
            self.gower.weights = initial_weights[i]
            S = pdist(X, metric=self.gower)
            initial_weights_scores.append(self._cophenetic_dist(S))

        initial_weights_scores.sort(key=lambda x: x[1][0], reverse=True)

        best_weights = (0.0,)

        for i in range(number_of_initial_weights):
            print(f"{i} set of weights")
            self.iteration = 0
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

            if opt_weights.fun < best_weights[0]:
                best_weights = (opt_weights.fun, opt_weights.x)

            if (
                self._cpcc_threshold is not None
                and best_weights[0] < -self._cpcc_threshold
            ):
                break

        self.gower.weights = best_weights[1]
        self.gower.weights /= np.sum(self.gower.weights)

        if self._save_computed_weights:
            self._save_weights()

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

        # X_values = np.linspace(3, 10, 7)
        # plt.plot(X_values, scores[3:])
        # plt.show()
