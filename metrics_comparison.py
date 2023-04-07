import os.path
from os import listdir
from os.path import isfile
from typing import Union

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (
    linkage,
    cophenet,
    fcluster,
)
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle

from utils import Dataset, Data, DataType
from GowerMetric import GowerMetric


DATASET_DIR = "./metrics_comparison_datasets/"

TEST_METRICS_NAMES = [
    "euclidean",
    "cosine",
    "minkowski",
    "dice",
    "jaccard"
]

LABELED_DATASETS = [
    "abalone",
    "adult",
    "arrhythmia",
    "bands",
    "pm2.5_data_of_5_chinese_cities",
    "spambase",
    "spam",
    "breast_cancer_wisconsin",
    "kr-vs-kp"
]

TASKS = [
    "hierarchical",
    "hdbscan",
    "knn"
]


def load_dataset(dataset_name: str):
    loaded_data = np.loadtxt(dataset_name, delimiter=",", dtype=object)
    cols_type = np.loadtxt(
        dataset_name[:-4] + "_cols_type.csv", delimiter=",", dtype=object
    )
    cols_type = np.array([Data.cols_type_mapping[k] for k in cols_type])
    labels = loaded_data[1, :]
    loaded_data = loaded_data[1:, :]
    return loaded_data, cols_type, labels


def load_sets():
    D_data = {}
    D_cols_type = {}
    D_labels = {}

    for file in listdir(os.path.abspath(DATASET_DIR)):
        if (
            isfile(os.path.abspath(DATASET_DIR) + "/" + file)
            and "_cols_type" not in file
        ):
            (
                D_data[file[:-4]],
                D_cols_type[file[:-4]],
                D_labels[file[:-4]],
            ) = load_dataset(os.path.abspath(DATASET_DIR) + "/" + file)
    D = Data(D_data, D_cols_type, D_labels)
    return D


def fill_na(data: np.array):
    data[data == ""] = -1
    data[data == "?"] = -1
    data[data == "NA"] = -1
    return data


def cpcc(X, Z):
    return cophenet(Z, X)


def ioa(O, P):
    O_ = np.average(O)
    return 1 - np.sum(np.power(P - O, 2)) / np.sum(
        np.power(
            np.absolute(P - O_) + np.absolute(O - O_),
            2,
        )
    )


def scores(metric: Union[str, GowerMetric], data: Data, name: str, labeled: bool, task: str):
    number_of_records = 500

    # -------------------------- Data preprocessing --------------------------
    cols_types = data.cols_type[name]
    df = np.copy(data.data[name])
    df = shuffle(df)
    df = df[:number_of_records]
    df = fill_na(df)

    enc = OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )

    # Categorical Nominal columns
    cat_nom_cols = [
        i
        for i in range(len(cols_types))
        if cols_types[i] == DataType.CATEGORICAL_NOMINAL
    ]

    if labeled:
        cat_nom_cols += [len(cols_types)]

    if len(cat_nom_cols) != 0:
        fit_df = df[:, cat_nom_cols]

        fit_df = pd.DataFrame(fit_df)
        fit_df = fit_df.astype("string")
        enc.fit(fit_df)
        fit_df = enc.transform(fit_df)

        df[:, cat_nom_cols] = fit_df

    y = None
    if labeled:
        y = df[:, -1]
        df = df[:, :-1]
        y = np.ndarray.astype(y, dtype=np.float64)

    df = np.ndarray.astype(df, dtype=np.float64)
    # ------------------------------------------------------------------------

    if isinstance(metric, GowerMetric):
        metric.fit(df)

    if task == "hierarchical":
        Z = linkage(df, method="average", metric=metric)

        num_of_clusters = (
            metric.number_of_clusters_
            if isinstance(metric, GowerMetric) and metric.number_of_clusters_ is not None
            else 3
        )

        dist_x = pdist(df, metric=metric)

        pred_labels = fcluster(Z, t=num_of_clusters, criterion="maxclust")

        c, cophenetic_distances = cpcc(dist_x, Z)
        i = ioa(dist_x, cophenetic_distances)
        knn_score = None

        if np.max(pred_labels) < 1:
            print("Predicted labels = 1!")
            return -1, -1, -1, -1, -1

    elif task == "hdbscan":
        clusterer = HDBSCAN(metric="precomputed")
        dist_matrix = pairwise_distances(X=df, metric=metric, n_jobs=-1)
        clusterer.fit(X=dist_matrix, y=y)
        pred_labels = clusterer.labels_

        c, i, knn_score = None, None, None
    elif task == "knn":
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2
        )
        knn = KNeighborsClassifier(
            n_neighbors=5, algorithm="auto", metric=metric
        )
        knn.fit(X_train, y_train)
        knn_score = knn.score(X_test, y_test)
        pred_labels = knn.predict(X_test)

        c, i = None, None
        s = silhouette_score(X_test, pred_labels, metric=metric)
        cal_halab = calinski_harabasz_score(X_test, pred_labels)
        dav_bould = davies_bouldin_score(X_test, pred_labels)
        return c, i, s, cal_halab, dav_bould, knn_score
    else:
        print("Wrong type of task")
        return -1, -1, -1, -1, -1

    s = silhouette_score(df, pred_labels, metric=metric)
    cal_halab = calinski_harabasz_score(df, pred_labels)
    dav_bould = davies_bouldin_score(df, pred_labels)

    return c, i, s, cal_halab, dav_bould, knn_score


def add_header(header, task):
    with open(f"./results/test_results_{task}.txt", "a") as file:
        file.write(f"\n{header}\n")


def save_result(metric, task, sil, cal_halab, dav_bould, c, i, knn_score):
    with open(f"./results/test_results_{task}.txt", "a") as file:
        file.write(f"{metric},{sil},{cal_halab},{dav_bould},{c},{i}")
        if task == "knn":
            file.write(f",{knn_score}\n")
        else:
            file.write("\n")


if __name__ == '__main__':
    D = load_sets()
    print(f"Loaded sets: {list(D.data.keys())}")
    datasets_names = list(D.data.keys())

    test_dataset_name = "kr-vs-kp"

    config = {
        "ratio_scale_normalization": "iqr",
        "ratio_scale_window": "kde",
        "kde_type": "cv_optuna",
        # "weights": "cpcc",
        "nan_values_handling": "max_dist",
        "number_of_clusters": -1
    }

    # gower = GowerMetric(
    #     D.cols_type[test_dataset_name],
    #     **config
    # )
    # c, i, sil, cal_halab, dav_bould, knn_score = scores(
    #     metric=gower,
    #     data=D,
    #     name=test_dataset_name,
    #     labeled=test_dataset_name in LABELED_DATASETS,
    #     task="hierarchical"
    # )
    #
    # print(f"CPCC: {c}\nIoA: {i}\nSilhouette: {sil}\nCalinski-Harabasz: {cal_halab}\nDavid-Bouldin: {dav_bould}\nKNN-Score: {knn_score}")

    # Only for gower
    # for completed, dataset_name in enumerate(datasets_names):
    #     print(f"Dataset: {dataset_name} ... {(completed / len(datasets_names) * 100):.2f}%")
    #     for task in TASKS:
    #         gower = GowerMetric(
    #             D.cols_type[dataset_name],
    #             **config
    #         )
    #
    #         try:
    #             c, i, sil, cal_halab, dav_bould = scores(
    #                 metric=gower,
    #                 data=D,
    #                 name=dataset_name,
    #                 labeled=dataset_name in LABELED_DATASETS,
    #                 task=task
    #             )
    #         except Exception:
    #             print(f"Error with gower metric, dataset: {dataset_name}, task: {task}")
    #             c, i, sil, cal_halab, dav_bould = -1, -1, -1, -1, -1
    #         print(f"Silhouette: {sil}\nCalinski-Harabasz: {cal_halab}\nDavid-Bouldin: {dav_bould}, CPCC: {c}, IoA: {i}")

    # All metrics
    for completed, dataset_name in enumerate(["breast_cancer_wisconsin"]):
        print(f"Dataset: {dataset_name} ... {(completed / len(datasets_names) * 100):.2f}%")
        for task in ["knn"]:
            add_header(dataset_name, task)
            gower = GowerMetric(
                D.cols_type[dataset_name],
                **config
            )

            try:
                c, i, sil, cal_halab, dav_bould, knn_score = scores(
                    metric=gower,
                    data=D,
                    name=dataset_name,
                    labeled=dataset_name in LABELED_DATASETS,
                    task=task
                )
            except Exception:
                print(f"Error with gower metric, dataset: {dataset_name}, task: {task}")
                c, i, sil, cal_halab, dav_bould, knn_score = -1, -1, -1, -1, -1, -1
            save_result("gower", task, sil, cal_halab, dav_bould, c, i, knn_score)

            for metric_name in TEST_METRICS_NAMES:
                try:
                    c, i, sil, cal_halab, dav_bould, knn_score = scores(
                        metric=metric_name,
                        data=D,
                        name=dataset_name,
                        labeled=dataset_name in LABELED_DATASETS,
                        task=task
                        )
                except Exception as e:
                    print(f"Error with {metric_name} metric, dataset: {dataset_name}, task: {task}")
                    print(e)
                    c, i, sil, cal_halab, dav_bould, knn_score = -1, -1, -1, -1, -1, -1
                save_result(metric_name, task, sil, cal_halab, dav_bould, c, i, knn_score)

    print("Completed ... 100%")
