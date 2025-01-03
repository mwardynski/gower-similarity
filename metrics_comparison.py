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
    rand_score,
    completeness_score,
    fowlkes_mallows_score,
    mutual_info_score
)
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle

from utils import Dataset, Data, DataType
from GowerMetric import MyGowerMetric


DATASET_DIR = "./tests/tests_files/"

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


def scores(metric: Union[str, MyGowerMetric], data: Data, name: str, labeled: bool, task: str, random_state: int = 0):
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

    # Binarize for jaccard and dice coefficients
    if metric == "jaccard" or metric == "dice":
        df = df != 0
        if y is not None:
            y = y != 0
    else:
        df = np.ndarray.astype(df, dtype=np.float64)
    # ------------------------------------------------------------------------

    if isinstance(metric, MyGowerMetric):
        metric.fit(df)

    if task == "hierarchical":
        Z = linkage(df, method="average", metric=metric)

        num_of_clusters = (
            metric.number_of_clusters_
            if isinstance(metric, MyGowerMetric) and metric.number_of_clusters_ is not None
            else 3
        )

        dist_x = pdist(df, metric=metric)

        pred_labels = fcluster(Z, t=num_of_clusters, criterion="maxclust")

        c, cophenetic_distances = cpcc(dist_x, Z)
        i = ioa(dist_x, cophenetic_distances)
        knn_score, f1 = None, None

        if np.max(pred_labels) < 1:
            print("Predicted labels = 1!")
            return -1, -1, -1, -1, -1, -1, -1, -1

    elif task == "hdbscan":
        clusterer = HDBSCAN(metric="precomputed")
        dist_matrix = pairwise_distances(X=df, metric=metric, n_jobs=-1)
        clusterer.fit(X=dist_matrix, y=y)
        pred_labels = clusterer.labels_

        c, i, knn_score, f1 = None, None, None, None
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

        f1 = f1_score(y_test, pred_labels, average="weighted")
        c, i, rand, complete, f_m_score, mutual = None, None, None, None, None, None
        return c, i, rand, complete, f_m_score, mutual, knn_score, f1
    else:
        print("Wrong type of task")
        return -1, -1, -1, -1, -1, -1, -1, -1

    rand = rand_score(y, pred_labels)
    complete = completeness_score(y, pred_labels)
    f_m_score = fowlkes_mallows_score(y, pred_labels)
    mutual = mutual_info_score(y, pred_labels)

    return c, i, rand, complete, f_m_score, mutual, knn_score, f1

def create_results_dir():
    if not os.path.exists("./results"):
        os.makedirs("./results")

def add_header(header, task):
    with open(f"./results/test_results_{task}.txt", "a") as file:
        file.write(f"\n{header}")
        if task == "hierarchical":
            file.write(f",Rand,Complete,F-M,Mutual,CPCC,IOA\n\n")
        elif task == "hdbscan":
            file.write(f",Rand,Complete,F-M,Mutual\n\n")
        elif task == "knn":
            file.write(f",KNN Score,F1\n\n")


def save_result(metric, task, c, i, rand, complete, f_m_score, mutual, knn_score, f1):
    with open(f"./results/test_results_{task}.txt", "a") as file:
        if task == "hierarchical":
            file.write(
                f"{metric},{rand},{complete},{f_m_score},{mutual},{c},{i}\n"
            )
        elif task == "hdbscan":
            file.write(
                f"{metric},{rand},{complete},{f_m_score},{mutual}\n"
            )
        elif task == "knn":
            file.write(
                f"{metric},{knn_score},{f1}\n"
            )


def calc_ranks():
    with open("./results/ranks_data_knn.txt", "r") as file:
        data = file.read()
    data = data.split("\n")

    for i in range(len(data)):
        data[i] = data[i].split("\t")

    table = [[]]
    for i in range(len(data)):
        if data[i][0] == '':
            table.append([])
        else:
            table[-1].append(data[i])

    table = np.array(table, dtype=np.float64)
    table[table == np.nan] = -1
    for e, batch in enumerate(table):
        print(e)
        for lane in batch:
            print(lane)

    ranks = np.zeros((len(table[0][0]), len(table[0])))
    print(f"Ranks shape: {ranks.shape}")

    for i in range(len(table)):
        for j in range(len(table[0][0])):
            # print(f"Sorting: {list(table[i][:, j])}")
            # to_sort = zip(list(table[i][:, j]), range(len(table[i][:, j])))
            # to_sort = sorted(list(to_sort), key=lambda x: x[0], reverse=True)
            # print("Sorted:", to_sort)
            # to_sort = np.array([x[1] for x in to_sort])
            # table[i][:, j] = to_sort
            # print(f"{table[i][:, j]}")
            order = np.argsort(-table[i][:, j])

            order = np.array([np.where(order == k)[0][0] for k in range(len(order))])
            table[i][:, j] = order
    table += 1

    for e, batch in enumerate(table):
        print(e)
        for lane in batch:
            print(lane)

    ranks = np.mean(table, axis=0)
    for rank in ranks:
        print(rank)

    with open("./results/ranks.txt", "w") as file:
        for rank in ranks:
            file.write(f"{','.join(list(rank.astype(str)))}\n")


if __name__ == '__main__':
    create_results_dir()
    # calc_ranks()
    # exit(0)

    D = load_sets()
    print(f"Loaded sets: {list(D.data.keys())}")
    datasets_names = list(D.data.keys())

    test_dataset_name = "spambase"

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
    # c, i, rand, complete, f_m_score, mutual, knn_score, f1 = scores(
    #     metric=gower,
    #     data=D,
    #     name=test_dataset_name,
    #     labeled=test_dataset_name in LABELED_DATASETS,
    #     task="hierarchical"
    # )
    #
    # print(f"CPCC: {c}, IOA: {i}, Rand: {rand}, Complete: {complete}, F-M: {f_m_score}, Mutual: {mutual}, KNN Score: {knn_score}, F1: {f1}")

    # Only for gower
    # for completed, dataset_name in enumerate(datasets_names):
    #     print(f"Dataset: {dataset_name} ... {(completed / len(datasets_names) * 100):.2f}%")
    #     for task in ["hdbscan", "knn"]:
    #         add_header(dataset_name, task)
    #         gower = GowerMetric(
    #             D.cols_type[dataset_name],
    #             **config
    #         )
    #
    #         try:
    #             c, i, rand, complete, f_m_score, mutual, knn_score, f1 = scores(
    #                 metric=gower,
    #                 data=D,
    #                 name=dataset_name,
    #                 labeled=dataset_name in LABELED_DATASETS,
    #                 task=task
    #             )
    #         except Exception:
    #             print(f"Error with gower metric, dataset: {dataset_name}, task: {task}")
    #             c, i, rand, complete, f_m_score, mutual, knn_score, f1 = -1, -1, -1, -1, -1, -1, -1, -1
    #         save_result("gower", task, c, i, rand, complete, f_m_score, mutual, knn_score, f1)
    #         print(f"CPCC: {c}, IOA: {i}, Rand: {rand}, Complete: {complete}, F-M: {f_m_score}, Mutual: {mutual}, KNN Score: {knn_score}, F1: {f1}")

    # All metrics
    for completed, dataset_name in enumerate(datasets_names):
        print(f"Dataset: {dataset_name} ... {(completed / len(datasets_names) * 100):.2f}%")
        for task in ["knn"]:
            add_header(dataset_name, task)
            gower = MyGowerMetric(
                D.cols_type[dataset_name],
                **config
            )

            try:
                c, i, rand, complete, f_m_score, mutual, knn_score, f1 = scores(
                    metric=gower,
                    data=D,
                    name=dataset_name,
                    labeled=dataset_name in LABELED_DATASETS,
                    task=task
                )
            except Exception:
                print(f"Error with gower metric, dataset: {dataset_name}, task: {task}")
                c, i, rand, complete, f_m_score, mutual, knn_score, f1 = -1, -1, -1, -1, -1, -1, -1, -1
            save_result("gower", task, c, i, rand, complete, f_m_score, mutual, knn_score, f1)

            for metric_name in TEST_METRICS_NAMES:
                try:
                    c, i, rand, complete, f_m_score, mutual, knn_score, f1 = scores(
                        metric=metric_name,
                        data=D,
                        name=dataset_name,
                        labeled=dataset_name in LABELED_DATASETS,
                        task=task
                        )
                except Exception as e:
                    print(f"Error with {metric_name} metric, dataset: {dataset_name}, task: {task}")
                    print(e)
                    c, i, rand, complete, f_m_score, mutual, knn_score, f1 = -1, -1, -1, -1, -1, -1, -1, -1
                save_result(metric_name, task, c, i, rand, complete, f_m_score, mutual, knn_score, f1)

    print("Completed ... 100%")
