import os.path
import timeit
from os import listdir
from os.path import isfile

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import (
    linkage,
    cophenet,
    fcluster,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from utils import Dataset, DataType, Data
from GowerMetric import GowerMetric


def cpcc(X, Z):
    return cophenet(Z, X)


def ioa(O, P):
    O_ = np.average(O)
    return 1 - np.sum(np.power(P - O, 2)) / np.sum(
        np.power(np.absolute(P - O_) + np.absolute(O - O_), 2,)
    )


def pca_test(df: np.ndarray, y: np.ndarray = None, labels=None):
    if y is None and labels is None:
        print(
            "PCA Test Error - data is not labeled neither by y nor by predicted labels!"
        )
        return

    colors = list(mcolors.CSS4_COLORS.values())
    df_cp = df.copy()

    df_cp = StandardScaler().fit_transform(df_cp)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_cp)

    if labels is not None:
        y = labels
        colors = colors[: np.max(labels)]
    else:
        colors = colors[: np.max(y)]

    y = y.copy()
    y = y.reshape(-1, 1)

    principal_components = np.concatenate((principal_components, y), axis=1)

    for e, color in enumerate(colors):
        plt.scatter(
            x=[
                principal_component[0]
                for principal_component in principal_components[
                    principal_components[:, 2] == e + 1
                ]
            ],
            y=[
                principal_component[1]
                for principal_component in principal_components[
                    principal_components[:, 2] == e + 1
                ]
            ],
            c=color,
            label=str(e + 1),
        )

    print(np.bincount(labels))
    plt.legend()
    plt.show()


def plot_clustering(X, y, predicted_labels):
    n_labels = int(np.max(predicted_labels))
    predicted_labels = np.ndarray.astype(predicted_labels, dtype=int)

    colors = ["red", "green", "blue", "orange", "purple"] + list(mcolors.CSS4_COLORS.values())
    pca = PCA(n_components=2)
    y = y.reshape((y.shape[0], 1))
    X = np.concatenate((X, y), axis=1)
    X = pca.fit_transform(X)
    X = X.reshape((X.shape[0], X.shape[1]))

    labeled_points = [[] for _ in range(n_labels+1)]
    for i in range(len(X)):
        labeled_points[predicted_labels[i]].append(X[i])

    for i in range(n_labels+1):
        plt.scatter([point[0] for point in labeled_points[i]], [point[1] for point in labeled_points[i]],
                    color=colors[i], label=f"Class {i}")

    plt.title("KNN Classification")
    plt.legend()
    plt.show()


def mertic_test(
    gower, dataset: Dataset, data: Data, number_of_records: int = None,
):
    if number_of_records is None:
        number_of_records = len(data.data[dataset.name])

    # -------------------------- Data preprocessing --------------------------
    df = np.copy(data.data[dataset.name][:number_of_records])
    df = fill_na(df)

    if dataset.metric == "gower":
        metric_func = gower
    else:
        metric_func = dataset.metric

    enc = OrdinalEncoder(
        categories="auto",
        dtype=np.float64,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan
    )

    # Categorical Nominal columns
    cat_nom_cols = [
        i
        for i in range(len(gower.dtypes))
        if gower.dtypes[i] == DataType.CATEGORICAL_NOMINAL
    ]

    if dataset.labeled:
        cat_nom_cols += [len(gower.dtypes)]

    if len(cat_nom_cols) != 0:
        fit_df = df[:, cat_nom_cols]

        fit_df = pd.DataFrame(fit_df)
        enc.fit(fit_df)
        fit_df = enc.transform(fit_df)

        df[:, cat_nom_cols] = fit_df

    y = None
    if dataset.labeled:
        y = df[:, -1]
        df = df[:, :-1]
        y = np.ndarray.astype(y, dtype=np.float64)

    df = np.ndarray.astype(df, dtype=np.float64)
    # ------------------------------------------------------------------------

    start = timeit.default_timer()
    if dataset.metric == "gower":
        gower.fit(df)
    print(f"Performing fit: {timeit.default_timer() - start}s")

    print(
        f"----------------- Test using {dataset.metric} metric -----------------"
    )

    if dataset.task == "cluster":
        # Hierarchical Clustering and dendrogram (without plotting)
        Z = linkage(df, method="average", metric=metric_func)

        num_of_clusters = (
            gower.number_of_clusters_ if dataset.metric == "gower" and gower.weights is not None else 3
        )

        start = timeit.default_timer()
        dist_x = pdist(df, metric=metric_func)
        print(f"Calculating dist matrix: {timeit.default_timer() - start}")
        pred_labels = fcluster(Z, t=num_of_clusters, criterion="maxclust")

        c, cophenetic_distances = cpcc(dist_x, Z)
        i = ioa(dist_x, cophenetic_distances)

        print(f"CPCC: {c}")
        print(f"IoA: {i}")

        if np.max(pred_labels) > 1:
            s = silhouette_score(df, pred_labels, metric=metric_func)
            cal_halab = calinski_harabasz_score(df, pred_labels)
            dav_bould = davies_bouldin_score(df, pred_labels)
            print(f"Silhouette: {s}")
            print(f"Calinski-Harabasz: {cal_halab}")
            print(f"Davies-Bouldin index: {dav_bould}")
        else:
            print("Predicted labels = 1!")

    elif dataset.task == "knn_class":
        if y is None:
            print("Dataset without labels!")
            exit(-1)

        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)

        knn = KNeighborsClassifier(
            n_neighbors=5,
            algorithm="auto",
            metric=metric_func
        )

        knn.fit(X_train, y_train)
        print(f"KNN Score: {knn.score(X_test, y_test)}")
        predicted_labels = knn.predict(X_test)
        plot_clustering(X_test, y_test, predicted_labels)

    else:
        print("Wrong task!")
    print(
        "--------------------------------------------------------------------\n"
    )


def load_dataset(dataset_name: str):
    loaded_data = np.loadtxt(dataset_name, delimiter=",", dtype=object)
    cols_type = np.loadtxt(
        dataset_name[:-4] + "_cols_type.csv", delimiter=",", dtype=object
    )
    cols_type = np.array([Data.cols_type_maping[k] for k in cols_type])
    labels = loaded_data[1, :]
    loaded_data = loaded_data[1:, :]
    return loaded_data, cols_type, labels


def fill_na(data: np.array):
    data[data == ""] = -1
    return data


def load_sets():
    D_data = {}
    D_cols_type = {}
    D_labels = {}

    for file in listdir(os.path.abspath("datasets")):
        if (
            isfile(os.path.abspath("datasets") + "/" + file)
            and "_cols_type" not in file
        ):
            (
                D_data[file[:-4]],
                D_cols_type[file[:-4]],
                D_labels[file[:-4]],
            ) = load_dataset(os.path.abspath("datasets") + "/" + file)
    D = Data(D_data, D_cols_type, D_labels)
    return D


if __name__ == "__main__":

    D = load_sets()
    # np.random.seed(1234)        # TODO - delete after testing

    print(f"Loaded sets: {list(D.data.keys())}")

    test_dataset_name = "adult"
    test_type = "cluster"
    labeled = True  # if dataset has column labels in same file as columns

    ds1 = Dataset(test_dataset_name, test_type, "gower", labeled)
    ds2 = Dataset(test_dataset_name, test_type, "bin", labeled)
    ds3 = Dataset(test_dataset_name, test_type, "euclidean", labeled)
    ds4 = Dataset(test_dataset_name, test_type, "cosine", labeled)
    ds5 = Dataset(test_dataset_name, test_type, "minkowski", labeled)
    ds6 = Dataset(test_dataset_name, test_type, "dice", labeled)
    ds7 = Dataset(test_dataset_name, test_type, "jaccard", labeled)

    n = 250

    gower = GowerMetric(
        D.cols_type[ds1.name],
        ratio_scale_normalization="iqr",
        ratio_scale_window="kde",
        kde_type="cv"
        # weights="cpcc",
        # precomputed_weights_file="gower_metric_saved_weights/saved_weights_quakes.csv",
    )

    # gower_2 = GowerMetric(
    #     D.cols_type[ds1.name],
    #     "iqr",
    #     # weights="cpcc"
    # )
    #
    # gower_3 = GowerMetric(
    #     D.cols_type[ds1.name],
    #     "kde"
    # )

    print(
        f"Dataset: {test_dataset_name}"
    )

    print(
        "=========================== Range ============================="
    )
    mertic_test(
        gower, ds1, D, n,
    )

    # print(
    #     "=========================== IQR ============================="
    # )
    # mertic_test(
    #     gower_2, ds1, D, n,
    # )
    #
    # print(
    #     "=========================== KDE ============================="
    # )
    # mertic_test(
    #     gower_3, ds1, D, n,
    # )
