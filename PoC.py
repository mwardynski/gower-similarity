import numpy as np
from enum import Enum


class DataType(Enum):
    BINARY_SYMMETRIC = 0
    BINARY_ASYMMETRIC = 1
    CATEGORICAL_NOMINAL = 2
    RATIO_SCALE = 3


class GowerMetric:
    def __init__(self, data_type: np.array):
        self.__data_type = data_type        # initialize with np.array of column data types
        self.__iqrs = []                    # iqr values in .__ratio_scale()
        self.__h = []                       # h values in .__ratio_scale()
        self.__data_fitted = False          # flag for running .fit()
        self.__sigma_sum = 0                # counter for existing values in vector

    def __bin_sym(self, value_1, value_2) -> float:
        if value_1 is None or value_2 is None:
            return 0.0

        self.__add_sigma()
        return 1.0 if value_1 == value_2 else 0.0

    def __bin_asym(self, value_1, value_2) -> float:
        if value_1 is None or value_2 is None:
            return 0.0

        if value_1 == value_2 and value_1 == 1:
            self.__add_sigma()
            return 1.0
        else:
            return 0.0

    def __cat_nom(self, value_1, value_2) -> float:
        if value_1 is None or value_2 is None:
            return 0.0

        self.__add_sigma()
        return 1.0 if value_1 == value_2 else 0.0

    def __ratio_scale(self, value_1, value_2, iqr, h) -> float:
        if value_1 is None or value_2 is None:
            return 0.0

        self.__add_sigma()
        absolute = np.abs(value_1 - value_2)

        if absolute >= iqr:
            return 0.0

        elif absolute <= h:
            return 1.0

        return 1.0 - absolute / iqr

    def __add_sigma(self):
        self.__sigma_sum += 1

    def __reset_sigma(self):
        self.__sigma_sum = 0

    def fit(self, x):
        for i in range(self.__data_type.size):
            if self.__data_type[i] == DataType.RATIO_SCALE:

                # IQR (g_t) - Interquartile Range
                q1, q3 = np.percentile(x[:, i].astype(np.int32), [25, 75])
                self.__iqrs.append(q3 - q1)

                # h_t - bandwidth in the kernel density estimation (Marcello D’Orazio - p. 9)
                if np.any(x[:, i].astype(np.int32) < 0):
                    c = 0.9
                else:
                    c = 1.06

                s = np.std(x[:, i].astype(np.int32))
                n = x[:, i].size
                self.__h.append(c / n ** (1/5) * np.min([s, self.__iqrs[-1] / 1.34]))

        self.__data_fitted = True
        print("IQRS:", self.__iqrs)
        print("h:", self.__h)

    def __call__(
        self, vector_1: np.array, vector_2: np.array,
    ):
        if len(vector_1) != len(vector_2):
            print("Vector sizes don't match!")
            return -1

        if not self.__data_fitted:
            print("Use GowerMetric.fit(x) first to initialize scales for data!")
            return -1

        d = 0
        ratios_index = 0

        for i in range(len(vector_1)):

            if self.__data_type[i] == DataType.BINARY_SYMMETRIC:
                d += 1 - self.__bin_sym(vector_1[i], vector_2[i])
            elif self.__data_type[i] == DataType.BINARY_ASYMMETRIC:
                d += 1 - self.__bin_asym(vector_1[i], vector_2[i])
            elif self.__data_type[i] == DataType.CATEGORICAL_NOMINAL:
                d += 1 - self.__cat_nom(vector_1[i], vector_2[i])
            elif self.__data_type[i] == DataType.RATIO_SCALE:
                d += 1 - self.__ratio_scale(
                    float(vector_1[i]), float(vector_2[i]), self.__iqrs[ratios_index], self.__h[ratios_index]
                )
                ratios_index += 1
            else:
                print(f"Wrong data type! - {self.__data_type[i]}")

        d /= self.__sigma_sum
        self.__reset_sigma()

        return d

    # Simple function for calculating dissimilarity matrix
    def make_matrix(self, data_frame: np.ndarray) -> np.array:
        n = len(data_frame)
        M = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i):
                if i != j:
                    M[i][j] = self.__call__(data_frame[i], data_frame[j])
                    M[j][i] = M[i][j]

        return np.array(M, dtype=np.float32)


if __name__ == "__main__":
    #   gender / age / grade
    data = np.array(
        [
            ["F", 15, 5],
            ["F", 36, 3],
            ["F", 58, 2],
            ["F", 78, 2],
            ["F", 100, 4],
            ["M", 15, 3],
            ["M", 36, 2],
            ["M", 58, 1],
            ["M", 78, 2],
            ["M", 100, 5],
        ]
    )

    types = np.array(
        [DataType.CATEGORICAL_NOMINAL, DataType.RATIO_SCALE, DataType.RATIO_SCALE]
    )
    gower = GowerMetric(types)
    gower.fit(data)

    # ------------------ Testowanie ------------------

    # Macierz odległości
    gower_matrix = gower.make_matrix(data)

    print("\t\t\t", *data)
    for e, lane in enumerate(gower_matrix):
        print(data[e], end=' ')
        for val in lane:
            print('{:.4f}'.format(val), end=',\t\t ')
        print("")

    # Pojedyncza odległość
    # print(gower(data[2], data[8]))
