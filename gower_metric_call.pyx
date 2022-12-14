import cython

import numpy as np
cimport numpy as np

def gower_metric_call(np.ndarray[np.float_t, ndim=1] vector_1,
                      np.ndarray[np.float_t, ndim=1] vector_2,
                      np.ndarray[np.float_t, ndim=1] weights,
                      int cat_nom_num,
                      int bin_asym_num,
                      int ratio_scale_num,
                      cat_nom_idx: np.ndarray,
                      bin_asym_idx: np.ndarray,
                      ratio_scale_idx: np.ndarray,
                      ratio_scale_normalization: str,
                      np.ndarray[np.float_t, ndim=1] ranges_,
                      np.ndarray[np.float_t, ndim=1] h_,
                      int n_features_in_
):

    cdef np.ndarray cat_nom_cols_1
    cdef np.ndarray cat_nom_cols_2
    cdef np.ndarray cat_nom_dist_vec
    cdef np.float_t cat_nom_dist

    cdef np.ndarray bin_asym_cols_1
    cdef np.ndarray bin_asym_cols_2
    cdef np.ndarray bin_asym_dist_vec
    cdef np.float_t bin_asym_dist

    cdef np.ndarray ratio_scale_cols_1
    cdef np.ndarray ratio_scale_cols_2
    cdef np.ndarray ratio_dist_vec
    cdef np.float_t ratio_dist

    cdef np.ndarray above_threshold
    cdef np.ndarray below_threshold

    cdef np.float_t distance

    if cat_nom_num > 0:
        cat_nom_cols_1 = vector_1[cat_nom_idx]
        cat_nom_cols_2 = vector_2[cat_nom_idx]
        cat_nom_dist_vec = 1.0 - (cat_nom_cols_1 == cat_nom_cols_2)

        if weights is not None:
            cat_nom_dist = cat_nom_dist_vec @ weights[cat_nom_idx]
        else:
            cat_nom_dist = cat_nom_dist_vec.sum()
    else:
        cat_nom_dist = 0.0

    if bin_asym_num > 0:
        bin_asym_cols_1 = vector_1[bin_asym_idx]
        bin_asym_cols_2 = vector_2[bin_asym_idx]

        # 0 if x1 == x2 == 1 or x1 != x2, so it's same as 1 if x1 == x2 == 0
        bin_asym_dist_vec = (bin_asym_cols_1 == 0) & (bin_asym_cols_2 == 0)

        if weights is not None:
            bin_asym_dist = bin_asym_dist_vec @ weights[bin_asym_idx]
        else:
            bin_asym_dist = bin_asym_dist_vec.sum()
    else:
        bin_asym_dist = 0.0

    if ratio_scale_num > 0:
        ratio_scale_cols_1 = vector_1[ratio_scale_idx]
        ratio_scale_cols_2 = vector_2[ratio_scale_idx]
        ratio_dist_vec = np.abs(ratio_scale_cols_1 - ratio_scale_cols_2)

        if ratio_scale_normalization == "kde":
            above_threshold = ratio_dist_vec >= ranges_
            below_threshold = ratio_dist_vec <= h_

        ratio_dist_vec = ratio_dist_vec / ranges_

        if ratio_scale_normalization == "kde":
            ratio_dist_vec[above_threshold] = 1.0
            ratio_dist_vec[below_threshold] = 0.0

        if weights is not None:
            ratio_dist = ratio_dist_vec @ weights[ratio_scale_idx]
        else:
            ratio_dist = ratio_dist_vec.sum()
    else:
        ratio_dist = 0.0

    distance = cat_nom_dist + bin_asym_dist + ratio_dist

    # Normalization
    distance /= n_features_in_

    return distance