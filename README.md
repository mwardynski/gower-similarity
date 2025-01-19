# GowerMetric

## Introduction
Distance quantifies how far apart two objects are and is synonymous with dissimilarity. Calculating distance between individuals or groups is essential in fields like biology, psychology, ecology, medical diagnosis, and agriculture. It also underpins statistical methods like discriminant analysis, classification, and clustering, as well as machine learning algorithms such as k-nearest neighbor (supervised learning) and k-means clustering (unsupervised learning).

Euclidean distance is the standard measure for continuous variables, while the simple matching coefficient is common for categorical ones. However, real-world data often combines continuous and categorical variables (mixed data). Although extensive research exists for either continuous or categorical data, mixed data poses challenges. Researchers often either treat categorical data as continuous or transform continuous data into categorical, both of which can result in information loss.

To preserve data integrity, tailored formulas for mixed data types are necessary and one of them is Gower's similarity.

Implementation of Gower's Metric in Python.

## Gower characteristics

In 1971, Gower introduced a general similarity coefficient that encompasses several existing measures as special cases, making it adaptable to various scenarios.

Two individuals, $i$ and $j$, can be compared on a variable $k$ and assigned a score $s_{ijk}$​. The similarity between $i$ and $j$ is calculated as the weighted average of these scores across all comparisons:

$$
S_{ij} = \frac{\sum_{k=1}^{p} s_{ijk}\delta_{ijk}}{\sum_{k=1}^{p} \delta_{ijk}}
$$

Let $\delta_{ijk}$​ represent the possibility of making a comparison. Specifically, $\delta_{ijk} = 1$ when variable $k$ can be compared for individuals $i$ and $j$, meaning no missing values exist for both.

The Gower's distance can by calculated in the following way: $d_{ij} = 1 - S_{ij}$, and the scores $s_{ijk}$ as follows:

#### Binary symmetric data:

$s_{ijk} = 1$ if $x_{ik} = x_{jk}$, $0$ otherwise.

#### Binary asymmetric data:

$s_{ijk} = 1$ if $x_{ik} = x_{jk} = 1$, $0$ otherwise.

#### Ratio scale

$s_{ijk} = 1 - \frac{|x_{ik} - x_{jk}|}{R_{k}}$, where $R_k = max(x_k) - min(x_k)$

#### Categorical nominal

$s_{ijk} = 1$, if variable $i$ equals to variable $j$ at $k$-th element, $0$ otherwise.

Additionaly, Gower proposed the inclusion of weights in the similarity coefficient.

## Metric enhancements

### Ordinal variables

The basic implementation of Gower’s distance does not account for ordinal variables. To address this, we can use the solution proposed by Podani in 1999.

$s_{ijk} = 1$ if $r_{ik} = r_{jk}$, otherwise:


$$s_{ijk} = 1 - \frac{r_{ik} - r_{jk} - \frac{T_{ik} - 1}{2} - \frac{T_{jk} - 1}{2}}{max(r_k) - min(r_k) - \frac{T_{max,k} - 1}{2} - \frac{T_{min,k} - 1}{2}}$$

$r_{ik}$ - rank of attribute $k$ at element $i$,  
$T_{ik}$ - the cardinality of elements with equel rank score to elament $i$ at the attribute $k$

Example of calculating rankings and cardinalities:

| Variable's value       | 1   | 2 | 1   | 4 | 1   | 2 | 2 | 1   |
|------------------------|-----|---|-----|---|-----|---|---|-----|
| Variable's rank        | 2.5 | 6 | 2.5 | 8 | 2.5 | 6 | 6 | 2.5 |
| T - rank's cardinality | 4   | 3 | 4   | 1 | 4   | 3 | 3 | 4   |

### Radio scale improvements

#### Outliers compensation
Problem: Outliers in numerical variables affect their contribution to the overall dissimilarity.  
Solution: replace $R_k$ with $IQR_k$, which is the Inter-Quartile Range (P75% - P25%), or even Inter-Decile  

$s_{ijk} = 1 - \frac{|x_{ik} - x_{jk}|}{IQR_{k}}$ if $|x_{ik} - x_{jk}| < IQR_{k}$, otherwise $s_{ijk} = 0$


## How to install

For now, the easiest way to get library is to clone the repository locally:
```bash
git clone https://github.com/mwardynski/gower-similarity.git
```

### Basic usage

In order to import class, which calculate Gower's metric, you need to import it as follows:
```python
from gowermetric.GowerMetric import MyGowerMetric
```


## Results for metrics comparison

Primary, we have focus on the comparison of the Gower's metric with the other metrics. Only three datasets files were used: adult.csv, car_insurance_csv and diabetes.csv. We have used [kNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [hierarchical clustering](https://docs.scipy.org/doc/scipy-1.15.0/reference/cluster.hierarchy.html) and [HDBSCAN](https://pypi.org/project/hdbscan/) algorithms. As background for hiperparameters improvement, we have used [optuna](https://optuna.org) framework. The results are shown in the following tables.

### - kNN
#### Adult Dataset
| Metric     | KNN Score | F1    |
|------------|-----------|-------|
| Gower      | 0.7200     | 0.6885|
| Euclidean  | 0.7000     | 0.6165|
| Cosine     | **0.7500** | **0.7070**|
| Minkowski  | **0.7500** | 0.6686|
| Dice       | **0.7600** | 0.6564|
| Jaccard    | 0.6900     | 0.7007|

---

#### Car Insurance Claim Dataset
| Metric     | KNN Score | F1    |
|------------|-----------|-------|
| Gower      | **0.7800** | **0.7676**|
| Euclidean  | 0.7300     | 0.6939|
| Cosine     | 0.7300     | 0.6783|
| Minkowski  | 0.6400     | 0.6110|
| Dice       | 0.6000     | 0.6122|
| Jaccard    | 0.7500     | 0.7342|

---

#### Diabetes Dataset
| Metric     | KNN Score | F1    |
|------------|-----------|-------|
| Gower      | 0.5900     | 0.4639|
| Euclidean  | **0.8100** | **0.8025**|
| Cosine     | 0.6800     | 0.6726|
| Minkowski  | 0.6500     | 0.6323|
| Dice       | 0.6600     | 0.5248|
| Jaccard    | 0.7100     | 0.5896|

---

### - Hierarchical clustering
#### Adult Dataset
| Metric     | Rand  | Complete | F-M   | Mutual | CPCC  | IOA   |
|------------|-------|----------|-------|--------|-------|-------|
| Gower      | 0.6281 | 0.0503   | 0.7807| 0.0070 | 0.7187| 0.8214|
| Euclidean  | 0.5324 | 0.0036   | 0.6358| 0.0020 | 0.7671| 0.8577|
| Cosine     | 0.6013 | **0.1998**   | 0.7713| 0.0102 | **0.9787**| **0.9891**|
| Minkowski  | **0.6406** | 0.0182   | **0.7889**| 0.0026 | 0.8732| 0.9292|
| Dice       | 0.5837 | 0.0179   | 0.7382| 0.0040 | 0.8575| 0.9196|
| Jaccard    | 0.5888 | 0.0329   | 0.7335| **0.0097** | 0.8540| 0.9173|

---

#### Car Insurance Claim Dataset
| Metric     | Rand  | Complete | F-M   | Mutual | CPCC  | IOA   |
|------------|-------|----------|-------|--------|-------|-------|
| Gower      | 0.4852 | 0.0200   | 0.4827| 0.0242 | 0.5300| 0.6521|
| Euclidean  | 0.6242 | **0.2312**   | **0.7843**| 0.0163 | 0.7883| **0.8709**|
| Cosine     | 0.5699 | 0.0757   | 0.7448| 0.0077 | 0.6582| 0.7719|
| Minkowski  | **0.6243** | 0.1704   | 0.7727| **0.0305** | **0.8209**| 0.8937|
| Dice       | 0.5890 | 0.0095   | 0.7552| 0.0011 | 0.6640| 0.7782|
| Jaccard    | 0.6126 | 0.0412   | 0.7578| 0.0095 | 0.6446| 0.7634|

---

#### Diabetes Dataset
| Metric     | Rand  | Complete | F-M   | Mutual | CPCC  | IOA   |
|------------|-------|----------|-------|--------|-------|-------|
| Gower      | 0.5964 | 0.1029   | 0.6507| 0.0735 | 0.6402| 0.7629|
| Euclidean  | 0.5517 | 0.0531   | **0.6979**| 0.0148 | **0.8347**| **0.9029**|
| Cosine     | 0.4990 | 0.0001   | 0.5197| 0.0001 | 0.7256| 0.8255|
| Minkowski  | **0.5530** | 0.0393   | 0.6894| 0.0126 | 0.8481| 0.9122|
| Dice       | 0.5441 | **1.0000** | 0.7376| 0.0000 | 0.0000| 0.0000|
| Jaccard    | 0.5465 | **1.0000** | 0.7393| 0.0000 | 0.0000| 0.0000|

### - HDBSCAN
#### Adult Dataset
| Metric     | Rand  | Complete | F-M   | Mutual |
|------------|-------|----------|-------|--------|
| Gower      | 0.5280 | 0.0132   | **0.6430** | 0.0095 |
| Euclidean  | 0.3938 | 0.0122   | 0.2411| 0.0344 |
| Cosine     | **0.5427** | **0.0385**   | 0.5893| 0.0451 |
| Minkowski  | 0.3608 | 0.0199   | 0.2190| 0.0561 |
| Dice       | 0.4137 | 0.0555   | 0.3554| **0.1081** |
| Jaccard    | 0.4368 | 0.0376   | 0.3721| 0.0639 |

---

#### Car Insurance Claim Dataset
| Metric     | Rand  | Complete | F-M   | Mutual |
|------------|-------|----------|-------|--------|
| Gower      | 0.6088 | 0.0332   | 0.7211| 0.0178 |
| Euclidean  | **0.6302** | 0.1364   | **0.7806**| 0.0215 |
| Cosine     | 0.5996 | **0.1501**   | 0.7649| 0.0152 |
| Minkowski  | 0.6265 | 0.1089   | 0.7775| 0.0169 |
| Dice       | 0.4108 | 0.0191   | 0.1567| **0.0647** |
| Jaccard    | 0.3794 | 0.0164   | 0.1721| 0.0532 |

---

#### Diabetes Dataset
| Metric     | Rand  | Complete | F-M   | Mutual |
|------------|-------|----------|-------|--------|
| Gower      | 0.5002 | 0.0038   | 0.5505| 0.0025 |
| Euclidean  | 0.5418 | 0.0164   | **0.6559** | 0.0086 |
| Cosine     | 0.4737 | 0.0108   | 0.3799| 0.0158 |
| Minkowski  | **0.5583** | **0.0346** | 0.6524| **0.0221** |
| Dice       | 0.5006 | 0.0020   | 0.5415| 0.0013 |
| Jaccard    | 0.5003 | 0.0026   | 0.5438| 0.0017 |