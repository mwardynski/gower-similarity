# GowerMetric
Implementation of Gower's Metric in Python.
  
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

## Datasets for testing:
https://drive.google.com/drive/folders/16_VCVOaOOnhilUJkSn6CaLpumHMzd7D-?usp=share_link

## Run pytest
Please feel free to run `pytest -v` from main repo root dir