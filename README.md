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

### Ratio scale improvements

#### Outliers compensation
Problem: Outliers in numerical variables affect their contribution to the overall dissimilarity.  
Solution: replace $R_k$ with $IQR_k$, which is the Inter-Quartile Range (P75% - P25%), or even Inter-Decile  

$s_{ijk} = 1 - \frac{|x_{ik} - x_{jk}|}{IQR_{k}}$ if $|x_{ik} - x_{jk}| < IQR_{k}$,

otherwise $s_{ijk} = 0$

#### Categorical variables dominance reduction
Problem: The Gower distance tends to treat units with the same categorical values as closer, giving less importance to the distance on ratio scaled variables.  
Solution: Discretizing the ratio scaled variables, preferring non-fixed discretization scheme, like *Kernel Density Estimation*.  

Two observations within the same moving "window" will have the similarity of 1, whereas for units where one or both fall outside the window, the similarity will be computed as usual:

$s_{ijk} = 1$ if $|x_{ik} - x_{jk}| \leq h_k$, otherwise:  
$s_{ijk} = \frac{|x_{ik} - x_{jk}|}{g_k}$, if $h_k < |x_{ik} - x_{jk}| < g_k$, or  
$s_{ijk} = 0$, if $|x_{ik} - x_{jk}| \geq g_k$

Where:
- $g_k$ can be the discussed previously $IQR_k$, or just the range of values at position $k$,
- $h_k$ is the KDE's bandwidth which can be estimated by following methods:
    - Silverman: $h_k = \frac{c}{\sqrt[5]{n}}min(std_k, \frac{IQR_k}{1.34})$
    - Scott: $h_k = \frac{c}{\sqrt[5]{n}}std_k$
    - Sheather-Jones minimizes asymptotic MEAN Integrated Square Error: $MISE(h) = E\int(\hat{f}(x;h) - f(x))^2dx$

Besides one can find the best fitting values using Grid Search or Optuna framework.

### Weights optimization

#### Cophenetic Correlation Coefficient (CPCC)
The problem of finding the optimal weights is expressed as follows:

How should one select the weights for Gower’s distance metric in order to optimize the Cophenetic Correlation Coefficient (CPCC) of the resulting hierarchical clustering?

Implementation based on `scipy.cluster.hierarchy.cophenet` which uses the formula:

$$C = \frac{\sum_{i<j}(x(i,j)-\bar{x})(t(i,j)-\bar{t})}{\sqrt{(\sum_{i<j}(x(i,j)-\bar{x})^2)(\sum_{i<j}(t(i,j)-\bar{t})^2)}}$$

where:  
$x(i,j)$ - Gower's distance between values at $i$ and $j$, with the global mean equals to $\bar{x}$  
$t(i,j)$ - cophenetic distance between values at $i$ and $j$, with the global meane equalts to $\bar{t}$

CPCC is differentiable, hence it's optimized using *L-BFGS-B*

#### Index of Agreement (IoA)

Shows how accurate a model fits the actual data

$$IoA = 1 - \frac{\sum_{i=1}^n(O_i-P_i)^2}{\sum_{i=1}^n(|P_i - \bar{O}| + |O_i - \bar{O}|)^2}$$

where:  
$P_i$ - predicted value of $i$, in our case Gower's distance  
$O_i$ - observed value of $i$, in our case cophenetic distance from hierarchical clustering  
$\bar{O}$ - mean of observed values  
$n$ - number of observations

### Unknown values handling

When a non-existing value is found, the implementation performs one of the following actions, depending on the user's choice:
- raise an exception
- omit the value
- set it to the maximal distance

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

---

### - kNN

#### Adult Dataset
| Metric    | KNN Score   | F1        |
|-----------|-------------|-----------|
| gower     | 0.7840      | 0.7759    |
| euclidean | 0.7780      | 0.7432    |
| cosine    | **0.7950**  | **0.7886**|
| minkowski | 0.7490      | 0.7050    |
| dice      | 0.7270      | 0.7418    |
| jaccard   | 0.7450      | 0.7620    |

---

#### Car Insurance Claim Dataset
| Metric    | KNN Score   | F1        |
|-----------|-------------|-----------|
| gower     | **0.7420**  | **0.7278**|
| euclidean | 0.7060      | 0.6675    |
| cosine    | 0.7000      | 0.6643    |
| minkowski | 0.6980      | 0.6586    |
| dice      | 0.7270      | 0.6999    |
| jaccard   | 0.6940      | 0.6922    |

---

#### Diabetes Dataset
| Metric    | KNN Score    | F1        |
|-----------|--------------|-----------|
| gower     | 0.6883       | 0.6272    |
| euclidean | 0.6883       | 0.6839    |
| cosine    | 0.7208       | 0.7202    |
| minkowski | **0.7727**   | **0.7733**|
| dice      | 0.6558       | 0.5195    |
| jaccard   | 0.6039       | 0.4548    |

---

### - Hierarchical clustering

#### Adult Dataset
| Metric    | Rand           | Complete         | F-M            | Mutual         | CPCC           | IOA            |
|-----------|----------------|------------------|----------------|----------------|----------------|----------------|
| gower     | 0.6333         | 0.0399           | 0.7948         | 0.0007         | 0.7162         | 0.8187         |
| euclidean | 0.6341         | 0.0184           | 0.7947         | 0.0005         | 0.7719         | 0.8599         |
| cosine    | 0.6353         | **0.1966**       | 0.7930         | **0.0103**     | **0.9103**     | **0.9509**     |
| minkowski | 0.6237         | 0.0045           | 0.7858         | 0.0002         | 0.7851         | 0.8702         |
| dice      | 0.5900         | 0.0195           | 0.7421         | 0.0043         | 0.8598         | 0.9211         |
| jaccard   | **0.6402**     | 0.0295           | **0.7998**     | 0.0002         | 0.8700         | 0.9276         |

---

#### Car Insurance Claim Dataset
| Metric    | Rand           | Complete         | F-M            | Mutual         | CPCC           | IOA            |
|-----------|----------------|------------------|----------------|----------------|----------------|----------------|
| gower     | 0.6096         | 0.0350           | 0.7691         | 0.0041         | 0.5539         | 0.6777         |
| euclidean | 0.4712         | 0.0003           | 0.4740         | 0.0003         | 0.6112         | 0.7316         |
| cosine    | **0.6140**     | **0.1100**       | **0.7833**     | 0.0006         | 0.6354         | 0.7523         |
| minkowski | 0.4680         | 0.0003           | 0.4639         | 0.0003         | 0.6306         | 0.7476         |
| dice      | 0.5962         | 0.0299           | 0.7149         | **0.0117**     | 0.6329         | 0.7523         |
| jaccard   | 0.6098         | 0.0193           | 0.7796         | 0.0004         | **0.6394**     | **0.7564**     |

---

#### Diabetes Dataset
| Metric    | Rand           | Complete         | F-M            | Mutual         | CPCC           | IOA            |
|-----------|----------------|------------------|----------------|----------------|----------------|----------------|
| gower     | 0.5371         | 0.0077           | **0.6928**     | 0.0021         | 0.6359         | 0.7490         |
| euclidean | **0.5530**     | **0.0318**       | 0.6903         | **0.0103**     | **0.8456**     | **0.9105**     |
| cosine    | 0.4994         | 0.0007           | 0.5186         | 0.0005         | 0.7340         | 0.8320         |
| minkowski | **0.5530**     | **0.0318**       | 0.6903         | **0.0103**     | **0.8456**     | **0.9105**     |
| dice      | 0.5455         | 1.0000           | 0.7386         | 0.0000         | 0.0000         | 0.0000         |
| jaccard   | 0.5455         | 1.0000           | 0.7386         | 0.0000         | 0.0000         | 0.0000         |

---

### - HDBSCAN

#### Adult Dataset
| Metric    | Rand           | Complete         | F-M            | Mutual         |
|-----------|----------------|------------------|----------------|----------------|
| gower     | 0.3983         | 0.0277           | 0.2850         | 0.0963         |
| euclidean | 0.3701         | 0.0127           | 0.1748         | 0.0596         |
| cosine    | **0.5075**     | 0.0316           | **0.5421**     | 0.0436         |
| minkowski | 0.3674         | 0.0131           | 0.1781         | 0.0618         |
| dice      | 0.4260         | **0.0507**       | 0.3718         | **0.1081**     |
| jaccard   | 0.4186         | 0.0476           | 0.3725         | 0.0890         |

---

#### Car Insurance Claim Dataset
| Metric    | Rand           | Complete         | F-M            | Mutual         |
|-----------|----------------|------------------|----------------|----------------|
| gower     | 0.5837         | 0.0104           | 0.6968         | 0.0064         |
| euclidean | 0.4653         | 0.0085           | **0.7806**     | 0.0218         |
| cosine    | **0.6077**     | 0.0095           | 0.7621         | 0.0016         |
| minkowski | 0.4729         | 0.0076           | 0.4812         | 0.0193         |
| dice      | 0.3972         | 0.0198           | 0.1298         | 0.0881         |
| jaccard   | 0.3945         | **0.0211**       | 0.1310         | **0.0931**     |

---

#### Diabetes Dataset
| Metric    | Rand           | Complete         | F-M            | Mutual         |
|-----------|----------------|------------------|----------------|----------------|
| gower     | 0.4996         | 0.0006           | 0.5205         | 0.0004         |
| euclidean | 0.5472         | **0.0193**       | 0.6705         | **0.0091**     |
| cosine    | 0.4848         | 0.0064           | 0.4190         | 0.0087         |
| minkowski | **0.5480**     | 0.0189           | **0.6718**     | 0.0089         |
| dice      | 0.5003         | 0.0035           | 0.5466         | 0.0022         |
| jaccard   | 0.5068         | 0.0013           | 0.5525         | 0.0009         |