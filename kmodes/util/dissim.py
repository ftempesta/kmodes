"""
Dissimilarity measures for clustering
"""

import numpy as np


def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    return np.sum(a != b, axis=1)


def jaccard_dissim_binary(a, b, **__):
    """Jaccard dissimilarity function for binary encoded variables"""
    if ((a == 0) | (a == 1)).all() and ((b == 0) | (b == 1)).all():
        numerator = np.sum(np.bitwise_and(a, b), axis=1)
        denominator = np.sum(np.bitwise_or(a, b), axis=1)
        if (denominator == 0).any(0):
            raise ValueError("Insufficient Number of data since union is 0")
        else:
            return 1 - numerator / denominator
    raise ValueError("Missing or non Binary values detected in Binary columns.")


def jaccard_dissim_label(a, b, **__):
    """Jaccard dissimilarity function for label encoded variables"""
    if np.isnan(a.astype('float64')).any() or np.isnan(b.astype('float64')).any():
        raise ValueError("Missing values detected in Numeric columns.")
    intersect_len = np.empty(len(a), dtype=int)
    union_len = np.empty(len(a), dtype=int)
    i = 0
    for row in a:
        intersect_len[i] = len(np.intersect1d(row, b))
        union_len[i] = len(np.unique(row)) + len(np.unique(b)) - intersect_len[i]
        i += 1
    if (union_len == 0).any():
        raise ValueError("Insufficient Number of data since union is 0")
    return 1 - intersect_len / union_len


def euclidean_dissim(a, b, **_):
    """Euclidean distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)

def chebyshev_dissim(a,b, **_):
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum(max(a - b), axis=1)

def mahalanobis(a=None, X=None, cov=None):
    a_minus_mean = a - np.mean(X)
    if not cov:
        cov = np.cov(X.values.T)
    inv_cov = sp.linalg.inv(cov)
    left_term = np.dot(a_minus_mean, inv_cov)
    mahala = np.dot(left_term, a_minus_mean.T)
    return mahala.diagonal()

#####
def covar(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    Cov_numerator = sum(((a - x_mean)*(b - y_mean)) for a, b in zip(x, y))
    Cov_denomerator = len(x) - 1
    Covariance = (Cov_numerator / Cov_denomerator)
    return  Covariance

def mahalanobis_dissim(x, y, cov=None):
    x_mean = np.mean(x)
    x_minus_mn_with_transpose =np.transpose(x- x_mean)
    Covariance = covar(x, np.transpose(y))
    inv_covmat = np.linalg.inv(Covariance)
    x_minus_mn = x - x_mean
    left_term = (x_minus_mn, inv_covmat)
    D_square = np.dot(left_term, x_minus_mn_with_transpose)
    return D_square

from scipy.spatial import distance

def maha(a,b, df):
    VI = df.cov()
    return distance.mahalanobis(a,b,VI)

###
    
    
def ng_dissim(a, b, X=None, membship=None):
    """Ng et al.'s dissimilarity measure, as presented in
    Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
    Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
    January, 2007

    This function can potentially speed up training convergence.

    Note that membship must be a rectangular array such that the
    len(membship) = len(a) and len(membship[i]) = X.shape[1]

    In case of missing membship, this function reverts back to
    matching dissimilarity (e.g., when predicting).
    """
    # Without membership, revert to matching dissimilarity
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(b, X, memj, idr):
        """Num objects w/ category value x_{i,r} for rth attr in jth cluster"""
        xcids = np.where(memj == 1)
        return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

    def calc_dissim(b, X, memj, idr):
        # Size of jth cluster
        cj = float(np.sum(memj))
        return (1.0 - (calc_cjr(b, X, memj, idr) / cj)) if cj != 0.0 else 0.0

    if len(membship) != a.shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError("'membship' must be a rectangular array where "
                         "the number of rows in 'membship' equals the "
                         "number of rows in 'a' and the number of "
                         "columns in 'membship' equals the number of rows in 'X'.")

    return np.array([np.array([calc_dissim(b, X, membship[idj], idr)
                               if b[idr] == t else 1.0
                               for idr, t in enumerate(val_a)]).sum(0)
                     for idj, val_a in enumerate(a)])
