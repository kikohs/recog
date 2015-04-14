# -*- coding: utf-8 -*-

__author__ = 'kikohs'

import os
import time
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
import scipy.sparse
import itertools
import math

from sklearn.preprocessing import normalize
# project imports
import utils


def soft_thresholding(data, value, substitute=0):
    mvalue = -value
    cond_less = np.less(data, value)
    cond_greater = np.greater(data, mvalue)
    data = np.where(cond_less & cond_greater, substitute, data)
    data = np.where(cond_less, data + value, data)
    data = np.where(cond_greater, data - value, data)
    return data


def create_recommendation_matrix(a_df, b_idx, gb_key, dataset_name, normalize=True, stop_criterion=1e-2, max_iter=1):
    # Create empty sparse matrix
    c = sp.sparse.dok_matrix((len(a_df), len(b_idx)), dtype=np.float64)
    # fill each playlist (row) with corresponding songs
    for i, (_, gb) in itertools.izip(itertools.count(), a_df.iterrows()):
        for b_id in gb[dataset_name + '_id']:
            # Get mapping form b_id to index position
            j = b_idx.get_loc(b_id)
            c[i, j] = 1.0
    # Convert to Compressed sparse row for speed
    c = sp.sparse.csr_matrix(c)
    if normalize:
        c = utils.create_double_stochastic_matrix(c, stop_criterion, max_iter)
    return c


def init_factor_matrices(nb_row, nb_col, rank):
    a = np.random.random((nb_row, rank))
    b = np.random.random((rank, nb_col))

    a = normalize(a, norm='l1', axis=0)
    b = normalize(b, norm='l1', axis=1)
    return a, b


def graph_gradient_operator(g, key='weight'):
    k = sp.sparse.dok_matrix((g.number_of_edges(), g.number_of_nodes()))
    for i, (src, tgt, data) in itertools.izip(itertools.count(), g.edges_iter(data=True)):
        k[i, src] = data[key]
        k[i, tgt] = -data[key]
    return sp.sparse.csr_matrix(k)


def update_step(theta_tv_a, theta_tv_b, a, b, ka, norm_ka, kb, norm_kb, omega, oc, stop_criterion, min_iter):
    b, a = update_factor(theta_tv_b, b, a, kb, norm_kb, omega, oc, stop_criterion, min_iter)
    a, b = update_factor(theta_tv_a, a.T, b.T, ka, norm_ka, omega.T, oc.T, stop_criterion, min_iter)
    a, b = a.T, b.T
    return a, b


def update_factor(theta_tv, X, Y, K, normK, omega, OC,
                  stop_criterion, min_iter=70, nb_iter_max=400):
    # L2-norm of columns
    # X = (X.T / np.linalg.norm(X, axis=1)).T
    # Y = Y / np.linalg.norm(Y, axis=0)

    # Primal variable
    Xb = X
    Xold = X
    # First dual variable
    P1 = Y.dot(X)
    # Second dual variable
    P2 = K.dot(X.T)

    # 2-norm largest singular value
    normY = np.linalg.norm(Y, 2)
    print 'norm Y:', normY
    # TODO remove
    normY = 10.0

    # Primal-dual parameters
    gamma1 = 1e-1
    gamma2 = 1e-1

    # init time-steps
    sigma1 = 1.0 / normY
    tau1 = 1.0 / normY
    sigma2 = 1.0 / normK
    tau2 = 1.0 / normK

    stop = False
    nb_iter = 0
    while not stop and nb_iter < nb_iter_max:
        # update P1 (NMF part)
        P1 += sigma1 * Y.dot(Xb)
        t = np.square((P1 - omega)) + 4 * sigma1 * OC
        P1 = 0.5 * (P1 + omega - np.sqrt(t))

        # update P2 (TV)
        P2 += sigma2 * K.dot(Xb.T)
        P2 -= sigma2 * soft_thresholding(P2 / sigma2, theta_tv / sigma2)

        # new primal variable
        X = X - tau1 * (Y.T.dot(P1)) - tau2 * (K.T.dot(P2)).T

        # set negative values to 0 (element wise)
        X = np.maximum(X, 0)

        # Acceleration, update time-steps
        # theta1 = 1. / np.sqrt(1 + 2 * gamma1 * tau1)
        # tau1 = tau1 * theta1
        # sigma1 = sigma1 / theta1
        # theta2 = 1. / np.sqrt(1 + 2 * gamma2 * tau2)
        # tau2 = tau2 * theta2
        # sigma2 = sigma2 / theta2
        theta1 = 1.0
        theta2 = 1.0

        # update primal variable for acceleration
        t = X - Xold
        Xb = X + 0.5 * theta1 * t + 0.5 * theta2 * t

        delta = np.linalg.norm(t)
        if math.isnan(delta) or delta <= 1e-9:
            # No enough iterations
            delta = 1

        if delta <= stop_criterion and nb_iter > min_iter:
            stop = True

        # update Xold
        Xold = X
        nb_iter += 1

    return X, Y


def proximal_training(C, WA, WB, rank, O=None, theta_tv_a=1e-4*5,
                      theta_tv_b=1e-4*5,
                      nb_iter_max=30, nb_min_iter=10, stop_criterion=1e-2,
                      stop_criterion_inner=5e-4, min_iter_inner=70, verbose=False):
    start = time.time()
    GA = utils.convert_adjacency_matrix(WA)
    GB = utils.convert_adjacency_matrix(WB)

    A, B = init_factor_matrices(C.shape[0], C.shape[1], rank)

    KA = graph_gradient_operator(GA)
    KB = graph_gradient_operator(GB)

    # For sparse matrix
    _, normKA, _ = sp.sparse.linalg.svds(KA, 1)
    _, normKB, _ = sp.sparse.linalg.svds(KB, 1)
    normKA = normKA[0]
    normKB = normKB[0]

    if O is None:  # no observation mask
        O = np.ones(C.shape)
        OC = C.copy()
    else:
        # Mask over rating matrix, computed once
        OC = O * C

    stop = False
    nb_iter = 0
    delta = 0
    error = False
    while not stop and nb_iter < nb_iter_max:
        old_A = A
        old_B = B
        A, B = update_step(theta_tv_a, theta_tv_b, A, B, KA, normKA, KB, normKB, O, OC,
                           stop_criterion_inner, min_iter_inner)
        nb_iter += 1

        deltaA = np.linalg.norm(A - old_A) / np.linalg.norm(A)
        deltaB = np.linalg.norm(B - old_B) / np.linalg.norm(B)

        if math.isnan(delta):
            stop = True
            error = True

        if verbose:
            print 'Step:', nb_iter, 'err:', np.linalg.norm(C - A.dot(B))
            print 'Delta A:', deltaA
            print 'Delta B:', deltaB
            print

        if deltaB <= stop_criterion and nb_iter > nb_min_iter:
            stop = True

    if not error:
        if delta <= stop_criterion:
            print 'Converged in', nb_iter, 'steps,', 'reconstruction error:', np.linalg.norm(C - A.dot(B))
        else:
            print 'Max iterations reached, did not converged after', nb_iter, 'steps,', \
                'reconstruction error:', np.linalg.norm(C - A.dot(B))
    else:
        print 'Error: try to increase min_iter_inner, the number of iteration for the inner loop'

    print 'Total elapsed time:', time.time() - start, 'seconds'
    return np.array(A), np.array(B)


def recommend(B, keypoints, k, idmap=None, threshold=1e-3):
    """Keypoints: list of tuple (movie, rating) or (song, rating), idmap: if given maps idspace to index in matrix"""
    rank = B.shape[0]
    length = B.shape[1]

    mask = np.zeros(length)
    if idmap is not None:
        mask_idx = map(lambda x: idmap[x[0]], keypoints)
    else:
        mask_idx = map(lambda x: x[0], keypoints)
    mask[mask_idx] = 1.0
    mask = np.diag(mask)

    ratings = np.zeros(length)
    ratings[mask_idx] = map(lambda x: x[1], keypoints)

    z = B.dot(mask).dot(B.T) + 1e-3 * np.eye(rank)
    q = B.dot(mask).dot(ratings)

    # Results
    t = np.linalg.solve(z, q)
    raw = np.array(t.T.dot(B))

    # Filter numeric errors
    mask = raw > threshold
    points = raw[mask]
    # Get valid subset of songs
    position = np.arange(len(mask))[mask]
    # Get unsorted subset index of k highest values
    ind = np.argpartition(points, -k)[-k:]
    # Get sorted subset index of highest values
    ind = ind[np.argsort(points[ind])]
    # map subset index to global position of k elements of highest value
    elems = position[ind]
    return elems, raw
