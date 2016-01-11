# -*- coding: utf-8 -*-

__author__ = 'kikohs'

import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.preprocessing import normalize


def convert_adjacency_matrix(W):
    G = W
    if isinstance(W, sp.sparse.base.spmatrix):
        G = nx.from_scipy_sparse_matrix(W)
    assert(isinstance(G, nx.Graph))
    # Remove self-edges
    G.remove_edges_from(G.selfloop_edges())
    return G


def create_double_stochastic_matrix(a, stop_criterion=1e-6, nb_iter_max=1, norm='l2'):
    c = sp.sparse.csr_matrix(a, copy=True)
    stop = False
    nb_iter = 0

    while not stop and nb_iter < nb_iter_max:
        old_c = c
        c = normalize(c, norm=norm, axis=0)
        c = normalize(c, norm=norm, axis=1)
        # frobenius norm of difference, sparse matrix
        t = c.data - old_c.data
        delta = np.sqrt(np.sum(t * t))
        if delta < stop_criterion and nb_iter > 2:
            stop = True
        nb_iter += 1
    return c


def plot_factor_mat(img, title='', cmap=None):
    plt.title(title)
    cm = None
    if cmap is not None:
        cm = plt.get_cmap(cmap)
    h = plt.imshow(img, interpolation='nearest', aspect='auto', cmap=cm, vmin=0, vmax=1)
    h.axes.set_position([0, 0, 1, 1])
    h.axes.set_xlim(-1, img.shape[1])
    h.axes.set_ylim(img.shape[0], -1)
