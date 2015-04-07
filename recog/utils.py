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


def convert_adjacency_matrix(W):
    G = W
    if isinstance(W, sp.sparse.base.spmatrix):
        G = nx.from_scipy_sparse_matrix(W)
    assert(isinstance(G, nx.Graph))
    # Remove self-edges
    G.remove_edges_from(G.selfloop_edges())
    return G


def plot_factor_mat(img):
    h = plt.imshow(img, interpolation='nearest', aspect='auto')
    h.axes.set_position([0, 0, 1, 1])
    h.axes.set_xlim(-1, img.shape[1])
    h.axes.set_ylim(img.shape[0], -1)
