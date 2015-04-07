# -*- coding: utf-8 -*-

__author__ = 'kikohs'

import os
import sys
import math
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from sklearn import neighbors


def create_song_graph(feat, n_neighbors, metadata=None, p=1, directed=False, with_eid=False):
    """Create song graph from dataframe of feature using k-nearest neighbors.
    The dataframe should be indexed by node_id. The metadata param is either a dataframe
    indexed by node_id or a list of tuple (node_id, attr_dict).
    """
    g = None
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    if metadata is not None:
        if isinstance(metadata, pd.DataFrame):
            m = metadata[metadata.index.isin(feat.index)]
            g.add_nodes_from(m.iterrows())
        else:
            g.add_nodes_from(metadata)

    # one more neighbors because we are skipping the self comparison
    near = neighbors.NearestNeighbors(n_neighbors + 1, algorithm='auto', p=p).fit(feat.values)
    distances, indices = near.kneighbors(feat)
    # Mean of the the kth nearest neighbors for each data point
    sigma = np.mean(distances[:, -1])

    # iter through each node to create edges
    for k, (i, d) in itertools.izip(itertools.count(), itertools.izip(indices, distances)):
        src = feat.index[k]
        # skip self edge
        for j, v in itertools.izip(i[1:], d[1:]):
            tgt = feat.index[j]
            data = dict()
            weight = float(math.exp(-(v / sigma)**2))
            data['weight'] = weight
            if with_eid:
                if not directed:
                    eid = str(tgt) + '-' + str(src) if src > tgt else str(src) + '-' + str(tgt)
                else:
                    eid = str(src) + '-' + str(tgt)
                data['eid'] = eid
            g.add_edge(src, tgt, data)
    return g


def create_playlist_graph():
    # TODO
    pass