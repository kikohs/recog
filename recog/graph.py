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


def create_playlist_graph(mix_df, playlist_df, playlist_id_key, song_id_key, gb_key, gb_key_weight=0.5):
    """Each playlist is a node, edges are created using similarity between playlists."""
    g = nx.Graph()

    # Create playlist nodes with all properties
    g.add_nodes_from(zip(mix_df.index, mix_df.to_dict(orient='records')))

    for song_id, group in playlist_df.groupby(song_id_key):
        # Get all playlists containing this song
        p = set(group[playlist_id_key].values)

        if len(p) < 2:
            continue

        # Create pairwise combinations to inc edge weights
        for (u, v) in itertools.combinations(p, 2):

            if not g.has_edge(u, v):
                g.add_edge(u, v, count=1)
            else:  # inc count
                g[u][v]['count'] += 1

    # Create or update edges within the same category
    for _, group in mix_df.groupby(gb_key):
        p = group.index.values
        if len(p) < 2:
            continue
        # index should be unique
        for (u, v) in itertools.combinations(p, 2):
            if not g.has_edge(u, v):
                g.add_edge(u, v, {gb_key: gb_key_weight})
            else:  # songs are common between the two playlists
                g[u][v][gb_key] = gb_key_weight

    # Reweight all edges
    for u, v, d in g.edges_iter(data=True):
        weight = 0
        if 'count' in d:
            cosine_sim = float(g[u][v]['count']) / (np.sqrt(g.node[u]['size']) * np.sqrt(g.node[v]['size']))
            # weight the cosine similarity to keep final weight between 0 and 1
            factor = 1.0 - gb_key_weight
            weight += factor * cosine_sim

        if gb_key in d:
            weight += gb_key_weight

        g[u][v]['weight'] = weight

    return g