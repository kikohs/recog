# -*- coding: utf-8 -*-

__author__ = 'kikohs'

import os
import sys
import math
import itertools
import time
import operator
import random

import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn import neighbors
from scipy import stats

import ncut


def create_song_graph_impl(feat, n_neighbors, metadata=None, p=1, directed=False, with_eid=False, relabel_nodes=False):
    g = None
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.name = 'Song graph'

    if metadata is not None:
        if isinstance(metadata, pd.DataFrame) or isinstance(metadata, pd.Series):
            m = metadata[metadata.index.isin(feat.index)]
            if relabel_nodes:
                m = m.reset_index()
            g.add_nodes_from(m.iterrows())
        else:
            g.add_nodes_from(metadata)
    else:
        if relabel_nodes:
            f = pd.DataFrame(index=feat.index).reset_index()
            g.add_nodes_from(f.iterrows())
        else:
            g.add_nodes_from(feat.index.values)

    # one more neighbors because we are skipping the self comparison
    near = neighbors.NearestNeighbors(n_neighbors + 1, algorithm='auto', p=p).fit(feat.values)
    distances, indices = near.kneighbors(feat)
    # Mean of the the kth nearest neighbors for each data point
    sigma = np.mean(distances[:, -1])

    # normalize distance for edge weights
    t = distances / sigma
    distances = np.exp(-(t * t))

    if relabel_nodes:
        feat = feat.reset_index()

    if with_eid:
        for i in xrange(indices.shape[0]):
            src = feat.index[i]
            for j in xrange(1, indices.shape[1]):
                tgt = feat.index[indices[i, j]]
                w = distances[i, j]
                data = dict()
                data['weight'] = w
                if not directed:
                    eid = str(tgt) + '-' + str(src) if src > tgt else str(src) + '-' + str(tgt)
                else:
                    eid = str(src) + '-' + str(tgt)
                data['eid'] = eid
                g.add_edge(src, tgt, data)
    else:
        for i in xrange(indices.shape[0]):
            src = feat.index[i]
            for j in xrange(1, indices.shape[1]):
                tgt = feat.index[indices[i, j]]
                w = distances[i, j]
                g.add_edge(src, tgt, weight=w)
    return g


def create_song_graph(feat_df, song_df, nb_neighbors=10, metadata=['artist_name', 'title', 'genre', 'genre_topic'],
                      ncut_key='genre_topic'):
    """Create song graph and label the nodes according to their N-cut cluster id"""
    start = time.time()
    g = create_song_graph_impl(feat_df, nb_neighbors, song_df[metadata], relabel_nodes=True)
    nb_cuts = len(np.unique(song_df[ncut_key]))
    print 'Ncuts clusters:', nb_cuts
    g, song_df = ncut_labeling(g, song_df, nb_cuts, ncut_key)
    print 'Created in:', time.time() - start, 'seconds \n'
    return g, song_df


def create_playlist_graph_impl(mix_df, playlist_df, playlist_id_key, song_id_key,
                               p_category_key, p_category_weight, max_category_edges, relabel_nodes=True):
    """Each playlist is a node, edges are created using similarity between playlists."""
    g = nx.Graph()
    g.name = 'Playlist graph'

    # Create playlist nodes with all properties
    if relabel_nodes:
        idmap = dict(itertools.izip(mix_df.index.values, itertools.count()))
        mix_df = mix_df.reset_index()
    g.add_nodes_from(mix_df.iterrows())

    for song_id, group in playlist_df.groupby(song_id_key):
        # Get all playlists containing this song
        p = set(group[playlist_id_key].values)

        if len(p) < 2:
            continue

        # Create pairwise combinations to inc edge weights
        for (u, v) in itertools.combinations(p, 2):
            if relabel_nodes:  # map from song_id to idx
                u, v = idmap[u], idmap[v]

            if not g.has_edge(u, v):
                g.add_edge(u, v, count=1)
            else:  # inc count
                g[u][v]['count'] += 1

    # Create or update edges within the same category
    for _, group in mix_df.groupby(p_category_key):
        p = group.index.values
        if len(p) < 2:
            continue
        # index should be unique
        max_pairs = len(p) * (len(p) - 1) / 2
        chosen_pairs = min(int(max_pairs * max_category_edges), max_pairs)
        pairs = random.sample(list(itertools.combinations(p, 2)), chosen_pairs)
        for (u, v) in pairs:
            if not g.has_edge(u, v):
                g.add_edge(u, v, {p_category_key: p_category_weight})
            else:  # songs are common between the two playlists
                g[u][v][p_category_key] = p_category_weight

    # Reweight all edges
    for u, v, d in g.edges_iter(data=True):
        weight = 0
        if 'count' in d:
            cosine_sim = float(g[u][v]['count']) / (np.sqrt(g.node[u]['size']) * np.sqrt(g.node[v]['size']))
            # weight the cosine similarity to keep final weight between 0 and 1
            factor = 1.0 - p_category_weight
            weight += factor * cosine_sim

        if p_category_key in d:
            weight += p_category_weight

        g[u][v]['weight'] = weight

    return g


def create_playlist_graph(mix_df, playlist_df, playlist_id_key, song_id_key, p_category_key,
                          p_category_weight=0.3, max_category_edges=0.1):
    start = time.time()
    g = create_playlist_graph_impl(mix_df, playlist_df, playlist_id_key,
                                   song_id_key, p_category_key, p_category_weight, max_category_edges, True)
    nb_cuts = len(np.unique(mix_df[p_category_key]))
    print 'Ncuts clusters:', nb_cuts
    g, mix_df = ncut_labeling(g, mix_df, nb_cuts, p_category_key)
    print 'Created in:', time.time() - start, 'seconds \n'
    return g, mix_df


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


def groupby_value(d):
    v = defaultdict(list)
    for key, value in sorted(d.iteritems()):
        v[value].append(key)

    return v


def _do_ncut(a, nb_classes):
    assert(nb_classes >= 2)
    # Apply ncut
    eigen_val, eigen_vec = ncut.ncut(a, nb_classes)
    # Coordinates for each point are the 2 first components of the eigenvectors
    x = np.array(eigen_vec[:, 0])
    y = np.array(eigen_vec[:, 1])
    # Create position
    pos = map(lambda x, y: (float(x), float(y)), x, y)

    # discretize to obtain cluster id
    eigenvec_discrete = ncut.discretisation(eigen_vec)
    res = eigenvec_discrete.dot(np.arange(1, nb_classes + 1))
    f = lambda k: k - 1  # remap classes between 0 and 3
    clusters = f(res)

    return clusters, pos


def apply_ncut(g, nb_classes):
    a = np.array(nx.to_numpy_matrix(g))
    clusters, pos = _do_ncut(a, nb_classes)
    for i, (n, d) in enumerate(g.nodes_iter(data=True)):
        d['ncut_id'] = int(clusters[i])
        d['x'] = float(pos[i][0])
        d['y'] = float(pos[i][1])
        d['pos'] = pos[i]


def ncut_purity(g, key='genre'):
    id2cluster = nx.get_node_attributes(g, 'ncut_id')
    id2genre = nx.get_node_attributes(g, key)
    cluster2id = groupby_value(id2cluster)

    cluster2genre = defaultdict(list)
    for k, v in cluster2id.iteritems(): # for each cluster
        for i in v: # for each id in cluster
            cluster2genre[k].append(id2genre[i])

    # Count number of rightly classified points
    good = 0
    clusterid2genre = dict()
    for k, v in cluster2genre.iteritems():
        val, count = stats.mode(np.array(v, dtype='object'))
        good += int(count[0])
        clusterid2genre[k] = val[0]

    return float(good) / g.number_of_nodes()


def ncut_labeling(g, df, nb_cuts, ncut_purity_key):
    apply_ncut(g, nb_cuts)
    # Get ncut_id from graph
    d = nx.get_node_attributes(g, 'ncut_id')
    # relabel nodes from the graph
    mapping_relabel = dict(zip(map(lambda x: x[0], sorted(d.items(), key=operator.itemgetter(1))), itertools.count()))
    g = nx.relabel_nodes(g, mapping_relabel, copy=True)
    # Sort nodes from graph according to their aotm_id, get the ncut_id value
    df['ncut_id'] = map(lambda x: x[1], sorted(d.items(), key=operator.itemgetter(0)))
    df.sort('ncut_id', inplace=True)
    return g, df