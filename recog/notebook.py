# -*- coding: utf-8 -*-

__author__ = 'kikohs'


import numpy as np
import pandas as pd
import itertools
import operator
from collections import defaultdict


# project imports
import recommender


# def create_custom_playlist_df(song_df, nb_mixes=100, playlist_size=10, gb_key='genre'):
#
#     def gen_playlist(elems):
#         res = []
#         for _ in xrange(to_pick_per_genre):
#             res.append(np.random.choice(elems, playlist_size))
#         return res
#
#     to_pick_per_genre = nb_mixes / len(np.unique(song_df[gb_key]))
#
#     raw_df = song_df.reset_index().groupby(gb_key)['aotm_id'].agg(gen_playlist)
#
#     res = []
#     last_mix_id = 0
#     for playlist_category, playlists in raw_df.iteritems():
#         for playlist in playlists:
#             last_mix_id += 1
#             for song in playlist:
#                 d = {'playlist_category': playlist_category,
#                      'mix_id': last_mix_id,
#                      'aotm_id': song
#                 }
#                 res.append(d)
#
#     return pd.DataFrame(res)
#
#
# def pick_playlist_category(mix_df, category_key='playlist_category'):
#     return np.random.choice(np.unique(mix_df[category_key]))

# def pick_random_sample_genre(song_df, genre, sample_size):
#     subset = song_df
#     if genre:
#         subset = song_df[song_df['genre'] == genre]
#     assert(len(subset) > 0)
#     assert(sample_size > 0)
#
#     if sample_size > len(subset):
#         sample_size = len(subset)
#
#     return np.random.choice(subset.index.values, sample_size)


def pick_random_sample(playlist_df, category, sample_size, song_id_key, category_key='playlist_category'):
    subset = playlist_df
    if category:
        subset = playlist_df[playlist_df[category_key] == category]

    assert(len(subset) > 0)
    assert(sample_size > 0)

    if sample_size > len(subset):
        sample_size = len(subset)

    return list(np.random.choice(subset[song_id_key].values, sample_size))


def playlist_key_score(reco_df, playlist_df, target, song_id_key, key):
    if reco_df.empty:
        return 0.0

    p_subset = playlist_df[playlist_df[song_id_key].isin(reco_df.index.values)]
    total = 0
    count = 0
    for i, row in p_subset.groupby(song_id_key):
        m = (row[key] == target).any()
        if m:
            count += 1
        total += 1
    return count / float(total)


def songs_key_score(df, target_key):
    res = df.groupby(target_key).size()
    res.sort(ascending=False)
    if res.empty:
        return 0.0
    score = res.iloc[0] / float(res.sum())
    return score


def genre_topics_score(input_df, reco_df):
    # normalized histogram of genre topics
    q = np.array(input_df['genre_topics'].values.tolist()).sum(axis=0) / len(input_df)
    if reco_df.empty:
        return np.linalg.norm(q, ord=2)

    p = np.array(reco_df['genre_topics'].values.tolist()).sum(axis=0) / len(reco_df)

    # if reco_df.empty:
    #     p = np.empty(1/float(len(q)))
    #     p.fill(len(q))
    # else:
    #     p = np.array(reco_df['genre_topics'].values.tolist()).sum(axis=0) / len(reco_df)
    #
    # # Use KL-divergence
    # result = 0
    # for i in xrange(len(p)):
    #     if p[i] < 1e-10:
    #         continue
    #     result += p[i] * (np.log(p[i]) - np.log(q[i]))

    result = np.linalg.norm(p - q, ord=2)

    return result


def song_graph_distance_score(df, pairs_distance, idmap):
    if len(df) < 2:
        return 1.0

    total = 0
    nb_pairs = 0
    for (u, v) in itertools.combinations(df.index.values, 2):
        src, tgt = idmap[u], idmap[v]
        total += pairs_distance[src][tgt]
        nb_pairs += 1
    return total / float(nb_pairs)


def recommend_score(reco_df, input_df, ptarget, ptarget_key, starget_key, playlist_df,
                    playlist_size, idmap, song_id_key, pair_distances=None, pcat_out_only=False):

    d = dict()
    p_reco_score = playlist_key_score(reco_df, playlist_df, ptarget, song_id_key, ptarget_key)
    if pcat_out_only:
        d[ptarget_key] = ptarget
        d['p_cat_out'] = p_reco_score
        return d

    # p_input_score = playlist_key_score(input_df, playlist_df, ptarget, song_id_key, ptarget_key)
    s_reco_score = songs_key_score(reco_df, starget_key)
    s_input_score = songs_key_score(input_df, starget_key)
    genre_score = genre_topics_score(input_df, reco_df)

    starget_key = 's_{}'.format(starget_key)
    d = {ptarget_key: ptarget,
        'starget_key': starget_key,
        'p_cat_out': p_reco_score,
        # 'p_cat_in': p_input_score,
        # 's_cluster_out': s_reco_score,
        # 's_cluster_in': s_input_score,
        's_cluster': s_reco_score - s_input_score,
        's_genre': genre_score
        }

    if pair_distances is not None:
        input_coherence = song_graph_distance_score(input_df, pair_distances, idmap)
        reco_coherence = song_graph_distance_score(reco_df, pair_distances, idmap)
        d['s_graph_dist_in'] = input_coherence
        d['s_graph_dist_out'] = reco_coherence

    return d


def recommend(songs, song_df, A, B, playlist_size, idmap, top_k_playlists=50, threshold=1e-6):
    # Map song ids to [(song_id1, 1), (song_id2, 1), ...]
    keypoints = map(lambda x: (x, 1), songs)
    reco_idx, _ = recommender.recommend_from_keypoints(A, B, keypoints,
                                                       playlist_size, idmap, threshold, top_k_playlists)
    return song_df.iloc[reco_idx]


# def playlist_graph_only_reco(mix_idx, pgraph, song_id_key, top_k_playlists=50, top_k_songs=30):
#     neighbors = pgraph[mix_idx]
#     res = []
#     for k, v in neighbors.iteritems():
#         res.append((k, v['weight']))
#
#     # Sorted neighbors by edge weight (bigger to smaller)
#     n = min(len(res), top_k_playlists)
#     sorted_neighbors = sorted(res, key=operator.itemgetter(1), reverse=True)[:n]
#
#     # All weight to each song in each playlist
#     hist = defaultdict(lambda: 0)
#     for (mix, weight) in sorted_neighbors:
#         for s in pgraph.node[mix][song_id_key]:
#             hist[s] += weight
#
#     n = min(len(hist), top_k_songs)
#     sorted_songs = sorted(hist.items(), key=operator.itemgetter(1), reverse=True)
#     res = sorted_songs[:n]
#     return map(lambda x: x[0], res)


def recommend_playlist_graph_only(songs, song_df, mix_df_train, playlist_df_train, song_id_key,
                                  playlist_id_key, top_k_playlists=50, top_k_songs=30):
    # Get all playlists with at least one song overlap
    subset = playlist_df_train[playlist_df_train[song_id_key].isin(songs)]
    counts = subset.groupby(playlist_id_key).size()
    # Compute cosine sim
    kept_playlists = pd.Series(index=counts.index)
    for mix, count in counts.iteritems():
        kept_playlists[mix] = count / (np.sqrt(mix_df_train.loc[mix]['size']) * np.sqrt(len(songs)))

    # Sort by closest playlist
    kept_playlists.sort(ascending=False)
    kept_playlists = kept_playlists[:top_k_playlists]

    # Create weighted histogram for each song in all kept playlists
    hist = defaultdict(lambda: 0)
    for mix, weight in kept_playlists.iteritems():
        for s in mix_df_train.loc[mix][song_id_key]:
            hist[s] += weight

    sorted_songs = sorted(hist.items(), key=operator.itemgetter(1), reverse=True)
    res = sorted_songs[:top_k_songs]
    return song_df.loc[map(lambda x: x[0], res)]


