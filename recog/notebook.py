# -*- coding: utf-8 -*-

__author__ = 'kikohs'


import numpy as np
import pandas as pd
import itertools


# project imports
import recommender


def create_custom_playlist_df(song_df, nb_mixes=100, playlist_size=10, gb_key='genre'):

    def gen_playlist(elems):
        res = []
        for _ in xrange(to_pick_per_genre):
            res.append(np.random.choice(elems, playlist_size))
        return res

    to_pick_per_genre = nb_mixes / len(np.unique(song_df[gb_key]))

    raw_df = song_df.reset_index().groupby(gb_key)['aotm_id'].agg(gen_playlist)

    res = []
    last_mix_id = 0
    for playlist_category, playlists in raw_df.iteritems():
        for playlist in playlists:
            last_mix_id += 1
            for song in playlist:
                d = {'playlist_category': playlist_category,
                     'mix_id': last_mix_id,
                     'aotm_id': song
                }
                res.append(d)

    return pd.DataFrame(res)


def pick_playlist_category(mix_df, category_key='playlist_category'):
    return np.random.choice(np.unique(mix_df[category_key]))


def pick_random_sample(playlist_df, category, sample_size, song_id_key, category_key='playlist_category'):
    subset = playlist_df
    if category:
        subset = playlist_df[playlist_df[category_key] == category]

    assert(len(subset) > 0)
    assert(sample_size > 0)

    if sample_size > len(subset):
        sample_size = len(subset)

    return list(np.random.choice(subset[song_id_key].values, sample_size))


def pick_random_sample_genre(song_df, genre, sample_size):
    subset = song_df
    if genre:
        subset = song_df[song_df['genre'] == genre]
    assert(len(subset) > 0)
    assert(sample_size > 0)

    if sample_size > len(subset):
        sample_size = len(subset)

    return np.random.choice(subset.index.values, sample_size)


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


def recommend_from_playlist(playlist, song_df, A, B, playlist_size, idmap, threshold=1e-4, use_both=True):
    # Map song ids to [(song_id1, 1), (song_id2, 1), ...]
    keypoints = map(lambda x: (x, 1), playlist)
    if use_both:
        reco_idx, raw = recommender.recommend2(A, B, keypoints, playlist_size, idmap, threshold)
    else:
        reco_idx, raw = recommender.recommend(B, keypoints, playlist_size, idmap, threshold)
    return song_df.iloc[reco_idx], song_df.loc[playlist], raw


def test_all_categories(selected_categories, playlist_df, song_df, sample_size, A,
                        B, playlist_size, idmap, song_id_key, playlist_cat_key, threshold=1e-4,
                        sample_from_random=False, pairs_distance=None, nb_playlist=1, use_both=True):
    results = []
    for p_category in selected_categories:

        query_cat = p_category
        if sample_from_random:
            query_cat = ''

        playlist_set = [pick_random_sample(playlist_df, query_cat, sample_size, song_id_key)
                        for _ in xrange(nb_playlist)]

        for i, playlist in enumerate(playlist_set):
            score = recommend_score(playlist, p_category, playlist_cat_key, 'cluster_id',
                                    playlist_df, song_df, A, B, playlist_size, idmap,
                                    song_id_key, pairs_distance, threshold)
            score['id'] = i
            results.append(score)

    return pd.DataFrame(results)


def sampled_vs_random(nb_laps, selected_categories, playlist_df, song_df, sample_size, A,
                        B, k, idmap, song_id_key, playlist_cat_key, pairs_distance=None, threshold=1e-4, use_both=True):

    results = test_all_categories(selected_categories, playlist_df, song_df, sample_size, A, B, k, idmap,
                                  song_id_key, playlist_cat_key, pairs_distance=pairs_distance, nb_playlist=nb_laps, use_both=use_both)

    m = results.groupby('p_' + playlist_cat_key).mean()
    m.drop('id', axis=1, inplace=True)

    results_random = test_all_categories(selected_categories, playlist_df, song_df,sample_size, A, B, k, idmap,
                                         song_id_key, playlist_cat_key, sample_from_random=True,
                                         pairs_distance=pairs_distance,
                                         nb_playlist=nb_laps, use_both=use_both)

    n = results_random.groupby('p_' + playlist_cat_key).mean()
    n.drop('id', axis=1, inplace=True)

    # res['sampled / random absolute gain'] = (res['sampled'] - res['random']) * 100
    # res['sampled / random relative gain'] = ((res['sampled'] / res['random']) - 1) * 100

    return m, n


def songs_key_score(df, target_key):
    res = df.groupby(target_key).size()
    if res.empty:
        return 0.0
    score = res.iloc[0] / float(res.sum())
    return score


def genre_topics_score(input_df, reco_df):
    # normalized historgram of genre topics
    x = np.array(reco_df['genre_topics'].values.tolist()).sum(axis=0) / len(reco_df)
    y = np.array(input_df['genre_topics'].values.tolist()).sum(axis=0) / len(reco_df)
    return np.linalg.norm(x - y, ord=2)


def song_graph_coherence_score(df, pairs_distance, idmap):
    if len(df) < 2:
        return 1.0

    total = 0
    nb_pairs = 0
    for (u, v) in itertools.combinations(df.index.values, 2):
        src, tgt = idmap[u], idmap[v]
        total += pairs_distance[src][tgt]
        nb_pairs += 1
    return total / float(nb_pairs)


def recommend_score(playlist, ptarget, ptarget_key, starget_key, playlist_df, song_df, A, B,
                    playlist_size, idmap, song_id_key, pairs_distance=None, threshold=1e-4):
    reco_df, input_df, _ = recommend_from_playlist(playlist, song_df, A, B, playlist_size, idmap, threshold, True)
    p_reco_score = playlist_key_score(reco_df, playlist_df, ptarget, song_id_key, ptarget_key)
    p_input_score = playlist_key_score(input_df, playlist_df, ptarget, song_id_key, ptarget_key)
    s_reco_score = songs_key_score(reco_df, starget_key)
    s_input_score = songs_key_score(input_df, starget_key)

    genre_score = genre_topics_score(input_df, reco_df)

    if pairs_distance is not None:
        input_coherence = song_graph_coherence_score(input_df, pairs_distance, idmap)
        reco_coherence = song_graph_coherence_score(reco_df, pairs_distance, idmap)

    ptarget_key = 'p_' + ptarget_key
    starget_key = 's_' + starget_key

    d = {ptarget_key: ptarget,
            'starget_key': starget_key,
            'p_reco': p_reco_score,
            'p_input': p_input_score,
            's_song_key_reco': s_reco_score,
            's_song_key_input': s_input_score,
            's_genre_topics': genre_score
        }

    if pairs_distance is not None:
        d['s_coherence_input'] = input_coherence
        d['s_coherence_reco'] = reco_coherence

    return d


def test_playlists(mix_df, playlist_df, song_df, A, B, playlist_size, idmap, song_id_key,
                   playlist_cat_key, pairs_distance=None, threshold=1e-4, use_both=True):
    results = []
    for mix_id, row in mix_df.iterrows():
        p_category = row[playlist_cat_key]
        playlist = row[song_id_key]
        score = recommend_score(playlist, p_category, playlist_cat_key, 'cluster_id',
                                playlist_df, song_df, A, B, playlist_size, idmap, song_id_key,
                                pairs_distance, threshold)

        score[mix_df.index.name] = mix_id
        results.append(score)
    return pd.DataFrame(results).set_index('mix_id')
