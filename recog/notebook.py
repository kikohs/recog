# -*- coding: utf-8 -*-

__author__ = 'kikohs'


import numpy as np
import pandas as pd


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


def playlist_category_score(reco_df, playlist_df, playlist_category, song_id_key, playlist_cat_key):

    if reco_df.empty:
        return 0.0

    p_subset = playlist_df[playlist_df[song_id_key].isin(reco_df.index.values)]
    # If song belong to different categories get current category
    total = 0
    count = 0
    for i, row in p_subset.groupby(song_id_key):
        m = (row[playlist_cat_key] == playlist_category).any()
        if m:
            count += 1
        total += 1
    return count / float(total)


def recommend_from_playlist(playlist, song_df, A, B, playlist_size, idmap, threshold=1e-4, use_both=False):
    # Map song ids to [(song_id1, 1), (song_id2, 1), ...]
    keypoints = map(lambda x: (x, 1), playlist)
    if use_both:
        reco_idx, raw = recommender.recommend2(A, B, keypoints, playlist_size, idmap, threshold)
    else:
        reco_idx, raw = recommender.recommend(B, keypoints, playlist_size, idmap, threshold)
    return song_df.iloc[reco_idx], song_df.loc[playlist], raw


def test_playlists(mix_df, playlist_df, song_df, A, B, playlist_size, idmap, song_id_key,
                   playlist_cat_key, threshold=1e-4, use_both=False):
    results = []
    for mix_id, row in mix_df.iterrows():
        p_category = row[playlist_cat_key]
        playlist = row[song_id_key]
        reco_df, _, _ = recommend_from_playlist(playlist, song_df, A, B, playlist_size, idmap, threshold, use_both)
        score = playlist_category_score(reco_df, playlist_df, p_category, song_id_key, playlist_cat_key)
        results.append({mix_df.index.name: mix_id, playlist_cat_key: p_category, 'score': score})

    return pd.DataFrame(results)


def test_all_categories(selected_categories, playlist_df, song_df, sample_size, A,
                        B, playlist_size, idmap, song_id_key, playlist_cat_key, threshold=1e-4,
                        sample_from_random=False, nb_playlist=1, use_both=False):
    results = []
    for p_category in selected_categories:

        query_cat = p_category
        if sample_from_random:
            query_cat = ''

        playlist_set = [pick_random_sample(playlist_df, query_cat, sample_size, song_id_key)
                        for _ in xrange(nb_playlist)]

        for i, playlist in enumerate(playlist_set):
            reco_df, _, _ = recommend_from_playlist(playlist, song_df, A, B, playlist_size, idmap, threshold, use_both)
            score = playlist_category_score(reco_df, playlist_df, p_category, song_id_key, playlist_cat_key)
            results.append({'mix_id': i, playlist_cat_key: p_category, 'score': score})

    return pd.DataFrame(results)


def sampled_vs_random(nb_laps, selected_categories, playlist_df, song_df, sample_size, A,
                        B, k, idmap, song_id_key, playlist_cat_key, threshold=1e-4, use_both=False):

    results = test_all_categories(selected_categories, playlist_df, song_df, sample_size, A, B, k, idmap,
                                  song_id_key, playlist_cat_key, nb_playlist=nb_laps, use_both=use_both)

    m = results.groupby(playlist_cat_key).agg({'score': np.mean})
    m.rename(columns={'score': 'sampled'}, inplace=True)

    results_random = test_all_categories(selected_categories, playlist_df, song_df,sample_size, A, B, k, idmap,
                                         song_id_key, playlist_cat_key, sample_from_random=True,
                                         nb_playlist=nb_laps, use_both=use_both)

    n = results_random.groupby(playlist_cat_key).agg({'score': np.mean})
    n.rename(columns={'score': 'random'}, inplace=True)

    res = pd.concat([m, n], axis=1)
    res['sampled / random absolute gain'] = (res['sampled'] - res['random']) * 100
    res['sampled / random relative gain'] = ((res['sampled'] / res['random']) - 1) * 100

    return res