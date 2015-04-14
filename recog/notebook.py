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

    return np.random.choice(subset[song_id_key].values, sample_size)


def pick_random_sample_genre(song_df, genre, sample_size):
    subset = song_df
    if genre:
        subset = song_df[song_df['genre'] == genre]
    assert(len(subset) > 0)
    assert(sample_size > 0)

    if sample_size > len(subset):
        sample_size = len(subset)

    return np.random.choice(subset.index.values, sample_size)


def recommend(playlist_df, song_df, playlist_category, sample_size, B, k, idmap, song_id_key, threshold=1e-6):
    picked_song_ids = pick_random_sample(playlist_df, playlist_category, sample_size, song_id_key)
    keypoints = zip(picked_song_ids, np.ones(len(picked_song_ids)))
    reco_idx, raw = recommender.recommend(B, keypoints, k, idmap, threshold)

    return song_df.iloc[reco_idx], song_df.loc[picked_song_ids], raw


def playlist_category_score(reco_df, playlist_df, playlist_category):
    hist = playlist_df[playlist_df['aotm_id'].isin(reco_df.index.values)]['playlist_category'].value_counts()
    f_hist = hist[hist.index == playlist_category]
    if f_hist.empty:
        return 0.0

    score = f_hist.values[0] / float(np.sum(hist))
    return score