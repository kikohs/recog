===============================
Song Recommendation with Non-Negative Matrix Factorization and Graph Total Variation
===============================

A graph audio recommender system

* Free software: GPLv3


Informations
------

This is the repository for the paper: Song Recommendation with Non-Negative Matrix Factorization and Graph Total Variation <http://arxiv.org/abs/1601.01892/> by Kirell Benzi, Vassilis Kalofolias, Xavier Bresson, Pierre Vandergheynst. It will presented in ICASSP 2016.

The code is in a *research state* (read not excellent) but it should work. To learn how to use the code please run the associated notebook. 
You cannot run the cells from 'Test on Real data' to 'Prepare graphs' because the full dataset is not uploaded here. However, the sampled test data is uploaded in **resources**.

If you use the code please cite the paper above, it means a lot to us.

Abstract
--------

This work formulates a novel song recommender system as a matrix completion problem that benefits from collaborative filtering through Non-negative Matrix Factorization (NMF) and content-based filtering via total variation (TV) on graphs. The graphs encode both playlist proximity information and song similarity, using a rich combination of audio, meta-data and social features. As we demonstrate, our hybrid recommendation system is very versatile and incorporates several well-known methods while outperforming them. Particularly, we show on real-world data that our model overcomes w.r.t. two evaluation metrics the recommendation of models solely based on low-rank information, graph-based information or a combination of both.




