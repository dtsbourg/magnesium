"""
MGCNN : Multi-Graph Convolutional Neural Networks

The code contained in this repository represents a TensorFlow implementation of the Recurrent Muli-Graph Convolutional Neural Network depicted in:

Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (in Proc. NIPS, 2017)
Federico Monti, Michael M. Bronstein, Xavier Bresson

https://arxiv.org/abs/1704.06803

License : GNU General Public License v3.0
by @fmonti (Frederico Monti)

Modifications : @dtsbourg (Dylan Bourgeois)
This code is an attempt to package the code presented in
https://github.com/fmonti/mgcnn for the Netflix challenge.

"""

import os,sys,inspect
import os
import joblib
import tensorflow as tf
import numpy as np
import h5py
import scipy.sparse.linalg as la
from scipy.sparse import csgraph
import scipy
import time
import pandas as pd

path_dataset = 'data/data_train.csv'

def interaction_matrix(W, O, ax):
    for k in range(O.shape[ax]):
        if ax==0:
            a = O[k,:].nonzero()
        else:
            a = O[:,k].nonzero()
        for i in a:
            for j in i:
                W[j, i[:j]] = 1
                W[j, i[j+1:]] = 1
    return W


def load_dataset(path=path_dataset, user_count=150, item_count=200, split=0.5):
    # M = ratings
    # O = data mask
    # Otraining = training mask
    # Otest = test mask
    # Wrow = user adjacency matrix
    # Wcol = movie adjacency matrix

    print("Loading dataset ...")
    ratings = pd.read_csv(path_dataset, dtype={'Prediction': np.float})
    print("Extracting index ...")
    idx = ratings.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)

    user_idx = idx[0].astype(int); user_idx = np.asarray(user_idx[user_idx <= user_count]);
    item_idx = idx[1].astype(int); item_idx = np.asarray(item_idx[item_idx <= item_count]);
    sz = (user_count, item_count)

    print("Computing test split ...")
    u_test = int(len(user_idx)*split)
    i_test = int(len(item_idx)*split)
    test_idx = list(zip(np.random.choice(user_idx,u_test),np.random.choice(item_idx,i_test)))

    print("Initialising model variables ...")
    M = np.zeros(sz, dtype=np.float)
    O = np.zeros(sz, dtype=np.int)
    Otraining = np.zeros(sz, dtype=np.int)
    Otest = np.zeros(sz, dtype=np.int)

    print("Building dataset ...")
    for j in range(len(user_idx)):
        u = user_idx[j]-1; i=item_idx[j]-1;
        #print(u,i)
        M[u,i] = ratings.Prediction[j]
        O[u,i] = 1

        if np.any(test_idx[:] == (u,i)):
            Otest[u,i] = 1
        else:
            Otraining[u,i] = 1

    print("Building user interaction matrix ...")
    # User interactions
    Wrow = np.zeros((user_count, user_count), dtype=np.int)
    Wrow = interaction_matrix(Wrow, O, 1)

    print("Building item interaction matrix ...")
    # Item interactions
    Wcol = np.zeros((item_count, item_count), dtype=np.int)
    Wcol = interaction_matrix(Wcol, O, 0)

    print("Computing Laplacian of interactions ...")
    Lrow = csgraph.laplacian(Wrow, normed=True)
    Lcol = csgraph.laplacian(Wcol, normed=True)

load_dataset()
