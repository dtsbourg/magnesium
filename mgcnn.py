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

---

mgcnn.py : The main interface

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
import matplotlib.pyplot as plt

from model import Train_test_matrix_completion

path_dataset = 'data/data_train.csv'

def interaction_matrix(W, O, ax):
    for k in range(O.shape[ax]):
        if ax==0:
            o = O[k,:].copy()
        else:
            o = O[:,k].copy()
        a = o.nonzero()
        for i in a:
            for j in i:
                o_ = o
                o_[j] = 0
                if ax == 0:
                    W[j,:] += o_
                    #W[j,:] = np.logical_or(W[j,:], o_).astype(int)
                else:
                    W[:,j] += o_
                    #W[:,j] = np.logical_or(W[:,j], o_).astype(int)
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
    idx = idx[(idx[0].astype(int)<user_count) & (idx[1].astype(int)<item_count)].reset_index(drop=True)

    user_idx = idx[0].astype(int)
    item_idx = idx[1].astype(int)
    sz = (user_count, item_count)
    user_range = len(idx)

    print("Initialising model variables ...")
    M = np.zeros(sz, dtype=np.float)
    O = np.zeros(sz, dtype=np.int)
    Otraining = np.zeros(sz, dtype=np.int)
    Otest = np.zeros(sz, dtype=np.int)

    print("Building dataset ...")
    for j in range(user_range):
        u = user_idx[j]-1; i=item_idx[j]-1;
        M[u,i] = ratings.Prediction[j]
        O[u,i] = 1

    print("Computing Leave-one-out test split ...")
    for u in user_idx:
        heldout = np.random.choice(list(O[u-1, :].nonzero())[0],1)
        Otest[u-1, heldout] = 1

    Otraining = O - Otest

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

    return M, Lrow, Lcol, O, Otraining, Otest
