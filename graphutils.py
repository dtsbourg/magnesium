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

graph.py : Some graph related utilies

"""

import numpy as np

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
