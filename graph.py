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
