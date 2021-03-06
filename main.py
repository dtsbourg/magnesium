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

main.py : The main interface to run MGCNN

"""
import pandas as pd
import numpy as np
import joblib

from mgcnn import MCSession
from plot import MCPlot

def load_netflix(path_dataset, user_count=150, item_count=200):
    print("Loading dataset ...")
    data = pd.read_csv(path_dataset, dtype={'Prediction': np.float})
    print("Extracting index ...")
    idx = data.Id.str.extract('r([0-9]+)_c([0-9]+)', expand=True)
    idx = idx[(idx[0].astype(int)<user_count) & (idx[1].astype(int)<item_count)].reset_index(drop=True)

    user_idx = idx[0].astype(int)
    item_idx = idx[1].astype(int)
    ratings = data.Prediction.astype(float)

    return [user_idx, item_idx, ratings]


if __name__ == '__main__':
    session = MCSession(1000)
    plotter = MCPlot(session)

    path_dataset = 'data/data_train.csv'
    interactions = load_netflix(path_dataset)

    session.load_dataset(interactions)

    session.load_existing = False

    if session.load_existing:
        session.load_saved_model()
    else:
        session.train()
        plotter.plot_loss()
    plotter.plot_pred()
