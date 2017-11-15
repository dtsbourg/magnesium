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

plot.py : Some plotting utilies

"""

import matplotlib.pyplot as plt
import numpy as np

class MCPlot():
    def __init__(self, session):
        self.session = session
        self.imgpath = 'res/img/'

    def plot_loss(self, train=True, test=True):
        fig, ax1 = plt.subplots(figsize=(20,10))

        ax2 = ax1.twinx()
        ax1.plot(np.arange(len(self.session.list_training_loss)), self.session.list_training_loss, 'g-')
        ax2.plot(np.arange(len(self.session.list_test_pred_error))*self.session.num_iter_test, self.session.list_test_pred_error, 'b-')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Training loss', color='g')
        ax2.set_ylabel('Test loss', color='b')

        figstr = 'Loss_u_' + str(self.session.graph.sz[0]) + '_i_' + str(self.session.graph.sz[1]) + '_iter_' + str(self.session.num_total_iter_training) + '_RMSE_' + str(self.session.RMSE) + '.png'
        plt.savefig(self.imgpath+figstr)

    def plot_pred(self):
        plt.figure(figsize=(20,10))

        plt.imshow(self.session.list_X[self.session.best_iter//self.session.num_iter_test])
        plt.colorbar()

        figstr = 'Prediction_u_' + str(self.session.graph.sz[0]) + '_i_' + str(self.session.graph.sz[1]) + '_iter_' + str(self.session.num_total_iter_training) + '_RMSE_' + str(self.session.RMSE) + '.png'
        plt.savefig(self.imgpath+figstr)
