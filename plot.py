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

class Plotter():
    def __init__(self, training_losses, test_losses, list_X):
        self.training_losses = training_losses
        self.test_losses = test_losses
        


def plot_loss(train=True, test=True):
    best_iter = (np.where(np.asarray(list_training_loss)==np.min(list_training_loss))[0][0]//num_iter_test)*num_iter_test
    best_pred_error = list_test_pred_error[best_iter//num_iter_test]
    print('Best predictions at iter: %d (error: %f)' % (best_iter, best_pred_error))
    RMSE = np.sqrt(np.square(best_pred_error)/np.sum(Otest))
    print('RMSE: %f' % RMSE)

    fig, ax1 = plt.subplots(figsize=(20,10))

    ax2 = ax1.twinx()
    ax1.plot(np.arange(len(list_training_loss)), list_training_loss, 'g-')
    ax2.plot(np.arange(len(list_test_pred_error))*num_iter_test, list_test_pred_error, 'b-')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Training loss', color='g')
    ax2.set_ylabel('Test loss', color='b')

    plt.savefig('loss.png')

def plot(list_training_loss, list_test_pred_error, list_X, Otest, num_iter_test):


    plt.figure(figsize=(20,10))
    plt.imshow(list_X[best_iter//num_iter_test])
    plt.colorbar()
    plt.savefig('results_10kval_n_iter_50')
