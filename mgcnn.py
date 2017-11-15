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

mgcnn.py : The interface to the MGCNN class

"""

import joblib
import numpy as np
from scipy.sparse import csgraph
import time
import pandas as pd
import matplotlib.pyplot as plt

from model import Train_test_matrix_completion
from graph import interaction_matrix

# M = ratings
# O = data mask
# Otraining = training mask
# Otest = test mask
# Wrow = user adjacency matrix
# Wcol = movie adjacency matrix
def load_dataset(interactions, user_count=150, item_count=200, split=0.5):
    # interactions = [us,is,rs]
    sz = (user_count, item_count); user_range = len(interactions[0])

    user_idx = interactions[0]
    item_idx = interactions[1]
    ratings  = interactions[2]

    print("Initialising model variables ...")
    M = np.zeros(sz, dtype=np.float)
    O = np.zeros(sz, dtype=np.int)
    Otraining = np.zeros(sz, dtype=np.int)
    Otest = np.zeros(sz, dtype=np.int)

    print("Building dataset ...")
    for j in range(user_range):
        u = user_idx[j]-1; i=item_idx[j]-1;
        M[u,i] = ratings[j]
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

def train(M, Lrow, Lcol, Odata, Otraining, Otest):
    ord_col = 5
    ord_row = 5

    learning_obj = Train_test_matrix_completion(M, Lrow, Lcol, Odata, Otraining, Otest,
                                                order_chebyshev_col = ord_col, order_chebyshev_row = ord_row,
                                                gamma=1e-8, learning_rate=1e-3)

    num_iter_test = 10
    num_total_iter_training = 50

    num_iter = 0

    list_training_loss = list()
    list_training_norm_grad = list()
    list_test_pred_error = list()
    list_predictions = list()
    list_X = list()

    list_training_times = list()
    list_test_times = list()
    list_grad_X = list()

    list_X_evolutions = list()

    num_iter = 0
    for k in range(num_iter, num_total_iter_training):

        tic = time.time()
        _, current_training_loss, norm_grad, X_grad = learning_obj.session.run([learning_obj.optimizer, learning_obj.loss,
                                                                                learning_obj.norm_grad, learning_obj.var_grad])
        training_time = time.time() - tic

        list_training_loss.append(current_training_loss)
        list_training_norm_grad.append(norm_grad)
        list_training_times.append(training_time)

        if (np.mod(num_iter, num_iter_test)==0):
            msg = "[TRN] iter = %03i, cost = %3.2e, |grad| = %.2e (%3.2es)" \
                                        % (num_iter, list_training_loss[-1], list_training_norm_grad[-1], training_time)
            print(msg)

            #Test Code
            tic = time.time()
            pred_error, preds, X = learning_obj.session.run([learning_obj.predictions_error, learning_obj.predictions,
                                                                                 learning_obj.norm_X])
            c_X_evolutions = learning_obj.session.run(learning_obj.list_X)
            list_X_evolutions.append(c_X_evolutions)

            test_time = time.time() - tic

            list_test_pred_error.append(pred_error)
            list_X.append(X)
            list_test_times.append(test_time)
            msg =  "[TST] iter = %03i, cost = %3.2e (%3.2es)" % (num_iter, list_test_pred_error[-1], test_time)
            print(msg)

        num_iter += 1



    best_iter = (np.where(np.asarray(list_training_loss)==np.min(list_training_loss))[0][0]//num_iter_test)*num_iter_test
    best_pred_error = list_test_pred_error[best_iter//num_iter_test]
    print('Best predictions at iter: %d (error: %f)' % (best_iter, best_pred_error))
    RMSE = np.sqrt(np.square(best_pred_error)/np.sum(Otest))
    print('RMSE: %f' % RMSE)

    joblib.dump( list_training_loss, open( "list_training_loss.p", "wb" ) )
    joblib.dump( list_test_pred_error, open( "test_pred_errors.p", "wb" ) )
    joblib.dump( list_X, open( "list_X.p", "wb" ) )
    #joblib.dump( learning_obj, open( "learning_obj.p", "wb" ) )

    return list_training_loss, list_test_pred_error, list_X, learning_obj



def plot(list_training_loss, list_test_pred_error, list_X, Otest, num_iter_test):
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

    plt.figure(figsize=(20,10))
    plt.imshow(list_X[best_iter//num_iter_test])
    plt.colorbar()
    plt.savefig('results_10kval_n_iter_50')
