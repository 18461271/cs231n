import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    #print(scores.shape)
    f_yi = scores[y[i]]
    #print(y[i], type(y[i]), np.unique(y) )
    #break
    correct_class = np.exp(f_yi)
    sum_scores = np.sum( [np.exp(scores[j]) for j in range(num_classes)])
    loss_i = -np.log(correct_class/sum_scores)
    #print(loss_i.shape)
    #break
    loss += loss_i
    for j in range(num_classes):
        p = np.exp(scores[j] ) / sum_scores
        dW[:,j] += (  p - (j==y[i]))*X[i]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss = loss/num_train
  dW /=num_train
  
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  
  exp_scores = np.exp(scores)
  
  #correct_class_scores = exp_scores[range(num_train), list(y)].reshape(-1,1) #(N, 1) 
  
  row_sum = np.sum( np.exp(scores), axis=1).reshape((num_train,1))
  #print("row_sum",row_sum)
  norm_exp_scores =  exp_scores/ row_sum
  
  data_loss = -np.sum( np.log( norm_exp_scores[range(num_train), y]))
  
  loss = data_loss/num_train + reg*np.sum(W * W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  norm_exp_scores[range(num_train), y] -= 1
  #coeff_mat = np.exp( scores )/row_sum
  
  dW = (X.T).dot(norm_exp_scores)
  dW = dW/num_train + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

