import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    #print(scores.shape)
    correct_class_score = scores[y[i]]
    #print(y[i], type(y[i]), np.unique(y) )
    #break
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T 
        dW[:,y[i]] += -X[i].T
        #print(X[i].shape,dW[:,j].shape, dW.shape )

    #W=dW
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  
  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  #print(scores.shape)
  #print(scores[range(num_train), list(y)].shape)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1) #(N, 1)
  #print(correct_class_scores.shape)
  margins = np.maximum(0, scores - correct_class_scores + 1)
  #print(margins.shape)
  margins[range(num_train), list(y)] = 0
  loss = np.sum(margins)/num_train + reg*np.sum(W * W)

  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1) #http://cs231n.github.io/optimization-1/
  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + 2*reg*W

  return loss, dW
