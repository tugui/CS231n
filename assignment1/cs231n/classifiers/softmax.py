import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    sum_score = np.sum(np.exp(scores))
    divider = np.exp(correct_class_score) * sum_score
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] += -np.exp(correct_class_score) * (sum_score - np.exp(correct_class_score)) * X[i].T / divider
      else:
        dW[:, j] += np.exp(correct_class_score) * np.exp(scores[j]) * X[i].T / divider
    loss += -np.log(np.exp(correct_class_score) / sum_score)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)

  correct_class_scores = scores[range(num_train), list(y)].reshape(-1, 1)
  sum_score = np.sum(np.exp(scores), axis=1).reshape(-1, 1)
  loss = np.sum(-np.log(np.exp(correct_class_scores) / sum_score)) / num_train

  divider = np.exp(correct_class_scores) * sum_score
  coef = np.exp(scores + correct_class_scores) / divider
  coef[range(num_train), list(y)] = 0

  tmp = np.zeros_like(scores)
  tmp[range(num_train), list(y)] = 1
  correct_class_coef = -np.exp(correct_class_scores) * (sum_score - np.exp(correct_class_scores)) / divider
  coef += correct_class_coef * tmp

  dW = np.dot(coef.T, X).T
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

