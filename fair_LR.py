# IMPLEMENTATION of Zafar work, directly get it from https://github.com/mbilalzafar/fair-classification
import sys
import os
import numpy as np
import scipy.special
from collections import defaultdict
import traceback
from copy import deepcopy

#MUST INSTALL THESE LIBRARIES TO run DISPARATE_MISTREATMENT
#!pip install cvxpy==0.4.11
#!pip install dccp==0.1.6


def _hinge_loss(w, X, y):
    yz = y * np.dot(X, w)  # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1 - yz))  # hinge function

    return sum(yz)


def _logistic_loss(w, X, y, return_arr=None):
    """Computes the logistic loss.
    This function is used from scikit-learn source code
    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Array of labels.
    """

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    if return_arr == True:
        out = -(log_logistic(yz))
    else:
        out = -np.sum(log_logistic(yz))
    return out


def _logistic_loss_l2_reg(w, X, y, lam=None):
    if lam is None:
        lam = 1.0

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(log_logistic(yz))
    l2_reg = (float(lam) / 2.0) * np.sum([elem * elem for elem in w])
    out = logistic_loss + l2_reg
    return out


def log_logistic(X):
    """ This function is used from scikit-learn source code. Source link below """

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0
    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function
    Returns
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    -----
    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X)  # same dimensions and data types

    idx = X > 0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out


from scipy.optimize import minimize


def get_constraints(model, x_arr, y, z, thresh):
    """

    set constraints, thresh >= abs( X^T*model * (z- mean(z)^T )
    where T is transpose operator
    """
    arr = np.dot(model, x_arr.T)  # the product with the weight vector -- the sign of this is the output label

    arr = np.array(arr, dtype=np.float64)
    cov = np.dot(z - np.mean(z), arr) / float(len(z))
    ans = thresh - abs(cov)
    return ans


def aistat_method(x_train, y_train, z_train, options={'c': 0.0, 'max_iter': 10000}):
    """
    c is the fairness constraint, the smaller the c, the more fairness level we want to achieve.
    """

    constraints = [({'type': 'ineq', 'fun': get_constraints, 'args': (x_train, y_train, z_train, options['c'])})]

    y_train[y_train == 0] = -1
    max_iter = options['max_iter']

    f_args = (x_train, y_train)
    loss_function = _logistic_loss  # can change to hingle loss if we want to consider SVM instead of LR

    x0 = np.random.rand(x_train.shape[1], 1)  # initial point for optimization

    w = minimize(fun=loss_function,
                 x0=x0,
                 args=f_args,
                 method='SLSQP',
                 options={"maxiter": max_iter},
                 constraints=constraints
                 )

    w_new = w.x

    return w_new


####### DISPARATE MISTREATMENT
# REMEMBER TO MODIFY the log_exp_sum, (1) disable first line
# and  (2) change scipy.misc to scipy.special

from __future__ import division
import os, sys
import traceback
import numpy as np
from random import seed, shuffle
from collections import defaultdict
from copy import deepcopy
import dccp
import cvxpy
from cvxpy import Problem, Minimize, Variable, logistic, ECOS

SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def train_model_disp_mist(x, y, x_control, loss_function, EPS, cons_params=None):
    max_iters = 100  # for the convex program
    max_iter_dccp = 50  # for the dccp algo

    num_points, num_features = x.shape
    w = Variable(num_features)  # this is the weight vector

    # initialize a random value of w
    np.random.seed(112233)
    w.value = np.random.rand(x.shape[1])

    if cons_params is None:  # just train a simple classifier, no fairness constraints
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, cons_params["sensitive_attrs_to_cov_thresh"],
                                              cons_params["cons_type"], w)

    if loss_function == "logreg":
        # constructing the logistic loss problem
        loss = cvxpy.sum_entries(logistic(
            cvxpy.mul_elemwise(-y, x * w))) / num_points  # we are converting y to a diagonal matrix for consistent

    # sometimes, its a good idea to give a starting point to the constrained solver
    # this starting point for us is the solution to the unconstrained optimization problem
    # another option of starting point could be any feasible solution
    if cons_params is not None:
        if cons_params.get("take_initial_sol") is None:  # true by default
            take_initial_sol = True
        elif cons_params["take_initial_sol"] == False:
            take_initial_sol = False

        if take_initial_sol == True:  # get the initial solution
            p = Problem(Minimize(loss), [])
            p.solve()

    # construct the cvxpy problem
    prob = Problem(Minimize(loss), constraints)

    try:

        tau, mu = 0.005, 1.2  # default dccp parameters, need to be varied per dataset
        if cons_params is not None:  # in case we passed these parameters as a part of dccp constraints
            if cons_params.get("tau") is not None: tau = cons_params["tau"]
            if cons_params.get("mu") is not None: mu = cons_params["mu"]

        prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10,
                   solver=ECOS, verbose=False,
                   feastol=EPS, abstol=EPS, reltol=EPS, feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
                   max_iters=max_iters, max_iter=max_iter_dccp)

        assert (prob.status == "Converged" or prob.status == "optimal")
        # print "Optimization done, problem status:", prob.status

    except:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)

    # check that the fairness constraint is satisfied
    for f_c in constraints:
        assert (
        f_c.value == True)  # can comment this out if the solver fails too often, but make sure that the constraints are satisfied empirically. alternatively, consider increasing tau parameter
        pass

    w = np.array(w.value).flatten()  # flatten converts it to a 1d array

    return w


def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs_to_cov_thresh, cons_type, w):
    constraints = []
    for attr in sensitive_attrs_to_cov_thresh.keys():

        attr_arr = x_control_train[attr]
        print(attr_arr)
        attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr.astype(int))

        if index_dict is None:  # binary attribute, in this case, the attr_arr_transformed is the same as the attr_arr

            s_val_to_total = {ct: {} for ct in [0, 1, 2]}  # constrain type -> sens_attr_val -> total number
            s_val_to_avg = {ct: {} for ct in [0, 1, 2]}
            cons_sum_dict = {ct: {} for ct in
                             [0, 1, 2]}  # sum of entities (females and males) in constraints are stored here

            for v in set(attr_arr):
                s_val_to_total[0][v] = np.sum(x_control_train[attr] == v)
                s_val_to_total[1][v] = np.sum(np.logical_and(x_control_train[attr] == v,
                                                             y_train == -1))  # FPR constraint so we only consider the ground truth negative dataset for computing the covariance
                s_val_to_total[2][v] = np.sum(np.logical_and(x_control_train[attr] == v, y_train == +1))

            for ct in [0, 1, 2]:
                s_val_to_avg[ct][0] = s_val_to_total[ct][1] / float(s_val_to_total[ct][0] + s_val_to_total[ct][
                    1])  # N1/N in our formulation, differs from one constraint type to another
                s_val_to_avg[ct][1] = 1.0 - s_val_to_avg[ct][0]  # N0/N

            for v in set(attr_arr):
                idx = x_control_train[attr] == v

                #################################################################
                # #DCCP constraints
                dist_bound_prod = cvxpy.mul_elemwise(y_train[idx], x_train[idx] * w)  # y.f(x)

                cons_sum_dict[0][v] = cvxpy.sum_entries(cvxpy.min_elemwise(0, dist_bound_prod)) * (
                s_val_to_avg[0][v] / len(x_train))  # avg misclassification distance from boundary
                cons_sum_dict[1][v] = cvxpy.sum_entries(
                    cvxpy.min_elemwise(0, cvxpy.mul_elemwise((1 - y_train[idx]) / 2.0, dist_bound_prod))) * (
                                      s_val_to_avg[1][v] / sum(y_train == -1))
                cons_sum_dict[2][v] = cvxpy.sum_entries(
                    cvxpy.min_elemwise(0, cvxpy.mul_elemwise((1 + y_train[idx]) / 2.0, dist_bound_prod))) * (
                                      s_val_to_avg[2][v] / sum(y_train == +1))
                #################################################################

            if cons_type == 4:
                cts = [1, 2]
            elif cons_type in [0, 1, 2]:
                cts = [cons_type]

            else:
                raise Exception("Invalid constraint type")

            #################################################################
            # DCCP constraints
            for ct in cts:
                thresh = abs(sensitive_attrs_to_cov_thresh[attr][ct][1] - sensitive_attrs_to_cov_thresh[attr][ct][0])
                constraints.append(cons_sum_dict[ct][1] <= cons_sum_dict[ct][0] + thresh)
                constraints.append(cons_sum_dict[ct][1] >= cons_sum_dict[ct][0] - thresh)

                #################################################################



        else:  # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately
            # need to fill up this part
            raise Exception("Fill the constraint code for categorical sensitive features... Exiting...")
            sys.exit(1)

    return constraints


def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            return None

    in_arr = np.array(in_arr, dtype=int)
    assert (len(in_arr.shape) == 1)  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    index_dict = {}  # value to the column number
    for i in range(0, len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []
    for i in range(0, len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict


def split_into_train_test(x_all, y_all, x_control_all, train_fold_size):
    split_point = int(round(float(x_all.shape[0]) * train_fold_size))
    x_all_train = x_all[:split_point]
    x_all_test = x_all[split_point:]
    y_all_train = y_all[:split_point]
    y_all_test = y_all[split_point:]
    x_control_all_train = {}
    x_control_all_test = {}
    for k in x_control_all.keys():
        x_control_all_train[k] = x_control_all[k][:split_point]
        x_control_all_test[k] = x_control_all[k][split_point:]

    return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test



