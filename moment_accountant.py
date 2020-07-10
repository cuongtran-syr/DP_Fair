import numpy as np
from scipy.stats import norm, laplace
#import cPickle as pickle
from math import exp
import matplotlib
from collections import Counter

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

l_max = 32  # lambda

# Sampling prob. for SGD
q = 100 / 60000.0
# epoch num for SGD
epoch_num = 30

# noise for k-means cluster size
sigma_cl_1 = 15
# noise for k-means cluster centers
sigma_cl_2 = 100
# noise for Gradient descent
sigma_gd = 1.3
# noise for Adaptive clipping in each GD step
sigma_dyn_clip = 4

delta = 10 ** -5
# k-means iterations
T_1 = 10
# SGD iterations
T_2 = epoch_num * int(1.0 / q)
# Total iterations
T = T_1 + T_2

# mu_0 = sigma* N(0,1)
# mu_1 = sigma* N(0,1) + 1
# mu = (1-q)* mu_0 + q*mu_1 = sigma*N(0,1) + q
# Probability is not incorporated when the the expected values are computed

import scipy.integrate as integrate


def compute_gauss(p_sigma, l, u, d=20, p_sens=1.0):
    e_1 = lambda x: norm.pdf(x, 0, p_sens * p_sigma) * (norm.pdf(x, 0, p_sens * p_sigma) / (
    (1 - q) * norm.pdf(x, 0, p_sens * p_sigma) +
    q * norm.pdf(x, p_sens, p_sens * p_sigma))) ** (u * l)
    e_2 = lambda x: ((1 - q) * norm.pdf(x, 0, p_sens * p_sigma) + q * norm.pdf(x, p_sens, p_sens * p_sigma)) * \
                    (((1 - q) * norm.pdf(x, 0, p_sens * p_sigma) + q * norm.pdf(x, p_sens, p_sens * p_sigma)) /
                     norm.pdf(x, 0, p_sens * p_sigma)) ** (u * l)
    E_1, _ = integrate.quad(e_1, -d, d)
    E_2, _ = integrate.quad(e_2, -d, d)
    return np.log(np.maximum(np.abs(E_1), np.abs(E_2)))


def compute_alpha(p_sigma, q, d):

    alpha_values = []
    for l in range(1, l_max + 1):
        e_1 = lambda x: norm.pdf(x, 0, p_sigma) * (norm.pdf(x, 0, p_sigma) / ((1 - q) * norm.pdf(x, 0, p_sigma) +
                                                                              q * norm.pdf(x, 1.0, p_sigma))) ** l
        e_2 = lambda x: ((1 - q) * norm.pdf(x, 0, p_sigma) + q * norm.pdf(x, 1.0, p_sigma)) * \
                        (((1 - q) * norm.pdf(x, 0, p_sigma) + q * norm.pdf(x, 1.0, p_sigma)) / (
                        norm.pdf(x, 0, p_sigma))) ** l

        E_1, _ = integrate.quad(e_1, -d, d)
        E_2, _ = integrate.quad(e_2, -d, d)
        val = np.log(max([abs(E_1), abs(E_2)]))
        print
        "l = ", l, " alpha = ", val
        alpha_values.append(val)

    return alpha_values


## Alpha computation

# Clustering

alpha_values_1 = compute_alpha(sigma_cl_1, 1.0, 100)
alpha_values_2 = compute_alpha(sigma_cl_2, 1.0, 500)

# DP SGD

alpha_values_3 = []

for l in range(1, l_max + 1):
    tmp = []
    for j in range(1, 10):
        u1 = j / 10.0
        u2 = (10 - j) / 10.0
        alpha_value1 = compute_gauss(sigma_dyn_clip, l, 1.0 / u1, d=50, p_sens=2.0 ** 0.5)
        alpha_value2 = compute_gauss(sigma_gd, l, 1.0 / u2, d=20)
        tmp.append(u1 * alpha_value1 + u2 * alpha_value2)
    val = min(tmp)

    alpha_values_3.append(val)

## Epsilon computation

eps_values = []
cnt = Counter()
for i in range(T):
    epsilon_values = []
    for l in range(1, l_max + 1):
        # Alpha of Clustering
        a_1 = min(i + 1, T_1) * (alpha_values_1[l - 1] + alpha_values_2[l - 1])
        # Alpha of GD
        a_2 = max(0, i + 1 - T_1) * alpha_values_3[l - 1]

        eps_t = (a_1 + a_2 - np.log(delta)) / float(l)
        epsilon_values.append(eps_t)

    idx = np.argmin(epsilon_values)
    cnt[idx] += 1
    eps_values.append(epsilon_values[idx])



print("Epsilon after clustering:", eps_values[T_1])
print("Epsilon after GD:", eps_values[T - 1])

eps_values = eps_values[T_1:]
eps_values = np.array(eps_values)[np.arange(int(1.0 / q) - 1, T_2, int(1.0 / q))]

pickle.dump(eps_values, open("moments.p", 'wb'))

for i in range(epoch_num):
    print("Epoch:", i + 1, "Eps:", eps_values[i])

# plt.xlabel("Epoch")
# plt.ylabel("Epsilon")
# plt.plot(np.arange(1,epochs), eps_values, marker="D")
# plt.savefig("epoch_epsilon2.pdf", bbox_inches="tight")
