# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import cvxpy as cp
import numpy as np


def qp_obj(P, q, r, x):
    first_term = np.dot(x, np.dot(P, x))
    second_term = 2.0 * np.dot(q, x)
    obj = first_term + second_term + r
    return obj


def new_origin(x0, P, q, r):
    q2 = np.dot(P, x0) + q
    r2 = qp_obj(P, q, r, x0)
    return (q2, r2)


def mulrandn(mu, A):
    n = mu.shape[0]
    z = np.random.randn(n, 1)
    x = mu + A @ z
    return x


def ONEOPT(x, P, q):
    n = P.shape[0]
    g = 2.0 * (P @ x + q.reshape(n, 1))
    v = np.diagonal(P).reshape(n, 1)  #
    iters = 0
    while True:
        iters = iters + 1
        if np.all(v >= np.abs(g)):
            break
        c = np.round(np.divide(-g, 2.0 * v)).reshape(n, 1)
        diffs = np.multiply(c**2, v) + np.multiply(c, g)
        i = np.argmin(diffs)
        x[i] = x[i] + c[i]
        g = g + 2.0 * c[i] * P[:, i].reshape(n, 1)

    val = x.T @ (P @ x) + 2.0 * q.T @ x
    return x, val


def oneopt_clipped(x, P, q, xflr, xmin, xmax):
    n = P.shape[0]
    xflr = xflr.reshape(n, 1)
    g = 2.0 * (P @ x + q.reshape(n, 1))
    v = np.diagonal(P).reshape(n, 1)  #
    iters = 0
    while True:
        iters = iters + 1
        if np.all(v >= np.abs(g)):
            break
        c = np.round(np.divide(-g, 2.0 * v)).reshape(n, 1)
        diffs = np.multiply(c**2, v) + np.multiply(c, g)
        i = np.argmin(diffs)
        x[i] = x[i] + c[i]
        g = g + 2.0 * c[i] * P[:, i].reshape(n, 1)

    xmin = xmin * np.ones_like(x)
    xmax = xmax * np.ones_like(x)

    idx_below = (x + xflr < xmin).flatten()
    x[idx_below] = xmin[idx_below] - xflr[idx_below]
    idx_above = (x + xflr > xmax).flatten()
    x[idx_above] = xmax[idx_above] - xflr[idx_above]
    val = x.T @ (P @ x) + 2.0 * q.T @ x
    return x, val


def icqm(P, q, r, K, xmin, xmax, verbose_solver):
    # ignore zero rows/columns in P to prevent numerical issues
    mask_nonzero = ~np.all(P == 0, axis=1)

    P = P[mask_nonzero, :]
    P = P[:, mask_nonzero]
    q = q[mask_nonzero]

    n = P.shape[0]
    xcts = -np.linalg.solve(P, q)
    xflr = np.floor(xcts)
    (q, r) = new_origin(xflr, P, q, r)

    X = cp.Variable((n, n), symmetric=True)
    x = cp.Variable((n, 1))

    P_cp = cp.Constant(P)
    q_cp = cp.Constant(q.reshape((P.shape[0], 1)))
    c_one_np = np.array([[1.0]])
    c_one = cp.Constant(c_one_np)

    S_upper = cp.hstack((X, x))
    S_lower = cp.hstack((x.T, c_one))
    S = cp.vstack((S_upper, S_lower))

    prod = 2.0 * q_cp.T @ x
    obj = cp.trace(P_cp @ X) + prod

    c1 = cp.diag(X) >= cp.vec(x)
    c2 = S >> 0.0
    constraints = [c1, c2]
    prob = cp.Problem(cp.Minimize(obj), constraints)

    print("Installed solvers:")
    print(cp.installed_solvers())
    lb = prob.solve(verbose=verbose_solver)

    X_val = X.value
    X_val = 0.5 * (X_val + X_val.transpose())
    x_val = x.value

    mu = x_val
    Sigma = X_val - x_val @ x_val.T
    D, V = np.linalg.eig(Sigma)

    A = V @ np.diag(np.sqrt(np.maximum(D, 0.0)))
    ub = 0
    xhat = np.zeros((n, 1), dtype="float64")
    for k in range(0, K):
        rand_vec = mulrandn(mu, A)
        x = np.round(rand_vec)
        x, val = oneopt_clipped(x, P, q, xflr, xmin, xmax)
        if ub > val:
            ub = val
            xhat = x
    lb = lb + r
    ub = ub + r
    xhat = xhat + xflr.reshape(n, 1)

    n_orig = mask_nonzero.shape[0]
    xhat_fullsize = np.zeros((n_orig, 1), dtype=np.float64)
    xhat_fullsize[np.nonzero(mask_nonzero)] = xhat
    return (lb, ub, xhat_fullsize)


def sdp_mixed_integer_qp_bounds(
    G: np.ndarray,
    w: np.ndarray,
    r: float,
    scale: float,
    verbose_solver: bool,
    int_min: float,
    int_max: float,
):
    G = G.astype("float64")
    w = w.astype("float64")

    P = G * scale**2
    q = -scale * np.dot(w, G).transpose()
    K = 3 * P.shape[0]

    (lb, ub, xhat) = icqm(P, q, r, K, int_min, int_max, verbose_solver)
    return (lb, ub.item(), xhat)
