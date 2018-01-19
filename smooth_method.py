import numpy as np
import time


class SmoothMethod(object):
    def __init__(self, A, b, mu, init_iteration=100, max_iteration=2000, init_tol=1e-7, train_tol=1e-9, eps=1e-6):
        self.x = None
        self.A = A
        self.m, self.n = self.A.shape
        self.q = np.dot(A.T, A)
        self.step_size = 1.0 / np.linalg.norm(self.q, 2)

        self.b = b
        self.Atb = np.dot(A.T, self.b)
        self.mu = mu

        self.init_iteration = init_iteration
        self.max_iteration = max_iteration
        self.obj_path = [1]
        self.init_tol = init_tol
        self.train_tol = train_tol
        self.eps = eps
        self.initers = 0
        self.iters = 0
        self.run_time = 0

    def loss(self, x):
        x = x.reshape(-1)
        return 0.5 * np.sum(np.square(np.dot(self.A, x) - self.b)) + self.mu * np.sum(np.abs(x))

    def pointwise_phi(self, x):  # use Huber penalty as smoothed absolute value
        if np.abs(x) <= self.eps:
            return np.square(x) / (2.0 * self.eps)
        else:
            return np.abs(x) - self.eps / 2.0

    def pointwise_phi_grad(self, x):  # define the gradient of Huber penalty
        return max(-1, min(x/self.eps, 1))

    def pointwise_prox(self, x, t):  # define pointwise Huber Pernalty's proximal function
        if x >= t + self.eps:
            return x - t
        elif x < -t - self.eps:
            return x + t
        else:
            return x / (1.0 + t/self.eps)

    def grad(self, mu, x):
        return np.dot(self.q, x) - self.Atb + mu * self.phig(x)

    def basic_step(self, mu, x, iter=1):
        x = x - self.step_size / iter * self.grad(mu, x)
        return x

    def BB_step(self, mu, x, g_, alpha):
        x_ = x.copy()
        g = self.grad(mu, x)
        x = x - alpha * g
        alpha = np.linalg.norm(x - x_)**2 / np.abs(np.dot(x - x_, g - g_))
        alpha = max(min(alpha, 1.2 * self.step_size), self.step_size*0.5)
        return x, g, alpha

    def fast_step(self, mu, x, x_, k):
        y = x + 1.0 * (k - 2) / (k + 1) * (x - x_)
        x_ = x.copy()
        g = np.dot(self.q, y) - self.Atb
        x = y - self.step_size * g
        x = self.prox(x, mu * self.step_size)
        return x, x_

    def nesterov_step(self, mu, x, v, k):
        theta = 2.0 / (k + 1)
        y = (1.0 - theta) * x + theta * v
        g = np.dot(self.q, y) - self.Atb
        tmp = v - self.step_size / theta * g
        v = self.prox(tmp, mu * self.step_size / theta)
        x = (1.0 - theta) * x + theta * v
        return x, v

    def train(self, mode="Basic"):
        t0 = time.time()
        self.phi = np.vectorize(self.pointwise_phi)
        self.phig = np.vectorize(self.pointwise_phi_grad)
        self.prox = np.vectorize(self.pointwise_prox)

        x = np.zeros(self.n)
        print("{} smooth method with Huber loss begins".format(mode))
        self.initers = 0
        if mode == 'Basic':
            for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2]:
                err_rate = 1.0
                in_iter = 1
                while err_rate > self.train_tol and in_iter < 2000:
                    x = self.basic_step(hot_mu, x)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    in_iter += 1
                self.initers += in_iter

            self.iters = 1
            err_rate = 1.0
            while err_rate > self.train_tol:
                x = self.basic_step(self.mu, x, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        elif mode == 'BB':
            alpha = self.step_size
            for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2]:
                err_rate = 1.0
                in_iter = 1
                while err_rate > self.train_tol and in_iter < 2000:
                    x = self.basic_step(hot_mu, x)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    in_iter += 1
                self.initers += in_iter

            self.iters = 1
            err_rate = 1.0
            g = self.grad(self.mu, x)
            while err_rate > self.train_tol:
                x, g, alpha = self.BB_step(self.mu, x, g, alpha)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        elif mode == 'FISTA':
            x_ = x.copy()
            for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
                err_rate = 1.0
                in_iter = 1
                while err_rate > self.init_tol and in_iter < self.init_iteration:
                    x, x_ = self.fast_step(hot_mu, x, x_, in_iter)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    in_iter += 1
                self.initers += in_iter

            self.iters = 1
            err_rate = 1.0
            while err_rate > self.train_tol and self.iters < self.max_iteration:
                x, x_ = self.fast_step(self.mu, x, x_, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        elif mode == 'Nesterov':
            v = x.copy()
            for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
                err_rate = 1.0
                in_iter = 1
                while err_rate > self.init_tol and in_iter < self.init_iteration:
                    x, v = self.nesterov_step(hot_mu, x, v, in_iter)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    in_iter += 1
                self.initers += in_iter

            self.iters = 1
            err_rate = 1.0
            while err_rate > self.train_tol and self.iters < self.max_iteration:
                x, v = self.nesterov_step(self.mu, x, v, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        else:
            raise ValueError("No {} mode. Choose modes from Basic(Default), BB, FISTA or Nesterov")

        self.x = x
        self.run_time = time.time() - t0
        print("{:s} smooth method obj: {: >4.9f}/ time: {: >4.4f} /initers: {}/ iters: {}".format(mode,
                                                                                                  self.obj_path[-1],
                                                                                                  self.run_time,
                                                                                                  self.initers,
                                                                                                  self.iters))