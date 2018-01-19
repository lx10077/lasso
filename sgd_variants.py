import numpy as np
import time


class SGD_variants(object):
    def __init__(self, A, b, mu, alpha=0.9, rho=0.5, rho1=0.9, rho2=0.999,
                 init_iteration=1000, max_iteration=2000, tol=1e-9, lr=None):
        self.x = None
        self.A = A
        self.At = self.A.T
        self.b = b
        self.Atb = np.dot(self.At, self.b)
        self.m, self.n = self.A.shape
        self.AtA = np.dot(A.T, A)
        self.AAt = np.dot(A, A.T)
        self.step_size = 1.0 / np.linalg.norm(self.AtA, 2) if not lr else lr

        self.alpha = alpha
        self.mu = mu
        self.rho = rho
        self.rho1 = rho1
        self.rho2 = rho2

        self.init_iteration = init_iteration
        self.max_iteration = max_iteration
        self.obj_path = [1]
        self.tol = tol
        self.iters = 0
        self.initers = 0
        self.run_time = 0

    def loss(self, x):
        x = x.reshape(-1)
        return 0.5 * np.sum(np.square(np.dot(self.A, x) - self.b)) + self.mu * np.sum(np.abs(x))

    def prox(self, u, t):
        return np.sign(u) * np.maximum(np.abs(u) - t, 0)

    def momentum_step(self, mu, x, delta_x):
        x_ = x.copy()
        g = np.dot(self.AtA, x) - self.Atb
        x = self.prox(x-self.step_size * g + self.alpha * delta_x, self.step_size * mu)
        delta_x = x - x_
        return x, delta_x

    def adagrad_step(self, mu, x, g):
        new_g = np.dot(self.AtA, x) - self.Atb
        g = g + new_g ** 2
        x = self.prox(x - self.step_size / (np.sqrt(g) + 1e-7) * new_g, self.step_size / (np.sqrt(g) + 1e-7) * mu)
        return x, g

    def adagrad_nesterov_step(self, mu, x, x_, g, k):
        y = x + (k - 2)/(k + 1) * (x - x_)
        x_ = x.copy()
        new_g = np.dot(self.AtA, y) - self.Atb
        g = g + new_g ** 2
        x = self.prox(y - self.step_size / (np.sqrt(g) + 1e-7) * new_g, self.step_size / (np.sqrt(g) + 1e-7) * mu)
        return x, x_, g

    def rmsprop_step(self, mu, x, g):
        new_g = np.dot(self.AtA, x) - self.Atb
        g = self.rho * g + (1 - self.rho) * (new_g ** 2)
        x = self.prox(x - self.step_size / np.sqrt(g + 1e-7) * new_g, self.step_size / np.sqrt(g + 1e-7) * mu)
        return x, g

    def rmsprop_nesterov_step(self, mu, x, x_, g, k):
        y = x + (k - 2)/(k + 1) * (x - x_)
        x_ = x.copy()
        new_g = np.dot(self.AtA, y) - self.Atb
        g = self.rho * g + (1 - self.rho) * (new_g ** 2)
        x = self.prox(y - self.step_size / np.sqrt(g + 1e-7) * new_g, self.step_size / np.sqrt(g + 1e-7) * mu)
        return x, x_, g

    def adam_step(self, mu, x, m, g, k):
        new_g = np.dot(self.AtA, x) - self.Atb
        m = self.rho1 * m + (1 - self.rho1) * new_g
        g = self.rho2 * g + (1 - self.rho2) * (new_g ** 2)
        m_hat = m / (1 - self.rho1 ** k)
        g_hat = g / (1 - self.rho2 ** k)
        x = self.prox(x - self.step_size / (np.sqrt(g_hat) + 1e-7) * m_hat,
                      self.step_size / (np.sqrt(g_hat) + 1e-7) * mu)
        return x, m, g

    def train(self, mode="Momentum"):
        t0 = time.time()
        print("{} sgd begins".format(mode))
        if mode == "Momentum":
            x = np.zeros(self.n, dtype=np.float64)
            delta_x = np.zeros(self.n, dtype=np.float64)
            self.initers = 0
            for hot_mu in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
                for i in range(self.init_iteration):
                    self.initers += 1
                    x, delta_x = self.momentum_step(hot_mu, x, delta_x)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < 1e-8:
                        break

            self.iters = 1
            err_rate = 1.
            while err_rate > self.tol and self.iters < self.max_iteration:
                self.iters += 1
                x, delta_x = self.momentum_step(self.mu, x, delta_x)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        elif mode == 'AdaGrad':
            x = np.zeros(self.n, dtype=np.float64)
            g = np.zeros(self.n, dtype=np.float64)
            self.initers = 0
            for hot_mu in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
                for i in range(self.init_iteration):
                    self.initers += 1
                    x, g = self.adagrad_step(hot_mu, x, g)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < 1e-8:
                        break

            self.iters = 1
            err_rate = 1.
            while err_rate > self.tol and self.iters < self.max_iteration:
                self.iters += 1
                x, g = self.adagrad_step(self.mu, x, g)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        elif mode == 'AdaGrad Nesterov':
            x = np.zeros(self.n, dtype=np.float64)
            x_ = np.zeros(self.n, dtype=np.float64)
            g = np.zeros(self.n, dtype=np.float64)
            self.initers = 0
            for hot_mu in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
                for k in range(self.init_iteration):
                    x, x_, g = self.adagrad_nesterov_step(hot_mu, x, x_, g, k)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < 1e-8:
                        break
                self.initers += k

            self.iters = 0
            err_rate = 1.
            while err_rate > self.tol and self.iters < self.max_iteration:
                self.iters += 1
                x, x_, g = self.adagrad_nesterov_step(self.mu, x, x_, g, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        elif mode == 'RMSProp':
            x = np.zeros(self.n, dtype=np.float64)
            g = np.zeros(self.n, dtype=np.float64)
            self.initers = 0
            for hot_mu in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
                for i in range(self.init_iteration):
                    self.initers += 1
                    x, g = self.rmsprop_step(hot_mu, x, g)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < 1e-7:
                        break
                self.step_size /= np.sqrt(10)

            self.iters = 1
            err_rate = 1.
            while err_rate > self.tol and self.iters < self.max_iteration:
                self.iters += 1
                x, g = self.rmsprop_step(self.mu, x, g)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        elif mode == 'RMSProp Nesterov':
            x = np.zeros(self.n, dtype=np.float64)
            x_ = np.zeros(self.n, dtype=np.float64)
            g = np.zeros(self.n, dtype=np.float64)
            self.initers = 0
            for hot_mu in [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
                for k in range(self.init_iteration):
                    x, x_, g = self.rmsprop_nesterov_step(hot_mu, x, x_, g, k)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < 1e-8:
                        break
                self.initers += k
                self.step_size /= np.sqrt(10)

            self.iters = 0
            err_rate = 1.
            while err_rate > self.tol and self.iters < self.max_iteration:
                self.iters += 1
                x, x_, g = self.rmsprop_nesterov_step(self.mu, x, x_, g, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        elif mode == 'Adam':
            x = np.zeros(self.n, dtype=np.float64)
            m = np.zeros(self.n, dtype=np.float64)
            g = np.zeros(self.n, dtype=np.float64)
            self.initers = 0
            for hot_mu in [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]:
                for k in range(1, self.init_iteration):
                    x, m, g = self.adam_step(hot_mu, x, m, g, k)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < 1e-7:
                        break
                self.initers += k

            self.iters = 1
            err_rate = 1.
            while err_rate > self.tol and self.iters < self.max_iteration:
                self.iters += 1
                x, m, g = self.adam_step(self.mu, x, m, g, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]



        else:
            raise ValueError("No {} mode. Choose modes from Momentum(Default), AdaGrad, AdaGrad Nesterov, " +
                             "RMSProp, RMSProp Nesterov or Adam")

        self.x = x
        self.run_time = time.time() - t0
        print("{:s} sgd obj: {: >4.9f}/ time: {: >4.4f} /initers: {}/ iters: {}".format(mode,
                                                                                        self.obj_path[-1],
                                                                                        self.run_time,
                                                                                        self.initers,
                                                                                        self.iters))
