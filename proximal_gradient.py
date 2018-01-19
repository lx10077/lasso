import numpy as np
import time


class ProxGradient(object):
    def __init__(self,  A, b, mu, init_iteration=100, max_iteration=2000, tol=1e-8):
        self.x = None
        self.A = A
        self.m, self.n = self.A.shape
        self.q = np.dot(A.T, A)
        self.step_size = 1.0 / np.linalg.norm(self.q, 2)

        self.b = b
        self.Atb = np.dot(self.A.T, self.b)
        self.mu = mu

        self.init_iteration = init_iteration
        self.max_iteration = max_iteration
        self.obj_path = [1]
        self.tol = tol
        self.iters = 1
        self.run_time = 0

    def loss(self, x):
        x = x.reshape(-1)
        return 0.5 * np.sum(np.square(np.dot(self.A, x) - self.b)) + self.mu * np.sum(np.abs(x))

    @staticmethod
    def prox(x, num):
        def pointwise_prox(u, t):
            if u >= t:
                return u - t
            elif u <= -t:
                return u + t
            else:
                return 0.0
        return np.vectorize(pointwise_prox)(x, num)

    def basic_step(self, mu, x):
        g = np.dot(self.q, x) - self.Atb
        x = x - self.step_size * g
        x = self.prox(x, mu * self.step_size)
        return x

    def fast_step(self, mu, x, x_, k):
        # we could reformulate FISTA in a nesterov-like form involving x and v
        y = x + 1.0 * (k - 2)/(k + 1) * (x - x_)
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

        x = np.random.normal(size=self.n)
        print("{} proximal gradient begins".format(mode))
        self.initers = 0
        hot_mus = [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]
        if mode == 'Basic':
            for i in range(len(hot_mus)):
                err_rate = 1.0
                in_iter = 1
                while err_rate > 10**(-5-i) and in_iter < self.max_iteration:
                    x = self.basic_step(hot_mus[i], x)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    in_iter += 1
                self.initers += in_iter

            self.iters = 1
            err_rate = 1.0
            while err_rate > self.tol and self.initers < self.max_iteration:
                x = self.basic_step(self.mu, x)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                print(err_rate)
                self.iters += 1

        elif mode == 'FISTA':
            x_ = x.copy()
            for i in range(len(hot_mus)):
                err_rate = 1.0
                in_iter = 1
                while err_rate > 10**(-5-i) and in_iter < self.init_iteration:
                    x, x_ = self.fast_step(hot_mus[i], x, x_, in_iter)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    in_iter += 1
                self.initers += in_iter

            self.iters = 1
            err_rate = 1.0
            while err_rate > self.tol and self.iters < self.max_iteration:
                x, x_ = self.fast_step(self.mu, x, x_, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        elif mode == 'Nesterov':
            v = x.copy()
            for i in range(len(hot_mus)):
                err_rate = 1.0
                in_iter = 1
                while err_rate > 10**(-5-i) and in_iter < self.init_iteration:
                    x, v = self.nesterov_step(hot_mus[i], x, v, in_iter)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    in_iter += 1
                self.initers += in_iter

            self.iters = 1
            err_rate = 1.0
            while err_rate > self.tol and self.max_iteration:
                x, v = self.nesterov_step(self.mu, x, v, self.iters)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        else:
            raise ValueError("No {} mode. Choose modes from Basic(Default), FISTA or Nesterov")

        self.x = x
        self.run_time = time.time() - t0
        print("{:s} proximal gradient obj: {: >4.9f}/ time: {: >4.4f} /initers: {}/ iters: {}".format(mode,
                                                                                                      self.obj_path[-1],
                                                                                                      self.run_time,
                                                                                                      self.initers,
                                                                                                      self.iters))