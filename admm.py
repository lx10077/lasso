import numpy as np
import time


# (g) Augmented Lagrangian method for the dual problem
# (h) Alternating direction method of multipliers for the dual problem
# (i) Alternating direction method of multipliers with linearization for the primal problem

class admm(object):
    def __init__(self, A, b, mu, rho=1., t=1., init_iteration=1000, max_iteration=2000, tol=1e-7, lr=None):
        self.x = None
        self.A = A
        self.m, self.n = self.A.shape
        self.AtA = np.dot(A.T, A)
        self.AAt = np.dot(A, A.T)
        self.At = self.A.T
        self.rho = rho
        self.t = t
        self.step_size = 2.0 / np.linalg.norm(self.AtA, 2) if not lr else lr
        self.prime_q = np.linalg.inv(np.eye(self.n) + 1 / self.rho * self.AtA)
        self.dual_q = np.linalg.inv(np.eye(self.m) + self.rho * self.AAt)

        self.b = b
        self.Atb = np.dot(A.T, self.b)
        self.mu = mu

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

    @staticmethod
    def shrinkage(x, num):
        def pointwise_shrinkage(u, t):
            return max(0., u - t) - max(0., - t - u)
        return np.vectorize(pointwise_shrinkage)(x, num)

    @staticmethod
    def proj(x, num):
        def pointwise_proj(u, t):
            return max(-t, min(u, t))
        return np.vectorize(pointwise_proj)(x, num)

    @staticmethod
    def soft_thresholding(x, num):
        def pointwise_soft(u, t):
            return np.sign(u) * max(abs(u)-t, 0.)
        return np.vectorize(pointwise_soft)(x, num)

    def dual_step(self, z, y, x, mu):
        z = self.proj(x - np.dot(self.At, y), mu * self.rho)
        y = np.dot(self.dual_q, self.rho * (np.dot(self.A, x - z) - self.b))
        x = x - self.t * (np.dot(self.At, y) + z)
        return z, y, x

    def prime_step(self, x, z, y, mu):
        x = np.dot(self.prime_q, self.Atb / self.rho + (z - y))
        z = self.shrinkage(x + y, mu / self.rho)
        y = y + self.t * (x - z)
        return x, z, y

    def linear_step(self, x, z, y, mu):
        g = self.rho * np.dot(self.AtA, x) - self.rho * self.Atb + self.rho * np.dot(self.At, y - z)
        x = self.soft_thresholding(x - self.step_size * g, self.step_size * mu)
        z = self.rho / (1.0 + self.rho) * (np.dot(self.A, x) + y - self.b)
        y = y + self.t * (np.dot(self.A, x) - self.b - z)
        return x, z, y

    def linear_step_v2(self, x, z, y, mu):
        g = np.dot(self.AtA, x) - self.Atb + self.rho * (x - z + y)
        x = x - self.step_size * g
        z = self.soft_thresholding(x + y, mu / self.rho)
        y = y + self.t * (x - z)
        return x, z, y, g

    def train(self, mode="Dual"):
        t0 = time.time()
        print("{} admm begins".format(mode))
        if mode == 'Prime':
            # In this mode, we wanna use ADMM to solve the prime prob:
            #  minimize 1/2*|| Ax - b ||_2^2 + \mu || z ||_1 s.t. x = z
            x = np.random.normal(size=self.n)
            z = np.zeros(self.n)
            y = np.zeros(self.n)
            self.initers = 0
            for hot_mu in [1e1, 1e-1, 1e-2, 1e-3]:
                for k in range(self.init_iteration):
                    self.initers += 1
                    x, z, y = self.prime_step(x, z, y, hot_mu)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < self.tol:
                        break

            self.iters = 0
            err_rate = 1.0
            while err_rate > self.tol and self.iters < self.init_iteration:
                x, z, y = self.prime_step(x, z, y, self.mu)
                self.iters += 1
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        elif mode == "Dual":
            # In this mode, we wanna use ADMM to solve the dual prob:
            # minimize b^Ty + 1/2*|| y ||_2 + 1_{|| z ||_inf <= \mu} s.t. A^Ty + z = 0
            y = np.random.normal(size=self.m)
            z = np.zeros(self.n)
            x = np.zeros(self.n)
            self.initers = 0
            for hot_mu in [1e1, 1, 1e-1, 1e-2, 1e-3]:
                for k in range(self.init_iteration):
                    self.initers += 1
                    z, y, x= self.dual_step(z, y, x, hot_mu)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < self.tol:
                        break

            self.iters = 0
            err_rate = 1.0
            while err_rate > self.tol and self.iters < self.init_iteration:
                z, y, x= self.dual_step(z, y, x, self.mu)
                self.iters += 1
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        elif mode == "Linear":
            # In this mode, we wanna use ADMM to solve the prime prob:
            # minimize 1/2*||z||_2^2 + \mu ||x||_1 s.t. Ax - b = z
            # and we update x by ridge regression
            x = np.random.normal(size=self.n)
            y = np.zeros(self.m)
            z = np.zeros(self.m)
            self.initers = 0
            for hot_mu in [1e1, 1e-1, 1e-2, 1e-3]:
                for k in range(self.init_iteration):
                    self.initers += 1
                    x, z, y = self.linear_step(x, z, y, hot_mu)
                    self.obj_path.append(self.loss(x))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    if err_rate < self.tol:
                        break

            self.iters = 0
            err_rate = 1.0
            while err_rate > self.tol:
                x, z, y = self.linear_step(x, z, y, self.mu)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        elif mode == "Linear_v2":
            # In this mode, we wanna use ADMM to solve the prime prob:
            #  minimize 1/2*|| Ax - b ||_2^2 + \mu || z ||_1 s.t. x = z
            # but we update x by single gradient update
            x = np.random.normal(size=self.n)
            z = np.random.normal(size=self.n)
            y = np.random.normal(size=self.n)
            self.initers = 0
            for hot_mu in [1e1, 1e-1, 1e-2, 1e-3]:
                for k in range(self.init_iteration):
                    self.initers += 1
                    x, z, y, g = self.linear_step_v2(x, z, y, hot_mu)
                    if np.linalg.norm(g) < self.tol / self.step_size:
                        break

            self.iters = 0
            err_rate = 1.0
            while err_rate > self.tol and self.iters < self.init_iteration:
                x, z, y, g = self.linear_step_v2(x, z, y, self.mu)
                self.obj_path.append(self.loss(x))
                self.iters += 1
                if np.linalg.norm(g) < self.tol / self.step_size:
                    break

        else:
            raise ValueError("No {} mode. Choose modes from Dual(Default), Prime, Linear or Linear_v2")

        self.x = x
        self.run_time = time.time() - t0
        print("{:s} admm gradient obj: {: >4.9f}/ time: {: >4.4f} /initers: {}/ iters: {}".format(mode,
                                                                                                  self.obj_path[-1],
                                                                                                  self.run_time,
                                                                                                  self.initers,
                                                                                                  self.iters))