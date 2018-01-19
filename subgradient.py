import numpy as np
import time


class SubGradient(object):
    def __init__(self, A, b, mu, iteration=2000, tol=1e-9):
        self.x = None
        self.A = A
        self.m, self.n = self.A.shape
        self.q = np.dot(A.T, A)
        self.step_size = 1.0 / np.linalg.norm(self.q, 2)

        self.b = b
        self.Atb = np.dot(A.T, self.b)
        self.mu = mu

        self.iteration = iteration
        self.obj_path = [1]
        self.tol = tol
        self.initers = 0
        self.iters = 0
        self.run_time = 0

    def loss(self, x):
        x = x.reshape(-1)
        return 0.5 * np.sum(np.square(np.dot(self.A, x) - self.b)) + self.mu * np.sum(np.abs(x))

    def fix_step(self, mu, x):
        g = np.dot(self.q, x) - self.Atb + np.sign(x) * mu
        x -= self.step_size * g
        return x

    def dimish_step(self, mu, x, iter):
        g = np.dot(self.q, x) - self.Atb + np.sign(x) * mu
        x -= self.step_size / iter * g
        return x

    def lenth_step(self, mu, x):
        g = np.dot(self.q, x) - self.Atb + np.sign(x) * mu
        x -= self.step_size / np.linalg.norm(g) * g
        return x

    def train(self):
        t0 = time.time()
        x = np.zeros(self.n)
        print("subgradient method begins")
        self.initers = 0
        for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2, 1e-3]:
            err_rate = 1.0
            in_iter = 1
            while err_rate > self.tol and in_iter < self.iteration:
                x = self.fix_step(hot_mu, x)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                in_iter += 1
            self.initers += in_iter

        self.iters = 1
        err_rate = 1.0
        while err_rate > self.tol:
            x = self.dimish_step(self.mu, x, self.iters)
            self.obj_path.append(self.loss(x))
            err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
            self.iters += 1

        self.x = x
        self.run_time = time.time() - t0
        print("subgradient obj: {: >4.9f}/ time: {: >4.4f} /initers: {}/ iters: {}".format(self.obj_path[-1],
                                                                                           self.run_time,
                                                                                           self.initers,
                                                                                           self.iters))