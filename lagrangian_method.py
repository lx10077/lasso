import numpy as np
import time


class lagrangian_method(object):
    def __init__(self, A, b, mu, rho=1., t=1., init_iteration=1000, max_iteration=2000, tol=1e-7, lr=None):
        # The augmented lagrangian is
        # L = 1/2||y||_2^2 + b^Ty + 1_{||z||_inf <= \mu} + rho/2 ||A^Ty + z - x||_2^2
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
    def proj(x, num):
        def pointwise_proj(u, t):
            return max(-t, min(u, t))
        return np.vectorize(pointwise_proj)(x, num)

    @staticmethod
    def shrinkage(x, num):
        def pointwise_shrinkage(u, t):
            return max(0., u - t) - max(0., - t - u)
        return np.vectorize(pointwise_shrinkage)(x, num)

    def dual_grad_step(self, z, y, x, mu):
        s = self.shrinkage(np.dot(self.At, y) - x, mu)
        g = self.b + y + self.rho * (np.dot(self.A, s))
        y -= self.step_size * g
        z = self.proj(x - np.dot(self.At, y), mu)
        x = x - self.t * (np.dot(self.At, y) + z)
        return z, y, x

    def train(self):
        t0 = time.time()
        print("augmented lagrangian begins")
        y = np.random.normal(size=self.m)
        z = np.zeros(self.n)
        x = np.zeros(self.n)
        self.initers = 0
        for hot_mu in [1e1, 1, 1e-1, 1e-2, 1e-3]:
            for k in range(self.init_iteration):
                self.initers += 1
                z, y, x = self.dual_grad_step(z, y, x, hot_mu)
                self.obj_path.append(self.loss(x))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                if err_rate < self.tol:
                    break

        self.iters = 0
        err_rate = 1.0
        while err_rate > self.tol and self.iters < self.init_iteration:
            z, y, x = self.dual_grad_step(z, y, x, self.mu)
            self.iters += 1
            self.obj_path.append(self.loss(x))
            err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]

        self.x = x
        self.run_time = time.time() - t0
        print("augmented lagrangian obj: {: >4.9f}/ time: {: >4.4f} /initers: {}/ iters: {}".format(self.obj_path[-1],
                                                                                                    self.run_time,
                                                                                                    self.initers,
                                                                                                    self.iters))
