import numpy as np
import time


class ProjectGradient(object):
    def __init__(self, A, b, mu, tol=1e-9):
        self.x = None
        self.A = A
        self.m, self.n = self.A.shape
        self.q = np.dot(A.T, A)
        self.step_size = 1.0 / np.linalg.norm(self.q, 2)  # 0.000334973485287

        self.b = b
        self.Atb = np.dot(A.T, self.b)
        self.mu = mu

        self.obj_path = [1]
        self.tol = tol
        self.initers = 0
        self.iters = 1
        self.run_time = 0

    def loss(self, x):
        x = x.reshape(-1)
        return 0.5 * np.sum(np.square(np.dot(self.A, x) - self.b)) + self.mu * np.sum(np.abs(x))

    def grad(self, mu, x_p, x_n):
        g = np.dot(self.q, x_p - x_n) - self.Atb
        return g + mu, -g + mu

    def step(self, mu, x_p, x_n):
        grad_xp, grad_xn = self.grad(mu, x_p, x_n)
        x_p -= self.step_size * grad_xp
        x_n -= self.step_size * grad_xn
        x_p[x_p <= 0.] = 0.
        x_n[x_n <= 0.] = 0.
        return x_p, x_n

    def BB_step(self, mu, x_p, x_n, alpha):
        x_p_, x_n_ = x_p.copy(), x_n.copy()
        grad_xp, grad_xn = self.grad(mu, x_p, x_n)
        x_p -= alpha * grad_xp
        x_n -= alpha * grad_xn
        x_p[x_p <= 0.] = 0.
        x_n[x_n <= 0.] = 0.
        s_p, s_n = x_p - x_p_, x_n - x_n_
        alpha = 0.5 * (np.linalg.norm(s_p)**2 + np.linalg.norm(s_n)**2) / np.linalg.norm(np.dot(self.A, s_p-s_n))**2
        alpha = max(min(alpha, self.step_size*2), self.step_size*0.5)
        return x_p, x_n, alpha

    def train(self, mode="Basic"):
        t0 = time.time()
        x_p = np.zeros(self.n)
        x_n = np.zeros(self.n)
        print("{} projection gradient method begins".format(mode))
        if mode == "Basic":
            self.initers = 0
            for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2]:
                err_rate = 1.0
                while err_rate > self.tol:
                    x_p, x_n = self.step(hot_mu, x_p, x_n)
                    self.obj_path.append(self.loss(x_p - x_n))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    self.initers += 1

            self.iters = 0
            err_rate = 1.0
            while err_rate > self.tol:
                x_p, x_n = self.step(self.mu, x_p, x_n)
                self.obj_path.append(self.loss(x_p - x_n))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        elif mode == "BB":
            alpha = self.step_size
            self.initers = 0
            for hot_mu in [1e3, 1e2, 1e1, 1e-1, 1e-2]:
                err_rate = 1.0
                while err_rate > self.tol:
                    x_p, x_n = self.step(hot_mu, x_p, x_n)
                    self.obj_path.append(self.loss(x_p - x_n))
                    err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                    self.initers += 1

            self.iters = 0
            err_rate = 1.0
            while err_rate > self.tol:
                x_p, x_n, alpha = self.BB_step(self.mu, x_p, x_n, alpha)
                self.obj_path.append(self.loss(x_p - x_n))
                err_rate = np.abs(self.obj_path[-1] - self.obj_path[-2]) / self.obj_path[-2]
                self.iters += 1

        else:
            raise ValueError("No {} mode. Choose modes from Basic(Default) and BB")

        self.x = x_p - x_n
        self.run_time = time.time() - t0
        print("projection gradient obj: {: >4.9f}/ time: {: >4.4f} /initers: {}/ iters: {}".format(self.obj_path[-1],
                                                                                                   self.run_time,
                                                                                                   self.initers,
                                                                                                   self.iters))