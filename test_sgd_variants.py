import numpy as np
import cvxpy as cvx
import time

np.random.seed(2333)
n = 1024
m = 512
A = np.random.randn(m, n)
u = np.random.randn(n) * np.random.binomial(1, 0.1, (n))
b = np.dot(A, u)
x0 = np.random.randn(n)
mu = 1e-3


def dif(cvx_x, solver):
    x = np.array(cvx_x.value).reshape(n,)
    err = np.linalg.norm(x - solver.x)/(1 + np.linalg.norm(x))
    print("Err: {:.7e}".format(err))
    return


cvx_t0 = time.time()
print("cvx begins")
x = cvx.Variable(n)
error = cvx.sum_squares(A*x - b)
obj = cvx.Minimize(0.5 * error + mu * cvx.norm(x, 1))
prob = cvx.Problem(obj)
a = prob.solve()
cvx_t1 = time.time()
print("cvx objective: {: >4.9f}/ time: {: >4.4f}".format(a, cvx_t1-cvx_t0))


from lasso.sgd_variants import SGD_variants
sv1 = SGD_variants(A, b, mu)
sv1.train()
dif(x, sv1)
sv2 = SGD_variants(A, b, mu, lr=1)
sv2.train("AdaGrad")
dif(x, sv2)
sv3 = SGD_variants(A, b, mu, init_iteration=500, lr=1)
sv3.train("AdaGrad Nesterov")
dif(x, sv3)
sv4 = SGD_variants(A, b, mu, lr=1/50, tol=5e-9)
sv4.train("RMSProp")
dif(x, sv4)
sv5 = SGD_variants(A, b, mu, rho=0.75, init_iteration=500, lr=1/150, tol=5e-9)
sv5.train("RMSProp Nesterov")
dif(x, sv5)
sv6 = SGD_variants(A, b, mu, init_iteration=100, lr=1/3.39, tol=1e-7)
sv6.train("Adam")
dif(x, sv6)


def plot_obj_path(save=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    plt.ylim(-3, 7)
    plt.xlim(0, 4000)
    plt.plot(range(len(sv1.obj_path)), np.log(sv1.obj_path), label="Momentum", linewidth=0.5)
    plt.plot(range(len(sv2.obj_path)), np.log(sv2.obj_path), label="AdaGrad", linewidth=0.5)
    plt.plot(range(len(sv3.obj_path)), np.log(sv3.obj_path), label="AdaGrad Nesterov", linewidth=0.5)

    plt.plot(range(len(sv4.obj_path)), np.log(sv4.obj_path), label="RMSProp", linewidth=0.5)
    plt.plot(range(len(sv5.obj_path)), np.log(sv5.obj_path), label="RMSProp Nesterov", linewidth=0.5)
    plt.plot(range(len(sv6.obj_path)), np.log(sv6.obj_path), label="Adam", linewidth=0.5)

    plt.hlines(np.log(a), 0, 4000, linewidth=1.5)
    plt.legend(loc="upper right")
    if save:
        plt.savefig('sgd.png', dpi=300)
    plt.show()

plot_obj_path()
