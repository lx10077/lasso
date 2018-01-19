import numpy as np
import cvxpy as cvx
import time

np.random.seed(23)
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

# def dif1(x, solver):
#     err = np.linalg.norm(x.x - solver.x)/(1 + np.linalg.norm(x.x))
#     print(np.linalg.norm(x.x - solver.x))
#     print("Err:", err)
#     return


cvx_t0 = time.time()
print("cvx begins")
x = cvx.Variable(n)
error = cvx.sum_squares(A*x - b)
obj = cvx.Minimize(0.5 * error + mu * cvx.norm(x, 1))
prob = cvx.Problem(obj)
a = prob.solve()
cvx_t1 = time.time()
print("cvx objective: {: >4.9f}/ time: {: >4.4f}".format(a, cvx_t1-cvx_t0))


from lasso.project_gradient import ProjectGradient
pg1 = ProjectGradient(A, b, mu)
pg1.train()
# dif(x,pg1)
pg2 = ProjectGradient(A, b, mu)
pg2.train("BB")
# dif(x, pg2)


from lasso.subgradient import SubGradient
sg = SubGradient(A, b, mu)
sg.train()
# dif(x, sg)


from lasso.proximal_gradient import ProxGradient
proxg1 = ProxGradient(A, b, mu)
proxg1.train()
# dif(x, proxg1)
proxg2 = ProxGradient(A, b, mu)
proxg2.train("FISTA")
# dif(x, proxg2)
proxg3 = ProxGradient(A, b, mu)
proxg3.train("Nesterov")
# dif(x, proxg3)


from lasso.smooth_method import SmoothMethod
sm1 = SmoothMethod(A, b, mu)
sm1.train()
# dif(x, sm1)
sm2 = SmoothMethod(A, b, mu)
sm2.train("BB")
# dif(x, sm2)
sm3 = SmoothMethod(A, b, mu)
sm3.train("FISTA")
# dif(x, sm3)
sm4 = SmoothMethod(A, b, mu)
sm4.train("Nesterov")
# dif(x, sm4)


from lasso.lagrangian_method import lagrangian_method
al = lagrangian_method(A, b, mu, lr=6e-4)
al.train()
# dif(x, al)


from lasso.admm import admm
ad1 = admm(A, b, mu, rho=1., init_iteration=1000, t=1.618, tol=1e-8)
ad1.train("Prime")
# dif(x, ad1)
ad2 = admm(A, b, mu, rho=1., init_iteration=1000, t=1.618, tol=1e-9)
ad2.train("Dual")
# dif(x, ad2)
ad3 = admm(A, b, mu, rho=1., init_iteration=1000, tol=1e-7)
ad3.train("Linear")
# dif(x, ad3)
ad4 = admm(A, b, mu, rho=50., init_iteration=1000, t=1.618, tol=1e-7, lr=6e-4)
ad4.train("Linear_v2")
# dif(x, ad4)


def plot_obj_path(save=False):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,8))
    plt.ylim(-3, 7)
    plt.xlim(0, 10000)
    plt.plot(range(len(sg.obj_path)), np.log(sg.obj_path), label="subgradient", linewidth=0.5)
    plt.plot(range(len(pg1.obj_path)), np.log(pg1.obj_path), label="proj-grad-basic", linewidth=0.5)
    plt.plot(range(len(pg2.obj_path)), np.log(pg2.obj_path), label="proj-grad-BB", linewidth=0.5)

    plt.plot(range(len(proxg1.obj_path)), np.log(proxg1.obj_path), label="prox-grad-basic", linewidth=0.5)
    plt.plot(range(len(proxg2.obj_path)), np.log(proxg2.obj_path), label="prox-grad-FISTA", linewidth=0.5)
    plt.plot(range(len(proxg3.obj_path)), np.log(proxg3.obj_path), label="prox-grad-Nesto", linewidth=0.5)

    plt.plot(range(len(sm1.obj_path)), np.log(sm1.obj_path), label="smooth-basic", linewidth=0.5)
    plt.plot(range(len(sm2.obj_path)), np.log(sm2.obj_path), label="smooth-BB", linewidth=0.5)
    plt.plot(range(len(sm3.obj_path)), np.log(sm3.obj_path), label="smooth-FISTA", linewidth=0.5)
    plt.plot(range(len(sm4.obj_path)), np.log(sm4.obj_path), label="smooth-Nesto", linewidth=0.5)

    plt.plot(range(len(al.obj_path)), np.log(al.obj_path), label="aug-lagrangian", linewidth=0.5)

    plt.plot(range(len(ad1.obj_path)), np.log(ad1.obj_path), label="prime-admm", linewidth=0.5)
    plt.plot(range(len(ad2.obj_path)), np.log(ad2.obj_path), label="dual-admm", linewidth=0.5)
    plt.plot(range(len(ad3.obj_path)), np.log(ad3.obj_path), label="linear-v1", linewidth=0.5)
    plt.plot(range(len(ad4.obj_path)), np.log(ad4.obj_path), label="linear-v2", linewidth=0.5)

    plt.hlines(np.log(a), 0, 10000, linewidth=1.5)
    plt.legend(loc="upper right")
    if save:
        plt.savefig('coarse.png', dpi=300)
    plt.show()

plot_obj_path()




