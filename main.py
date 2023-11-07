import os
import numpy as np
from numpy import linalg
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main_f(x1, x2, x3):
    return 6 * x1 ** 2 + 4 * x2 ** 2 + 3 * x3 ** 2 + 3 * x1 * x2 - x2 * x3 + 10 * x2


def grad_f(z1: float, z2: float, z3: float):
    x1 = tf.Variable([[z1]])
    x2 = tf.Variable([[z2]])
    x3 = tf.Variable([[z3]])
    with tf.GradientTape() as tape:
        f = main_f(x1, x2, x3)

    t = tape.gradient(f, [x1, x2, x3])
    return np.array([float(t[0]), float(t[1]), float(t[2])])


def grad_f_fast(x1: float, x2: float, x3: float):
    return np.array([12*x1+3*x2, 8*x2+3*x1-x3+10, 6*x3-x2])


def argmin_t(x1, x2, x3, p1, p2, p3):

    def one_main_f(t):
        return main_f(x1+t*p1, x2+t*p2, x3+t*p3)

    def der_one_main_f(t: float):
        t_ins = tf.Variable([[t]])
        with tf.GradientTape() as tape:
            f = one_main_f(t_ins)
        return float(tape.gradient(f, t_ins))

    t = 0.0
    eps = 0.0001
    step = 0.1
    der = der_one_main_f(t)

    while der > eps:

        t_up = t - step * der
        while one_main_f(t) <= one_main_f(t_up):
            step /= 2
            t_up = t - step * der

        t = t_up

        der = der_one_main_f(t)
    t = t - step * der
    return t


def argmin_t_fast(x1, x2, x3, p1, p2, p3):
    return -(12*x1*p1 + 8*x2*p2 + 6*x3*p3 + 3*x1*p2 + 3*x2*p1 - x2*p3 - x3*p2 + 10*p2)\
        / (12*p1**2 + 8*p2**2 + 6*p3**2 + 6*p1*p2 - 2*p2*p3)


class Tasks:
    def __init__(self, eps, x0, fastmode):
        self.eps = eps
        self.xk = x0
        if fastmode:
            self.gradfunc = grad_f_fast
            self.argminfunc = argmin_t_fast
        else:
            self.gradfunc = grad_f
            self.argminfunc = argmin_t

    def grad_spusk(self):

        eps = self.eps
        xk = self.xk
        grudfunc = self.gradfunc

        tk = 0.01
        gradient = grudfunc(xk[0], xk[1], xk[2])
        gradient_norm = linalg.norm(gradient)
        print('x0=', list(xk), end='\n')
        k = 1

        while gradient_norm > eps:
            xk = xk + -tk * gradient
            print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ',
                  "{0:.3f}".format(xk[2]), '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))
            gradient = grudfunc(xk[0], xk[1], xk[2])
            gradient_norm = linalg.norm(gradient)
            k += 1

        xk = xk + -tk * gradient
        print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ', "{0:.3f}".format(xk[2])
              , '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))


    def grad_spusk_with_opt_step(self):

        eps = self.eps
        xk = self.xk
        grudfunc = self.gradfunc

        tk = 0.1
        gradient = grudfunc(xk[0], xk[1], xk[2])
        gradient_norm = linalg.norm(gradient)
        print('x0=', list(xk), end='\n')
        k = 1

        while gradient_norm > eps:

            xk_up = xk + -tk * gradient
            while main_f(xk[0], xk[1], xk[2]) <= main_f(xk_up[0], xk_up[1], xk_up[2]):
                tk /= 2
                xk_up = xk + -tk * gradient
            xk = xk_up

            print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ',
                  "{0:.3f}".format(xk[2]), '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))
            gradient = grudfunc(xk[0], xk[1], xk[2])
            gradient_norm = linalg.norm(gradient)
            k += 1

        xk = xk + -tk * gradient
        print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ', "{0:.3f}".format(xk[2])
              , '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))


    def grad_spusk_with_argmin_opt(self):

        eps = self.eps
        xk = self.xk
        grudfunc = self.gradfunc
        argminfunc = self.argminfunc

        gradient = grudfunc(xk[0], xk[1], xk[2])
        gradient_norm = linalg.norm(gradient)

        tk = abs(argminfunc(xk[0], xk[1], xk[2], gradient[0], gradient[1], gradient[2]))

        print('x0=', list(xk), end='\n')
        k = 1

        while gradient_norm > eps:
            xk = xk + -tk * gradient
            print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ',
                  "{0:.3f}".format(xk[2]), '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))
            gradient = grudfunc(xk[0], xk[1], xk[2])
            gradient_norm = linalg.norm(gradient)
            tk = abs(argminfunc(xk[0], xk[1], xk[2], gradient[0], gradient[1], gradient[2]))
            k += 1

        xk = xk + -tk * gradient
        print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ', "{0:.3f}".format(xk[2])
              , '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))


Launch = Tasks(0.01, np.array([0.0, 0.0, 0.0]), True)
Launch.grad_spusk_with_argmin_opt()