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


def argmin_t(x1, x2, x3, p1, p2, p3):

    def one_main_f(t):
        return main_f(x1+t*p1, x2+t*p2, x3+t*p3)

    def der_one_main_f(t: float):
        t_ins = tf.Variable([[t]])
        with tf.GradientTape() as tape:
            f = one_main_f(t_ins)
        return float(tape.gradient(f, t_ins))

    t = 0.0
    eps = 0.001
    step = 0.01
    der = der_one_main_f(t)

    while der > eps:
        t = t - step * der
        der = der_one_main_f(t)
    t = t - step * der
    return t

def grad_spusk():
    eps = 0.01
    xk = np.array([0.0, 0.0, 0.0])
    tk = 0.01
    gradient = grad_f(xk[0], xk[1], xk[2])
    gradient_norm = linalg.norm(gradient)
    print('x0=', list(xk), end='\n')
    k = 1

    while gradient_norm > eps:
        xk = xk + -tk * gradient
        print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ',
              "{0:.3f}".format(xk[2]), '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))
        gradient = grad_f(xk[0], xk[1], xk[2])
        gradient_norm = linalg.norm(gradient)
        k += 1

    xk = xk + -tk * gradient
    print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ', "{0:.3f}".format(xk[2])
          , '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))

def grad_spusk_with_opt_step():
    eps = 0.01
    xk = np.array([0.0, 0.0, 0.0])
    tk = 0.1
    gradient = grad_f(xk[0], xk[1], xk[2])
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
        gradient = grad_f(xk[0], xk[1], xk[2])
        gradient_norm = linalg.norm(gradient)
        k += 1

    xk = xk + -tk * gradient
    print('k = ', k, '  ', 'xk = [', "{0:.3f}".format(xk[0]), ' ', "{0:.3f}".format(xk[1]), ' ', "{0:.3f}".format(xk[2])
          , '] ||grad(xk)|| = ', "{0:.3f}".format(gradient_norm))


print(argmin_t(0,0,0, 1, 1,1))