#!/usr/bin/env python
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


class IV_Problem(object):
    """
    Initial value problem (IVP) class"""
    def __init__(self, rhs, y0, interval, name='IVP'):
        """
        rhs         'right hand side' function of the orindary differential
                    equation f(t, y)
        y0          array with initial values
        interval    start and end value of the interval of independent
                    variables; often initial and end times
        name        descriptive name of the problem
        """
        self.rhs = rhs
        self.y0 = y0
        self.t0, self.tend = interval


def rhs(t, y):
    g = 9.81
    l = 1.
    yprime = np.array([y[1], g / l * np.sin(y[0])])
    return yprime


class IVPsolver(object):
    """
    IVP solver class for explicit one-step discretization methods with
    constant step size
    """
    def __init__(self, problem, discretization, stepsize):
        self.problem = problem
        self.discretization = discretization
        self.stepsize = stepsize

    def one_stepper(self):
        yield self.problem.t0, self.problem.y0
        ys = self.problem.y0
        ts = self.problem.t0

        while ts <= self.problem.tend:
            ts, ys = self.discretization(self.problem.rhs, ts, ys,
                                         self.stepsize)
            yield ts, ys

    def solve(self):
        return list(self.one_stepper())


def expliciteuler(rhs, ts, ys, h):
    return ts + h, ys + h * rhs(ts, ys)


def rungekutta4(rhs, ts, ys, h):
    k1 = h * rhs(ts, ys)
    k2 = h * rhs(ts + h / 2., ys + 1 / 2. * k1)
    k3 = h * rhs(ts + h / 2., ys + 1 / 2. * k2)
    k4 = h * rhs(ts + h, ys + k3)
    return ts + h, ys + 1. / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == "__main__":
    pendulum = IV_Problem(rhs, np.array([np.pi / 2, 0]), [0., 10.],
                          'mathem. pendulum')
    pendulum_Euler = IVPsolver(pendulum, expliciteuler, 0.001)
    pendulum_RK4 = IVPsolver(pendulum, rungekutta4, 0.001)

    sol_Euler = pendulum_Euler.solve()
    sol_RK4 = pendulum_RK4.solve()
    tEuler, yEuler = zip(*sol_Euler)
    tRK4, yRK4 = zip(*sol_RK4)
    plt.subplot(1, 2, 1), plt.plot(tEuler, yEuler), \
        plt.title('Pendulum result with Explicit Euler'), \
        plt.xlabel('time'), plt.ylabel('Angle and angular velocity')
    plt.subplot(1, 2, 2),
    plt.plot(tRK4, np.abs(np.array(yRK4) - np.array(yEuler))), \
        plt.title('Difference between both methods'), \
        plt.xlabel('Time'), plt.ylabel('Angle and angular velocity')
