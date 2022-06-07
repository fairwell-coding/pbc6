#!/usr/bin/env python3
#
# neural_sampler.py
#

import itertools
import numpy as np


def logsig(x):
    xa = np.asarray(x)
    m = (xa > logsig.theta)
    y = np.zeros_like(xa, dtype=float)

    y[m] = 1. / (1. + np.exp(-xa[m]))

    if np.ndim(y) == 0:
        y = float(y)

    return y

logsig.theta = -np.log(np.finfo(float).max)


class NeuralSampler:
    """
    Implements a neural sampling network with absolute refractory
    mechanism according to Buesing et al. (2011).
    """
    def __init__(self, W, b=None, tau=10, dt=1, update_random=True):
        """
        Create a NeuralSampler.

        :param W: (D, D)-shaped weight matrix (symmetric, 0 on main diagonal)
        :param b: (D,)-shaped bias vector, if None (default): use zeros
        :param tau: PSP / state length in ms, default: 10
        :param dt: time step in ms, default: 1
        :param update_random: update units in random order, default: True
        """

        # check weights

        self.W = np.asarray(W)
        assert self.W.ndim == 2, 'weights must be a matrix'
        assert self.W.shape[0] == self.W.shape[1], 'weights must be a square matrix'
        assert (np.diag(W) == 0).all(), 'cannot have autapses (main diagonal of weight matrix must be zero)'
        assert np.isclose(W, W.T).all(), 'weight matrix must be symmetric'
        self.D = self.W.shape[0]

        # check / create biases

        self.b = np.asarray(b) if b is not None else np.zeros(self.W.shape[0])
        assert self.b.ndim == 1, 'biases must be a vector'
        assert self.b.shape[0] == self.D, 'bias vector dimensionality must match weight matrix'

        # check other args

        self.tau = tau
        assert self.tau > 0

        self.dt = dt
        assert self.tau > 0

        self.tau_in_dt = round(tau / dt)
        self.log_tau_in_dt = np.log(tau / dt)

        self.update_random = update_random

        # setup state

        self.reset()  # reset to random initial state
        self.reset_clamp()  # don't clamp any units

    def reset(self, z0=None, z0_random=True):
        """
        Reset the state of the NeuralSampler.

        :param z0: (D,)-shaped state vector to use, if None (default): create one (see other args)
        :param z0_random: whether to draw the initial state at random if z0 is not passed, default: True. If False, zeros will be used. Setting this to True will also result in random values of the auxilliary variables if z0 is passed.
        """

        # set states

        if z0 is None:
            if not z0_random:
                z0 = np.zeros(self.D, dtype=int)
            else:
                z0 = np.random.randint(2, size=self.D, dtype=int)
        else:
            z0 = np.asarray(z0)

        assert z0.ndim == 1, 'state must be a vector'
        assert z0.shape[0] == self.D, 'state vector dimensionality must match weight matrix'
        self.z = z0

        # set auxiliary variables

        zeta0 = np.zeros(self.D, dtype=int)

        if z0_random:
            m = (z0 > 0)
            zeta0[m] = np.random.randint(self.tau_in_dt+1, size=m.sum(), dtype=int)

        assert zeta0.ndim == 1, 'state must be a vector'
        assert zeta0.shape[0] == self.D, 'state vector dimensionality must match weight matrix'
        self.zeta = zeta0

        # save states

        self.z_ = []

    def clamp_on(self, units):
        """
        Clamp some units to have state = 1 at all times. These units
        are not updated as usual.

        :param units: list containing indices of neurons which should be clamped.
        """
        units = np.asarray(units)
        assert units.ndim == 1
        assert units.min() >= 0
        assert units.max() < self.D

        free_units = set(range(self.D)) - set(units.tolist())
        self._free_units = list(free_units)

        # set state
        self.z[units] = 1
        self.zeta[units] = self.tau_in_dt

    def reset_clamp(self):
        """
        Resets clamping, i.e. all units will be updated.
        """
        self._free_units = list(np.arange(self.D))

    def step(self):
        """
        Perform a single update step using the transition operator.
        Only non-clamped units are updated.

        :returns: state after step
        """

        update_indices = self._free_units  # only update non-clamped

        if self.update_random:
            np.random.shuffle(update_indices)

        for k in update_indices:
            if self.zeta[k] > 1:  # is refractory
                self.zeta[k] -= 1
                continue

            u_k = self.W[:,k].T.dot(self.z) + self.b[k]
            p_k = logsig(u_k - self.log_tau_in_dt)
            rnd = np.random.rand()
            s_k = int(p_k > rnd)  # spike

            self.zeta[k] = self.tau_in_dt * s_k
            self.z[k] = s_k

        z_new = self.z.copy()
        self.z_ += [z_new]

        return z_new

    def run(self, T):
        """
        Run the sampler for a given period of time (using the time
        step dt specified when creating the NeuralSampler).

        :param T: time to run in ms
        :returns: state after running
        """

        assert T > 0
        steps = round(T / self.dt)

        for n in range(steps):
            z = self.step()

        return z

    def get_unnormalized_state_probability(self, state):
        """
        Get the unnormalized probability of a given state, i.e. the
        value under the Boltzmann distribution without taking the
        normalization constant into account..

        :param state: state
        :returns: unnormalized probability
        """
        state = np.asarray(state)
        assert state.ndim == 1 and state.shape[0] == self.D

        return np.exp(state.dot(self.W).dot(state)/2. + state.dot(self.b))

    def get_all_state_probabilities(self):
        """
        Enumerate all possible states and compute the probability of
        each under the Boltzmann distribution. Note that this may
        take very long if the state space is large. Returns two
        arrays. The first contains all possible states, the second
        the probability of each state.

        :returns: tuple of state array, probability array
        """
        all_states = np.asarray([*itertools.product([0, 1], repeat=self.D)])
        all_probabilities = np.asarray([self.get_unnormalized_state_probability(s) for s in all_states])

        return all_states, all_probabilities / all_probabilities.sum()

    @property
    def states(self):
        """
        All states since creating or resetting the NeuralSampler.
        """
        return np.asarray(self.z_)

