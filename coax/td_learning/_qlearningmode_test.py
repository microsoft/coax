# ------------------------------------------------------------------------------------------------ #
# MIT License                                                                                      #
#                                                                                                  #
# Copyright (c) 2020, Microsoft Corporation                                                        #
#                                                                                                  #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software    #
# and associated documentation files (the "Software"), to deal in the Software without             #
# restriction, including without limitation the rights to use, copy, modify, merge, publish,       #
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the    #
# Software is furnished to do so, subject to the following conditions:                             #
#                                                                                                  #
# The above copyright notice and this permission notice shall be included in all copies or         #
# substantial portions of the Software.                                                            #
#                                                                                                  #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING    #
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,     #
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.          #
# ------------------------------------------------------------------------------------------------ #

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

from .._base.test_case import TestCase, DiscreteEnv, BoxEnv
from .._core.value_q import Q
from .._core.policy import Policy
from ..utils import get_transition
from ._qlearningmode import QLearningMode

env_discrete = DiscreteEnv(random_seed=13)
env_boxspace = BoxEnv(random_seed=17)


def func_type1_discrete(S, A, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1), jnp.ravel,
    ))
    S = hk.Flatten()(S)
    A_onehot = jax.nn.one_hot(A, env_discrete.action_space.n)
    try:
        X = jnp.concatenate((S, A_onehot), axis=-1)
    except:
        print(f"\n  S: {S}\n  A: {A}\n  A_onehot: {A_onehot}\n")
        raise
    return seq(X)


def func_type1_boxspace(S, A, is_training):
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1), jnp.ravel,
    ))
    S = hk.Flatten()(S)
    A = hk.Flatten()(A)
    try:
        X = jnp.concatenate((S, A), axis=-1)
    except:
        print(f"\n  S: {S}\n  A: {A}")
        raise
    return seq(X)


def func_type2(S, is_training):
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env_discrete.action_space.n),
    ))
    return seq(S)


def func_pi_discrete(S, is_training):
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(env_discrete.action_space.n),
    ))
    return {'logits': seq(S)}


def func_pi_boxspace(S, is_training):
    mu = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(jnp.prod(env_boxspace.action_space.shape)),
    ), name='mu')
    logvar = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(hk.BatchNorm(False, False, 0.99), is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(jnp.prod(env_boxspace.action_space.shape)),
    ), name='logvar')
    return {'mu': mu(S), 'logvar': logvar(S)}


class TestQLearningMode(TestCase):

    def setUp(self):
        self.transition_discrete = get_transition(self.env_discrete).to_batch()
        self.transition_boxspace = get_transition(self.env_box).to_batch()

    def test_update_discrete(self):
        q = Q(func_type1_discrete, env_discrete.observation_space, env_discrete.action_space)
        pi = Policy(func_pi_discrete, env_discrete.observation_space, env_discrete.action_space)
        q_targ = q.copy()
        updater = QLearningMode(q, pi, q_targ)

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_discrete)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_boxspace(self):
        q = Q(func_type1_boxspace, env_boxspace.observation_space, env_boxspace.action_space)
        pi = Policy(func_pi_boxspace, env_boxspace.observation_space, env_boxspace.action_space)
        q_targ = q.copy()
        updater = QLearningMode(q, pi, q_targ)

        params = deepcopy(q.params)
        function_state = deepcopy(q.function_state)

        updater.update(self.transition_boxspace)

        self.assertPytreeNotEqual(params, q.params)
        self.assertPytreeNotEqual(function_state, q.function_state)

    def test_update_type2(self):
        q = Q(func_type2, env_discrete.observation_space, env_discrete.action_space)
        pi = Policy(func_pi_discrete, env_discrete.observation_space, env_discrete.action_space)
        q_targ = q.copy()

        with self.assertRaisesRegex(TypeError, "q must be a type-1 q-function, got type-2"):
            QLearningMode(q, pi, q_targ)
