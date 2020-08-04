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

from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk

from .._base.test_case import TestCase
from ..utils import get_transition
from .value_v import V


def func(S, is_training):
    rng1, rng2, rng3 = hk.next_rng_keys(3)
    rate = 0.25 if is_training else 0.
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, rng1, rate),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, rng2, rate),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, rng3, rate),
        partial(batch_norm, is_training=is_training),
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel,
    ))
    return seq(S)


class TestV(TestCase):
    margin = 0.1
    decimal = 6

    def setUp(self):
        self.v = V(func, self.env_discrete.observation_space, random_seed=13)
        self.transition = get_transition(self.env_discrete)
        self.transition_batch = self.transition.to_batch()

    def tearDown(self):
        del self.v, self.transition

    def test_batch_eval(self):
        V = self.v.batch_eval(self.transition_batch.S)
        self.assertArrayAlmostEqual(V, [0.])

    def test_call(self):
        v = self.v(self.transition.s)
        self.assertAlmostEqual(v, 0.)

    def test_soft_update(self):
        tau = 0.13
        v = self.v
        v_targ = v.copy()
        v.params = jax.tree_map(jnp.zeros_like, v.params)
        v_targ.params = jax.tree_map(jnp.ones_like, v.params)
        expected = jax.tree_map(lambda a: jnp.full_like(a, tau), v.params)
        v.soft_update(v_targ, tau=tau)
        self.assertPytreeAlmostEqual(v.params, expected)

    def test_function_state(self):
        print(self.v.function_state)
        self.assertArrayAlmostEqual(
            self.v.function_state['batch_norm/~/mean_ema']['average'],
            jnp.array([[0, 0.139146, 1.829144, 0, 0, 0.954466, 0, 0]]))
