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

import gym
import jax
import jax.numpy as jnp
import numpy as onp
import haiku as hk

from .._base.test_case import TestCase
from ..utils import safe_sample
from .policy import Policy


discrete = gym.spaces.Discrete(7)
boxspace = gym.spaces.Box(low=0, high=1, shape=(3, 5))
multidiscrete = gym.spaces.MultiDiscrete([11, 5, 7])


def func_discrete(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(discrete.n),
    ))
    return {'logits': seq(S)}


def func_boxspace(S, is_training):
    batch_norm = hk.BatchNorm(False, False, 0.99)
    mu = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape)),
        hk.Reshape(boxspace.shape),
    ))
    logvar = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jnp.tanh,
        hk.Linear(onp.prod(boxspace.shape)),
        hk.Reshape(boxspace.shape),
    ))
    return {'mu': mu(S), 'logvar': logvar(S)}


class TestPolicy(TestCase):
    def test_init(self):
        # cannot define a type-2 q-function on a non-discrete action space
        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\(dict\[\['logvar', 'mu'\]\], \[\*,\*\]\), "
            r"got: PyTreeDef\(dict\[\['logits'\]\], \[\*\]\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            Policy(func_discrete, boxspace, boxspace)

        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\(dict\[\['logits'\]\], \[\*\]\), "
            r"got: PyTreeDef\(dict\[\['logvar', 'mu'\]\], \[\*,\*\]\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            Policy(func_boxspace, boxspace, discrete)

        # these should all be fine
        Policy(func_discrete, boxspace, discrete)
        Policy(func_boxspace, boxspace, boxspace)

    def test_call_discrete(self):
        s = safe_sample(boxspace, seed=17)
        pi = Policy(func_discrete, boxspace, discrete, random_seed=19)

        a = pi(s)
        print(a, discrete)
        self.assertTrue(discrete.contains(a))
        self.assertEqual(a, 3)

    def test_call_box(self):
        s = safe_sample(boxspace, seed=17)
        pi = Policy(func_boxspace, boxspace, boxspace, random_seed=19)

        a = pi(s)
        print(type(a), a.shape, a.dtype)
        print(a)
        self.assertTrue(boxspace.contains(a))
        self.assertArrayShape(a, (3, 5))
        self.assertArraySubdtypeFloat(a)

    def test_greedy_discrete(self):
        s = safe_sample(boxspace, seed=17)
        pi = Policy(func_discrete, boxspace, discrete, random_seed=19)

        a = pi.greedy(s)
        self.assertTrue(discrete.contains(a))

    def test_greedy_box(self):
        s = safe_sample(boxspace, seed=17)
        pi = Policy(func_boxspace, boxspace, boxspace, random_seed=19)

        a = pi.greedy(s)
        print(type(a), a.shape, a.dtype)
        print(a)
        self.assertTrue(boxspace.contains(a))
        self.assertArrayShape(a, (3, 5))
        self.assertArraySubdtypeFloat(a)

    def test_function_state(self):
        pi = Policy(func_discrete, boxspace, discrete, random_seed=19)
        print(pi.function_state)
        batch_norm_avg = pi.function_state['batch_norm/~/mean_ema']['average']
        self.assertArrayShape(batch_norm_avg, (1, 8))
        self.assertArrayNotEqual(batch_norm_avg, jnp.zeros_like(batch_norm_avg))

    def test_bad_input_signature(self):
        def badfunc(S, is_training, x):
            pass
        msg = (
            r"func has bad signature; "
            r"expected: func\(S, is_training\), got: func\(S, is_training, x\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            Policy(badfunc, boxspace, discrete, random_seed=13)

    def test_bad_output_structure(self):
        def badfunc(S, is_training):
            dist_params = func_discrete(S, is_training)
            dist_params['foo'] = jnp.zeros(1)
            return dist_params
        msg = (
            r"func has bad return tree_structure, "
            r"expected: PyTreeDef\(dict\[\['logits'\]\], \[\*\]\), "
            r"got: PyTreeDef\(dict\[\['foo', 'logits'\]\], \[\*,\*\]\)"
        )
        with self.assertRaisesRegex(TypeError, msg):
            Policy(badfunc, boxspace, discrete, random_seed=13)
