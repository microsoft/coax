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
from .value_q import Q


discrete = gym.spaces.Discrete(7)
boxspace = gym.spaces.Box(low=0, high=1, shape=(2, 3))


def func_type1(S, A, is_training):
    if jnp.issubdtype(S.dtype, jnp.integer):
        S = jax.nn.one_hot(S, discrete.n)
    if jnp.issubdtype(A.dtype, jnp.integer):
        A = jax.nn.one_hot(A, discrete.n)

    flatten = hk.Flatten()
    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1), jnp.ravel,
    ))
    X = jnp.concatenate((flatten(S), flatten(A)), axis=-1)
    return seq(X)


def func_type2(S, is_training):
    if jnp.issubdtype(S.dtype, jnp.integer):
        S = jax.nn.one_hot(S, discrete.n)

    batch_norm = hk.BatchNorm(False, False, 0.99)
    seq = hk.Sequential((
        hk.Flatten(),
        hk.Linear(8), jax.nn.relu,
        partial(hk.dropout, hk.next_rng_key(), 0.25 if is_training else 0.),
        partial(batch_norm, is_training=is_training),
        hk.Linear(8), jax.nn.relu,
        hk.Linear(discrete.n),
    ))
    return seq(S)


class TestQ(TestCase):
    decimal = 5

    def test_init(self):
        # cannot define a type-2 q-function on a non-discrete action space
        msg = r"type-2 q-functions are only well-defined for Discrete action spaces"
        with self.assertRaisesRegex(TypeError, msg):
            Q(func_type2, boxspace, boxspace)
        with self.assertRaisesRegex(TypeError, msg):
            Q(func_type2, discrete, boxspace)

        # these should all be fine
        Q(func_type2, boxspace, discrete)
        Q(func_type1, boxspace, boxspace)
        Q(func_type1, boxspace, discrete)
        Q(func_type2, discrete, discrete)
        Q(func_type1, discrete, boxspace)
        Q(func_type1, discrete, discrete)

    def test_call_type1_discrete(self):
        s = safe_sample(boxspace, seed=19)
        a = safe_sample(discrete, seed=19)
        q = Q(func_type1, boxspace, discrete)

        # without a
        q_s = q(s)
        self.assertArrayShape(q_s, (discrete.n,))
        self.assertArraySubdtypeFloat(q_s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_s)
        self.assertArrayAlmostEqual(q_sa, q_s[a])

    def test_call_type2_discrete(self):
        s = safe_sample(boxspace, seed=23)
        a = safe_sample(discrete, seed=23)
        q = Q(func_type2, boxspace, discrete)

        # without a
        q_s = q(s)
        self.assertArrayShape(q_s, (discrete.n,))
        self.assertArraySubdtypeFloat(q_s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_s)
        self.assertArrayAlmostEqual(q_sa, q_s[a])

    def test_call_type1_box(self):
        s = safe_sample(boxspace, seed=29)
        a = safe_sample(boxspace, seed=29)
        q = Q(func_type1, boxspace, boxspace)

        # type-1 requires a if actions space is non-discrete
        msg = r"input 'A' is required for type-1 q-function when action space is non-Discrete"
        with self.assertRaisesRegex(ValueError, msg):
            q(s)

        # with a
        q_sa = q(s, a)
        self.assertArrayShape(q_sa, ())
        self.assertArraySubdtypeFloat(q_sa)

    def test_apply_q1_as_q2(self):
        n = discrete.n  # num_actions
        q = Q(func_type1, boxspace, discrete)

        def q1_func(params, state, rng, S, A, is_training):
            return jnp.array([encode(s, a) for s, a in zip(S, A)]), state

        def encode(s, a):
            return 2 ** a + 2 ** (s + n)

        def decode(i):
            b = onp.array(list(bin(i)))[::-1]
            a = onp.argwhere(b[:n] == '1').item()
            s = onp.argwhere(b[n:] == '1').item()
            return s, a

        q._function = q1_func
        rng = jax.random.PRNGKey(0)
        params = ()
        state = ()
        is_training = True

        S = jnp.array([5, 7, 11, 13, 17, 19, 23])
        encoded_rows, _ = q.function_type2(params, state, rng, S, is_training)
        for s, encoded_row in zip(S, encoded_rows):
            for a, x in enumerate(encoded_row):
                s_, a_ = decode(x)
                self.assertEqual(s_, s)
                self.assertEqual(a_, a)

    def test_apply_q2_as_q1(self):
        n = discrete.n  # num_actions
        q = Q(func_type2, boxspace, discrete)

        def q2_func(params, state, rng, S, is_training):
            return jnp.tile(jnp.arange(n), reps=(S.shape[0], 1)), state

        q._function = q2_func
        rng = jax.random.PRNGKey(0)
        params = ()
        state = ()
        is_training = True

        S = jnp.array([5, 7, 11, 13, 17, 19, 23])
        A = jnp.array([2, 0, 1, 1, 0, 1, 2])
        Q_sa, _ = q.function_type1(params, state, rng, S, A, is_training)
        self.assertArrayAlmostEqual(Q_sa, A)

    def test_soft_update(self):
        tau = 0.13
        q = Q(func_type1, boxspace, discrete)
        q_targ = q.copy()
        q.params = jax.tree_map(jnp.ones_like, q.params)
        q_targ.params = jax.tree_map(jnp.zeros_like, q.params)
        expected = jax.tree_map(lambda a: jnp.full_like(a, tau), q.params)
        q_targ.soft_update(q, tau=tau)
        self.assertPytreeAlmostEqual(q_targ.params, expected)

    def test_function_state(self):
        q = Q(func_type1, boxspace, discrete, random_seed=11)
        print(q.function_state)
        self.assertArrayAlmostEqual(
            q.function_state['batch_norm/~/mean_ema']['average'],
            jnp.array([[0, 0.28637, 0, 0.03926, 0, 0.09223, 0, 0]]))

    def test_bad_input_signature(self):
        def badfunc(S, A, is_training, x):
            pass
        msg = (
            r"func has bad signature; "
            r"expected: func\(S, A, is_training\) or func\(S, is_training\), "
            r"got: func\(S, A, is_training, x\)")
        with self.assertRaisesRegex(TypeError, msg):
            Q(badfunc, boxspace, discrete)

    def test_bad_output_type(self):
        def badfunc(S, A, is_training):
            return 'garbage'
        msg = r"func has bad return type; expected jnp\.ndarray, got str"
        with self.assertRaisesRegex(TypeError, msg):
            Q(badfunc, boxspace, discrete)

    def test_bad_output_shape(self):
        def badfunc(S, A, is_training):
            Q = func_type1(S, A, is_training)
            return jnp.expand_dims(Q, axis=-1)
        msg = r"func has bad return shape; expected ndim=1, got ndim=2"
        with self.assertRaisesRegex(TypeError, msg):
            Q(badfunc, boxspace, discrete)

    def test_bad_output_dtype(self):
        def badfunc(S, A, is_training):
            Q = func_type1(S, A, is_training)
            return Q.astype('int32')
        msg = r"func has bad return dtype; expected a subdtype of jnp\.floating, got dtype=int32"
        with self.assertRaisesRegex(TypeError, msg):
            Q(badfunc, boxspace, discrete)
