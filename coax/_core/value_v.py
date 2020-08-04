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

from inspect import signature

import jax
import jax.numpy as jnp

from ..utils import single_to_batch, batch_to_single
from .base_func import BaseFunc


__all__ = (
    'V',
)


class V(BaseFunc):
    r""" A state value function :math:`v(s)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass. The function signature must be the
        same as this example:

        .. code:: python

            import jax
            import jax.numpy as jnp
            import haiku as hk
            from functools import partial

            def func(S, is_training):
                rng1, rng2. rng3 = hk.next_rng_keys(3)
                rate = 0.25 if is_training else 0.
                seq = hk.Sequential((
                    hk.Linear(8), jax.nn.relu, partial(hk.dropout, rng1, rate),
                    hk.Linear(8), jax.nn.relu, partial(hk.dropout, rng2, rate),
                    hk.Linear(8), jax.nn.relu, partial(hk.dropout, rng3, rate),
                    hk.Linear(1), jnp.ravel,
                ))
                return seq(S)

        The input ``S`` is a batch of state observations and ``is_training`` is single boolean flag
        that indicates whether or not to run the forward-pass in training mode.

    s : state observation

        An example state observation, e.g.

        .. code:: python

            s = env.observation_space.sample()

        This is used to initialize the underlying function approximator in the following steps:

        .. code:: python

            S = coax.utils.single_to_batch(s)
            transformed = hk.transform_with_state(func)
            self.function = transform.apply
            self.params, self.function_state = transformed.init(rng, S)

    optimizer : optix optimizer, optional

        An optix-style optimizer. The default optimizer is :func:`optix.adam(1e-3)
        <jax.experimental.optix.adam>`.

    """

    def __init__(self, func, s, optimizer=None, random_seed=None):
        super().__init__(
            func,
            example_inputs=(single_to_batch(s), True),
            optimizer=optimizer,
            random_seed=random_seed)

        def apply_single(params, state, rng, s):
            S = single_to_batch(s)
            V, _ = self.function(params, state, rng, S, False)
            v = batch_to_single(V)
            return v

        self._apply_single = jax.jit(apply_single)

    def __call__(self, s):
        r"""

        Evaluate the value function on a state observation :math:`s`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        Returns
        -------
        v : ndarray, shape: ()

            The estimated expected value associated with the input state observation ``s``.

        """
        s = self._preprocess_state(s)
        v = self._apply_single(self.params, self.function_state, self.rng, s)
        return v

    def batch_eval(self, S):
        r"""

        Evaluate the value function on a batch of state observations.

        Parameters
        ----------
        S : ndarray

            A batch of state observations :math:`s`.

        Returns
        -------
        V : ndarray, shape: (batch_size,)

            The estimated expected value associated with the input state observations ``S``.

        """
        V, _ = self.function(self.params, self.function_state, self.rng, S, False)
        return V

    def _check_argspec(self, func):
        if tuple(signature(func).parameters) != ('S', 'is_training'):
            argspec = ', '.join(signature(func).parameters)
            raise TypeError(
                f"func has bad argspec; expected: func(S, is_training), got: func({argspec})")
        return 1 + 3  # static_argnums; the +3 offset is due to (params, state, rng) args

    def _check_output(self, example_output):
        if not isinstance(example_output, jnp.ndarray):
            class_name = example_output.__class__.__name__
            raise TypeError(f"func has bad return type; expected jnp.ndarray, got {class_name}")

        if not jnp.issubdtype(example_output.dtype, jnp.floating):
            dt = example_output.dtype
            raise TypeError(
                f"func has bad return dtype; expected a subdtype of jnp.floating, got dtype={dt}")

        if example_output.ndim != 1:
            ndim = example_output.ndim
            raise TypeError(f"func has bad return shape; expected ndim=1, got ndim={ndim}")
