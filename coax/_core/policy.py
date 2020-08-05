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
import haiku as hk

from .._base.mixins import PolicyMixin
from ..utils import docstring, single_to_batch, batch_to_single, safe_sample
from ..proba_dists import default_proba_dist
from .base_func import BaseFunc


class Policy(BaseFunc, PolicyMixin):
    r"""

    A parametrized (i.e. learnable) policy :math:`\pi_\theta(a|s)`.

    Parameters
    ----------
    func : function

        A Haiku-style function that specifies the forward pass. The function signature must be the
        same as the example below.

    observation_space : gym.Space

        The observation space of the environment. This is used to generate example input for
        initializing :attr:`params`.

    action_space : gym.Space

        The action space of the environment. This may be used to generate example input for
        initializing :attr:`params` or to validate the output structure.

    optimizer : optix optimizer, optional

        An optix-style optimizer. The default optimizer is :func:`optix.adam(1e-3)
        <jax.experimental.optix.adam>`.

    proba_dist : ProbaDist, optional

        A probability distribution that is used to interpret the output of :paramref:`func
        <coax.Policy.func>`. Check out the :mod:`coax.proba_dists` module for available options.

        If left unspecified, we try to get the default proba_dist
        :func:`coax.proba_dists.default_proba_dist` helper function.

    random_seed : int, optional

        Sets the random state to get reproducible results.

    """

    def __init__(
            self, func, observation_space, action_space,
            optimizer=None, proba_dist=None, random_seed=None):

        self.proba_dist = default_proba_dist(action_space) if proba_dist is None else proba_dist
        super().__init__(
            func,
            observation_space=observation_space,
            action_space=action_space,
            optimizer=optimizer,
            random_seed=random_seed)

        def sample_func(params, state, rng, S):
            rngs = hk.PRNGSequence(rng)
            dist_params, _ = self.function(params, state, next(rngs), S, False)
            A = self.proba_dist.sample(dist_params, next(rngs))
            logP = self.proba_dist.log_proba(dist_params, A)
            return A, logP

        def mode_func(params, state, rng, S):
            dist_params, _ = self.function(params, state, rng, S, False)
            A = self.proba_dist.mode(dist_params)
            return A

        self._sample_func = jax.jit(sample_func)
        self._mode_func = jax.jit(mode_func)

    @docstring(PolicyMixin.__call__)
    def __call__(self, s, return_logp=False):
        S = self._preprocess_state(s)
        A, logP = self._sample_func(self.params, self.function_state, self.rng, S)
        a = self.proba_dist.postprocess_variate(A)
        return (a, batch_to_single(logP)) if return_logp else a

    def dist_params(self, s):
        S = self._preprocess_state(s)
        dist_params = self.function(self.params, self.function_state, self.rng, S)
        return batch_to_single(dist_params)

    def greedy(self, s):
        S = self._preprocess_state(s)
        A = self._mode_func(self.params, self.function_state, self.rng, S)
        a = self.proba_dist.postprocess_variate(A)
        return a

    @property
    def sample_func(self):
        r""" The function used for sampling random actions, defined as a JIT-compiled pure function.

        .. code:: python

            output = obj.sample_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        return self._sample_func

    @property
    def mode_func(self):
        r""" The function used for sampling greedy actions, defined as a JIT-compiled pure function.

        .. code:: python

            output = obj.mode_func(obj.params, obj.function_state, obj.rng, *inputs)

        """
        return self._mode_func

    def _check_signature(self, func):
        if tuple(signature(func).parameters) != ('S', 'is_training'):
            sig = ', '.join(signature(func).parameters)
            raise TypeError(
                f"func has bad signature; expected: func(S, is_training), got: func({sig})")

        # example inputs
        S = single_to_batch(safe_sample(self.observation_space, seed=self.random_seed))
        is_training = True

        example_inputs = (S, is_training)
        static_argnums = (1,)

        return example_inputs, static_argnums

    def _check_output(self, example_output):
        if jax.tree_structure(example_output) != self.proba_dist.dist_params_structure:
            raise TypeError(
                f"func has bad return tree_structure; "
                f"expected: {jax.tree_structure(example_output)}, "
                f"got: {self.proba_dist.dist_params_structure}")
