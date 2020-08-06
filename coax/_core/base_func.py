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

from abc import ABC, abstractmethod
from copy import deepcopy

import jax
import numpy as onp
import haiku as hk
from jax.experimental import optix
from gym.wrappers.frame_stack import LazyFrames

from .._base.mixins import RandomStateMixin
from ..utils import single_to_batch


class BaseFunc(ABC, RandomStateMixin):
    """ Abstract base class for function approximators: coax.V, coax.Q, coax.Policy """

    def __init__(self, func, observation_space, action_space, optimizer=None, random_seed=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.random_seed = random_seed

        # Haiku-transform the provided func
        example_inputs, static_argnums = self._check_signature(func)
        static_argnums = tuple(i + 3 for i in static_argnums)  # offset args: (params, state, rng)
        transformed = hk.transform_with_state(func)
        self._function = jax.jit(transformed.apply, static_argnums=static_argnums)

        # init function params and state
        self._params, self._function_state = transformed.init(self.rng, *example_inputs)

        # check if output has the expected shape etc.
        example_output, _ = \
            self._function(self.params, self.function_state, self.rng, *example_inputs)
        self._check_output(example_output)

        # optimizer
        self._optimizer = optix.adam(1e-3) if optimizer is None else optimizer
        self._optimizer_state = self.optimizer.init(self.params)

        # construct jitted param update function
        def apply_grads_func(opt, opt_state, params, grads):
            updates, new_opt_state = opt.update(grads, opt_state)
            new_params = optix.apply_updates(params, updates)
            return new_opt_state, new_params

        def soft_update_func(old, new, tau):
            return jax.tree_multimap(lambda a, b: (1 - tau) * a + tau * b, old, new)

        self._apply_grads_func = jax.jit(apply_grads_func, static_argnums=0)
        self._soft_update_func = jax.jit(soft_update_func)

    def apply_grads(self, grads):
        r""" Apply gradients to update the underlying :attr:`params`.


        Parameters
        ----------
        grads : pytree with ndarray leaves

            Gradients of some (auxiliary) loss function with respect to the model parameters. Note
            that the pytree structure must be the same as :attr:`params`.

        """
        self.optimizer_state, self.params = \
            self._apply_grads_func(self.optimizer, self.optimizer_state, self.params, grads)

    def soft_update(self, other, tau):
        r""" Synchronize the current instance with ``other`` through exponential smoothing:

        .. math::

            \theta\ \leftarrow\ \theta + \tau\, (\theta_\text{new} - \theta)

        Parameters
        ----------
        other

            A seperate copy of the current object. This object will hold the new parameters
            :math:`\theta_\text{new}`.

        tau : float between 0 and 1, optional

            If we set :math:`\tau=1` we do a hard update. If we pick a smaller value, we do a smooth
            update.

        """
        if not isinstance(other, self.__class__):
            raise TypeError("'self' and 'other' must be of the same type")

        self.params = self._soft_update_func(self.params, other.params, tau)

    def copy(self):
        """ Create a deep copy.

        Returns
        -------
        copy

            A deep copy of the current instance.

        """
        return deepcopy(self)

    @property
    def params(self):
        """ The parameters (weights) of the function approximator. """
        return self._params

    @params.setter
    def params(self, new_params):
        if jax.tree_structure(new_params) != jax.tree_structure(self._params):
            raise TypeError("new params must have the same structure as old params")
        self._params = new_params

    @property
    def function(self):
        r"""

        The function approximator itself, defined as a JIT-compiled pure function. This function may
        be called directly as:

        .. code:: python

            output, function_state = obj.function(obj.params, obj.function_state, obj.rng, *inputs)

        """
        return self._function

    @property
    def function_state(self):
        """ The state of the function approximator, see :func:`haiku.transform_with_state`. """
        return self._function_state

    @function_state.setter
    def function_state(self, new_function_state):
        if jax.tree_structure(new_function_state) != jax.tree_structure(self._function_state):
            raise TypeError("new function_state must have the same structure as old function_state")
        self._function_state = new_function_state

    @property
    def optimizer(self):
        """ The optimizer, see :mod:`jax.experimental.optix`. """
        return self._optimizer

    @property
    def optimizer_state(self):
        """ The state of the optimizer. """
        return self._optimizer_state

    @optimizer_state.setter
    def optimizer_state(self, new_optimizer_state):
        self._optimizer_state = new_optimizer_state

    @abstractmethod
    def _check_signature(self, func):
        """ Check if func has expected input signature; returns static_argnums; raises TypeError """

    @abstractmethod
    def _check_output(self, example_output):
        """ Check if func has expected output signature; raises TypeError """

    def _preprocess_state(self, s):
        if isinstance(s, LazyFrames):
            return onp.asanyarray(s)

        return single_to_batch(s)
