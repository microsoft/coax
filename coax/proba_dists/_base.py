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

import gym
import jax

from ..utils import batch_to_single


class ProbaDist(ABC):
    r"""

    Abstract base class for probability distributions. Check out
    :class:`coax.proba_dists.CategoricalDist` for a specific example.

    """
    __slots__ = (
        'space',
        '_sample_func',
        '_mode_func',
        '_log_proba_func',
        '_entropy_func',
        '_cross_entropy_func',
        '_kl_divergence_func',
        '_default_priors_func',
    )

    def __init__(self, space):
        if not isinstance(space, gym.Space):
            raise TypeError("space must be derived from gym.Space")
        self.space = space

    @property
    def hyperparams(self):
        r""" Hyperparameters specific to this probability distribution. """
        return {}

    @property
    def sample(self):
        r"""

        JIT-compiled function that generates differentiable variates.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        rng : PRNGKey

            A key for seeding the pseudo-random number generator.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._sample_func

    @property
    def mode(self):
        r"""

        JIT-compiled functions that generates differentiable modes of the distribution.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        X : ndarray

            A batch of differentiable variates.

        """
        return self._mode_func

    @property
    def log_proba(self):
        r"""

        JIT-compiled function that evaluates log-probabilities.

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        X : ndarray

            A batch of variates, e.g. a batch of actions :math:`a` collected from experience.

        Returns
        -------
        logP : ndarray of floats

            A batch of log-probabilities associated with the provided variates.

        """
        return self._log_proba_func

    @property
    def entropy(self):
        r"""

        JIT-compiled function that computes the entropy of the distribution.

        .. math::

            H\ =\ -\mathbb{E}_p \log p


        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        H : ndarray of floats

            A batch of entropy values.

        """
        return self._entropy_func

    @property
    def cross_entropy(self):
        r"""

        JIT-compiled function that computes the cross-entropy of a distribution :math:`q` relative
        to another categorical distribution :math:`p`:

        .. math::

            \text{CE}[p,q]\ =\ -\mathbb{E}_p \log q

        Parameters
        ----------
        dist_params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        dist_params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution :math:`q`.

        """
        return self._cross_entropy_func

    @property
    def kl_divergence(self):
        r"""

        JIT-compiled function that computes the Kullback-Leibler divergence of a categorical
        distribution :math:`q` relative to another distribution :math:`p`:

        .. math::

            \text{KL}[p,q]\ = -\mathbb{E}_p \left(\log q -\log p\right)

        Parameters
        ----------
        dist_params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        dist_params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution :math:`q`.

        """
        return self._kl_divergence_func

    @property
    def dist_params_structure(self):
        r"""

        The tree structure of the distribution parameters.

        """
        return jax.tree_structure(self.default_priors(shape=()))

    @staticmethod
    @abstractmethod
    def default_priors(shape):
        r"""

        The default distribution parameters.

        Parameters
        ----------
        shape : tuple of ints

            The shape of the distribution parameters.

        Returns
        -------
        dist_params_prior : pytree with ndarray leaves

            The distribution parameters that represent the default priors.

        """
        pass

    def postprocess_variate(self, X):
        r"""

        The post-processor specific to variates drawn from this ditribution.

        This method provides the interface between differentiable, batched variates, i.e. outputs
        of :func:`sample` and :func:`mode` and the provided gym space.

        Parameters
        ----------
        X : variates

            A batch of variates sampled from this proba_dist. This will be converted into a single
            variate. Note that if the batch size is greater than one, all but the first variate are
            ignored.

        Returns
        -------
        x : variate

            A single variate that should satisfy :code:`self.space.contains(a)`.

        """
        x = batch_to_single(X)
        assert self.space.contains(x), \
            f"{self.__class__.__name__}.postprocessor_variate failed for X: {X}"
        return x
