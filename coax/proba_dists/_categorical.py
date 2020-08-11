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

from gym.spaces import Discrete
import jax
import jax.numpy as jnp
import numpy as onp

from ._base import BaseProbaDist


__all__ = (
    'CategoricalDist',
)


class CategoricalDist(BaseProbaDist):
    r"""

    A differentiable categorical distribution.

    The input ``dist_params`` to each of the functions is expected to be of the form:

    .. code:: python

        dist_params = {'logits': array([...])}

    which represent the (conditional) distribution parameters. The ``logits``, denoted
    :math:`z\in\mathbb{R}^n`, are related to the categorical distribution parameters
    :math:`p\in\Delta^n` via a softmax:

    .. math::

        p_k\ =\ \text{softmax}_k(z)\ =\ \frac{\text{e}^{z_k}}{\sum_j\text{e}^{z_j}}


    Parameters
    ----------
    space : gym.spaces.Discrete

        The gym-style space that specifies the domain of the distribution.

    gumbel_softmax_tau : positive float, optional

        The parameter :math:`\tau` specifies the sharpness of the Gumbel-softmax sampling (see
        :func:`sample` method below). A good value for :math:`\tau` balances the trade-off between
        getting proper deterministic variates (i.e. one-hot vectors) versus getting smooth
        differentiable variates.

    """
    __slots__ = (*BaseProbaDist.__slots__, '_gumbel_softmax_tau')

    def __init__(self, space, gumbel_softmax_tau=0.2):
        if not isinstance(space, Discrete):
            raise TypeError(f"{self.__class__.__name__} can only be defined over Discrete spaces")

        super().__init__(space)
        self._gumbel_softmax_tau = gumbel_softmax_tau

        def sample(dist_params, rng):
            logp = jax.nn.log_softmax(dist_params['logits'])
            u = jax.random.uniform(rng, logp.shape)
            g = -jnp.log(-jnp.log(u))  # g ~ Gumbel(0,1)
            return jax.nn.softmax((g + logp) / self.gumbel_softmax_tau)

        def mode(dist_params):
            logp = jax.nn.log_softmax(dist_params['logits'])
            return jax.nn.softmax(logp / self.gumbel_softmax_tau)

        def log_proba(dist_params, X):
            Z = dist_params['logits']
            assert X.ndim == 2 and X.shape[1] == self.space.n, f"unexpected X.shape: {X.shape}"
            assert Z.ndim == 2 and Z.shape[1] == self.space.n, f"unexpected logits.shape: {Z.shape}"
            return jnp.einsum('ij,ij->i', X, jax.nn.log_softmax(Z))

        def entropy(dist_params):
            logp = jax.nn.log_softmax(dist_params['logits'])
            return jnp.einsum('ij,ij->i', jnp.exp(logp), -logp)

        def cross_entropy(dist_params_p, dist_params_q):
            p = jax.nn.softmax(dist_params_p['logits'])
            logq = jax.nn.log_softmax(dist_params_q['logits'])
            return jnp.einsum('ij,ij->i', p, -logq)

        def kl_divergence(dist_params_p, dist_params_q):
            logp = jax.nn.log_softmax(dist_params_p['logits'])
            logq = jax.nn.log_softmax(dist_params_q['logits'])
            return jnp.einsum('ij,ij->i', jnp.exp(logp), logp - logq)

        self._sample_func = jax.jit(sample)
        self._mode_func = jax.jit(mode)
        self._log_proba_func = jax.jit(log_proba)
        self._entropy_func = jax.jit(entropy)
        self._cross_entropy_func = jax.jit(cross_entropy)
        self._kl_divergence_func = jax.jit(kl_divergence)

    @property
    def hyperparams(self):
        return {'gumbel_softmax_tau': self.gumbel_softmax_tau}

    @property
    def gumbel_softmax_tau(self):
        return self._gumbel_softmax_tau

    @property
    def sample(self):
        r"""

        JIT-compiled function that generates differentiable variates using Gumbel-softmax sampling.
        :math:`x\sim\text{Cat}(p)` is implemented as

        .. math::

            u_k\ &\sim\ \text{Unif}(0, 1) \\
            g_k\ &=\ -\log(-\log(u_k)) \\
            x_k\ &=\ \text{softmax}_k\left(
                \frac{g_k + \log p_k}{\tau} \right)

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        rng : PRNGKey

            A key for seeding the pseudo-random number generator.

        Returns
        -------
        X : ndarray

            A batch of variates :math:`x\sim\text{Cat}(p)`. In order to ensure differentiability of
            the variates this is not an integer, but instead an *almost* one-hot encoded version
            thereof.

            For example, instead of sampling :math:`x=2` from a 4-class categorical distribution,
            Gumbel-softmax will return a vector like :math:`x=[0.05, 0.02, 0.86, 0.07]`. The latter
            representation can be viewed as an *almost* one-hot encoded version of the former.

        """
        return self._sample_func

    @property
    def mode(self):
        r"""

        JIT-compiled functions that generates differentiable modes of the distribution, for which we
        use a similar trick as in Gumbel-softmax sampling:

        .. math::

            \text{mode}_k\ =\ \text{softmax}_k\left( \frac{\log p_k}{\tau} \right)

        Parameters
        ----------
        dist_params : pytree with ndarray leaves

            A batch of distribution parameters.

        Returns
        -------
        X : ndarray

            A batch of variates :math:`x\sim\text{Cat}(p)`. In order to ensure differentiability of
            the variates this is not an integer, but instead an *almost* one-hot encoded version
            thereof.

            For example, instead of sampling :math:`x=2` from a 4-class categorical distribution,
            Gumbel-softmax will return a vector like :math:`x=(0.05, 0.02, 0.86, 0.07)`. The latter
            representation can be viewed as an *almost* one-hot encoded version of the former.

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

            H\ =\ -\sum_k p_k \log p_k


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

        JIT-compiled function that computes the cross-entropy of a categorical distribution
        :math:`q` relative to another categorical distribution :math:`p`:

        .. math::

            \text{CE}[p,q]\ =\ -\sum_k p_k \log q_k

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
        distribution :math:`q` relative to another categorical distribution :math:`p`:

        .. math::

            \text{KL}[p,q]\ =\ -\sum_k p_k \left(\log q_k -\log p_k\right)

        Parameters
        ----------
        dist_params_p : pytree with ndarray leaves

            The distribution parameters of the *base* distribution :math:`p`.

        dist_params_q : pytree with ndarray leaves

            The distribution parameters of the *auxiliary* distribution :math:`q`.

        """
        return self._kl_divergence_func

    @property
    def default_priors(self):
        return {'logits': jnp.zeros((1, self.space.n))}

    def postprocess_variate(self, X, batch_mode=False):
        assert X.ndim == 2
        assert X.shape[1] == self.space.n
        X = onp.argmax(X, axis=1)
        x = X[0]
        assert self.space.contains(x), \
            f"{self.__class__.__name__}.postprocessor_variate failed for X: {X}"
        return X if batch_mode else x

    def preprocess_variate(self, X):
        X = jnp.asarray(X)
        assert X.ndim <= 1, f"unexpected X.shape: {X.shape}"
        assert jnp.issubdtype(X.dtype, jnp.integer), f"expected an integer dtype, got {X.dtype}"
        return jax.nn.one_hot(X, self.space.n).reshape(-1, self.space.n)
