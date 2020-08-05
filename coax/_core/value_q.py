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

import gym
import jax
import jax.numpy as jnp

from ..utils import single_to_batch, batch_to_single, safe_sample
from .base_func import BaseFunc


__all__ = (
    'Q',
)


class Q(BaseFunc):
    r"""

    A state-action value function :math:`q(s,a)`.

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


    Examples
    --------

    Here's an example where the observation space and the action space are both a
    :class:`gym.spaces.Box`.

    .. code:: python

        from functools import partial

        import gym
        import coax
        import jax
        import jax.numpy as jnp
        import haiku as hk
        from jax.experimental import optix


        def func(S, A, is_training):
            rng1, rng2, rng3 = hk.next_rng_keys(3)
            rate = 0.25 if is_training else 0.
            seq = hk.Sequential((
                hk.Linear(8), jax.nn.relu, partial(hk.dropout, rng1, rate),
                hk.Linear(8), jax.nn.relu, partial(hk.dropout, rng2, rate),
                hk.Linear(8), jax.nn.relu, partial(hk.dropout, rng3, rate),
                hk.Linear(1, w_init=jnp.zeros), jnp.ravel,
            ))
            return seq(jnp.concatenate((S, A), axis=-1))


        env = gym.make('Pendulum-v0')

        # the state value function
        q = coax.Q(func, env.observation_space, env.action_space, optimizer=optix.adam(0.01))

        # example usage:
        s = env.observation_space.sample()
        a = env.action_space.sample()
        q(s, a)  # returns a float


    The inputs ``S`` and ``A`` are batches of state observations and actions respectively, and
    ``is_training`` is a single boolean flag that indicates whether or not to run the forward-pass
    in training mode.

    **Discrete action spaces**

    Here's an example where the action space is :class:`gym.spaces.Discrete`. Having a discrete
    action space reveals some additional functionality. This comes from the fact that we may model
    the q-function in two different ways, which we refer to type-1 and type-2:

    .. math::

        (s,a)   &\mapsto q(s,a)\in\mathbb{R}    &\qquad (\text{qtype} &= 1) \\
        s       &\mapsto q(s,.)\in\mathbb{R}^n  &\qquad (\text{qtype} &= 2)

    where :math:`n` is the number of discrete actions. A **type-1** q-function is defined in the
    standard way, i.e. similar to what we did for a non-discrete actions (above):

    .. code:: python

        env = gym.make('CartPole-v0')

        def func_type1(S, A, is_training):
            seq = hk.Sequential((
                hk.Linear(8), jax.nn.relu,
                hk.Linear(8), jax.nn.relu,
                hk.Linear(8), jax.nn.relu,
                hk.Linear(1, w_init=jnp.zeros), jnp.ravel,
            ))
            A = jax.nn.one_hot(A, env.action_space.n)
            return seq(jnp.concatenate((S, A), axis=-1))

        q = coax.Q(func_type1, env.observation_space, env.action_space)

    A **type-2** q-function is defined differently. The forward-pass function omits the action input
    and returns a vector of size :math:`n`:

    .. code:: python

        def func_type2(S, is_training):
            seq = hk.Sequential((
                hk.Linear(8), jax.nn.relu,
                hk.Linear(8), jax.nn.relu,
                hk.Linear(8), jax.nn.relu,
                hk.Linear(env.action_space.n, w_init=jnp.zeros),
            ))
            return seq(S)

        q = coax.Q(func_type2, env.observation_space, env.action_space)

    An additional feature of discrete action spaces is that we may omit the action inputs, e.g.

    .. code:: python

        q(s, a)  # returns a single float
        q(s)     # returns a vector of floats

    This functionality works for both type-1 and type-2 q-functions. Finally, we remark that the
    q-function type is accessible as the :attr:`qtype` property.

    """

    def __init__(self, func, observation_space, action_space, optimizer=None, random_seed=None):
        super().__init__(
            func,
            observation_space=observation_space,
            action_space=action_space,
            optimizer=optimizer,
            random_seed=random_seed)

        def apply_single_type1(params, state, rng, s, a):
            S = single_to_batch(s)
            A = single_to_batch(a)
            Q_sa, _ = self.function_type1(params, state, rng, S, A, False)
            q_sa = batch_to_single(Q_sa)
            return q_sa

        def apply_single_type2(params, state, rng, s):
            S = single_to_batch(s)
            Q_s, _ = self.function_type2(params, state, rng, S, False)
            q_s = batch_to_single(Q_s)
            return q_s

        self._apply_single_type1 = jax.jit(apply_single_type1)
        self._apply_single_type2 = jax.jit(apply_single_type2)

    def __call__(self, s, a=None):
        r"""

        Evaluate the state-action function on a state observation :math:`s` or
        on a state-action pair :math:`(s, a)`.

        Parameters
        ----------
        s : state observation

            A single state observation :math:`s`.

        a : action

            A single action :math:`a`.

        Returns
        -------
        q_sa or q_s : ndarray

            Depending on whether ``a`` is provided, this either returns a scalar representing
            :math:`q(s,a)\in\mathbb{R}` or a vector representing :math:`q(s,.)\in\mathbb{R}^n`,
            where :math:`n` is the number of discrete actions. Naturally, this only applies for
            discrete action spaces.

        """
        s = self._preprocess_state(s)
        assert self.observation_space.contains(s), f"bad state: {s}"
        if a is None:
            return self._apply_single_type2(self.params, self.function_state, self.rng, s)

        assert self.action_space.contains(a), f"bad action: {a}"
        return self._apply_single_type1(self.params, self.function_state, self.rng, s, a)

    def batch_eval(self, S, A=None):
        r"""

        Evaluate the value function on a batch of state observations.

        This modifies the :attr:`func_approx.params <coax.FuncApprox.params>`
        attribute.

        Parameters
        ----------
        S : ndarray

            A batch of state observations :math:`s`.

        A : ndarray, optional

            A batch of actions :math:`a`. This may be omitted if the action space is discrete.

        Returns
        -------
        Q_sa or Q_s : ndarray

            Depending on whether ``A`` is provided, this either returns a batch of scalars
            representing :math:`q(s,a)\in\mathbb{R}` or a batch of vectors representing
            :math:`q(s,.)\in\mathbb{R}^n`, where :math:`n` is the number of discrete actions.
            Naturally, this only applies for discrete action spaces.

        """
        if A is None:
            Q, _ = self.function_type2(self.params, self.function_state, self.rng, S, False)
        else:
            Q, _ = self.function_type1(self.params, self.function_state, self.rng, S, A, False)
        return Q

    @property
    def function_type1(self):
        r"""

        JIT-compiled function responsible for the forward-pass through the underlying function
        approximator. This function is used by the :attr:`batch_eval` and :attr:`__call__` methods.

        This is the type-1 version of the apply-function, regardless of the underlying
        :attr:`qtype`.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        S : state observations

            A batch of state observations.

        A : actions

            A batch of actions.

        is_training : bool

            A flag that indicates whether we are in training mode.

        Returns
        -------
        Q_sa : ndarray

            A batch of state-action values :math:`q(s,a)`.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        """
        if self.qtype == 1:
            return self.function

        assert isinstance(self.action_space, gym.spaces.Discrete)
        n = self.action_space.n

        def q1_func(q2_params, q2_state, rng, S, A, is_training):
            A_onehot = jax.nn.one_hot(A, n)
            Q_s, state_new = self.function(q2_params, q2_state, rng, S, is_training)
            Q_sa = jnp.einsum('ij,ij->i', A_onehot, Q_s)
            return Q_sa, state_new

        return q1_func

    @property
    def function_type2(self):
        r"""

        JIT-compiled function responsible for the forward-pass through the underlying function
        approximator. This function is used by the :attr:`batch_eval` and :attr:`__call__` methods.

        This is the type-2 version of the apply-function, regardless of the underlying
        :attr:`qtype`.

        Parameters
        ----------
        params : pytree with ndarray leaves

            The model parameters (weights) used by the underlying q-function.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        rng : PRNGKey

            A key to seed JAX's pseudo-random number generator.

        S : state observations

            A batch of state observations.

        is_training : bool

            A flag that indicates whether we are in training mode.

        Returns
        -------
        Q_s : ndarray

            A batch of vector-valued state-action values :math:`q(s,.)`, one for each discrete
            action.

        state : pytree

            The internal state of the forward-pass function. See :attr:`function_state` and
            :func:`haiku.transform_with_state` for more details.

        """
        if self.qtype == 2:
            return self.function

        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise ValueError(
                "input 'A' is required for type-1 q-function when action space is non-Discrete")

        n = self.action_space.n

        def q2_func(q1_params, q1_state, rng, S, is_training):
            # example: let S = [7, 2, 5, 8] and num_actions = 3, then
            # S_rep = [7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8]  # repeated
            # A_rep = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]  # tiled
            S_rep = jnp.repeat(S, n, axis=0)
            A_rep = jnp.tile(jnp.arange(n), S.shape[0])

            # evaluate on replicas => output shape: (batch * num_actions, 1)
            Q_sa_rep, state_new = self.function(q1_params, q1_state, rng, S_rep, A_rep, is_training)
            Q_s = Q_sa_rep.reshape(-1, n)  # shape: (batch, num_actions)

            return Q_s, state_new

        return q2_func

    @property
    def qtype(self):
        r"""

        Specifier for how the q-function is modeled, i.e.

        .. math::

            (s,a)   &\mapsto q(s,a)\in\mathbb{R}    &\qquad (\text{qtype} &= 1) \\
            s       &\mapsto q(s,.)\in\mathbb{R}^n  &\qquad (\text{qtype} &= 2)

        Note that qtype=2 is only well-defined if the action space is :class:`Discrete
        <gym.spaces.Discrete>`. Namely, :math:`n` is the number of discrete actions.

        """
        return self._qtype

    def _check_signature(self, func):
        sig_type1 = ('S', 'A', 'is_training')
        sig_type2 = ('S', 'is_training')

        discrete = isinstance(self.action_space, gym.spaces.Discrete)
        sig = tuple(signature(func).parameters)

        if sig not in (sig_type1, sig_type2):
            sig = ', '.join(sig)
            alt = ' or func(S, is_training)' if discrete else ''
            raise TypeError(
                f"func has bad signature; expected: func(S, A, is_training){alt}, got: func({sig})")

        if sig == sig_type2 and not discrete:
            raise TypeError("type-2 q-functions are only well-defined for Discrete action spaces")

        # example inputs
        S = single_to_batch(safe_sample(self.observation_space, seed=self.random_seed))
        A = single_to_batch(safe_sample(self.action_space, seed=self.random_seed))
        is_training = True

        if sig == sig_type1:
            self._qtype = 1
            example_inputs = (S, A, is_training)
            static_argnums = (2,)
        else:
            self._qtype = 2
            example_inputs = (S, is_training)
            static_argnums = (1,)

        return example_inputs, static_argnums

    def _check_output(self, example_output):
        if not isinstance(example_output, jnp.ndarray):
            class_name = example_output.__class__.__name__
            raise TypeError(f"func has bad return type; expected jnp.ndarray, got {class_name}")

        if not jnp.issubdtype(example_output.dtype, jnp.floating):
            dt = example_output.dtype
            raise TypeError(
                f"func has bad return dtype; expected a subdtype of jnp.floating, got dtype={dt}")

        if self.qtype == 1 and example_output.ndim != 1:
            ndim = example_output.ndim
            raise TypeError(f"func has bad return shape; expected ndim=1, got ndim={ndim}")

        if self.qtype == 2 and example_output.ndim != 2:
            ndim = example_output.ndim
            raise TypeError(f"func has bad return shape; expected ndim=2, got ndim={ndim}")

        if self.qtype == 2 and example_output.shape[1] != self.action_space.n:
            k = example_output.shape[1]
            raise TypeError(f"func has bad return shape; expected shape=(?, 2), got shape=(?, {k})")
