Policies
========

There are generally two distinct ways of constructing a policy
:math:`\pi(a|s)`. One method uses a function approximator to parametrize a
state-action value function :math:`q_\theta(s,a)` and then derives a policy
from this q-function. The other method uses a function approximator to
parametrize the policy directly, i.e. :math:`\pi(a|s)=\pi_\theta(a|s)`. The
methods are called *value-based* methods and *policy gradient* methods,
respectively.


Parametrized policies
---------------------

Let's start with the policy-gradient style function approximator :math:`\pi_\theta(a|s)`, which is
implemented by :class:`coax.Policy`.


Let's suppose we wish to construct a policy function approximator to build an agent that solve
the *Pendulum* environment. We'll start by creating some example data:

.. code:: python

    env = gym.make('Pendulum-v0')
    data = coax.example_data.policy_data(env.observation_space, env.action_space)

    print(data)
    # ExampleData(input=Inputs(S=DeviceArray([[0.2517, 1.3511 , 0.0255]], dtype=float32),
    #                          is_training=True),
    #             output={'mu': DeviceArray([[1.7124]], dtype=float32),
    #                     'logvar': DeviceArray([[-0.8897]], dtype=float32)})

Now, our task is to come up with a Haiku-style function that generates this output given the
input. To be clear, our task is not to recreate the exact values; the example data is only there
to give you an idea of the structure (shapes, dtypes, etc.).

Here's an example of how to create a valid policy function approximator for the Pendulum
environment:

.. code:: python

    @coax.policy(env)
    def pi(S, is_training):
        shared = hk.Sequential((
            hk.Flatten(),
            hk.Linear(8), jax.nn.relu,
            hk.Linear(8), jax.nn.relu,
        ))
        mu = hk.Sequential((
            shared,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(jnp.prod(env.action_space.shape)),
            hk.Reshape(env.action_space.shape),
        ))
        logvar = hk.Sequential((
            shared,
            hk.Linear(8), jax.nn.relu,
            hk.Linear(jnp.prod(env.action_space.shape)),
            hk.Reshape(env.action_space.shape),
        ))
        return {'mu': mu(S), 'logvar': logvar(S)}

    s = env.observation_space.sample()
    a = pi(s)

    print(a)
    # array([0.39267802], dtype=float32)

Here's an alternative way of defining ``pi`` that doesn't use a decorator:

.. code:: python

    def func(S, is_training):
        mu = hk.Sequential((
            ...
        ))
        logvar = hk.Sequential((
            ...
        ))
        return {'mu': mu(S), 'logvar': logvar(S)}

    pi = coax.Policy(func, env.observation_space, env.action_space)

If something goes wrong and you'd like to inspect what's going on, here's an example of you might
proceed:

.. code:: python

    rngs = hk.PRNGSequence(42)
    func = hk.transform_with_state(func)
    params, function_state = func.init(next(rngs), *data.input)
    output, function_state = func.apply(params, function_state, next(rngs), *data.input)

The code above is what the :attr:`coax.Policy.__init__` runs under the hood.


Value-based policies
--------------------

Value-based policies are defined indirectly, via a :doc:`q-function <value_functions>`. Examples of
value-based policies are :class:`coax.EpsilonGreedy` (see example below) and
:class:`coax.BoltzmannPolicy`.

.. code:: python

    pi = coax.EpsilonGreedy(q, epsilon=0.1)
    pi = coax.BoltzmannPolicy(q, temperature=0.02)


Random policy
-------------

The :class:`coax.RandomPolicy` doesn't depend on any function approximator. It
merely calls ``.sample()`` on the action space of the underlying gym
enviroment. This policy is particularly useful if you want to have a quick look
at an environment.


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.Policy
    coax.EpsilonGreedy
    coax.BoltzmannPolicy
    coax.RandomPolicy

.. autoclass:: coax.Policy
.. autoclass:: coax.EpsilonGreedy
.. autoclass:: coax.BoltzmannPolicy
.. autoclass:: coax.RandomPolicy
