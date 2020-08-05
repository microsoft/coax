Value Functions
===============

The are two kinds of value functions, state value functions :math:`v(s)` and state-action value
functions (or q-functions) :math:`q(s,a)`. The state value function evaluates the expected
(discounted) return, defined as:

.. math::

    v(s)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma\,R_{t+1} + \gamma^2 R_{t+2} + \dots \,\Big|\, S_t=s
    \right\}

The operator :math:`\mathbb{E}_t` takes the expectation value over all transitions (indexed by
:math:`t`). The :math:`v(s)` function is implemented by the :class:`coax.V` class. The state-action
value is defined in a similar way:

.. math::

    q(s,a)\ =\ \mathbb{E}_t\left\{
        R_t + \gamma\,R_{t+1} + \gamma^2 R_{t+2} + \dots \,\Big|\, S_t=s, A_t=a
    \right\}

This is implemented by the :class:`coax.Q` class.


Object Reference
----------------

.. autosummary::
    :nosignatures:

    coax.V
    coax.Q

.. autoclass:: coax.V
.. autoclass:: coax.Q
