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

r"""
.. autosummary::
    :nosignatures:

    coax.Agent

----

Agents
======

This module provides the abstractions required for building distributed agents.

The way this works in **coax** is to define a :class:`coax.Agent` class and then to create multiple
instances of that class, which can play different roles. For instance, below is an example of an
Ape-X DQN agent.

Example: Ape-X DQN
------------------

.. code:: python

    class ApexDQN(coax.Agent):
        def __init__(self, env, q_updater, tracer, buffer=None, param_store=None, name=None):
            self.q_updater = q_updater
            self.q = self.q_updater.q
            self.q_targ = self.q_updater.q_targ
            super().__init__(
                env=env,
                pi=coax.BoltzmannPolicy(self.q, temperature=0.015),
                tracer=tracer,
                buffer=buffer,
                param_store=param_store,
                name=name)

        def get_state(self):
            return (
                self.q.params,
                self.q.function_state,
                self.q_targ.params,
                self.q_targ.function_state,
            )

        def set_state(self, state):
            (
                self.q.params,
                self.q.function_state,
                self.q_targ.params,
                self.q_targ.function_state,
            ) = state

        def update(self, s, a, r, done, logp):
            self.tracer.add(s, a, r, done, logp)
            self.q_targ.soft_update(self.q, tau=0.001)
            if done:
                transition_batch = self.tracer.flush()
                td_error = self.q_updater.td_error(transition_batch)
                self.buffer_add(transition_batch, td_error)

        def batch_update(self, transition_batch):
            metrics, td_error = self.q_updater.update(transition_batch, return_td_error=True)
            self.buffer_update(transition_batch.idx, td_error)
            self.push_metrics(metrics)


    def make_env():
        env = gym.make('PongNoFrameskip-v4')  # AtariPreprocessing will do frame skipping
        env = gym.wrappers.AtariPreprocessing(env)
        env = coax.wrappers.FrameStacking(env, num_frames=3)
        env = coax.wrappers.TrainMonitor(env, name=name)
        env.spec.reward_threshold = 19.
        return env


    def forward_pass(S, is_training):
        seq = hk.Sequential((
            coax.utils.diff_transform,
            hk.Conv2D(16, kernel_shape=8, stride=4), jax.nn.relu,
            hk.Conv2D(32, kernel_shape=4, stride=2), jax.nn.relu,
            hk.Flatten(),
            hk.Linear(256), jax.nn.relu,
            hk.Linear(make_env().action_space.n, w_init=jnp.zeros),
        ))
        X = jnp.stack(S, axis=-1) / 255.  # stack frames
        return seq(X)


    # function approximator
    q = coax.Q(forward_pass, make_env())

    # updater
    qlearning = coax.td_learning.QLearning(q, q_targ=q.copy(), optimizer=adam(3e-4))

    # reward tracer and replay buffer
    tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)

    # ray-remote versions of our agent and replay buffer
    RemoteApexDQN = ray.remote(ApexDQN)
    RemoteBuffer = ray.remote(coax.experience_replay.PrioritizedReplayBuffer)

    buffer = RemoteBuffer.remote(capacity=1000000)
    param_store = RemoteApexDQN.remote(q_updater, tracer, buffer, name='param_store')

    actors = [
        RemoteApexDQN.remote(make_env, q_updater, tracer, buffer, param_store, name=f'actor_{i}')
        for i in range(4)]

    learner = RemoteApexDQN.remote(make_env, q_updater, tracer, buffer, param_store, name='learner')

    # block until one of the remote processes terminates
    ray.wait([
        learner.batch_update_loop.remote(),
        *(actor.rollout_loop.remote() for actor in actors)
    ])




Object Reference
----------------

.. autoclass:: coax.envs.ConnectFourEnv

"""
import time
import inspect
from abc import ABC, abstractmethod
from copy import deepcopy

import gym
import ray
from ray.actor import ActorHandle

from ..wrappers import TrainMonitor


__all__ = (
    'Worker',
)


class WorkerError(Exception):
    pass


class Worker(ABC):
    def __init__(self, env, pi, tracer, is_driver=False, buffer=None, param_store=None, name=None):
        self.env = _check_env(env, name)
        self.pi = deepcopy(pi)
        self.tracer = deepcopy(tracer)
        self.buffer = deepcopy(buffer)
        self.param_store = param_store
        self.name = name

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def trace(self, s, a, r, done, logp):
        pass

    @abstractmethod
    def learn(self, transition_batch):
        pass

    def rollout(self):
        s = self.env.reset()
        for t in range(self.env.spec.max_episode_steps):
            a, logp = self.pi(s, return_logp=True)
            s_next, r, done, info = self.env.step(a)

            self.update(s, a, r, done, logp)

            if done:
                break

            s = s_next

    def rollout_loop(self, max_total_steps, reward_threshold=None):
        reward_threshold = _check_reward_threshold(reward_threshold, self.env)
        while self.env.T < max_total_steps and self.env.avg_G < reward_threshold:
            self.pull_state()
            self.rollout()
            metrics = self.pull_metrics()
            metrics['throughput/rollout_loop'] = 1000 / self.env.dt_ms
            self.env.record_metrics(metrics)

    def learn_loop(self, max_total_steps, batch_size=32):
        total_steps = 0
        throughput = 0.
        while total_steps < max_total_steps:
            t_start = time.time()
            self.pull_state()
            metrics = self.learn(self.buffer_sample(batch_size=batch_size))
            metrics['throughput/learn_loop'] = throughput
            self.push_state()
            self.push_metrics(metrics)
            throughput = batch_size / (time.time() - t_start)

    def buffer_len(self):
        assert self.buffer is not None
        if self.param_store is None:
            len_ = len(self.buffer)
        elif isinstance(self.param_store, ActorHandle):
            len_ = ray.get(self.param_store.buffer_len.remote())
        else:
            len_ = self.param_store.buffer_len()
        return len_

    def buffer_add(self, transition_batch, Adv=None):
        assert self.buffer is not None
        if self.param_store is None:
            if 'Adv' in inspect.signature(self.buffer.add).parameters:  # duck typing
                self.buffer.add(deepcopy(transition_batch), Adv=deepcopy(Adv))
            else:
                self.buffer.add(deepcopy(transition_batch))
        elif isinstance(self.param_store, ActorHandle):
            ray.get(self.param_store.buffer_add.remote(transition_batch, Adv=Adv))
        else:
            self.param_store.buffer_add(transition_batch, Adv=Adv)

    def buffer_update(self, transition_batch_idx, Adv):
        assert self.buffer is not None
        if self.param_store is None:
            self.buffer.update(deepcopy(transition_batch_idx), Adv=deepcopy(Adv))
        elif isinstance(self.param_store, ActorHandle):
            ray.get(self.param_store.buffer_update.remote(transition_batch_idx, Adv))
        else:
            self.param_store.buffer_update(transition_batch_idx, Adv)

    def buffer_sample(self, batch_size=32):
        assert self.buffer is not None
        wait_secs = 1 / 1024.
        while self.buffer_len() < batch_size:
            self.env.logger.info(f"waiting for buffer to be populated for {wait_secs}s")
            time.sleep(wait_secs)
            wait_secs = min(30, wait_secs * 2)  # wait at most 30s between tries

        if self.param_store is None:
            transition_batch = self.buffer.sample(batch_size=batch_size)
        elif isinstance(self.param_store, ActorHandle):
            transition_batch = ray.get(self.param_store.buffer_sample.remote(batch_size=batch_size))
        else:
            transition_batch = self.param_store.buffer_sample(batch_size=batch_size)
        assert transition_batch is not None
        return transition_batch

    def pull_state(self):
        if self.param_store is None:
            raise
        if isinstance(self.param_store, ActorHandle):
            self.set_state(ray.get(self.param_store.get_state.remote()))
        else:
            self.set_state(self.param_store.get_state())

    def push_state(self):
        assert self.param_store is not None
        if isinstance(self.param_store, ActorHandle):
            ray.get(self.param_store.set_state.remote(self.get_state()))
        else:
            self.param_store.set_state(self.get_state())

    def pull_metrics(self):
        if self.param_store is None:
            metrics = self.env.get_metrics()
        elif isinstance(self.param_store, ActorHandle):
            metrics = ray.get(self.param_store.pull_metrics.remote()).copy()
        else:
            metrics = self.param_store.pull_metrics()
        return metrics

    def push_metrics(self, metrics):
        if self.param_store is None:
            self.env.record_metrics(metrics)
        elif isinstance(self.param_store, ActorHandle):
            ray.get(self.param_store.push_metrics.remote(metrics))
        else:
            self.param_store.push_metrics(metrics)

    def pull_getattr(self, name, default_value=...):
        if self.param_store is None:
            value = _getattr_recursive(self, name, deepcopy(default_value))
        elif isinstance(self.param_store, ActorHandle):
            value = ray.get(self.pull_getattr.remote(name, default_value))
        else:
            value = self.pull_getattr(name, default_value)
        return value

    def push_setattr(self, name, value):
        if self.param_store is None:
            _setattr_recursive(self, name, deepcopy(value))
        elif isinstance(self.param_store, ActorHandle):
            ray.get(self.push_setattr.remote(name, value))
        else:
            self.push_setattr(name, value)


# -- some helper functions (boilerplate) --------------------------------------------------------- #


def _check_env(env, name):
    if isinstance(env, gym.Env):
        pass
    elif isinstance(env, str):
        env = gym.make(env)
    elif hasattr(env, '__call__'):
        env = env()
    else:
        raise TypeError(f"env must be a gym.Env, str or callable; got: {type(env)}")

    if getattr(getattr(env, 'spec', None), 'max_episode_steps', None) is None:
        raise ValueError(
            "env.spec.max_episode_steps not set; please register env with "
            "gym.register('Foo-v0', entry_point='foo.Foo', max_episode_steps=...) "
            "or wrap your env with: env = gym.wrappers.TimeLimit(env, max_episode_steps=...)")

    if not isinstance(env, TrainMonitor):
        env = TrainMonitor(env, name=name, log_all_metrics=True)

    return env


def _check_reward_threshold(reward_threshold, env):
    if reward_threshold is None:
        reward_threshold = getattr(getattr(env, 'spec', None), 'reward_threshold', None)
    if reward_threshold is None:
        reward_threshold = float('inf')
    return reward_threshold


def _getattr_recursive(obj, name, default=...):
    if '.' not in name:
        return getattr(obj, name) if default is Ellipsis else getattr(obj, name, default)

    name, subname = name.split('.', 1)
    return _getattr_recursive(getattr(obj, name), subname, default)


def _setattr_recursive(obj, name, value):
    if '.' not in name:
        return setattr(obj, name, value)

    name, subname = name.split('.', 1)
    return _setattr_recursive(getattr(obj, name), subname, value)
