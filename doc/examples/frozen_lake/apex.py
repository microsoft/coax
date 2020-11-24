from time import sleep
from collections import deque

import gym
import ray
import jax
import jax.numpy as jnp
import optax
import coax
import haiku as hk


@ray.remote(num_cpus=1, num_gpus=0)
class ParameterStore:
    def __init__(self, q):
        self._q = q
        self._metrics = deque(maxlen=1000)

    def get(self):
        return self._q.params, self._q.function_state

    def set(self, params, function_state):
        self._q.params, self._q.function_state = params, function_state

    def record_metrics(self, metrics):
        self._metrics.append(metrics)

    def get_metrics(self):
        metrics = list(self._metrics)  # shallow copy
        self._metrics = deque()
        return metrics


@ray.remote(num_cpus=1, num_gpus=1)
class Learner(coax._base.mixins.LoggerMixin):
    def __init__(self, updater, buffer, param_store):
        coax.utils.enable_logging()
        self.updater = updater
        self.buffer = buffer
        self.param_store = param_store

    def load_params(self):
        params, function_state = ray.get(self.param_store.get.remote())
        self.updater.q.params, self.updater.q.function_state = params, function_state

    def run(self):
        while True:
            self._wait_for_buffer_if_empty()
            transition_batch = ray.get(self.buffer.sample.remote(batch_size=32))
            metrics, td_error = self.updater.update(transition_batch, return_td_error=True)
            ray.get([
                self.buffer.update.remote(transition_batch.idx, td_error),
                self.param_store.set.remote(self.updater.q.params, self.updater.q.function_state),
                self.param_store.record_metrics.remote(metrics),
            ])

    def _wait_for_buffer_if_empty(self):
        wait_sec = 1 / 1024.
        while not ray.get(self.buffer.__bool__.remote()):
            self.logger.info(f"buffer is empty, waiting for {wait_sec:g}s")
            sleep(wait_sec)
            wait_sec = min(60, 2 * wait_sec)


@ray.remote(num_cpus=1, num_gpus=0)
class Actor(coax._base.mixins.LoggerMixin):
    def __init__(self, env, pi, tracer, buffer, updater, param_store, i):
        coax.utils.enable_logging()
        self.i = i
        self.env = env
        self.pi = pi
        self.tracer = tracer
        self.buffer = buffer
        self.updater = updater
        self.param_store = param_store

    def load_params(self):
        params, function_state = ray.get(self.param_store.get.remote())
        self.pi.q.params, self.pi.q.function_state = params, function_state
        self.updater.q.params, self.updater.q.function_state = params, function_state

    def run(self):
        while True:
            self.load_params()

            s = self.env.reset()
            for t in range(self.env.spec.max_episode_steps):
                a, logp = self.pi(s, return_logp=True)
                s_next, r, done, info = self.env.step(a)
                if s_next == s:
                    r = -0.001

                self.tracer.add(s, a, r, done, logp)

                if done:
                    transition_batch = self.tracer.flush()
                    td_error = self.updater.td_error(transition_batch)
                    ray.get(self.buffer.add.remote(transition_batch, td_error))
                    for metrics in ray.get(self.param_store.get_metrics.remote()):
                        self.env.record_metrics(metrics)
                    break

                s = s_next

            if self.env.ep >= 500 or self.env.avg_G >= self.env.spec.reward_threshold:
                self.logger.info(f"[{self.i}] stopping after T={self.env.T} steps")
                return


ray.init()


# the MDP
env = gym.make('FrozenLakeNonSlippery-v0')
env = coax.wrappers.TrainMonitor(env)


def func(S, A, is_training):
    value = hk.Sequential((hk.Flatten(), hk.Linear(1, w_init=jnp.zeros), jnp.ravel))
    X = jax.vmap(jnp.kron)(S, A)  # S and A are one-hot encoded
    return value(X)


# function approximator
q = coax.Q(func, env)
pi = coax.BoltzmannPolicy(q, temperature=0.1)


# experience tracer
tracer = coax.reward_tracing.NStep(n=1, gamma=0.9)


# updater
qlearning = coax.td_learning.QLearning(q, optimizer=optax.adam(0.02))


# shared "actors"
buffer = ray.remote(coax.experience_replay.PrioritizedReplayBuffer).remote(capacity=10000)
param_store = ParameterStore.remote(q)


learner = Learner.remote(qlearning, buffer, param_store)
actors = [Actor.remote(env, pi, tracer, buffer, qlearning, param_store, i) for i in range(2)]

# run
learner.run.remote()
ray.wait([actor.run.remote() for actor in actors])
