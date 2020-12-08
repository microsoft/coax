import os
from threading import Thread, Lock

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # don't use all gpu mem

import gym
# import ray
import jax
import jax.numpy as jnp
import coax
import haiku as hk
import optax


# name of this script
name, _ = os.path.splitext(os.path.basename(__file__))


class ApexWorker(coax.Worker):
    learn_lock = Lock()
    rollout_lock = Lock()

    def __init__(self, name, make_env, param_store=None, tensorboard_dir=None):
        env = make_env(name, tensorboard_dir)
        super().__init__(env=env, pi=None, param_store=param_store, name=name)

        # function approximator
        self.q = coax.Q(forward_pass, self.env)
        self.pi = coax.BoltzmannPolicy(self.q, temperature=0.015)

        # target network
        self.q_targ = self.q.copy()

        # tracer and updater
        self.tracer = coax.reward_tracing.NStep(n=1, gamma=0.99)
        self.q_updater = coax.td_learning.QLearning(
            self.q, q_targ=self.q_targ, optimizer=optax.adam(3e-4))

        # replay buffer
        self.buffer = coax.experience_replay.PrioritizedReplayBuffer(capacity=1000000, alpha=0.6)
        # self.buffer_warmup = 50000
        self.beta = coax.utils.StepwiseLinearFunction((0, 0.4), (1000000, 1))

    def get_state(self):
        return self.q.params, self.q.function_state, self.q_targ.params, self.q_targ.function_state

    def set_state(self, state):
        self.q.params, self.q.function_state, self.q_targ.params, self.q_targ.function_state = state

    def trace(self, s, a, r, done, logp):
        self.tracer.add(s, a, r, done, logp)
        self.q_targ.soft_update(self.q, tau=0.001)
        if done:
            transition_batch = self.tracer.flush()
            for chunk in coax.utils.chunks_pow2(transition_batch):
                td_error = self.q_updater.td_error(chunk)
                self.buffer_add(chunk, td_error)
            self.push_setattr('buffer.beta', self.beta(self.env.T))

    def learn(self, transition_batch):
        self.learn_lock.acquire()
        # self.env.logger.debug("learn")
        metrics, td_error = self.q_updater.update(transition_batch, return_td_error=True)
        self.buffer_update(transition_batch.idx, td_error)
        self.rollout_lock.release()
        return metrics

    def rollout(self):
        assert self.pi is not None

        s = self.env.reset()

        for t in range(self.env.spec.max_episode_steps):
            # acquire lock
            self.rollout_lock.acquire()
            # self.env.logger.debug("rollout")

            a, logp = self.pi(s, return_logp=True)
            s_next, r, done, info = self.env.step(a)

            self.trace(s, a, r, done, logp)

            if done:
                self.push_setattr('env.T', self.pull_getattr('env.T') + t)

            # release lock
            if self.buffer_len() < (self.buffer_warmup or 32):
                self.rollout_lock.release()
            else:
                self.learn_lock.release()

            if done:
                break

            s = s_next


def make_env(name=None, tensorboard_dir=None):
    import logging
    fmt = '[%(threadName)s|%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    env = gym.make('PongNoFrameskip-v4')  # AtariPreprocessing will do frame skipping
    env = gym.wrappers.AtariPreprocessing(env)
    env = coax.wrappers.FrameStacking(env, num_frames=3)
    env = coax.wrappers.TrainMonitor(env, tensorboard_dir=tensorboard_dir)
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


# start ray cluster
# ray.init(num_cpus=8, num_gpus=3)
# ApexWorker = @ay.remote(num_cpus=1, num_gpus=0.1)(ApexWorker)


# # the central parameter store
# param_store = ApexWorker.remote('param_store', make_env)


# # concurrent rollout workers
# actors = [
#     ApexWorker.remote(
#         f'actor_{i}', make_env, param_store,
#         tensorboard_dir=f'data/tensorboard/apex_dqn/actor_{i}')
#     for i in range(6)]


# # one learner
# learner = ApexWorker.remote('learner', make_env, param_store)


# # block until one of the remote processes terminates
# ray.wait([
#     learner.learn_loop.remote(max_total_steps=3000000),
#     *(actor.rollout_loop.remote(max_total_steps=3000000) for actor in actors)
# ])


param_store = ApexWorker('param_store', make_env)

actor = ApexWorker(
    'actor', make_env, param_store,
    tensorboard_dir='data/tensorboard/apex_dqn/singlethreaded')
actor.learn_lock.acquire()


learner = ApexWorker('learner', make_env, param_store)


threads = [
    Thread(target=actor.rollout_loop, args=(3000000,), name='actor'),
    Thread(target=learner.learn_loop, args=(3000000,), name='learner'),
]


for t in threads:
    t.start()


for t in threads:
    t.join()
