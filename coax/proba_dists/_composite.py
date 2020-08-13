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

from enum import Enum

import gym
import numpy as onp
import haiku as hk

from ._base import BaseProbaDist
from ._categorical import CategoricalDist
from ._normal import NormalDist


__all__ = (
    'ProbaDist',
)


class StructureType(Enum):
    LEAF = 0
    LIST = 1
    DICT = 2


class ProbaDist(BaseProbaDist):
    r"""

    A composite probability distribution. This consists of a nested structure, whose leaves are
    either :class:`coax.proba_dists.CategoricalDist` or :class:`coax.proba_dists.NormalDist`
    instances.

    Parameters
    ----------
    space : gym.Space

        The gym-style space that specifies the domain of the distribution. This may be any space
        included in the :mod:`gym.spaces` module.

    """
    __slots__ = BaseProbaDist.__slots__ + ('_structure', '_structure_type')

    def __init__(self, space):
        super().__init__(space)

        if isinstance(self.space, gym.spaces.Discrete):
            self._structure_type = StructureType.LEAF
            self._structure = CategoricalDist(space)
        elif isinstance(self.space, gym.spaces.Box):
            self._structure_type = StructureType.LEAF
            self._structure = NormalDist(space)
        elif isinstance(self.space, gym.spaces.MultiDiscrete):
            self._structure_type = StructureType.LIST
            self._structure = [self.__class__(gym.spaces.Discrete(n)) for n in space.nvec]
        elif isinstance(self.space, gym.spaces.MultiBinary):
            self._structure_type = StructureType.LIST
            self._structure = [self.__class__(gym.spaces.Discrete(2)) for _ in range(space.n)]
        elif isinstance(self.space, gym.spaces.Tuple):
            self._structure_type = StructureType.LIST
            self._structure = [self.__class__(sp) for sp in space.spaces]
        elif isinstance(self.space, gym.spaces.Dict):
            self._structure_type = StructureType.DICT
            self._structure = {k: self.__class__(sp) for k, sp in space.spaces.items()}
        else:
            raise TypeError(f"unsupported space: {space.__class__.__name__}")

        def sample_func(dist_params, rng):
            if self._structure_type == StructureType.LEAF:
                return self._structure.sample(dist_params, rng)

            rngs = hk.PRNGSequence(rng)
            if self._structure_type == StructureType.LIST:
                return [
                    dist.sample(dist_params[i], next(rngs))
                    for i, dist in enumerate(self._structure)]

            if self._structure_type == StructureType.DICT:
                return {
                    k: dist.sample(dist_params[k], next(rngs))
                    for k, dist in self._structure.items()}

            raise AssertionError(f"bad structure_type: {self._structure_type}")

        def mode_func(dist_params):
            if self._structure_type == StructureType.LEAF:
                return self._structure.mode(dist_params)

            if self._structure_type == StructureType.LIST:
                return [
                    dist.mode(dist_params[i])
                    for i, dist in enumerate(self._structure)]

            if self._structure_type == StructureType.DICT:
                return {
                    k: dist.mode(dist_params[k])
                    for k, dist in self._structure.items()}

            raise AssertionError(f"bad structure_type: {self._structure_type}")

        def log_proba_func(dist_params, X):
            if self._structure_type == StructureType.LEAF:
                return self._structure.log_proba(dist_params, X)

            if self._structure_type == StructureType.LIST:
                return sum(
                    dist.log_proba(dist_params[i], X[i])
                    for i, dist in enumerate(self._structure))

            if self._structure_type == StructureType.DICT:
                return sum(
                    dist.log_proba(dist_params[k], X[k])
                    for k, dist in self._structure.items())

        def entropy_func(dist_params):
            if self._structure_type == StructureType.LEAF:
                return self._structure.entropy(dist_params)

            if self._structure_type == StructureType.LIST:
                return sum(
                    dist.entropy(dist_params[i])
                    for i, dist in enumerate(self._structure))

            if self._structure_type == StructureType.DICT:
                return sum(
                    dist.entropy(dist_params[k])
                    for k, dist in self._structure.items())

            raise AssertionError(f"bad structure_type: {self._structure_type}")

        def cross_entropy_func(dist_params_p, dist_params_q):
            if self._structure_type == StructureType.LEAF:
                return self._structure.cross_entropy(dist_params_p, dist_params_q)

            if self._structure_type == StructureType.LIST:
                return sum(
                    dist.cross_entropy(dist_params_p[i], dist_params_q[i])
                    for i, dist in enumerate(self._structure))

            if self._structure_type == StructureType.DICT:
                return sum(
                    dist.cross_entropy(dist_params_p[k], dist_params_q[k])
                    for k, dist in self._structure.items())

            raise AssertionError(f"bad structure_type: {self._structure_type}")

        def kl_divergence_func(dist_params_p, dist_params_q):
            if self._structure_type == StructureType.LEAF:
                return self._structure.kl_divergence(dist_params_p, dist_params_q)

            if self._structure_type == StructureType.LIST:
                return sum(
                    dist.kl_divergence(dist_params_p[i], dist_params_q[i])
                    for i, dist in enumerate(self._structure))

            if self._structure_type == StructureType.DICT:
                return sum(
                    dist.kl_divergence(dist_params_p[k], dist_params_q[k])
                    for k, dist in self._structure.items())

            raise AssertionError(f"bad structure_type: {self._structure_type}")

        self._sample_func = sample_func
        self._mode_func = mode_func
        self._log_proba_func = log_proba_func
        self._entropy_func = entropy_func
        self._cross_entropy_func = cross_entropy_func
        self._kl_divergence_func = kl_divergence_func

    @property
    def hyperparams(self):
        if self._structure_type == StructureType.LEAF:
            return self._structure.hyperparams

        if self._structure_type == StructureType.LIST:
            return tuple(dist.hyperparams for dist in self._structure)

        if self._structure_type == StructureType.DICT:
            return {k: dist.hyperparams for k, dist in self._structure.items()}

        raise AssertionError(f"bad structure_type: {self._structure_type}")

    @property
    def default_priors(self):
        if self._structure_type == StructureType.LEAF:
            return self._structure.default_priors

        if self._structure_type == StructureType.LIST:
            return tuple(dist.default_priors for dist in self._structure)

        if self._structure_type == StructureType.DICT:
            return {k: dist.default_priors for k, dist in self._structure.items()}

        raise AssertionError(f"bad structure_type: {self._structure_type}")

    def postprocess_variate(self, X, batch_mode=False):
        if self._structure_type == StructureType.LEAF:
            return self._structure.postprocess_variate(X, batch_mode)

        if isinstance(self.space, (gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
            assert self._structure_type == StructureType.LIST
            return onp.stack([
                dist.postprocess_variate(X[i], batch_mode)
                for i, dist in enumerate(self._structure)], axis=-1)

        if isinstance(self.space, gym.spaces.Tuple):
            assert self._structure_type == StructureType.LIST
            return tuple(
                dist.postprocess_variate(X[i], batch_mode)
                for i, dist in enumerate(self._structure))

        if isinstance(self.space, gym.spaces.Dict):
            assert self._structure_type == StructureType.DICT
            return {
                k: dist.postprocess_variate(X[k], batch_mode)
                for k, dist in self._structure.items()}

        raise AssertionError(
            f"postprocess_variate not implemented for space: {self.space.__class__.__name__}; "
            "please send us a bug report / feature request")

    def preprocess_variate(self, X):
        if self._structure_type == StructureType.LEAF:
            return self._structure.preprocess_variate(X)

        if self._structure_type == StructureType.LIST:
            return [dist.preprocess_variate(X[i]) for i, dist in enumerate(self._structure)]

        if self._structure_type == StructureType.DICT:
            return {k: dist.preprocess_variate(X[k]) for k, dist in self._structure.items()}

        raise AssertionError(f"bad structure_type: {self._structure_type}")
