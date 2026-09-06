"""Abstract bases, mirroring `ruptures.base`.

Kept so that `isinstance(x, BaseCost)` checks in user code — and in this
package's own `custom_cost` handling — behave as they do in `ruptures`.
"""

import abc

from .utils import pairwise


class BaseEstimator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def fit_predict(self, *args, **kwargs):
        ...


class BaseCost(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def error(self, start, end):
        ...

    def sum_of_costs(self, bkps):
        return sum(self.error(start, end) for start, end in pairwise([0] + list(bkps)))

    @property
    @abc.abstractmethod
    def model(self):
        ...
