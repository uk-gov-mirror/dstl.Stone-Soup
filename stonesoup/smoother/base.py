# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base, Property
from ..models.transition import TransitionModel


class Smoother(Base):
    """Smoother Base Class."""

    transition_model: TransitionModel = Property(default=None, doc="Transition Model.")

    @abstractmethod
    def smooth(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def track_smooth(self, *args, **kwargs):
        raise NotImplementedError
