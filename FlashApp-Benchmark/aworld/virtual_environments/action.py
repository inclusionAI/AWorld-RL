# coding: utf-8
# Copyright (c) 2025 inclusionAI.

import abc
from typing import Tuple, Any

from aworld.core.common import ActionModel, ActionResult


class ExecutableAction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, action: ActionModel, **kwargs) -> Tuple[ActionResult, Any]:
        """"""

    @abc.abstractmethod
    async def async_act(self, action: ActionModel, **kwargs) -> Tuple[ActionResult, Any]:
        """"""

