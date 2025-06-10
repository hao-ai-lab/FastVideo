# SPDX-License-Identifier: Apache-2.0
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed.checkpoint.stateful
from torch.distributed.checkpoint.state_dict import (StateDictOptions,
                                                     get_model_state_dict,
                                                     get_optimizer_state_dict,
                                                     set_model_state_dict,
                                                     set_optimizer_state_dict)


class ModelWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def state_dict(self) -> Dict[str, Any]:
        return get_model_state_dict(self.model)  # type: ignore[no-any-return]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_model_state_dict(
            self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(strict=False),
        )


class OptimizerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        return get_optimizer_state_dict(  # type: ignore[no-any-return]
            self.model,
            self.optimizer,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


class SchedulerWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def state_dict(self) -> Dict[str, Any]:
        return {"scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict["scheduler"])


class RandomStateWrapper(torch.distributed.checkpoint.stateful.Stateful):

    def __init__(self,
                 noise_generator: Optional[torch.Generator] = None) -> None:
        self.noise_generator = noise_generator

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
            if torch.cuda.device_count() > 1:
                state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        if self.noise_generator is not None:
            state["noise_generator_state"] = self.noise_generator.get_state()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if "torch_rng_state" in state_dict:
            torch.set_rng_state(state_dict["torch_rng_state"])

        if "numpy_rng_state" in state_dict:
            np.random.set_state(state_dict["numpy_rng_state"])

        if "python_rng_state" in state_dict:
            random.setstate(state_dict["python_rng_state"])

        # Restore CUDA random state
        if torch.cuda.is_available():
            if "cuda_rng_state" in state_dict:
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
            if "cuda_rng_state_all" in state_dict:
                torch.cuda.set_rng_state_all(state_dict["cuda_rng_state_all"])

        # Restore noise generator state
        if "noise_generator_state" in state_dict and self.noise_generator is not None:
            self.noise_generator.set_state(state_dict["noise_generator_state"])
