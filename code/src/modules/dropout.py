import math

import torch
import torch.nn as nn

from src.modules.base_generator import GeneratorAbstract

class Dropout(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)

class DropoutGenerator(GeneratorAbstract):
    """Fire block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        """call method.

        Args: p: float
        """
        module = []
        p = self.args[0]  # c is equivalent as self.out_channel
        if repeat>1:
            for i in range(repeat):
                module.append(
                    self.base_module(p)
                )
        else:
            module=self.base_module(p)
        return self._get_module(module)