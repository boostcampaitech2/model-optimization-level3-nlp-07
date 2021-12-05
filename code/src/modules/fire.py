import math

import torch
import torch.nn as nn

from src.modules.base_generator import GeneratorAbstract

class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )

class FireGenerator(GeneratorAbstract):
    """Fire block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return int(self.args[1]+self.args[2])

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        """call method.

        Args: squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int
        """
        module = []
        sq,ex1,ex3 = self.args  # c is equivalent as self.out_channel
        inp = self.in_channel
        if repeat>1:
            for i in range(repeat):
                module.append(
                    self.base_module(
                        in_planes=inp,
                        squeeze_planes=sq,
                        expand1x1_planes=ex1,
                        expand3x3_planes=ex3,
                    )
                )
                inp = self.out_channel
        else:
            module=self.base_module(
                inplanes=inp,
                squeeze_planes=sq,
                expand1x1_planes=ex1,
                expand3x3_planes=ex3,
            )
        return self._get_module(module)
