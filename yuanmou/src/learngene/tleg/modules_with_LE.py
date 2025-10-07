import torch
from torch import Tensor, Size
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import init
from timm.models.layers import trunc_normal_

import numbers
import math
from typing import Union, List, Tuple


class LELayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LELayerNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  
            
        self.normalized_shape = tuple(normalized_shape)  
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight_base = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias_base = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.weight_ilayer = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias_ilayer = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight_base', None)
            self.register_parameter('bias_base', None)
            self.register_parameter('weight_ilayer', None)
            self.register_parameter('bias_ilayer', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight_base)
            init.zeros_(self.bias_base)
            init.zeros_(self.weight_ilayer)
            init.zeros_(self.bias_ilayer)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_base is not None:
            weight = self.weight_base + self._ilayer * self.weight_ilayer
            bias = self.bias_base + self._ilayer * self.bias_ilayer
        else:
            weight = self.weight_base
            bias = self.bias_base
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class LELinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LELinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight_ilayer = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_base = Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias_base = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_ilayer = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_base', None)
            self.register_parameter('bias_ilayer', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight_base, std=.02)
        trunc_normal_(self.weight_ilayer, std=.02)
        if self.bias_base is not None:
            init.zeros_(self.bias_base)
            init.zeros_(self.bias_ilayer)

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_base + self._ilayer * self.weight_ilayer
        if self.bias_base is not None:
            bias = self.bias_base + self._ilayer * self.bias_ilayer
        else:
            bias = None
        return F.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_base is not None
        )
        