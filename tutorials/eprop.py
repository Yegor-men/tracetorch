import torch
from torch import nn

import tracetorch
from tracetorch import eprop

foo = eprop.LIF(10, 10)
