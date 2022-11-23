import functools
import numpy as np
import torch
import torch.nn as nn
import itertools
from functools import reduce

import torch.nn.functional as F
import math


def linear_regression():
    