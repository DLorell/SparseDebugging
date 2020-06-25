import pytest
import pytest_check as check
import sys
import torch
import torch.nn as nn
sys.path.append('src')
from functional import tester   # nopep8



def test_batch_omp():
    