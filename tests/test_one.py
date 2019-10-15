import pytest
import torch

class TestClass:
    def test_one(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert device =="cpu"

    def test_two(self):
        x = 1
        assert x==1
