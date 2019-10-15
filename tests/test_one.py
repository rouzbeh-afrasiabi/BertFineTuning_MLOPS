import pytest

class TestClass:
    def test_one(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert device =="cuda:0"

    def test_two(self):
        x = "hello"
        assert hasattr(x, 'check')
