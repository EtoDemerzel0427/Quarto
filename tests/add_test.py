import unittest
import pytest

from Quarto.Tensor import Tensor

class TestAdd(unittest.TestCase):
    def test_simple_add(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        assert c.data.tolist() == [5, 7, 9]

        c.backward(Tensor([1,1,1]))
        assert a.grad.data.tolist() == [1, 1, 1]
        assert b.grad.data.tolist() == [1, 1, 1]

    def test_broadcast_add(self):
        a = Tensor([1,2,3], requires_grad=True)
        b = Tensor([[1,1,1], [2,2,2]], requires_grad=True)

        c = a + b
        assert c.data.tolist() == [[2,3,4], [3,4,5]]
        c.backward(Tensor([[1, 1, 1],[1,1,1]]))
        assert a.grad.data.tolist() == [2, 2, 2]
        assert b.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]

        a = Tensor(1, requires_grad=True)
        b = Tensor([1,2,3,4], requires_grad=True)
        c = a + b
        assert c.data.tolist() == [2,3,4,5]
        c.backward(Tensor([1,1,1,1]))
        assert a.grad.data.tolist() == 4
        assert b.grad.data.tolist() == [1,1,1,1]

        x = Tensor([1,2,3], requires_grad=True)
        y = Tensor([[2]], requires_grad=True)
        z = x + y
        assert z.data.tolist() == [[3, 4, 5]]
        z.backward(Tensor([[1,1,1]]))
        assert x.grad.data.tolist() == [1, 1, 1]
        assert y.grad.data.tolist() == [[3]]

    def test_iadd(self):
        pass

    def test_radd(self):
        pass
