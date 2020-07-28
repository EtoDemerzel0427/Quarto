from typing import List, NamedTuple, Callable, Optional, Union
import numpy as np

# todo: only tensors of floating point dtype can require gradients, solve this
class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[int, float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable

    return np.array(arrayable)


Tensorable = Union['Tensor', int, float, np.ndarray]

def ensure_tensor(tensorable: Tensorable, requires_grad: bool=False) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable

    return Tensor(tensorable, requires_grad=requires_grad)

class Tensor:
    def __init__(self, data: Arrayable, requires_grad: bool=False, is_leaf: bool = True,
                 depends_on: List[Dependency]=None):
        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.shape = self._data.shape
        self.grad: Optional['Tensor'] = None  # grad should be tensor to allow higher-order derivatives
        self.is_leaf = is_leaf

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: Arrayable):
        self._data = ensure_array(new_data)
        self.shape = self._data.shape
        self.grad = None  # change data will invalidate the grad

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad: 'Tensor'=None) -> None:
        if not self.requires_grad:
            raise RuntimeError("element 0 of tensors does not require grad and does not have a grad_fn")
        elif self.grad is None:
            self.zero_grad()

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")

        self.grad.data = self.grad.data + grad.data
        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))  # todo: grad's require_grad could enable higher-order grad

    def sum(self):
        pass

    def __repr__(self) -> str:
        return f"tensor({self._data}, requires_grad={self.requires_grad})"

    # todo: deal with require_grad and non-Tensor add.
    def __add__(self, other: Tensorable) -> 'Tensor':
        return _add(self, ensure_tensor(other))

    def __iadd__(self, other):
        if self.requires_grad:
            raise RuntimeError("a leaf Variable that requires grad is being used in an in-place operation.")

        t_other = ensure_tensor(other)
        if t_other.requires_grad:
            self.requires_grad = True

        return _add(self, t_other)

    def __radd__(self, other):
        return _add(ensure_tensor(other), self)


# TODO: add error handling
def _add(a: Tensor, b: Tensor) -> Tensor:
    data = a.data + b.data
    requires_grad = a.requires_grad or b.requires_grad

    depends_on: List[Dependency] = []

    if a.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            # deal with broadcasting, extra dimensions, e.g, (2,3,5) vs (5,)
            ndims_added = grad.ndim - a.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(a.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(a, grad_fn1))

    if b.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            # deal with broadcasting, extra dimensions, e.g, (2,3,5) vs (5,)
            ndims_added = grad.ndim - b.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(b.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(b, grad_fn2))

    return Tensor(data, requires_grad=requires_grad, depends_on=depends_on, is_leaf=False)











