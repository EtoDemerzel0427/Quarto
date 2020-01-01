# Quarto
Quarto is a toy-level, self-implemented deep learning library with autograd.

## Content

Quarto is now under planning. Similar to PyTorch, in the first stage, we will implement:

1. A Tensor module. There should be a Tensor class based on NumPy array.
2. An `autograd` module.
3. Layers. A base Layer class, and some common layers, e.g, `conv2d`. (should beLealongs to `nn` module)
4. Functional. (should belongs to `nn` module)
5. init. (should belongs to `nn` module)

## Todo
1. Learn about reverse-mode autograd.
2. Learn about static typing in Python with `mypy`.


## Reference
1. To understand the concept of automatic differentiation: [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/pdf/1502.05767.pdf)
2. To quickly get the big picture of automatic differentiation: [UWashington slides](http://dlsys.cs.washington.edu/pdf/lecture4.pdf)
3. To quickly understand Pytorch internals: [](http://blog.ezyang.com/2019/05/pytorch-internals/)
