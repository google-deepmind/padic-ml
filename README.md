# padic-ml

**WORK IN PROGRESS. Paper code is being refactored into this library.
Thanks for your patience!**

This is a JAX library for differentiable training of _p_-adic neural networks,
implementing methods described in *[Continuous Optimization for p-adic Models](https://arxiv.org/abs/2609.25501)*
(arXiv 2026).

Install with `pip install padic-ml`.

**Disclaimer:** This is not an officially supported Google product.

## Usage

`PArray` is an array primitive for "learnable" _p_-adic numbers (points in the _p_-adic injective hull $\Gamma_p$ in Berkovich space).

With radius 0 (no second argument), these become ordinary _p_-adic numbers $\mathbb{Q}_p$.

```python
>>> import padic_ml as pml
>>> p1 = pml.array([3, 4], radius=[1, 3], p=3)
>>> p2 = pml.array([7, 1], p=3)
>>> p1*p2
PArray(unit=Array([7, 4], dtype=int32), valuation=Array([1, 0], dtype=int32), log_radius=Array([0., 1.], dtype=float32), p=3)
```
