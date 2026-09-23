# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data structures for Berkovich space representation and auto-diff."""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import jaxtyping as jt
import sympy


@functools.cache
def _is_prime(p: int) -> bool:
  """Cached primality check using sympy.isprime."""
  return bool(sympy.isprime(p))


def _matching_float_dtype(int_dtype: jnp.dtype) -> jnp.dtype:
  """Returns the floating-point dtype of equal bitwidth to an integer dtype."""
  return jnp.dtype(f'float{int_dtype.itemsize * 8}')


class IntOperand:
  """Union type alias for int | jt.Int[jt.Array, Shape].

  Usage:
    IntOperand['*N']   -> int | jt.Int[jt.Array, '*N']
    IntOperand['*#N']  -> int | jt.Int[jt.Array, '*#N']
    IntOperand['...']  -> int | jt.Int[jt.Array, '...']
  """

  def __class_getitem__(cls, shape):
    return int | jt.Int[jt.Array, shape]


class FloatOperand:
  """Union type alias for float | jt.Float[jt.Array, Shape].

  Usage:
    FloatOperand['*N']   -> float | jt.Float[jt.Array, '*N']
    FloatOperand['*#N']  -> float | jt.Float[jt.Array, '*#N']
    FloatOperand['...']  -> float | jt.Float[jt.Array, '...']
  """

  def __class_getitem__(cls, shape):
    return float | jt.Float[jt.Array, shape]


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=('unit', 'valuation', 'log_radius'),
    meta_fields=('p',),
)
@dataclasses.dataclass(eq=False, frozen=True, kw_only=True)
class PArray:
  """DO NOT INIT DIRECTLY; use `(as)array()`. Array of 'continuous' p-adics.

  These are elements zeta_{unit * p^valuation, p^log_radius} of the p-adic
  convex hull of Qp. In a normalized (unit, valuation) pair, unit is coprime to
  p, or (0,0) if c = 0, such that c = unit * p^valuation. Details on this
  representation:
  https://doc.sagemath.org/html/en/reference/padics/sage/rings/padics/tutorial.html#terminology-and-types-of-p-adics
  https://flintlib.org/doc/padic.html

  NOTE: With integer units, we can only represent Z[1/p] in Qp exactly. However,
  this set is dense, so for non-zero radii, we can choose the center to be in
  Z[1/p].

  Attributes:
    unit: Integer unit representative coprime to p (or 0 if center is 0). int.
    valuation: Exact p-adic valuation v_p(c) (or 0 if center is 0). int.
    log_radius: Log-radius of the disk. float.
    p: Prime base p.
  """

  unit: jt.Int[jt.Array, '*N']
  valuation: jt.Int[jt.Array, '*N']
  log_radius: jt.Float[jt.Array, '*N']
  p: int

  @property
  def center(self) -> jt.Int[jt.Array, '*N']:
    """Reconstructs the integer center u * p^v. Requires valuation >= 0."""
    p_pow = jnp.asarray(self.p, dtype=self.unit.dtype) ** jnp.maximum(
        0, self.valuation
    )
    return self.unit * p_pow

  # The below two properties (shape and dtype) give a duck-typed array for JAX
  # typing. See: https://docs.kidger.site/jaxtyping/api/array/#array

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape of the array."""
    return self.unit.shape

  @property
  def dtype(self) -> jnp.dtype:
    """Returns the data type of the unit array."""
    return self.unit.dtype

  # Imports of ops are inside the functions to avoid circular dependency.
  #
  # We do not add shape annotations for these wrappers, as such annotations only
  # get used in runtime type checking. We consolidate those in `ops`. Also see
  # warning which discourages strings or from __future__ import annotations as
  # array terms of the jt type annotation.
  # https://docs.kidger.site/jaxtyping/api/runtime-type-checking/.
  #
  # pylint: disable=g-import-not-at-top

  def __add__(self, other: PArray | IntOperand['...']) -> PArray:
    from padic_ml import ops

    return ops.add(self, other)

  def __radd__(self, other: PArray | IntOperand['...']) -> PArray:
    return self.__add__(other)

  def __mul__(self, other: PArray | IntOperand['...']) -> PArray:
    from padic_ml import ops

    return ops.mul(self, other)

  def __rmul__(self, other: PArray | IntOperand['...']) -> PArray:
    return self.__mul__(other)

  def __neg__(self) -> PArray:
    from padic_ml import ops

    return ops.neg(self)

  def __pow__(self, power: int) -> PArray:
    from padic_ml import ops

    return ops.power(self, power)

  def __sub__(self, other: PArray | IntOperand['...']) -> PArray:
    from padic_ml import ops

    return ops.sub(self, other)

  def abs(self) -> jax.Array:
    from padic_ml import ops

    return ops.absolute(self)

  def is_padic(self) -> jax.Array:
    from padic_ml import ops

    return ops.is_padic(self)

  # pylint: enable=g-import-not-at-top


def array(
    center: IntOperand['*N'] | None = None,
    radius: FloatOperand['*#N'] | None = None,
    *,
    p: int,
    log_radius: FloatOperand['*#N'] | None = None,
    unit: IntOperand['*N'] | None = None,
    valuation: IntOperand['*#N'] | None = None,
) -> PArray:
  """Constructs a PArray, validating and canonicalizing its arguments.

  This is the eager entry point for building a PArray, analogous to `jnp.array`.
  All validation and casework lives here rather than in `PArray.__init__`, which
  must stay free of logic because JAX calls it to rebuild the pytree on every
  transformation boundary.

  Supported invocation modes:
    1. From an integer center:
       `array(center, radius=None, *, p, log_radius=None)`
    2. From pre-decomposed unit and valuation (e.g. elements of Z[1/p]):
       `array(radius=None, *, p, log_radius=None, unit=..., valuation=...)`

  Args:
    center: Integer center of the disk, u * p^v. If provided, unit and valuation
      are computed automatically via `decompose(center, p)`.
    radius: Non-negative radius r = p^log_radius of the disk. Mutually exclusive
      with `log_radius`. Converted internally to `log_radius`.
    p: Prime base p (must be a prime integer >= 2).
    log_radius: Log-radius of the disk (base p). Defaults to -inf (a strictly
      p-adic point) if both `radius` and `log_radius` are omitted.
    unit: Integer unit representative coprime to p (or 0 if center is 0). Must
      be provided together with `valuation` when `center` is omitted.
    valuation: Exact p-adic valuation v_p(c) (or 0 if center is 0). Must be
      provided together with `unit` when `center` is omitted.

  Returns:
    A PArray in canonical (unit, valuation, log_radius) form.

  Raises:
    ValueError: If p is not prime, if both `radius` and `log_radius` are given,
      if `radius` is negative, if the integer arguments are not of integer
      dtype, or if neither `center` nor `(unit, valuation)` is provided.
  """
  if p < 2 or not _is_prime(p):
    raise ValueError(f'Prime base p must be a prime >= 2, got {p}.')
  if radius is not None and log_radius is not None:
    raise ValueError('Cannot specify both radius and log_radius.')

  # Resolve the canonical (unit, valuation) pair.
  if unit is not None and valuation is not None:
    u = jnp.asarray(unit)
    v = jnp.asarray(valuation)
    if not jnp.issubdtype(u.dtype, jnp.integer):
      raise ValueError(f'unit must have integer dtype, got {u.dtype}.')
    if not jnp.issubdtype(v.dtype, jnp.integer):
      raise ValueError(f'valuation must have integer dtype, got {v.dtype}.')
    # Zero has no valuation; pin it to 0 so equal values compare equal.
    v = jnp.broadcast_to(jnp.where(u == 0, 0, v), u.shape)
  elif center is not None:
    c = jnp.asarray(center)
    if not jnp.issubdtype(c.dtype, jnp.integer):
      raise ValueError(f'center must have integer dtype, got {c.dtype}.')
    # Deferred to break the _array <-> ops cycle, as in the PArray operator
    # methods above.
    from padic_ml import ops  # pylint: disable=g-import-not-at-top

    u, v = ops.decompose(c, p)
  else:
    raise ValueError('Either center or (unit, valuation) must be provided.')

  # Resolve the log-radius, matching the bitwidth of the integer payload so that
  # 32-bit inputs do not silently incur 64-bit intermediates under jit.
  float_dtype = _matching_float_dtype(u.dtype)
  if log_radius is not None:
    r = jnp.asarray(log_radius, dtype=float_dtype)
  elif radius is not None:
    rad = jnp.asarray(radius, dtype=float_dtype)
    if jnp.any(rad < 0):
      raise ValueError('Radius must be non-negative.')
    log_p = jnp.log(jnp.asarray(p, dtype=float_dtype))
    r = jnp.where(
        rad == 0,
        jnp.asarray(-jnp.inf, dtype=float_dtype),
        jnp.log(rad) / log_p,
    )
  else:
    r = jnp.full(u.shape, -jnp.inf, dtype=float_dtype)

  return PArray(
      unit=u, valuation=v, log_radius=jnp.broadcast_to(r, u.shape), p=p
  )


@jt.jaxtyped
def asarray(
    x: jt.Shaped[PArray, '*N'] | IntOperand['*N'],
    *,
    p: int,
) -> jt.Shaped[PArray, '*N']:
  """Converts an integer or integer array operand to a PArray with prime base p.

  Idempotent on PArray inputs, analogous to `jnp.asarray`.

  Raises:
    ValueError: If x is a PArray whose prime base differs from p.
  """
  if isinstance(x, PArray):
    if x.p != p:
      raise ValueError(
          f'Cannot convert PArray with prime base p={x.p} to p={p}.'
      )
    return x
  return array(x, p=p)
