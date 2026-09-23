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

"""Functional arithmetic operations for PArray (Berkovich disk algebra).

Architecture and Conventions:
-----------------------------
1. Container + Functional Math Separation:
   - `PArray` (defined in `_array.py`) is the Berkovich disk container.
   - It delegates to the arithmetic logic here (the functional `ops.*` API).

2. Type Enforcement via jaxtyping:
   - `ops.*` functions are decorated with `@jt.jaxtyped` to enforce dtype
     contracts and broadcasting rules (`*#N -> *N`) across operands.

3. Subclass and Custom Type Construction Contract:
   - Each arithmetic implementation explicitly constructs and returns a base
     `PArray(unit=..., valuation=..., log_radius=..., p=...)`. That constructor
     is canonical-only and keyword-only; it performs no validation, because JAX
     calls it to rebuild the pytree on every transformation boundary.
   - To build a PArray from a center, a radius, or a plain integer operand, use
     `_array.array(...)` or `_array.asarray(...)` instead.
   - Subclasses (e.g. `LegacyPArray`) override the relevant operator
     methods, call `super()` or `ops.*` for the base container, and
     explicitly construct their own class instance.
"""

import functools
import math

import jax
import jax.numpy as jnp
import jaxtyping as jt
from padic_ml import _array

IntOperand = _array.IntOperand
PArray = _array.PArray


@jt.jaxtyped
def absolute(x: jt.Shaped[PArray, '*N']) -> jt.Float[jax.Array, '*N']:
  """Computes the p-adic norm |x|_p = p^{-v_p(x)} (0 if unit == 0)."""
  # log_radius already carries the float dtype matching the integer payload.
  float_dtype = x.log_radius.dtype
  return jnp.where(
      x.unit == 0,
      jnp.asarray(0.0, dtype=float_dtype),
      jnp.asarray(x.p, dtype=float_dtype) ** (-x.valuation),
  )


@jt.jaxtyped
def is_padic(x: jt.Shaped[PArray, '*N']) -> jt.Bool[jax.Array, '*N']:
  """Checks if the array is strictly p-adic (log_radius -inf)."""
  return jnp.isneginf(x.log_radius)


@jt.jaxtyped
def add(
    a: jt.Shaped[PArray, '*#N'],
    b: jt.Shaped[PArray, '*#N'] | IntOperand['*#N'],
) -> jt.Shaped[PArray, '*N']:
  """Translates the center of a Berkovich disk."""

  if isinstance(b, PArray):
    if a.p != b.p:
      raise ValueError(
          f'Addition between PArrays with different primes {a.p} and {b.p} is'
          ' not supported.'
      )
    if is_padic(b):
      return add(a, b.center)
    elif is_padic(a):
      return add(b, a.center)
    else:
      raise NotImplementedError(
          'Addition between two PArrays with positive radii is not supported'
          ' outside computation graphs.'
      )

  return _array.array(a.center + b, log_radius=a.log_radius, p=a.p)


@jt.jaxtyped
def mul(
    a: jt.Shaped[PArray, '*#N'],
    b: jt.Shaped[PArray, '*#N'] | IntOperand['*#N'],
) -> jt.Shaped[PArray, '*N']:
  """Multiplicativity of seminorms on the Berkovich line."""
  b = _array.asarray(b, p=a.p)
  new_unit = a.unit * b.unit
  new_valuation = jnp.where(new_unit == 0, 0, a.valuation + b.valuation)

  term_one = jnp.where(
      a.unit == 0,
      -jnp.inf,
      -a.valuation.astype(b.log_radius.dtype) + b.log_radius,
  )
  term_two = jnp.where(
      b.unit == 0,
      -jnp.inf,
      -b.valuation.astype(a.log_radius.dtype) + a.log_radius,
  )
  term_three = a.log_radius + b.log_radius

  new_log_radius = jnp.maximum(jnp.maximum(term_one, term_two), term_three)
  return PArray(
      unit=new_unit,
      valuation=new_valuation,
      log_radius=new_log_radius,
      p=a.p,
  )


@jt.jaxtyped
def neg(a: jt.Shaped[PArray, '*N']) -> jt.Shaped[PArray, '*N']:
  """Negation on Berkovich disk."""
  return mul(a, -1)


@jt.jaxtyped
def power(a: jt.Shaped[PArray, '*N'], exponent: int) -> jt.Shaped[PArray, '*N']:
  """Integer power on the Berkovich line via binomial expansion."""

  new_unit = a.unit**exponent
  new_valuation = jnp.where(new_unit == 0, 0, a.valuation * exponent)

  terms_list = []
  for k in range(1, exponent + 1):
    coefficient = math.comb(exponent, k)
    _, v_comb = decompose(coefficient, a.p)
    is_zero_term = (a.unit == 0) & (k < exponent)
    v_term = v_comb + (exponent - k) * a.valuation
    term_log_radius = jnp.where(
        is_zero_term,
        -jnp.inf,
        -v_term.astype(a.log_radius.dtype) + k * a.log_radius,
    )
    terms_list.append(term_log_radius)

  terms_stacked = jnp.stack(terms_list, axis=0)
  new_log_radius = jnp.max(terms_stacked, axis=0)

  return PArray(
      unit=new_unit,
      valuation=new_valuation,
      log_radius=new_log_radius,
      p=a.p,
  )


@jt.jaxtyped
def sub(
    a: jt.Shaped[PArray, '*#N'],
    b: jt.Shaped[PArray, '*#N'] | IntOperand['*#N'],
) -> jt.Shaped[PArray, '*N']:
  """Subtraction on Berkovich disk."""
  return add(a, -b)


# ---------------------------------------------------------------------------
# Non-PArray utility functions
# ---------------------------------------------------------------------------


@functools.cache
def _max_valuation(p: int, dtype: jnp.dtype) -> int:
  """Maximum p-adic valuation representable by an integer dtype.

  Computes floor(log_p(max_val)) using integer arithmetic to avoid
  floating-point precision issues with math.log.

  Args:
    p: Prime base.
    dtype: Integer dtype whose range bounds the valuation.

  Returns:
    The largest v such that p^v fits within the dtype's range.
  """
  v, n = 0, jnp.iinfo(dtype).max
  while n >= p:
    n //= p
    v += 1
  return v


@jt.jaxtyped
def decompose(
    x: IntOperand['*N'],
    p: int,
) -> tuple[jt.Int[jt.Array, '*N'], jt.Int[jt.Array, '*N']]:
  """Decomposes an integer array into canonical (unit, valuation) pair.

  For each element, factors x = u * p^v such that p does not divide u
  (or u = 0, v = 0 if x = 0).

  Uses fori_loop with a static bound (the maximum valuation for the input
  dtype) instead of while_loop. This eliminates the per-iteration jnp.any()
  global synchronization barrier that while_loop's cond_fun requires, making
  each iteration purely element-wise.

  Returns:
    (unit, valuation): unit tensor u in Z and int valuation tensor v in Z.
  """
  x_arr = jnp.asarray(x)
  abs_x = jnp.abs(x_arr)
  is_zero = abs_x == 0
  # Replace zeros with 1 to avoid division-by-zero in the loop body.
  safe_x = jnp.where(is_zero, 1, abs_x)

  def body_fun(_, state):
    curr_x, val = state
    divisible = curr_x % p == 0
    # Only divide elements still divisible by p; others pass through.
    next_x = jnp.where(divisible, curr_x // p, curr_x)
    next_val = jnp.where(divisible, val + 1, val)
    return next_x, next_val

  # Upper bound on iterations: no integer of this dtype can have a p-adic
  # valuation exceeding this, so the loop is guaranteed to finish all elements.
  max_iters = _max_valuation(p, x_arr.dtype)
  init_val = jnp.zeros_like(safe_x)
  final_x, final_val = jax.lax.fori_loop(
      0, max_iters, body_fun, (safe_x, init_val)
  )

  # Restore sign and canonicalize zeros to (u=0, v=0).
  unit = jnp.where(is_zero, 0, jnp.sign(x_arr) * final_x)
  valuation = jnp.where(is_zero, 0, final_val)
  return unit, valuation


@jt.jaxtyped
def to_expansion(
    value: int | jt.Int[jt.Array, ''],
    p: int,
    terms: int = 8,
) -> str:
  """Formats a standard integer into a p-adic string for representation."""
  value_modulo = int(value) % (p**terms)
  digits_list = []
  for _ in range(terms):
    digits_list.append(str(value_modulo % p))
    value_modulo //= p
  return '...' + ''.join(reversed(digits_list))
