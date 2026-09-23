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

"""Unit tests for PArray and p-adic operations in ops module."""

from absl.testing import absltest
from absl.testing import parameterized
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import padic_ml as pml
from padic_ml import ops

PArray = pml.PArray
array = pml.array


class OpsTest(absltest.TestCase):
  """Tests for PArray dataclass and p-adic operations."""

  def test_absolute(self):
    # |9|_3 = 3^{-2} = 1/9
    nine = array(jnp.array(9), p=3)
    self.assertAlmostEqual(nine.abs(), 1.0 / 9.0)
    self.assertAlmostEqual(ops.absolute(nine), 1.0 / 9.0)

    # |5|_3 = 3^{-0} = 1
    five = array(jnp.array(5), p=3)
    self.assertAlmostEqual(five.abs(), 1.0)
    self.assertAlmostEqual(ops.absolute(five), 1.0)

    # |0|_3 = 0.0
    zero = array(jnp.array(0), p=3)
    self.assertAlmostEqual(zero.abs(), 0.0)
    self.assertAlmostEqual(ops.absolute(zero), 0.0)

  def test_absolute_dtype_x64(self):
    """|x|_p preserves the input bitwidth instead of upcasting to float64.

    Deliberately runs under `jax.enable_x64(True)`: in JAX's default x32 mode
    float64 is unreachable, so asserting a float32 result would be vacuous.
    """
    with jax.enable_x64(True):
      arr32 = array(jnp.array([9, 0, 5], dtype=jnp.int32), p=3)
      self.assertEqual(ops.absolute(arr32).dtype, jnp.float32)

      arr64 = array(jnp.array([9, 0, 5], dtype=jnp.int64), p=3)
      self.assertEqual(ops.absolute(arr64).dtype, jnp.float64)

  def test_asarray(self):
    # asarray now lives in _array.py and is exported as pml.asarray; ops no
    # longer re-exports it (container/math separation).
    self.assertFalse(hasattr(ops, 'asarray'))

    # Converts int to PArray with prime p
    arr = pml.asarray(9, p=3)
    self.assertIsInstance(arr, PArray)
    self.assertEqual(arr.unit, 1)
    self.assertEqual(arr.valuation, 2)
    self.assertEqual(arr.p, 3)

    # Returns existing PArray unchanged if p matches
    same = pml.asarray(arr, p=3)
    self.assertIs(same, arr)

    # Raises ValueError if prime p mismatches
    with self.assertRaises(ValueError):
      pml.asarray(arr, p=5)

  def test_is_padic(self):
    a = array(jnp.array([1, 2]), log_radius=jnp.array([-1.0, -2.0]), p=2)
    np.testing.assert_array_equal(a.is_padic(), jnp.array([False, False]))
    np.testing.assert_array_equal(ops.is_padic(a), jnp.array([False, False]))

    b = array(jnp.array([1, 2]), log_radius=jnp.array([3.0, -jnp.inf]), p=3)
    np.testing.assert_array_equal(b.is_padic(), jnp.array([False, True]))
    np.testing.assert_array_equal(ops.is_padic(b), jnp.array([False, True]))

    # Default log_radius is -inf, so is_padic is True.
    c = array(jnp.array([1, 2]), p=3)
    np.testing.assert_array_equal(c.is_padic(), jnp.array(True))
    np.testing.assert_array_equal(ops.is_padic(c), jnp.array(True))

  def test_add(self):
    # Scalar translation: center translated, log_radius unchanged
    a = array(5, log_radius=-2.0, p=3)
    res_scalar = a + 4
    self.assertEqual(res_scalar.unit, 1)
    self.assertEqual(res_scalar.valuation, 2)
    self.assertEqual(res_scalar.log_radius, -2.0)

    res_scalar_fn = ops.add(a, 4)
    self.assertEqual(res_scalar_fn.unit, 1)
    self.assertEqual(res_scalar_fn.valuation, 2)
    self.assertEqual(res_scalar_fn.log_radius, -2.0)

    # Left scalar addition:
    res_left = 4 + a
    self.assertEqual(res_left.unit, 1)
    self.assertEqual(res_left.valuation, 2)
    self.assertEqual(res_left.log_radius, -2.0)

    # Point + Disk translation:
    b = array(3, log_radius=-jnp.inf, p=3)
    res_point = a + b
    self.assertEqual(res_point.unit, 8)
    self.assertEqual(res_point.valuation, 0)
    self.assertEqual(res_point.log_radius, -2.0)

    # Two positive radii disks should raise NotImplementedError:
    c = array(3, log_radius=-1.0, p=3)
    with self.assertRaises(NotImplementedError):
      _ = a + c
    with self.assertRaises(NotImplementedError):
      ops.add(a, c)

  def test_commutativity(self):
    """Addition is commutative: a+b == b+a in unit, valuation, and log_radius."""
    a = array(5, log_radius=-2.0, p=3)
    b = array(3, log_radius=-jnp.inf, p=3)
    ab = a + b
    ba = b + a
    np.testing.assert_allclose(ab.log_radius, ba.log_radius)
    np.testing.assert_allclose(ab.unit, ba.unit)
    np.testing.assert_allclose(ab.valuation, ba.valuation)

  def test_mul_cross_terms(self):
    """Verify that mul accounts for cross terms in radius."""
    x = array(1, log_radius=-2.0, p=3)
    y = array(1, log_radius=-3.0, p=3)
    f = x * y
    # max(v_1*(-1)+rho_2, v_2*(-1)+rho_1, rho_1 +rho_2) = max(-3,-2,-5) = -2.0
    self.assertAlmostEqual(f.log_radius, -2.0)
    self.assertEqual(f.unit, 1)
    self.assertEqual(f.valuation, 0)

    # Scalar multiplication:
    f_scalar = x * 9  # v_3(9) = 2 -> rho - v = -2 - 2 = -4.0
    self.assertEqual(f_scalar.unit, 1)
    self.assertEqual(f_scalar.valuation, 2)
    self.assertAlmostEqual(f_scalar.log_radius, -4.0)

    f_scalar_left = 9 * x
    self.assertEqual(f_scalar_left.unit, 1)
    self.assertEqual(f_scalar_left.valuation, 2)
    self.assertAlmostEqual(f_scalar_left.log_radius, -4.0)

    # Pre-decomposed Z[1/p] multiplication
    frac_a = array(unit=2, valuation=-3, p=3)  # 2/27
    frac_b = array(unit=5, valuation=2, p=3)  # 45
    prod = frac_a * frac_b
    self.assertEqual(prod.unit, 10)
    self.assertEqual(prod.valuation, -1)
    self.assertTrue(jnp.isneginf(prod.log_radius))

  def test_power(self):
    """Verify that pow accounts for cross terms in radius."""
    # c=1, rho=-2.0
    x = array(1, log_radius=-2.0, p=3)
    f = x**2
    # Correct radius: max(log|2c| + rho, 2*rho) = max(0 - 2, -4) = -2.0
    self.assertAlmostEqual(f.log_radius, -2.0)
    self.assertEqual(f.unit, 1)
    self.assertEqual(f.valuation, 0)

    # Power with non-trivial unit and valuation: 45 = 5 * 3^2
    a = array(45, log_radius=-1.0, p=3)
    a2 = a**2
    self.assertEqual(a2.unit, 25)
    self.assertEqual(a2.valuation, 4)

    a3 = a**3
    self.assertEqual(a3.unit, 125)
    self.assertEqual(a3.valuation, 6)

  def test_zero_edge_cases(self):
    """Verify operations involving exact zeros (u=0, v=0)."""
    zero = array(0, p=3)
    five = array(5, p=3)

    # Zero * Non-zero PArray
    mul_zero = zero * five
    self.assertEqual(mul_zero.unit, 0)
    self.assertEqual(mul_zero.valuation, 0)
    self.assertTrue(jnp.isneginf(mul_zero.log_radius))

    # Non-zero * Scalar 0
    scalar_zero = five * 0
    self.assertEqual(scalar_zero.unit, 0)
    self.assertEqual(scalar_zero.valuation, 0)
    self.assertTrue(jnp.isneginf(scalar_zero.log_radius))

    # Zero + Non-zero PArray
    add_zero = zero + five
    self.assertEqual(add_zero.unit, 5)
    self.assertEqual(add_zero.valuation, 0)

    # Power of zero
    pow_zero = zero**2
    self.assertEqual(pow_zero.unit, 0)
    self.assertEqual(pow_zero.valuation, 0)

  def test_cancellation_cases(self):
    """Verify p-adic cancellation dynamics in addition and multiplication."""
    # --- Addition Cancellation (valuation jumps) ---
    # 1 + 2 = 3 at p=3: valuation jumps from 0 to 1
    a = array(1, p=3)
    b = array(2, p=3)
    sum_ab = a + b
    self.assertEqual(sum_ab.unit, 1)
    self.assertEqual(sum_ab.valuation, 1)

    # 5 + 4 = 9 = 1 * 3^2 at p=3: valuation jumps from 0 to 2
    c = array(5, p=3)
    d = array(4, p=3)
    sum_cd = c + d
    self.assertEqual(sum_cd.unit, 1)
    self.assertEqual(sum_cd.valuation, 2)

    # Complete cancellation to exact 0: 5 + (-5) = 0
    neg_c = array(-5, p=3)
    sum_zero = c + neg_c
    self.assertEqual(sum_zero.unit, 0)
    self.assertEqual(sum_zero.valuation, 0)

    # --- Multiplication: Euclid's lemma guarantees NO cancellation ---
    # (4 * 3^1) * (5 * 3^2) = 20 * 3^3 = 540
    m1 = array(12, p=3)  # u=4, v=1
    m2 = array(45, p=3)  # u=5, v=2
    prod = m1 * m2  # u=20, v=3
    self.assertEqual(prod.unit, 20)
    self.assertEqual(prod.valuation, 3)

  def test_neg_and_sub(self):
    a = array(5, log_radius=-2.0, p=3)
    neg_a = -a
    self.assertEqual(neg_a.unit, -5)
    self.assertEqual(neg_a.valuation, 0)
    self.assertEqual(neg_a.log_radius, -2.0)
    self.assertEqual(ops.neg(a).unit, -5)
    self.assertEqual(ops.neg(a).valuation, 0)

    diff = a - 2
    self.assertEqual(diff.unit, 1)
    self.assertEqual(diff.valuation, 1)
    self.assertEqual(diff.log_radius, -2.0)

    diff_fn = ops.sub(a, 2)
    self.assertEqual(diff_fn.unit, 1)
    self.assertEqual(diff_fn.valuation, 1)
    self.assertEqual(diff_fn.log_radius, -2.0)

  def test_to_expansion(self):
    exp = ops.to_expansion(14, p=3, terms=4)
    self.assertEqual(exp, '...0112')


class DecomposeTest(absltest.TestCase):
  """Tests for decompose edge cases and fori_loop bound assumptions."""

  def test_max_representable_power_int32(self):
    """Validates the fori_loop bound adapts to int32 dtype."""
    max_v = ops._max_valuation(3, jnp.dtype(jnp.int32))
    val = 3**max_v
    u, v = ops.decompose(jnp.array(val, dtype=jnp.int32), p=3)
    np.testing.assert_array_equal(u, jnp.array(1, dtype=jnp.int32))
    np.testing.assert_array_equal(v, jnp.array(max_v, dtype=jnp.int32))

  def test_max_representable_power_int64(self):
    """Validates fori_loop bound for int64 using jax.enable_x64 context."""
    with jax.enable_x64():
      max_v = ops._max_valuation(3, jnp.dtype(jnp.int64))
      val = 3**max_v
      u, v = ops.decompose(jnp.array(val, dtype=jnp.int64), p=3)
      np.testing.assert_array_equal(u, jnp.array(1, dtype=jnp.int64))
      np.testing.assert_array_equal(v, jnp.array(max_v, dtype=jnp.int64))

  def test_coprime_to_p(self):
    """Value coprime to p should have valuation 0."""
    u, v = ops.decompose(jnp.array(7), p=3)
    np.testing.assert_array_equal(u, jnp.array(7))
    np.testing.assert_array_equal(v, jnp.array(0))

  def test_batch_mixed_valuations(self):
    """Validates vectorized decomposition across varied elements."""
    x = jnp.array([1, 3, 3**5, 0, -(3**3)])
    u, v = ops.decompose(x, p=3)
    np.testing.assert_array_equal(u, jnp.array([1, 1, 1, 0, -1]))
    np.testing.assert_array_equal(v, jnp.array([0, 1, 5, 0, 3]))

  def test_max_valuation_values(self):
    """Verifies _max_valuation returns correct bounds for known cases."""
    # 2^62 fits in int64 (max 2^63-1); 2^63 does not.
    self.assertEqual(ops._max_valuation(2, jnp.dtype(jnp.int64)), 62)
    # 2^30 fits in int32 (max 2^31-1); 2^31 does not.
    self.assertEqual(ops._max_valuation(2, jnp.dtype(jnp.int32)), 30)
    # 3^39 fits in int64; 3^40 does not.
    self.assertEqual(ops._max_valuation(3, jnp.dtype(jnp.int64)), 39)


class BitwidthTest(parameterized.TestCase):
  """No operation may introduce a 64-bit intermediate for a 32-bit PArray.

  A silent float64 upcast (classically from `jnp.float_`, which is float64
  under x64) stays numerically correct while doubling memory traffic, and f64
  is emulated on TPU. No correctness test would catch it, hence this
  white-box scan of the traced jaxpr.

  These run under `jax.enable_x64(True)` on purpose: in the default x32 mode
  float64 is unreachable, so the assertion would hold vacuously.

  `neg` and `sub` are omitted because they are defined as `mul(a, -1)` and
  `add(a, -b)`, so they are covered transitively. `is_padic` returns bool and
  `to_expansion` is pure Python.
  """

  @parameterized.named_parameters(
      ('absolute', ops.absolute),
      ('mul', lambda x: ops.mul(x, x)),
      ('power', lambda x: ops.power(x, 3)),
  )
  def test_no_64bit_intermediates(self, op_fn):
    with jax.enable_x64(True):
      arr32 = array(jnp.array([9, 0, 5], dtype=jnp.int32), p=3)
      jaxpr = jax.make_jaxpr(op_fn)(arr32)
      eqn_vars = itertools.chain.from_iterable(
          (*eqn.invars, *eqn.outvars) for eqn in jaxpr.jaxpr.eqns
      )
      eqn_dtypes = {v.aval.dtype for v in eqn_vars}
      const_dtypes = {c.dtype for c in jaxpr.consts}
      self.assertEmpty(
          (eqn_dtypes | const_dtypes) & {jnp.float64, jnp.int64},
          f'64-bit types in a 32-bit jaxpr: eqns={eqn_dtypes},'
          f' consts={const_dtypes}',
      )


if __name__ == '__main__':
  absltest.main()
