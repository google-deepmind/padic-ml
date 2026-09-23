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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import padic_ml as pml
from padic_ml import _array

PArray = _array.PArray
array = pml.array


class PArrayTest(absltest.TestCase):
  """PArray methods that are stubs in _array.py are tested in ops_test.py."""

  def test_init(self):
    # Default log_radius is -inf
    arr = array(jnp.array([1, 2]), p=3)
    np.testing.assert_array_equal(arr.center, jnp.array([1, 2]))
    np.testing.assert_array_equal(
        arr.log_radius, jnp.array([-jnp.inf, -jnp.inf])
    )
    self.assertEqual(arr.p, 3)

    # Casts inputs to array
    arr = array([1, 2], p=2)
    self.assertIsInstance(arr.center, jax.Array)
    np.testing.assert_array_equal(arr.center, jnp.array([1, 2]))

    # Broadcasts radius inputs (r=4.0 at p=2 -> log_radius=2.0)
    arr = array(jnp.array([1, 2]), 4.0, p=2)
    self.assertIsInstance(arr.log_radius, jax.Array)
    np.testing.assert_allclose(arr.log_radius, jnp.array([2.0, 2.0]))

    # Radius 0.0 converts to log_radius -inf
    arr_zero_rad = array(jnp.array([1, 2]), 0.0, p=2)
    np.testing.assert_array_equal(
        arr_zero_rad.log_radius, jnp.array([-jnp.inf, -jnp.inf])
    )

    # Keyword log_radius input
    arr_log = array(jnp.array([1, 2]), log_radius=1.0, p=2)
    np.testing.assert_array_equal(arr_log.log_radius, jnp.array([1.0, 1.0]))

  def test_construction_and_properties(self):
    x = array(jnp.array([1, 2, 3], dtype=jnp.int32), p=5)
    self.assertEqual(x.shape, (3,))
    self.assertEqual(x.dtype, jnp.dtype(jnp.int32))
    self.assertEqual(x.p, 5)

  def test_unit_valuation_init(self):
    # Eager canonicalization of integer centers
    arr = array(45, p=3)
    np.testing.assert_array_equal(arr.unit, jnp.array(5))
    np.testing.assert_array_equal(arr.valuation, jnp.array(2))
    np.testing.assert_array_equal(arr.center, jnp.array(45))

    # Exact zero center (u=0, v=0)
    zero = array(0, p=3)
    np.testing.assert_array_equal(zero.unit, jnp.array(0))
    np.testing.assert_array_equal(zero.valuation, jnp.array(0))
    np.testing.assert_array_equal(zero.center, jnp.array(0))

    # Negative integer center
    neg_arr = array(-12, p=3)
    np.testing.assert_array_equal(neg_arr.unit, jnp.array(-4))
    np.testing.assert_array_equal(neg_arr.valuation, jnp.array(1))
    np.testing.assert_array_equal(neg_arr.center, jnp.array(-12))

    # Pre-decomposed unit/valuation (Z[1/p] representation)
    frac = array(unit=2, valuation=-3, p=3)
    np.testing.assert_array_equal(frac.unit, jnp.array(2))
    np.testing.assert_array_equal(frac.valuation, jnp.array(-3))

    # Pre-decomposed with log-radius
    disk = array(unit=5, valuation=2, log_radius=-1.0, p=3)
    np.testing.assert_array_equal(disk.center, jnp.array(45))
    np.testing.assert_array_equal(disk.log_radius, jnp.array(-1.0))

  def test_center_clamps_negative_valuation(self):
    """Documents (does not endorse) the .center precondition.

    2 * 3^-3 is not an integer, so there is no correct integer to return. The
    property clamps to p^max(0, v) and thus reports the unit. This is asserted
    so the documented precondition stays honest rather than drifting silently;
    exact work must use (unit, valuation).
    """
    frac = array(unit=2, valuation=-3, p=3)
    np.testing.assert_array_equal(frac.center, jnp.array(2))

  def test_direct_construction_accepts_float0(self):
    """The canonical constructor must not validate dtypes.

    `jax.grad(..., allow_int=True)` assigns float0 tangents to integer leaves
    and then rebuilds the pytree by calling this constructor. A dtype assertion
    here would reject JAX's own gradient machinery, so this must not raise.
    """
    float0 = jax.dtypes.float0
    zero = np.zeros((), dtype=float0)
    arr = PArray(
        unit=zero,
        valuation=zero,
        log_radius=jnp.asarray(-jnp.inf),
        p=3,
    )
    self.assertEqual(arr.unit.dtype, float0)

  def test_init_error(self):
    # Raising ValueError when neither center nor (unit, valuation) is passed
    with self.assertRaises(ValueError):
      array(p=3)
    # Raising ValueError when only unit is passed
    with self.assertRaises(ValueError):
      array(unit=5, p=3)
    # Raising ValueError when only valuation is passed
    with self.assertRaises(ValueError):
      array(valuation=2, p=3)
    # Raising ValueError when not prime
    with self.assertRaises(ValueError):
      array(5, p=1)
    with self.assertRaises(ValueError):
      array(5, p=4)
    with self.assertRaises(ValueError):
      array(5, p=9)
    # Raising ValueError when center is not integer dtype
    with self.assertRaises(ValueError):
      array(1.5, p=3)
    # Raising ValueError when unit or valuation is not integer dtype
    with self.assertRaises(ValueError):
      array(unit=1.5, valuation=2, p=3)
    with self.assertRaises(ValueError):
      array(unit=1, valuation=2.5, p=3)
    # Raising ValueError when both radius and log_radius are specified
    with self.assertRaises(ValueError):
      array(5, 1.0, log_radius=0.0, p=3)
    # Raising ValueError when radius is negative
    with self.assertRaises(ValueError):
      array(5, -1.0, p=3)

  def test_pytree_round_trip(self):
    """register_dataclass must rebuild an identical PArray from its leaves.

    This is the container contract the logic-free constructor exists to serve:
    JAX calls it via `unflatten_func` on every transformation boundary.
    """
    x = array(45, log_radius=-1.0, p=3)
    leaves, treedef = jax.tree_util.tree_flatten(x)
    y = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(y.unit, x.unit)
    self.assertEqual(y.valuation, x.valuation)
    self.assertEqual(y.log_radius, x.log_radius)
    self.assertEqual(y.p, x.p)

  def test_asarray(self):
    # Idempotent on PArray inputs
    x = array(45, p=3)
    self.assertIs(pml.asarray(x, p=3), x)
    # Converts integer operands
    converted = pml.asarray(45, p=3)
    np.testing.assert_array_equal(converted.unit, jnp.array(5))
    # Rejects a prime-base mismatch
    with self.assertRaises(ValueError):
      pml.asarray(x, p=5)


if __name__ == '__main__':
  absltest.main()
