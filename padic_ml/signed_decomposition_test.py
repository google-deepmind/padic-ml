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

"""Check integer decomposition against unbounded Python integer arithmetic."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import padic_ml as pml
from padic_ml import ops


def reference(value, p):
  value = int(value)
  valuation = 0
  while value and value % p == 0:
    value //= p
    valuation += 1
  return value, valuation


class SignedDecompositionTest(parameterized.TestCase):

  @parameterized.product(
      x64=(False, True), p=(2, 3, 5, 7), compiled=(False, True)
  )
  def test_signed_endpoints_and_neighbours(self, x64, p, compiled):
    with jax.enable_x64(x64):
      dtype = jnp.asarray(0).dtype
      info = np.iinfo(dtype)
      values = np.array(
          [
              info.min,
              info.min + 1,
              info.min + 2,
              info.min // 2,
              -81,
              -27,
              -8,
              -1,
              0,
              1,
              8,
              27,
              81,
              info.max - 1,
              info.max,
          ],
          dtype=dtype,
      )
      function = lambda x: ops.decompose(x, p)
      if compiled:
        function = jax.jit(function)
      unit, valuation = function(jnp.asarray(values))
      expected = [reference(x, p) for x in values]
      np.testing.assert_array_equal(unit, [u for u, _ in expected])
      np.testing.assert_array_equal(valuation, [v for _, v in expected])
      self.assertEqual(unit.dtype, dtype)
      self.assertEqual(valuation.dtype, dtype)
      for x, u, v in zip(values, np.asarray(unit), np.asarray(valuation)):
        self.assertEqual(int(u) * p ** int(v), int(x))
        self.assertTrue(int(u) == 0 or int(u) % p != 0)

  @parameterized.product(x64=(False, True), p=(2, 3, 5))
  def test_public_constructor_and_absolute_value(self, x64, p):
    with jax.enable_x64(x64):
      dtype = jnp.asarray(0).dtype
      value = jnp.asarray(jnp.iinfo(dtype).min, dtype=dtype)
      expected_u, expected_v = reference(int(value), p)
      for constructor in (
          lambda x: pml.array(x, log_radius=-2.0, p=p),
          jax.jit(lambda x: pml.array(x, log_radius=-2.0, p=p)),
      ):
        result = constructor(value)
        self.assertEqual(int(result.unit), expected_u)
        self.assertEqual(int(result.valuation), expected_v)
        self.assertEqual(int(result.center), int(value))
        np.testing.assert_allclose(result.abs(), float(p) ** -expected_v)
        self.assertEqual(float(result.log_radius), -2.0)

  @parameterized.product(x64=(False, True), p=(2, 3))
  def test_batched_and_empty_inputs(self, x64, p):
    with jax.enable_x64(x64):
      dtype = jnp.asarray(0).dtype
      low = jnp.iinfo(dtype).min
      values = jnp.array([[low, 0, 1], [low + 1, -(p**3), p**3]], dtype=dtype)
      direct = ops.decompose(values, p)
      batched = jax.jit(jax.vmap(lambda x: ops.decompose(x, p)))(values)
      for actual, expected in zip(batched, direct):
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, values.shape)
      for shape in ((0,), (2, 0)):
        result = jax.jit(lambda x: ops.decompose(x, p))(
            jnp.empty(shape, dtype=dtype)
        )
        for component in result:
          self.assertEqual(component.shape, shape)
          self.assertEqual(component.dtype, values.dtype)

  @parameterized.parameters(2, 3, 5, 7)
  def test_every_int8_value_matches_exact_reference(self, p):
    # A deliberately explicit narrow input tests its complete finite domain.
    values = np.arange(-128, 128, dtype=np.int8)
    unit, valuation = jax.jit(lambda x: ops.decompose(x, p))(
        jnp.asarray(values)
    )
    expected = [reference(x, p) for x in values]
    np.testing.assert_array_equal(unit, [u for u, _ in expected])
    np.testing.assert_array_equal(valuation, [v for _, v in expected])
    self.assertEqual(unit.dtype, values.dtype)

  def test_narrow_inputs_preserve_their_dtype_with_x64_enabled(self):
    with jax.enable_x64(True):
      values = jnp.array([-(2**31), 0, 1], dtype=jnp.int32)
      unit, valuation = jax.jit(lambda x: ops.decompose(x, 2))(values)
      self.assertEqual(unit.dtype, values.dtype)
      self.assertEqual(valuation.dtype, values.dtype)
      np.testing.assert_array_equal(unit, [-1, 0, 1])
      np.testing.assert_array_equal(valuation, [31, 0, 0])


if __name__ == '__main__':
  absltest.main()
