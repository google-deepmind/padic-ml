Precision Convention:
   - We should not hardcode precision. Use `jnp.int_` or `jnp.float_`. User
    scripts should control precision via globally setting `jax_enable_x64`
    themselves.
