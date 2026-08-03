import jax


def same_pytree_spec(a, b, comparison_fn=None):
    """Return whether two pytrees have matching structure and leaf specs.

    By default, leaves match when both their `shape` and `dtype` match. This
    works for JAX arrays and `jax.ShapeDtypeStruct` specs.
    """
    a_leaves, a_def = jax.tree.flatten(a)
    b_leaves, b_def = jax.tree.flatten(b)

    if comparison_fn is None:

        def _default_cmp(x, y):
            return (x.shape == y.shape) and (x.dtype == y.dtype)

        comparison_fn = _default_cmp

    return a_def == b_def and all(
        comparison_fn(x, y) for x, y in zip(a_leaves, b_leaves)
    )
