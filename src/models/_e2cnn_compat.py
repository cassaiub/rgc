"""Compatibility shims so the (unmaintained) ``e2cnn`` library runs on NumPy >= 1.24 / 2.x.

``e2cnn``'s last release predates NumPy 1.24 and relies on two APIs that newer
NumPy changed or removed:

1. ``numpy.ndarray.tostring()`` — removed in NumPy 1.24. It was always an exact
   alias of ``tobytes()``. ``e2cnn`` only calls it inside four ``__hash__`` methods
   (used as kernel-basis cache keys) in ``e2cnn.kernels.irreps_basis``. Since
   ``numpy.ndarray`` is an immutable C type we cannot re-add ``tostring`` to it, so
   we rebind those four ``__hash__`` methods to use ``tobytes()``. The bytes — and
   therefore the hash values — are identical, so the cache stays correct.

2. ``numpy.array(x, copy=False)`` — NumPy 2.0 raises when a copy is unavoidable
   instead of silently copying. We fall back to ``np.asarray`` in that case.

Importing this module applies both patches once (idempotent). It is imported by
``models.dsteerablelenet`` so every entry point — training scripts and notebooks —
gets the fix automatically before any steerable layer is built.
"""

from __future__ import annotations

import numpy as np


def _patch_numpy_array_copy() -> None:
    """e2cnn uses ``np.array(x, copy=False)``; NumPy 2.x raises if a copy is required."""
    if getattr(np, "_e2cnn_copy_patch", False):
        return
    _orig = np.array

    def _array(*args, **kwargs):
        if kwargs.get("copy") is False and len(args) == 1:
            try:
                return _orig(*args, **kwargs)
            except ValueError as err:
                if "Unable to avoid copy" in str(err):
                    dtype = kwargs.get("dtype")
                    return np.asarray(args[0], dtype=dtype) if dtype is not None else np.asarray(args[0])
                raise
        return _orig(*args, **kwargs)

    np.array = _array
    np._e2cnn_copy_patch = True


def _patch_ndarray_tostring() -> None:
    """Rebind the e2cnn ``__hash__`` methods that call the removed ``ndarray.tostring()``."""
    if getattr(np, "_e2cnn_tostring_patch", False):
        return
    try:
        from e2cnn.kernels import irreps_basis as _ib
    except Exception:
        return

    def _make_hash(gamma_is_array: bool):
        if gamma_is_array:
            def __hash__(self):
                return (hash(self.in_irrep) + hash(self.out_irrep)
                        + hash(self.mu.tobytes()) + hash(self.gamma.tobytes()))
        else:
            def __hash__(self):
                return (hash(self.in_irrep) + hash(self.out_irrep)
                        + hash(self.mu.tobytes()) + hash(self.gamma))
        return __hash__

    # class name -> whether `self.gamma` is a numpy array (vs. a plain scalar)
    specs = {
        "R2FlipsSolution": False,
        "R2ContinuousRotationsSolution": True,
        "R2FlipsDiscreteRotationsSolution": False,
        "R2FlipsContinuousRotationsSolution": False,
    }
    for name, gamma_is_array in specs.items():
        cls = getattr(_ib, name, None)
        if cls is not None:
            cls.__hash__ = _make_hash(gamma_is_array)

    np._e2cnn_tostring_patch = True


def apply() -> None:
    _patch_numpy_array_copy()
    _patch_ndarray_tostring()


apply()
