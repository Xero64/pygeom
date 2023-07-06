from typing import TYPE_CHECKING

from numpy import allclose, hsplit, hstack, number
from numpy.linalg import solve

from .arrayvector2d import ArrayVector2D as ArrayVector2D
from .arrayvector2d import zero_arrayvector2d as zero_arrayvector2d
from .arraytensor2d import ArrayTensor2D as ArrayTensor2D
from .arraytensor2d import zero_arraytensor2d as zero_arraytensor2d

if TYPE_CHECKING:
    from numpy import ndarray

def solve_arrayvector2d(a: 'ndarray', b: 'ArrayVector2D') -> 'ArrayVector2D':
    newb = hstack(b.to_xy())
    newc = solve(a, newb)
    x, y = hsplit(newc, 2)
    return ArrayVector2D(x, y)

def arrayvector2d_allclose(a: ArrayVector2D, b: ArrayVector2D,
                           rtol: 'number'=1e-09, atol: 'number'=0.0) -> bool:
    """Returns True if two ArrayVector2Ds are close enough to be considered equal."""
    return allclose(a.x, b.x, rtol=rtol, atol=atol) and \
        allclose(a.y, b.y, rtol=rtol, atol=atol)

def matmul_arrayvector2d(a: 'ndarray', b: 'ArrayVector2D') -> 'ArrayVector2D':
    """Returns the matrix multiplication of a and b."""
    x = a @ b.x
    y = a @ b.y
    return ArrayVector2D(x, y)
