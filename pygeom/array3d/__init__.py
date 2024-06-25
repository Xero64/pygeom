from typing import TYPE_CHECKING

from numpy import allclose, float64, hsplit, hstack
from numpy.linalg import solve

from .arrayvector import ArrayVector as ArrayVector
from .arrayvector import arrayvector3d_from_2d as arrayvector3d_from_2d
from .arrayvector import zero_arrayvector as zero_arrayvector

if TYPE_CHECKING:
    from numpy.typing import NDArray

def solve_arrayvector(a: 'NDArray[float64]', b: 'ArrayVector') -> 'ArrayVector':
    newb = hstack(b.to_xyz())
    newc = solve(a, newb)
    x, y, z = hsplit(newc, 3)
    return ArrayVector(x, y, z)

def arrayvector_allclose(a: ArrayVector, b: ArrayVector,
                         rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two ArrayVectors are close enough to be considered equal."""
    return allclose(a.x, b.x, rtol=rtol, atol=atol) and \
        allclose(a.y, b.y, rtol=rtol, atol=atol) and \
            allclose(a.z, b.z, rtol=rtol, atol=atol)

def matmul_arrayvector(a: 'NDArray[float64]', b: 'ArrayVector') -> 'ArrayVector':
    """Returns the matrix multiplication of a and b."""
    x = a @ b.x
    y = a @ b.y
    z = a @ b.z
    return ArrayVector(x, y, z)
