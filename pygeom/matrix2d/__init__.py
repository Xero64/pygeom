from typing import TYPE_CHECKING

from numpy.matlib import zeros as zero_matrix

from ..geom2d.vector2d import Vector2D
from .matrixvector2d import MatrixVector2D, zero_matrix_vector
from .matrixtensor2d import MatrixTensor2D as MatrixTensor2D
from .matrixtensor2d import zero_matrix_tensor as zero_matrix_tensor

if TYPE_CHECKING:
    from numpy.matlib import matrix
    from ..geom2d.coordinate2d import Coordinate2D

def solve_matrix_vector(a: 'matrix', b: 'MatrixVector2D') -> 'MatrixVector2D':
    from numpy.linalg import solve
    newb = zero_matrix((b.shape[0], b.shape[1]*2), dtype=b.dtype)
    for i in range(b.shape[1]):
        newb[:, 2*i+0] = b[:, i].x
        newb[:, 2*i+1] = b[:, i].y
    newc = solve(a, newb)
    c = zero_matrix_vector(b.shape, dtype=b.dtype)
    for i in range(b.shape[1]):
        c[:, i] = MatrixVector2D(newc[:, 2*i+0], newc[:, 2*i+1])
    return c

def vector2d_to_global(crd: 'Coordinate2D', vec: 'MatrixVector2D') -> 'MatrixVector2D':
    """Transforms a vector from this local coordinate system to global"""
    dirx = Vector2D(crd.dirx.x, crd.diry.x)
    diry = Vector2D(crd.dirx.y, crd.diry.y)
    x = vec*dirx
    y = vec*diry
    return MatrixVector2D(x, y)

def vector2d_to_local(crd: 'Coordinate2D', vec: 'MatrixVector2D') -> 'MatrixVector2D':
    """Transforms a vector from global  to this local coordinate system"""
    x = vec*crd.dirx
    y = vec*crd.diry
    return MatrixVector2D(x, y)
