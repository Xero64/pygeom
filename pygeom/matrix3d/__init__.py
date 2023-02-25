from typing import TYPE_CHECKING, Tuple

from numpy.matlib import zeros as zero_matrix

from ..geom3d.vector import Vector
from .matrixvector import MatrixVector

if TYPE_CHECKING:
    from numpy.matlib import matrix
    from ..geom3d.coordinate import Coordinate

def zero_matrix_vector(shape: Tuple['int', 'int'],
                       dtype=float, order='C') -> 'MatrixVector':
    x = zero_matrix(shape, dtype=dtype, order=order)
    y = zero_matrix(shape, dtype=dtype, order=order)
    z = zero_matrix(shape, dtype=dtype, order=order)
    return MatrixVector(x, y, z)

def solve_matrix_vector(a: 'matrix', b: 'MatrixVector') -> 'MatrixVector':
    from numpy.linalg import solve
    newb = zero_matrix((b.shape[0], b.shape[1]*3))
    for i in range(b.shape[1]):
        newb[:, 3*i+0] = b[:, i].x
        newb[:, 3*i+1] = b[:, i].y
        newb[:, 3*i+2] = b[:, i].z
    newc = solve(a, newb)
    c = zero_matrix_vector(b.shape)
    for i in range(b.shape[1]):
        c[:, i] = MatrixVector(newc[:, 3*i+0], newc[:, 3*i+1], newc[:, 3*i+2])
    return c

def vector_to_global(crd: 'Coordinate', vec: 'MatrixVector') -> 'MatrixVector':
    """Transforms a matrix vector from this local coordinate system to global"""
    dirx = Vector(crd.dirx.x, crd.diry.x, crd.dirz.x)
    diry = Vector(crd.dirx.y, crd.diry.y, crd.dirz.y)
    dirz = Vector(crd.dirx.z, crd.diry.z, crd.dirz.z)
    x = dirx*vec
    y = diry*vec
    z = dirz*vec
    return MatrixVector(x, y, z)

def vector_to_local(crd: 'Coordinate', vec: 'MatrixVector') -> 'MatrixVector':
    """Transforms a vector from global to this local coordinate system"""
    dirx = Vector(crd.dirx.x, crd.dirx.y, crd.dirx.z)
    diry = Vector(crd.diry.x, crd.diry.y, crd.diry.z)
    dirz = Vector(crd.dirz.x, crd.dirz.y, crd.dirz.z)
    x = dirx*vec
    y = diry*vec
    z = dirz*vec
    return MatrixVector(x, y, z)
