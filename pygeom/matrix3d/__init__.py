from typing import TYPE_CHECKING

from numpy.matlib import zeros

from ..geom3d.vector import Vector
from .matrixvector import MatrixVector, zero_matrix_vector

if TYPE_CHECKING:
    from numpy.matlib import matrix
    from ..geom3d.transform import Transform

def solve_matrix_vector(a: 'matrix', b: 'MatrixVector') -> 'MatrixVector':
    from numpy.linalg import solve
    newb = zeros((b.shape[0], b.shape[1]*3))
    for i in range(b.shape[1]):
        newb[:, 3*i+0] = b[:, i].x
        newb[:, 3*i+1] = b[:, i].y
        newb[:, 3*i+2] = b[:, i].z
    newc = solve(a, newb)
    c = zero_matrix_vector(b.shape)
    for i in range(b.shape[1]):
        c[:, i] = MatrixVector(newc[:, 3*i+0], newc[:, 3*i+1], newc[:, 3*i+2])
    return c

def vector_to_global(tfm: 'Transform', vec: 'MatrixVector') -> 'MatrixVector':
    """Transforms a matrix vector from this local coordinate system to global"""
    dirx = Vector(tfm.dirx.x, tfm.diry.x, tfm.dirz.x)
    diry = Vector(tfm.dirx.y, tfm.diry.y, tfm.dirz.y)
    dirz = Vector(tfm.dirx.z, tfm.diry.z, tfm.dirz.z)
    x = vec*dirx
    y = vec*diry
    z = vec*dirz
    return MatrixVector(x, y, z)

def vector_to_local(tfm: 'Transform', vec: 'MatrixVector') -> 'MatrixVector':
    """Transforms a vector from global to this local coordinate system"""
    x = vec*tfm.dirx
    y = vec*tfm.diry
    z = vec*tfm.dirz
    return MatrixVector(x, y, z)
