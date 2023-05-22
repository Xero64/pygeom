from typing import TYPE_CHECKING, Any, List, Tuple, Union

from numpy.matlib import zeros
from numpy.matlib import divide, matrix, multiply, sqrt, square

from pygeom.geom3d.vector import Vector

if TYPE_CHECKING:
    VectorLike = Union['Vector', 'MatrixVector']

class MatrixVector():
    """MatrixVector Class"""
    x: 'matrix' = None
    y: 'matrix' = None
    z: 'matrix' = None
    def __init__(self, x: 'matrix', y: 'matrix', z: 'matrix') -> None:
        """Initialise matrixvector object

        Args:
            x (matrix): x component of vector
            y (matrix): y component of vector
            z (matrix): z component of vector
        """
        if isinstance(x, matrix) and isinstance(y, matrix) and isinstance(z, matrix):
            self.x = x
            self.y = y
            self.z = z
        else:
            raise TypeError('Vector arguments must all be matrix.')
    def return_magnitude(self) -> 'matrix':
        """Returns the magnitude matrix of this matrixvector"""
        return sqrt(square(self.x) + square(self.y) + square(self.z))
    def to_unit(self) -> 'MatrixVector':
        """Returns the unit matrixvector of this matrixvector"""
        mag = self.return_magnitude()
        return elementwise_divide(self, mag)
    def __getitem__(self, key) -> 'VectorLike':
        x = self.x[key]
        y = self.y[key]
        z = self.z[key]
        if isinstance(x, matrix) and isinstance(y, matrix) and isinstance(z, matrix):
            return MatrixVector(x, y, z)
        else:
            return Vector(x, y, z)
    def __setitem__(self, key, value: 'VectorLike') -> None:
        if isinstance(key, tuple):
            self.x[key] = value.x
            self.y[key] = value.y
            self.z[key] = value.z
        else:
            raise IndexError()
    @property
    def shape(self) -> Tuple[int, int]:
        if self.x.shape == self.y.shape and self.y.shape == self.z.shape:
            return self.x.shape
        else:
            raise ValueError('MatrixVector x, y and z should have the same shape.')
    @property
    def dtype(self):
        if self.x.dtype is self.y.dtype and self.y.dtype is self.z.dtype:
            return self.x.dtype
        else:
            raise ValueError('MatrixVector x, y and z should have the same dtype.')
    def transpose(self) -> 'MatrixVector':
        x = self.x.transpose()
        y = self.y.transpose()
        z = self.z.transpose()
        return MatrixVector(x, y, z)
    def sum(self, axis=None, dtype=None, out=None) -> 'VectorLike':
        x = self.x.sum(axis=axis, dtype=dtype, out=out)
        y = self.y.sum(axis=axis, dtype=dtype, out=out)
        z = self.z.sum(axis=axis, dtype=dtype, out=out)
        if isinstance(x, matrix) and isinstance(y, matrix) and isinstance(z, matrix):
            return MatrixVector(x, y, z)
        else:
            return Vector(x, y, z)
    def repeat(self, repeats, axis=None) -> 'MatrixVector':
        x = self.x.repeat(repeats, axis=axis)
        y = self.y.repeat(repeats, axis=axis)
        z = self.z.repeat(repeats, axis=axis)
        return MatrixVector(x, y, z)
    def reshape(self, shape, order='C') -> 'MatrixVector':
        x = self.x.reshape(shape, order=order)
        y = self.y.reshape(shape, order=order)
        z = self.z.reshape(shape, order=order)
        return MatrixVector(x, y, z)
    def tolist(self) -> List[List['Vector']]:
        lst: List['Vector'] = []
        for i in range(self.shape[0]):
            lsti: List['Vector'] = []
            for j in range(self.shape[1]):
                x = self.x[i, j]
                y = self.y[i, j]
                z = self.z[i, j]
                lsti.append(Vector(x, y, z))
            lst.append(lsti)
        return lst
    def copy(self, order='C') -> 'MatrixVector':
        x = self.x.copy(order=order)
        y = self.y.copy(order=order)
        z = self.z.copy(order=order)
        return MatrixVector(x, y, z)
    def to_xyz(self) -> Tuple['matrix', 'matrix', 'matrix']:
        """Returns the x, y and z values of this matrix vector"""
        return self.x, self.y, self.z
    def dot(self, vec: 'VectorLike') -> 'matrix':
        if isinstance(vec, (MatrixVector, Vector)):
            return self.x*vec.x + self.y*vec.y + self.z*vec.z
        else:
            err = 'MatrixVector dot product must be with VectorLike object.'
            raise TypeError(err)
    def cross(self, vec: 'Vector') -> 'MatrixVector':
        if isinstance(vec, (Vector, MatrixVector)):
            x = self.y*vec.z - self.z*vec.y
            y = self.z*vec.x - self.x*vec.z
            z = self.x*vec.y - self.y*vec.x
            return MatrixVector(x, y, z)
        else:
            err = 'MatrixVector cross product must be with VectorLike object.'
            raise TypeError(err)
    def __abs__(self) -> 'matrix':
        return self.return_magnitude()
    def __mul__(self, obj: Any) -> Union['matrix', 'MatrixVector']:
        if isinstance(obj, (Vector, MatrixVector)):
            return self.x*obj.x + self.y*obj.y + self.z*obj.z
        else:
            x = self.x*obj
            y = self.y*obj
            z = self.z*obj
            return MatrixVector(x, y, z)
    def __rmul__(self, obj: Any) -> Union['matrix', 'MatrixVector']:
        if isinstance(obj, (Vector, MatrixVector)):
            return obj.x*self.x + obj.y*self.y + obj.z*self.z
        else:
            x = obj*self.x
            y = obj*self.y
            z = obj*self.z
            return MatrixVector(x, y, z)
    def __truediv__(self, obj: Any):
        x = self.x/obj
        y = self.y/obj
        z = self.z/obj
        return MatrixVector(x, y, z)
    def __pow__(self, obj: Any)  -> 'MatrixVector':
        if isinstance(obj, (Vector, MatrixVector)):
            x = self.y*obj.z - self.z*obj.y
            y = self.z*obj.x - self.x*obj.z
            z = self.x*obj.y - self.y*obj.x
            return MatrixVector(x, y, z)
        else:
            x = self.x**obj
            y = self.y**obj
            z = self.z**obj
            return MatrixVector(x, y, z)
    def __rpow__(self, obj: Any)  -> 'MatrixVector':
        if isinstance(obj, (Vector, MatrixVector)):
            x = obj.y*self.z - obj.z*self.y
            y = obj.z*self.x - obj.x*self.z
            z = obj.x*self.y - obj.y*self.x
            return MatrixVector(x, y, z)
        else:
            raise TypeError('MatrixVector cannot be an exponent of a power.')
    def __add__(self, obj: Any) -> 'MatrixVector':
        if isinstance(obj, (MatrixVector, Vector)):
            x = self.x + obj.x
            y = self.y + obj.y
            z = self.z + obj.z
            return MatrixVector(x, y, z)
        else:
            err = 'MatrixVector object can only be added to VectorLike object.'
            raise TypeError(err)
    def __radd__(self, obj: Any) -> 'MatrixVector':
        if obj is None:
            return self
        elif obj == Vector(0.0, 0.0, 0.0):
            return self
        elif obj == 0:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj: Any) -> 'MatrixVector':
        if isinstance(obj, (MatrixVector, Vector)):
            x = self.x - obj.x
            y = self.y - obj.y
            z = self.z - obj.z
            return MatrixVector(x, y, z)
        else:
            err = 'MatrixVector object can only be subtracted from VectorLike object.'
            raise TypeError(err)
    def __rsub__(self, obj: Any) -> 'MatrixVector':
        if isinstance(obj, (Vector, MatrixVector)):
            return -self.__sub__(obj)
        else:
            err = 'MatrixVector object cannot be reverse subtracted'
            err += f' from {type(obj)} object.'
            raise TypeError()
    def __pos__(self) -> 'MatrixVector':
        return MatrixVector(self.x, self.y, self.z)
    def __neg__(self) -> 'MatrixVector':
        return MatrixVector(-self.x, -self.y, -self.z)
    def __repr__(self) -> str:
        return '<MatrixVector: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __str__(self) -> str:
        return 'x:\n{:}\ny:\n{:}\nz:\n{:}'.format(self.x, self.y, self.z)
    def __format__(self, frm: str) -> str:
        frmstr = 'x:\n{:' + frm + '}\ny:\n{:' + frm + '}\nz:\n{:' + frm + '}'
        return frmstr.format(self.x, self.y, self.z)

def elementwise_multiply(a: 'MatrixVector',
                         b: 'matrix') -> 'MatrixVector':
    if a.shape == b.shape:
        x = multiply(a.x, b)
        y = multiply(a.y, b)
        z = multiply(a.z, b)
        return MatrixVector(x, y, z)
    else:
        raise ValueError('The shape of a and b need to be the same.')

def elementwise_divide(a: 'MatrixVector',
                       b: 'matrix') -> 'MatrixVector':
    if a.shape == b.shape:
        x = divide(a.x, b)
        y = divide(a.y, b)
        z = divide(a.z, b)
        return MatrixVector(x, y, z)
    else:
        raise ValueError('The shape of a and b need to be the same.')

def elementwise_dot_product(a: 'MatrixVector',
                            b: 'MatrixVector') -> 'matrix':
    if a.shape == b.shape:
        return multiply(a.x, b.x) + multiply(a.y, b.y) + multiply(a.z, b.z)
    else:
        raise ValueError('The shape of a and b need to be the same.')

def elementwise_cross_product(a: 'MatrixVector',
                              b: 'MatrixVector') -> 'MatrixVector':
    if a.shape == b.shape:
        x = multiply(a.y, b.z) - multiply(a.z, b.y)
        y = multiply(a.z, b.x) - multiply(a.x, b.z)
        z = multiply(a.x, b.y) - multiply(a.y, b.x)
        return MatrixVector(x, y, z)
    else:
        raise ValueError('The shape of a and b need to be the same.')

def zero_matrix_vector(shape: Tuple[int, int],
                       **kwargs) -> 'MatrixVector':
    x = zeros(shape, **kwargs)
    y = zeros(shape, **kwargs)
    z = zeros(shape, **kwargs)
    return MatrixVector(x, y, z)
