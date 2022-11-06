from typing import Tuple, Union, List
from pygeom.geom2d import Vector2D, Coordinate2D
from numpy.matlib import matrix, zeros, multiply, divide, arctan2, sqrt, square

class MatrixVector2D():
    """MatrixVector2D Class"""
    x: matrix = None
    y: matrix = None
    def __init__(self, x: matrix, y: matrix) -> None:
        self.x = x
        self.y = y
    def to_unit(self) -> 'MatrixVector2D':
        """Returns the unit matrix vector of this matrix vector"""
        mag = self.return_magnitude()
        return elementwise_divide(self, mag)
    def return_magnitude(self) -> matrix:
        """Returns the magnitude matrix of this matrix vector"""
        return sqrt(square(self.x) + square(self.y))
    def return_angle(self) -> matrix:
        """Returns the angle matrix of this matrix vector from the x axis"""
        return arctan2(self.y, self.x)
    def __getitem__(self, key) -> Union[Vector2D, 'MatrixVector2D']:
        x = self.x[key]
        y = self.y[key]
        if isinstance(x, matrix) and isinstance(y, matrix):
            output = MatrixVector2D(x, y)
        else:
            output = Vector2D(x, y)
        return output
    def __setitem__(self, key, value: Vector2D) -> Union[Vector2D, 'MatrixVector2D']:
        if isinstance(key, tuple):
            self.x[key] = value.x
            self.y[key] = value.y
        else:
            raise IndexError()
    @property
    def shape(self) -> Tuple[int]:
        return self.x.shape
    def transpose(self) -> 'MatrixVector2D':
        x = self.x.transpose()
        y = self.y.transpose()
        return MatrixVector2D(x, y)
    def sum(self, axis=None, dtype=None, out=None):
        x = self.x.sum(axis=axis, dtype=dtype, out=out)
        y = self.y.sum(axis=axis, dtype=dtype, out=out)
        if isinstance(x, matrix) and isinstance(y, matrix):
            return MatrixVector2D(x, y)
        else:
            return Vector2D(x, y)
    def repeat(self, repeats, axis=None) -> 'MatrixVector2D':
        x = self.x.repeat(repeats, axis=axis)
        y = self.y.repeat(repeats, axis=axis)
        return MatrixVector2D(x, y)
    def reshape(self, shape, order='C') -> 'MatrixVector2D':
        x = self.x.reshape(shape, order=order)
        y = self.y.reshape(shape, order=order)
        return MatrixVector2D(x, y)
    def tolist(self) -> List[List[Vector2D]]:
        lst = []
        for i in range(self.shape[0]):
            lstj = []
            for j in range(self.shape[1]):
                x = self.x[i, j]
                y = self.y[i, j]
                lstj.append(Vector2D(x, y))
            lst.append(lstj)
        return lst
    def copy(self, order='C') -> 'MatrixVector2D':
        x = self.x.copy(order=order)
        y = self.y.copy(order=order)
        return MatrixVector2D(x, y)
    def to_xy(self) -> Tuple[matrix]:
        """Returns the x and y values of this matrix vector"""
        return self.x, self.y
    def __mul__(self, obj) -> 'MatrixVector2D':
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return self.x*obj.x+self.y*obj.y
        else:
            x = self.x*obj
            y = self.y*obj
            return MatrixVector2D(x, y)
    def __rmul__(self, obj) -> 'MatrixVector2D':
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return obj.x*self.x+obj.y*self.y
        else:
            x = obj*self.x
            y = obj*self.y
            return MatrixVector2D(x, y)
    def __truediv__(self, obj) -> 'MatrixVector2D':
        if isinstance(obj, (int, float, complex)):
            x = self.x/obj
            y = self.y/obj
            return MatrixVector2D(x, y)
        else:
            raise TypeError('Invalid type for MatrixVector2D true division.')
    def __pow__(self, obj) -> 'MatrixVector2D':
        if isinstance(obj, Vector2D):
            z = self.x*obj.y-self.y*obj.x
            return z
        elif isinstance(obj, MatrixVector2D):
            z = self.x*obj.y-self.y*obj.x
            return z
        else:
            raise TypeError('Invalid type for MatrixVector2D cross product.')
    def __add__(self, obj) -> 'MatrixVector2D':
        if isinstance(obj, (MatrixVector2D, Vector2D)):
            x = self.x+obj.x
            y = self.y+obj.y
            return MatrixVector2D(x, y)
        else:
            raise TypeError('Invalid type for MatrixVector2D addition.')
    def __radd__(self, obj) -> 'MatrixVector2D':
        if obj is None:
            return self
        elif obj == 0:
            return self
        else:
            if isinstance(obj, (MatrixVector2D, Vector2D)):
                return self.__add__(obj)
            else:
                raise TypeError('Invalid type for MatrixVector2D right addition.')
    def __sub__(self, obj) -> 'MatrixVector2D':
        if isinstance(obj, (MatrixVector2D, Vector2D)):
            x = self.x-obj.x
            y = self.y-obj.y
            return MatrixVector2D(x, y)
        else:
            raise TypeError('Invalid type for MatrixVector2D subtraction.')
    def __pos__(self) -> 'MatrixVector2D':
        return MatrixVector2D(self.x, self.y)
    def __neg__(self) -> 'MatrixVector2D':
        return MatrixVector2D(-self.x, -self.y)
    def __repr__(self) -> str:
        return '<MatrixVector2D: {:}, {:}>'.format(self.x, self.y)
    def __str__(self) -> str:
        return 'x:\n{:}\ny:\n{:}'.format(self.x, self.y)
    def __format__(self, format_spec: str) -> str:
        frmstr = 'x:\n{:'+format_spec+'}\ny:\n{:'+format_spec+'}'
        return frmstr.format(self.x, self.y)

def zero_matrix_vector(shape: tuple, dtype=float, order='C') -> MatrixVector2D:
    x = zeros(shape, dtype=dtype, order=order)
    y = zeros(shape, dtype=dtype, order=order)
    return MatrixVector2D(x, y)

def solve_matrix_vector(a: matrix, b: MatrixVector2D) -> MatrixVector2D:
    from numpy.linalg import solve
    newb = zeros((b.shape[0], b.shape[1]*2))
    for i in range(b.shape[1]):
        newb[:, 2*i+0] = b[:, i].x
        newb[:, 2*i+1] = b[:, i].y
    newc = solve(a, newb)
    c = zero_matrix_vector(b.shape)
    for i in range(b.shape[1]):
        c[:, i] = MatrixVector2D(newc[:, 2*i+0], newc[:, 2*i+1])
    return c

def elementwise_multiply(a: MatrixVector2D, b: matrix) -> MatrixVector2D:
    if a.shape == b.shape:
        x = multiply(a.x, b)
        y = multiply(a.y, b)
        return MatrixVector2D(x, y)
    else:
        raise ValueError('MatrixVector2D and matrix shapes not the same.')

def elementwise_divide(a: MatrixVector2D, b: matrix) -> MatrixVector2D:
    if a.shape == b.shape:
        x = divide(a.x, b)
        y = divide(a.y, b)
        return MatrixVector2D(x, y)
    else:
        raise ValueError('MatrixVector2D and matrix shapes not the same.')

def elementwise_dot_product(a: MatrixVector2D, b: MatrixVector2D) -> matrix:
    if a.shape == b.shape:
        return multiply(a.x, b.x) + multiply(a.y, b.y)
    else:
        raise ValueError('MatrixVector2D shapes not the same.')

def elementwise_cross_product(a: MatrixVector2D, b: MatrixVector2D) -> matrix:
    if a.shape == b.shape:
        z = multiply(a.x, b.y)-multiply(a.y, b.x)
        return z
    else:
        raise ValueError('MatrixVector2D shapes not the same.')

def vector2d_to_global(crd: Coordinate2D, vec: MatrixVector2D) -> MatrixVector2D:
    """Transforms a vector from this local coordinate system to global"""
    dirx = Vector2D(crd.dirx.x, crd.diry.x)
    diry = Vector2D(crd.dirx.y, crd.diry.y)
    x = dirx*vec
    y = diry*vec
    return MatrixVector2D(x, y)

def vector2d_to_local(crd: Coordinate2D, vec: MatrixVector2D) -> MatrixVector2D:
    """Transforms a vector from global  to this local coordinate system"""
    dirx = crd.dirx
    diry = crd.diry
    x = dirx*vec
    y = diry*vec
    return MatrixVector2D(x, y)
