from typing import List, Tuple, Union
from numpy.matlib import arctan2, divide, matrix, multiply, sqrt, square
from ..geom2d.vector2d import Vector2D

class MatrixVector2D():
    """MatrixVector2D Class"""
    x: 'matrix' = None
    y: 'matrix' = None
    def __init__(self, x: 'matrix', y: 'matrix') -> None:
        self.x = x
        self.y = y
    def to_unit(self) -> 'MatrixVector2D':
        """Returns the unit matrix vector of this matrix vector"""
        mag = self.return_magnitude()
        return elementwise_divide(self, mag)
    def return_magnitude(self) -> 'matrix':
        """Returns the magnitude matrix of this matrix vector"""
        return sqrt(square(self.x) + square(self.y))
    def return_angle(self) -> 'matrix':
        """Returns the angle matrix of this matrix vector from the x axis"""
        return arctan2(self.y, self.x)
    def __getitem__(self, key) -> Union['Vector2D', 'MatrixVector2D']:
        x = self.x[key]
        y = self.y[key]
        if isinstance(x, matrix) and isinstance(y, matrix):
            output = MatrixVector2D(x, y)
        else:
            output = Vector2D(x, y)
        return output
    def __setitem__(self, key, value: 'Vector2D') -> Union['Vector2D',
                                                           'MatrixVector2D']:
        if isinstance(key, tuple):
            self.x[key] = value.x
            self.y[key] = value.y
        else:
            raise IndexError()
    @property
    def shape(self) -> Tuple['int']:
        if self.x.shape == self.y.shape:
            return self.x.shape
        else:
            raise ValueError('MatrixVector2D x and y should have the same shape.')
    @property
    def dtype(self):
        if self.x.dtype is self.y.dtype:
            return self.x.dtype
        else:
            raise ValueError('MatrixVector2D x and y should have the same dtype.')
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
    def tolist(self) -> List[List['Vector2D']]:
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
    def to_xy(self) -> Tuple['matrix']:
        """Returns the x and y values of this matrix vector"""
        return self.x, self.y
    def astype(self, type) -> 'MatrixVector2D':
        return MatrixVector2D(self.x.astype(type), self.y.astype(type))
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
    def __repr__(self) -> 'str':
        return '<MatrixVector2D: {:}, {:}>'.format(self.x, self.y)
    def __str__(self) -> 'str':
        return '\nx:\n{:}\ny:\n{:}'.format(self.x, self.y)
    def __format__(self, format_spec: str) -> 'str':
        frmstr = '\nx:\n{:'+format_spec+'}\ny:\n{:'+format_spec+'}'
        return frmstr.format(self.x, self.y)

def elementwise_multiply(a: 'MatrixVector2D',
                         b: 'matrix') -> 'MatrixVector2D':
    if a.shape == b.shape:
        x = multiply(a.x, b)
        y = multiply(a.y, b)
        return MatrixVector2D(x, y)
    else:
        raise ValueError('MatrixVector2D and matrix shapes not the same.')

def elementwise_divide(a: 'MatrixVector2D',
                       b: 'matrix') -> 'MatrixVector2D':
    if a.shape == b.shape:
        x = divide(a.x, b)
        y = divide(a.y, b)
        return MatrixVector2D(x, y)
    else:
        raise ValueError('MatrixVector2D and matrix shapes not the same.')

def elementwise_dot_product(a: 'MatrixVector2D',
                            b: 'MatrixVector2D') -> 'matrix':
    if a.shape == b.shape:
        return multiply(a.x, b.x) + multiply(a.y, b.y)
    else:
        raise ValueError('MatrixVector2D shapes not the same.')

def elementwise_cross_product(a: 'MatrixVector2D',
                              b: 'MatrixVector2D') -> 'matrix':
    if a.shape == b.shape:
        z = multiply(a.x, b.y) - multiply(a.y, b.x)
        return z
    else:
        raise ValueError('MatrixVector2D shapes not the same.')
