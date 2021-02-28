from pygeom.geom2d import Vector2D, Coordinate2D
from numpy.matlib import matrix, zeros, multiply, divide, arctan2, sqrt, square

class MatrixVector2D(object):
    """MatrixVector2D Class"""
    x = None
    y = None
    def __init__(self, x: matrix, y: matrix):
        self.x = x
        self.y = y
    def to_unit(self):
        """Returns the unit matrix vector of this matrix vector"""
        mag = self.return_magnitude()
        return elementwise_divide(self, mag)
    def return_magnitude(self):
        """Returns the magnitude matrix of this matrix vector"""
        return sqrt(square(self.x) + square(self.y))
    def return_angle(self):
        """Returns the angle matrix of this matrix vector from the x axis"""
        return arctan2(self.y, self.x)
    def __getitem__(self, key):
        x = self.x[key]
        y = self.y[key]
        if isinstance(x, matrix) and isinstance(y, matrix):
            output = MatrixVector2D(x, y)
        else:
            output = Vector2D(x, y)
        return output
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.x[key] = value.x
            self.y[key] = value.y
        else:
            raise IndexError()
    @property
    def shape(self):
        return self.x.shape
    def transpose(self):
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
    def repeat(self, repeats, axis=None):
        x = self.x.repeat(repeats, axis=axis)
        y = self.y.repeat(repeats, axis=axis)
        return MatrixVector2D(x, y)
    def reshape(self, shape, order='C'):
        x = self.x.reshape(shape, order=order)
        y = self.y.reshape(shape, order=order)
        return MatrixVector2D(x, y)
    def tolist(self):
        lst = []
        for i in range(self.shape[0]):
            lst.append([])
            for j in range(self.shape[1]):
                x = self.x[i, j]
                y = self.y[i, j]
                lst[-1].append(Vector2D(x, y))
        return lst
    def copy(self, order='C'):
        x = self.x.copy(order=order)
        y = self.y.copy(order=order)
        return MatrixVector2D(x, y)
    def to_xy(self):
        """Returns the x and y values of this matrix vector"""
        return self.x, self.y
    def __mul__(self, obj):
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return self.x*obj.x+self.y*obj.y
        else:
            x = self.x*obj
            y = self.y*obj
            return MatrixVector2D(x, y)
    def __rmul__(self, obj):
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return obj.x*self.x+obj.y*self.y
        else:
            x = obj*self.x
            y = obj*self.y
            return MatrixVector2D(x, y)
    def __truediv__(self, obj):
        if isinstance(obj, (int, float, complex)):
            x = self.x/obj
            y = self.y/obj
            return MatrixVector2D(x, y)
        else:
            raise TypeError()
    def __pow__(self, obj):
        if isinstance(obj, Vector2D):
            z = self.x*obj.y-self.y*obj.x
            return z
        elif isinstance(obj, MatrixVector2D):
            z = self.x*obj.y-self.y*obj.x
            return z
        else:
            raise TypeError()
    def __add__(self, obj):
        if isinstance(obj, (MatrixVector2D, Vector2D)):
            x = self.x+obj.x
            y = self.y+obj.y
            return MatrixVector2D(x, y)
    def __radd__(self, obj):
        if obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj):
        if isinstance(obj, (MatrixVector2D, Vector2D)):
            x = self.x-obj.x
            y = self.y-obj.y
            return MatrixVector2D(x, y)
        else:
            raise TypeError()
    def __pos__(self):
        return MatrixVector2D(self.x, self.y)
    def __neg__(self):
        return MatrixVector2D(-self.x, -self.y)
    def __repr__(self):
        return '<MatrixVector2D: {:}, {:}>'.format(self.x, self.y)
    def __str__(self):
        return 'x:\n{:}\ny:\n{:}'.format(self.x, self.y)
    def __format__(self, format_spec):
        frmstr = 'x:\n{:'+format_spec+'}\ny:\n{:'+format_spec+'}'
        return frmstr.format(self.x, self.y)

def zero_matrix_vector(shape: tuple, dtype=float, order='C'):
    x = zeros(shape, dtype=dtype, order=order)
    y = zeros(shape, dtype=dtype, order=order)
    return MatrixVector2D(x, y)

def solve_matrix_vector(a: matrix, b: MatrixVector2D):
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
        raise ValueError()

def elementwise_divide(a: MatrixVector2D, b: matrix) -> MatrixVector2D:
    if a.shape == b.shape:
        x = divide(a.x, b)
        y = divide(a.y, b)
        return MatrixVector2D(x, y)
    else:
        raise ValueError()

def elementwise_dot_product(a: MatrixVector2D, b: MatrixVector2D) -> matrix:
    if a.shape == b.shape:
        return multiply(a.x, b.x) + multiply(a.y, b.y)
    else:
        raise ValueError()

def elementwise_cross_product(a: MatrixVector2D, b: MatrixVector2D) -> matrix:
    if a.shape == b.shape:
        z = multiply(a.x, b.y)-multiply(a.y, b.x)
        return z
    else:
        raise ValueError()

def vector2d_to_global(self: Coordinate2D, vec: MatrixVector2D) -> MatrixVector2D:
    """Transforms a vector from this local coordinate system to global"""
    dirx = Vector2D(self.dirx.x, self.diry.x)
    diry = Vector2D(self.dirx.y, self.diry.y)
    x = dirx*vec
    y = diry*vec
    return MatrixVector2D(x, y)

def vector2d_to_local(self: Coordinate2D, vec: MatrixVector2D) -> MatrixVector2D:
    """Transforms a vector from global  to this local coordinate system"""
    dirx = self.dirx
    diry = self.diry
    x = dirx*vec
    y = diry*vec
    return MatrixVector2D(x, y)
