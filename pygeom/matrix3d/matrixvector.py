from pygeom.geom3d import Vector, Point, Coordinate
from numpy.matlib import matrix, zeros, multiply, divide

class MatrixVector(object):
    """Vector Class"""
    x = None
    y = None
    z = None
    def __init__(self, x: matrix, y: matrix, z: matrix):
        self.x = x
        self.y = y
        self.z = z
    def to_unit(self):
        """Returns the unit matrixvector of this matrixvector"""
        mag = self.return_magnitude()
        return elementwise_divide(self, mag)
    def return_magnitude(self):
        """Returns the magnitude matrix of this matrixvector"""
        from numpy.matlib import sqrt
        return sqrt(elementwise_dot_product(self, self))
    def __getitem__(self, key):
        x = self.x[key]
        y = self.y[key]
        z = self.z[key]
        if isinstance(x, matrix) and isinstance(y, matrix) and isinstance(z, matrix):
            return MatrixVector(x, y, z)
        else:
            return Vector(x, y, z)
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self.x[key] = value.x
            self.y[key] = value.y
            self.z[key] = value.z
        else:
            raise IndexError()
    @property
    def shape(self):
        return self.x.shape
    def transpose(self):
        x = self.x.transpose()
        y = self.y.transpose()
        z = self.z.transpose()
        return MatrixVector(x, y, z)
    def sum(self, axis=None, dtype=None, out=None):
        x = self.x.sum(axis=axis, dtype=dtype, out=out)
        y = self.y.sum(axis=axis, dtype=dtype, out=out)
        z = self.z.sum(axis=axis, dtype=dtype, out=out)
        if isinstance(x, matrix) and isinstance(y, matrix) and isinstance(z, matrix):
            return MatrixVector(x, y, z)
        else:
            return Vector(x, y, z)
    def repeat(self, repeats, axis=None):
        x = self.x.repeat(repeats, axis=axis)
        y = self.y.repeat(repeats, axis=axis)
        z = self.z.repeat(repeats, axis=axis)
        return MatrixVector(x, y, z)
    def reshape(self, shape, order='C'):
        x = self.x.reshape(shape, order=order)
        y = self.y.reshape(shape, order=order)
        z = self.z.reshape(shape, order=order)
        return MatrixVector(x, y, z)
    def tolist(self):
        lst = []
        for i in range(self.shape[0]):
            lst.append([])
            for j in range(self.shape[1]):
                x = self.x[i, j]
                y = self.y[i, j]
                z = self.z[i, j]
                lst[-1].append(Vector(x, y, z))
        return lst
    def copy(self, order='C'):
        x = self.x.copy(order=order)
        y = self.y.copy(order=order)
        z = self.z.copy(order=order)
        return MatrixVector(x, y, z)
    def to_xyz(self):
        """Returns the x, y and z values of this matrix vector"""
        return self.x, self.y, self.z
    def __mul__(self, obj):
        if isinstance(obj, (Vector, MatrixVector)):
            return self.x*obj.x+self.y*obj.y+self.z*obj.z
        else:
            x = self.x*obj
            y = self.y*obj
            z = self.z*obj
            return MatrixVector(x, y, z)
    def __rmul__(self, obj):
        if isinstance(obj, (Vector, MatrixVector)):
            return obj.x*self.x+obj.y*self.y+obj.z*self.z
        else:
            x = obj*self.x
            y = obj*self.y
            z = obj*self.z
            return MatrixVector(x, y, z)
    def __truediv__(self, obj):
        if isinstance(obj, (int, float, complex)):
            x = self.x/obj
            y = self.y/obj
            z = self.z/obj
            return MatrixVector(x, y, z)
        else:
            raise TypeError()
    def __pow__(self, obj):
        if isinstance(obj, Vector):
            x = self.y*obj.z-self.z*obj.y
            y = self.z*obj.x-self.x*obj.z
            z = self.x*obj.y-self.y*obj.x
            return MatrixVector(x, y, z)
        elif isinstance(obj, MatrixVector):
            x = self.y*obj.z-self.z*obj.y
            y = self.z*obj.x-self.x*obj.z
            z = self.x*obj.y-self.y*obj.x
            return MatrixVector(x, y, z)
        else:
            raise TypeError()
    def __add__(self, obj):
        if isinstance(obj, (MatrixVector, Vector)):
            x = self.x+obj.x
            y = self.y+obj.y
            z = self.z+obj.z
            return MatrixVector(x, y, z)
    def __radd__(self, obj):
        if obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj):
        if isinstance(obj, (MatrixVector, Vector)):
            x = self.x-obj.x
            y = self.y-obj.y
            z = self.z-obj.z
            return MatrixVector(x, y, z)
        else:
            raise TypeError()
    def __pos__(self):
        return MatrixVector(self.x, self.y, self.z)
    def __neg__(self):
        return MatrixVector(-self.x, -self.y, -self.z)
    def __repr__(self):
        return '<MatrixVector: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __str__(self):
        return 'x:\n{:}\ny:\n{:}\nz:\n{:}'.format(self.x, self.y, self.z)
    def __format__(self, format_spec):
        frmstr = 'x:\n{:'+format_spec+'}\ny:\n{:'+format_spec+'}\nz:\n{:'+format_spec+'}'
        return frmstr.format(self.x, self.y, self.z)

def zero_matrix_vector(shape: tuple, dtype=float, order='C'):
    x = zeros(shape, dtype=dtype, order=order)
    y = zeros(shape, dtype=dtype, order=order)
    z = zeros(shape, dtype=dtype, order=order)
    return MatrixVector(x, y, z)

def solve_matrix_vector(a: matrix, b: MatrixVector):
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

def elementwise_multiply(a: MatrixVector, b: matrix) -> MatrixVector:
    if a.shape == b.shape:
        x = multiply(a.x, b)
        y = multiply(a.y, b)
        z = multiply(a.z, b)
        return MatrixVector(x, y, z)
    else:
        raise ValueError()

def elementwise_divide(a: MatrixVector, b: matrix) -> MatrixVector:
    if a.shape == b.shape:
        x = divide(a.x, b)
        y = divide(a.y, b)
        z = divide(a.z, b)
        return MatrixVector(x, y, z)
    else:
        raise ValueError()

def elementwise_dot_product(a: MatrixVector, b: MatrixVector) -> matrix:
    if a.shape == b.shape:
        return multiply(a.x, b.x) + multiply(a.y, b.y) + multiply(a.z, b.z)
    else:
        raise ValueError()

def elementwise_cross_product(a: MatrixVector, b: MatrixVector) -> MatrixVector:
    if a.shape == b.shape:
        x = multiply(a.y, b.z)-multiply(a.z, b.y)
        y = multiply(a.z, b.x)-multiply(a.x, b.z)
        z = multiply(a.x, b.y)-multiply(a.y, b.x)
        return MatrixVector(x, y, z)
    else:
        raise ValueError()

def vector_to_global(crd: Coordinate, vec: MatrixVector) -> MatrixVector:
    """Transforms a matrix vector from this local coordinate system to global"""
    dirx = Vector(crd.dirx.x, crd.diry.x, crd.dirz.x)
    diry = Vector(crd.dirx.y, crd.diry.y, crd.dirz.y)
    dirz = Vector(crd.dirx.z, crd.diry.z, crd.dirz.z)
    x = dirx*vec
    y = diry*vec
    z = dirz*vec
    return MatrixVector(x, y, z)

def vector_to_local(crd: Coordinate, vec: MatrixVector) -> MatrixVector:
    """Transforms a vector from global to this local coordinate system"""
    dirx = Vector(crd.dirx.x, crd.dirx.y, crd.dirx.z)
    diry = Vector(crd.diry.x, crd.diry.y, crd.diry.z)
    dirz = Vector(crd.dirz.x, crd.dirz.y, crd.dirz.z)
    x = dirx*vec
    y = diry*vec
    z = dirz*vec
    return MatrixVector(x, y, z)
