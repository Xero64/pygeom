from math import atan2, cos, sin

class Vector2D(object):
    """Vector2D Class"""
    x = None
    y = None
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def to_unit(self):
        """Returns the unit vector of this vector"""
        mag = self.return_magnitude()
        if mag == 0:
            return Vector2D(self.x, self.y)
        else:
            x = self.x/mag
            y = self.y/mag
            return Vector2D(x, y)
    def to_point(self):
        """Returns the end point position of this vector"""
        from .point2d import Point2D
        return Point2D(self.x, self.y)
    def to_vector(self):
        """Returns a copy of this vector"""
        return Vector2D(self.x, self.y)
    def rotate(self, rot):
        """Rotates this vector by an input angle in radians"""
        mag = self.return_magnitude()
        ang = self.return_angle()
        x = mag*cos(ang+rot)
        y = mag*sin(ang+rot)
        return Vector2D(x, y)
    def return_angle(self):
        """Returns the angle of this vector from the x axis"""
        return atan2(self.y, self.x)
    def to_complex(self):
        """Returns the complex number of this vector"""
        cplx = self.x+1j*self.y
        return cplx
    def return_magnitude(self):
        """Returns the magnitude of this vector"""
        return (self.x**2+self.y**2)**0.5
    def to_xy(self):
        """Returns the x, y values of this vector"""
        return self.x, self.y
    def __mul__(self, obj):
        from numpy.matlib import matrix
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return self.x*obj.x+self.y*obj.y
        elif isinstance(obj, matrix):
            x = self.x*obj
            y = self.y*obj
            return MatrixVector2D(x, y)
        else:
            x = self.x*obj
            y = self.y*obj
            return Vector2D(x, y)
    def __rmul__(self, obj):
        from numpy.matlib import matrix
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return self.x*obj.x+self.y*obj.y
        elif isinstance(obj, matrix):
            x = obj*self.x
            y = obj*self.y
            return MatrixVector2D(x, y)
        else:
            x = obj*self.x
            y = obj*self.y
            return Vector2D(x, y)
    def __truediv__(self, obj):
        x = self.x/obj
        y = self.y/obj
        return Vector2D(x, y)
    def __pow__(self, obj):
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return self.x*obj.y-self.y*obj.x
    def __add__(self, obj):
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, Vector2D):
            x = self.x+obj.x
            y = self.y+obj.y
            return Vector2D(x, y)
        elif isinstance(obj, MatrixVector2D):
            x = self.x+obj.x
            y = self.y+obj.y
            return MatrixVector2D(x, y)
    def __radd__(self, obj):
        if obj == 0 or obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj):
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, Vector2D):
            x = self.x-obj.x
            y = self.y-obj.y
            return Vector2D(x, y)
        elif isinstance(obj, MatrixVector2D):
            x = self.x-obj.x
            y = self.y-obj.y
            return MatrixVector2D(x, y)
    def __pos__(self):
        return Vector2D(self.x, self.y)
    def __neg__(self):
        return Vector2D(-self.x, -self.y)
    def __repr__(self):
        return '<Vector2D: {:}, {:}>'.format(self.x, self.y)
    def __str__(self):
        return '<{:}, {:}>'.format(self.x, self.y)
    def __format__(self, format_spec):
        frmstr = '<{:'+format_spec+'}, {:'+format_spec+'}>'
        return frmstr.format(self.x, self.y)

def vector2d_from_complex(cplx):
    """Create a Vector2D from a complex number"""
    x = cplx.real
    y = cplx.imag
    return Vector2D(x, y)

def vector2d_from_points(pnta, pntb):
    """Create a Vector2D from two Point2Ds"""
    x = pntb.x-pnta.x
    y = pntb.y-pnta.y
    return Vector2D(x, y)

def vector2d_from_magang(mag, ang):
    """Create a Vector2D from magnatude and angle from x direction"""
    x = mag*cos(ang)
    y = mag*sin(ang)
    return Vector2D(x, y)

def vector2d_from_lists(x, y):
    """Create a list of Vector2D objects"""
    n = len(x)
    if len(y) == n:
        vecs = [Vector2D(x[i], y[i]) for i in range(n)]
        return vecs

ihat2d = Vector2D(1.0, 0.0)
jhat2d = Vector2D(0.0, 1.0)
zero_vector2d = Vector2D(0.0, 0.0)
