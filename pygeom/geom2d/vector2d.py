from typing import TYPE_CHECKING, Tuple, Union, Optional, Any
from math import atan2, cos, sin
from ..geom3d.vector import Vector

if TYPE_CHECKING:
    from numpy.matlib import matrix
    from pygeom.matrix2d import MatrixVector2D

class Vector2D():
    """Vector2D Class"""
    x: 'float' = None
    y: 'float' = None
    def __init__(self, x: 'float', y: 'float') -> None:
        self.x = x
        self.y = y
    def to_unit(self) -> 'Vector2D':
        """Returns the unit vector of this vector"""
        mag = self.return_magnitude()
        if mag == 0:
            return Vector2D(self.x, self.y)
        else:
            x = self.x/mag
            y = self.y/mag
            return Vector2D(x, y)
    def to_point(self) -> 'Vector2D':
        """Returns the end point position of this vector"""
        return Vector2D(self.x, self.y)
    def copy(self) -> 'Vector2D':
        """Returns a copy of this vector"""
        return Vector2D(self.x, self.y)
    def return_magnitude(self):
        """Returns the magnitude of this vector"""
        return (self.x**2 + self.y**2)**0.5
    def return_angle(self) -> 'float':
        """Returns the angle of this vector from the x axis"""
        return atan2(self.y, self.x)
    def rotate(self, rot: 'float') -> 'Vector2D':
        """Rotates this vector by an input angle in radians"""
        mag = self.return_magnitude()
        ang = self.return_angle()
        x = mag*cos(ang + rot)
        y = mag*sin(ang + rot)
        return Vector2D(x, y)
    def to_complex(self) -> 'complex':
        """Returns the complex number of this vector"""
        cplx = self.x + 1j*self.y
        return cplx
    def to_xy(self) -> Tuple['float', 'float']:
        """Returns the x, y values of this vector"""
        return self.x, self.y
    def to_3d(self) -> 'Vector':
        """Returns the 2D vector as 3D vector with zero z value"""
        return Vector(self.x, self.y, 0.0)
    def __mul__(self, obj: Any) -> Union['Vector2D', 'MatrixVector2D']:
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
    def __rmul__(self, obj: Any) -> Union['Vector2D', 'MatrixVector2D']:
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
    def __truediv__(self, obj: 'float') -> 'Vector2D':
        x = self.x/obj
        y = self.y/obj
        return Vector2D(x, y)
    def __pow__(self, obj: Union['Vector2D', 'MatrixVector2D']) -> Union['float',
                                                                         'matrix']:
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, (Vector2D, MatrixVector2D)):
            return self.x*obj.y-self.y*obj.x
        else:
            err = 'Can only cross product Vector2D to Vector2D or MatrixVector2D.'
            raise ValueError(err)
    def __add__(self, obj: Union['Vector2D', 'MatrixVector2D']) -> Union['Vector2D',
                                                                         'MatrixVector2D']:
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, Vector2D):
            x = self.x+obj.x
            y = self.y+obj.y
            return Vector2D(x, y)
        elif isinstance(obj, MatrixVector2D):
            x = self.x+obj.x
            y = self.y+obj.y
            return MatrixVector2D(x, y)
        else:
            raise ValueError('Can only add Vector2D to Vector2D or MatrixVector2D.')
    def __radd__(self, obj: Optional[Union['Vector2D', 'MatrixVector2D']]):
        if obj == 0 or obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj: Union['Vector2D', 'MatrixVector2D']):
        from pygeom.matrix2d import MatrixVector2D
        if isinstance(obj, Vector2D):
            x = self.x-obj.x
            y = self.y-obj.y
            return Vector2D(x, y)
        elif isinstance(obj, MatrixVector2D):
            x = self.x-obj.x
            y = self.y-obj.y
            return MatrixVector2D(x, y)
    def __pos__(self) -> 'Vector2D':
        return self
    def __neg__(self) -> 'Vector2D':
        return Vector2D(-self.x, -self.y)
    def __repr__(self) -> 'str':
        return '<Vector2D: {:}, {:}>'.format(self.x, self.y)
    def __str__(self) -> 'str':
        return '<{:}, {:}>'.format(self.x, self.y)
    def __format__(self, format_spec) -> 'str':
        frmstr: 'str' = '<{:'+format_spec+'}, {:'+format_spec+'}>'
        return frmstr.format(self.x, self.y)

def zero_vector2d() -> 'Vector2D':
    return Vector2D(0.0, 0.0)

def vector2d_from_points(pnta: 'Vector2D', pntb: 'Vector2D') -> 'Vector2D':
    """Create a Vector2D from two Point2Ds"""
    x = pntb.x-pnta.x
    y = pntb.y-pnta.y
    return Vector2D(x, y)
