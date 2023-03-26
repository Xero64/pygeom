from math import atan2, cos, sin, sqrt, isclose
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..geom3d.vector import Vector
    from .point2d import Point2D

class Vector2D():
    """Vector2D Class"""
    x: float = None
    y: float = None
    def __init__(self, x: float, y: float) -> None:
        if isinstance(x, float) and isinstance(y, float):
            self.x = x
            self.y = y
        else:
            raise TypeError('Vector2D arguments must all be float.')
    def to_unit(self) -> 'Vector2D':
        """Returns the unit vector of this vector2d"""
        mag = self.return_magnitude()
        if mag == 0:
            return Vector2D(self.x, self.y)
        else:
            x = self.x/mag
            y = self.y/mag
            return Vector2D(x, y)
    def to_point(self) -> 'Point2D':
        """Returns the end point position of this vector2d"""
        from .point2d import Point2D
        return Point2D(self.x, self.y)
    def copy(self) -> 'Vector2D':
        """Returns a copy of this vector2d"""
        return Vector2D(self.x, self.y)
    def return_magnitude(self):
        """Returns the magnitude of this vector2d"""
        return sqrt(self.x**2 + self.y**2)
    def return_angle(self) -> float:
        """Returns the angle of this vector from the x axis"""
        return atan2(self.y, self.x)
    def rotate(self, rot: float) -> 'Vector2D':
        """Rotates this vector by an input angle in radians"""
        mag = self.return_magnitude()
        ang = self.return_angle()
        x = mag*cos(ang + rot)
        y = mag*sin(ang + rot)
        return Vector2D(x, y)
    def to_complex(self) -> complex:
        """Returns the complex number of this vector"""
        cplx = self.x + 1j*self.y
        return cplx
    def to_xy(self) -> Tuple[float, float]:
        """Returns the x, y values of this vector"""
        return self.x, self.y
    def to_3d(self) -> 'Vector':
        from ..geom3d.vector import Vector
        """Returns the 2D vector as 3D vector with zero z value"""
        return Vector(self.x, self.y, 0.0)
    def dot(self, obj: 'Vector2D') -> float:
        return self.x*obj.x + self.y*obj.y
    def cross(self, obj: 'Vector2D') -> float:
        return self.x*obj.y - self.y*obj.x
    def __abs__(self) -> float:
        return self.return_magnitude()
    def __mul__(self, obj: Any) -> Union[float, 'Vector2D']:
        if isinstance(obj, Vector2D):
            return self.dot(obj)
        else:
            x = self.x*obj
            y = self.y*obj
            return Vector2D(x, y)
    def __rmul__(self, obj: Any) -> Union[float, 'Vector2D']:
        if isinstance(obj, Vector2D):
            return obj.dot(self)
        else:
            x = obj*self.x
            y = obj*self.y
            return Vector2D(x, y)
    def __truediv__(self, obj: Any) -> 'Vector2D':
        x = self.x/obj
        y = self.y/obj
        return Vector2D(x, y)
    def __pow__(self, obj: Any) -> Union[float, 'Vector2D']:
        if isinstance(obj, Vector2D):
            return self.cross(obj)
        else:
            x = self.x**obj
            y = self.y**obj
            return Vector2D(x, y)
    def __rpow__(self, obj: Any) -> 'Vector':
        return obj**self
    def __add__(self, obj: 'Vector2D') -> 'Vector2D':
        if isinstance(obj, Vector2D):
            x = self.x + obj.x
            y = self.y + obj.y
            return Vector2D(x, y)
        else:
            return obj.__add__(self)
    def __radd__(self, obj: Optional['Vector2D']=None) -> 'Vector2D':
        if obj is None:
            return self
        elif obj == 0:
            return self
        elif obj == Vector2D(0.0, 0.0):
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj: 'Vector2D') -> 'Vector2D':
        if isinstance(obj, Vector2D):
            x = self.x - obj.x
            y = self.y - obj.y
            return Vector2D(x, y)
        else:
            err = 'Vector2D object can only be subtracted from Vector2D object.'
            raise TypeError(err)
    def __pos__(self) -> 'Vector2D':
        return self
    def __neg__(self) -> 'Vector2D':
        return Vector2D(-self.x, -self.y)
    def __eq__(self, obj: Any) -> 'bool':
        if isinstance(obj, Vector2D):
            if obj.x == self.x and obj.y == self.y:
                return True
        return False
    def __neq__(self, obj: Any) -> 'bool':
        if isinstance(obj, Vector2D):
            if obj.x != self.x or obj.y != self.y:
                return True
        return False
    def __repr__(self) -> str:
        return '<Vector2D: {:}, {:}>'.format(self.x, self.y)
    def __str__(self) -> str:
        return '<{:}, {:}>'.format(self.x, self.y)
    def __format__(self, frm: str) -> str:
        frmstr: str = '<{:'+ frm +'}, {:'+ frm +'}>'
        return frmstr.format(self.x, self.y)

def zero_vector2d() -> 'Vector2D':
    return Vector2D(0.0, 0.0)

def vector2d_from_points(pnta: 'Point2D', pntb: 'Point2D') -> 'Vector2D':
    """Create a Vector2D from two Point2Ds"""
    x = pntb.x - pnta.x
    y = pntb.y - pnta.y
    return Vector2D(x, y)

def vector2d_isclose(a: Vector2D, b: Vector2D,
                     rel_tol: float=1e-09, abs_tol: float=0.0) -> bool:
    return isclose(a.x, b.x, rel_tol=rel_tol, abs_tol=abs_tol) and \
        isclose(a.y, b.y, rel_tol=rel_tol, abs_tol=abs_tol)
