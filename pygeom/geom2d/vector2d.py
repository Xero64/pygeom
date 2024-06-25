from typing import Any, Tuple, Union

from numpy import arctan2, cos, isclose, sin, sqrt, square


class Vector2D():
    """Vector2D Class"""
    x: float = None
    y: float = None

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def to_unit(self, return_magnitude: bool = False) -> Union['Vector2D',
                                                               Tuple['Vector2D',
                                                                     float]]:
        """Returns the unit vector of this vector2d"""
        mag = self.return_magnitude()
        if mag != 0.0:
            x = self.x/mag
            y = self.y/mag
        else:
            x = self.x
            y = self.y
        if return_magnitude:
            return Vector2D(x, y), mag
        else:
            return Vector2D(x, y)

    def return_magnitude(self):
        """Returns the magnitude of this vector2d"""
        return sqrt(square(self.x) + square(self.y))

    def return_angle(self) -> float:
        """Returns the angle of this vector from the x axis"""
        return arctan2(self.y, self.x)

    def dot(self, obj: 'Vector2D') -> float:
        try:
            return self.x*obj.x + self.y*obj.y
        except AttributeError:
            err = 'Vector2D object can only be dotted with Vector2D object.'
            raise TypeError(err)

    def cross(self, obj: 'Vector2D') -> float:
        try:
            return self.x*obj.y - self.y*obj.x
        except AttributeError:
            err = 'Vector2D object can only be crossed with Vector2D object.'
            raise TypeError(err)

    def __abs__(self) -> float:
        return self.return_magnitude()

    def __mul__(self, obj: Any) -> 'Vector2D':
        x = self.x*obj
        y = self.y*obj
        return Vector2D(x, y)

    def __rmul__(self, obj: Any) -> 'Vector2D':
        x = obj*self.x
        y = obj*self.y
        return Vector2D(x, y)

    def __truediv__(self, obj: Any) -> 'Vector2D':
        x = self.x/obj
        y = self.y/obj
        return Vector2D(x, y)

    def __pow__(self, obj: Any) -> 'Vector2D':
        x = self.x**obj
        y = self.y**obj
        return Vector2D(x, y)

    def __rpow__(self, obj: Any) -> 'Vector2D':
        x = obj**self.x
        y = obj**self.y
        return Vector2D(x, y)

    def __add__(self, obj: 'Vector2D') -> 'Vector2D':
        try:
            x = self.x + obj.x
            y = self.y + obj.y
            return Vector2D(x, y)
        except AttributeError:
            err = 'Vector2D object can only be added to Vector2D object.'
            raise TypeError(err)

    def __sub__(self, obj: 'Vector2D') -> 'Vector2D':
        try:
            x = self.x - obj.x
            y = self.y - obj.y
            return Vector2D(x, y)
        except AttributeError:
            err = 'Vector2D object can only be subtracted from Vector2D object.'
            raise TypeError(err)

    def __pos__(self) -> 'Vector2D':
        return self

    def __neg__(self) -> 'Vector2D':
        return Vector2D(-self.x, -self.y)

    def __repr__(self) -> str:
        return '<Vector2D: {:}, {:}>'.format(self.x, self.y)

    def __str__(self) -> str:
        return '<{:}, {:}>'.format(self.x, self.y)

    def __format__(self, frm: str) -> str:
        frmstr: str = '<{:'+ frm +'}, {:'+ frm +'}>'
        return frmstr.format(self.x, self.y)

    def __eq__(self, obj: 'Vector2D') -> 'bool':
        try:
            if obj.x == self.x and obj.y == self.y:
                return True
            else:
                return False
        except AttributeError:
            return False

    def __neq__(self, obj: 'Vector2D') -> 'bool':
        try:
            if obj.x != self.x or obj.y != self.y:
                return True
            else:
                return False
        except AttributeError:
            return False

    def rotate(self, rot: float) -> 'Vector2D':
        """Rotates this vector by an input angle in radians"""
        mag = self.return_magnitude()
        ang = self.return_angle()
        x = mag*cos(ang + rot)
        y = mag*sin(ang + rot)
        return Vector2D(x, y)

    def to_complex(self) -> float:
        """Returns the complex float of this vector"""
        cplx = self.x + 1j*self.y
        return cplx

    def to_xy(self) -> Tuple[float, float]:
        """Returns the x, y values of this vector"""
        return self.x, self.y

def zero_vector2d() -> Vector2D:
    return Vector2D(0.0, 0.0)

def vector2d_from_points(pnta: Vector2D, pntb: Vector2D) -> Vector2D:
    """Create a Vector2D from two Point2Ds"""
    x = pntb.x - pnta.x
    y = pntb.y - pnta.y
    return Vector2D(x, y)

def vector2d_isclose(a: Vector2D, b: Vector2D,
                     rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two Vector2Ds are close enough to be considered equal."""
    return isclose(a.x, b.x, rtol=rtol, atol=atol) and \
        isclose(a.y, b.y, rtol=rtol, atol=atol)
