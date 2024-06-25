from typing import TYPE_CHECKING, Any, Tuple, Union

from numpy import isclose, sqrt, square

class Vector():
    """Vector Class"""
    x: float = None
    y: float = None
    z: float = None

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def to_unit(self, return_magnitude: bool = False) -> Union['Vector',
                                                               Tuple['Vector',
                                                                     float]]:
        """Returns the unit vector of this vector"""
        mag = self.return_magnitude()
        if mag != 0.0:
            x = self.x/mag
            y = self.y/mag
            z = self.z/mag
        else:
            x = self.x
            y = self.y
            z = self.z
        if return_magnitude:
            return Vector(x, y, z), mag
        else:
            return Vector(x, y, z)

    def return_magnitude(self) -> float:
        """Returns the magnitude of this vector"""
        return sqrt(square(self.x) + square(self.y) + square(self.z))

    def to_xyz(self) -> Tuple[float, float, float]:
        """Returns the x, y and z values of this vector"""
        return self.x, self.y, self.z

    def dot(self, vec: 'Vector') -> float:
        try:
            return self.x*vec.x + self.y*vec.y + self.z*vec.z
        except AttributeError:
            err = 'Vector dot product must be with Vector object.'
            raise TypeError(err)

    def cross(self, vec: 'Vector') -> 'Vector':
        try:
            x = self.y*vec.z - self.z*vec.y
            y = self.z*vec.x - self.x*vec.z
            z = self.x*vec.y - self.y*vec.x
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector cross product must be with Vector object.'
            raise TypeError(err)

    def rcross(self, vec: 'Vector') -> 'Vector':
        try:
            x = vec.y*self.z - vec.z*self.y
            y = vec.z*self.x - vec.x*self.z
            z = vec.x*self.y - vec.y*self.x
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector cross product must be with Vector object.'
            raise TypeError(err)

    def __abs__(self) -> float:
        return self.return_magnitude()

    def __mul__(self, obj: Any) -> 'Vector':
        x = self.x*obj
        y = self.y*obj
        z = self.z*obj
        return Vector(x, y, z)

    def __rmul__(self, obj: Any) -> 'Vector':
        x = obj*self.x
        y = obj*self.y
        z = obj*self.z
        return Vector(x, y, z)

    def __truediv__(self, obj: Any) -> 'Vector':
        x = self.x/obj
        y = self.y/obj
        z = self.z/obj
        return Vector(x, y, z)

    def __pow__(self, obj: Any) -> 'Vector':
        x = self.x**obj
        y = self.y**obj
        z = self.z**obj
        return Vector(x, y, z)

    def __rpow__(self, obj: Any) -> 'Vector':
        x = obj**self.x
        y = obj**self.y
        z = obj**self.z
        return Vector(x, y, z)

    def __add__(self, obj: 'Vector') -> 'Vector':
        try:
            x = self.x + obj.x
            y = self.y + obj.y
            z = self.z + obj.z
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector object can only be added to Vector object.'
            raise TypeError(err)

    def __sub__(self, obj: 'Vector') -> 'Vector':
        try:
            x = self.x - obj.x
            y = self.y - obj.y
            z = self.z - obj.z
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector object can only be subtracted from Vector object.'
            raise TypeError(err)

    def __pos__(self) -> 'Vector':
        return self

    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        return '<Vector: {:}, {:}, {:}>'.format(self.x, self.y, self.z)

    def __str__(self) -> str:
        return '<{:}, {:}, {:}>'.format(self.x, self.y, self.z)

    def __format__(self, frm: str) -> str:
        frmstr = '<{:' + frm + '}, {:' + frm + '}, {:' + frm + '}>'
        return frmstr.format(self.x, self.y, self.z)

    def __eq__(self, obj: 'Vector') -> 'bool':
        try:
            if obj.x == self.x and obj.y == self.y and obj.z == self.z:
                return True
            else:
                return False
        except AttributeError:
            return False

    def __neq__(self, obj: 'Vector') -> 'bool':
        try:
            if obj.x != self.x or obj.y != self.y or obj.z != self.z:
                return True
            else:
                return False
        except AttributeError:
            return False

def zero_vector() -> Vector:
    return Vector(0.0, 0.0, 0.0)

def vector_isclose(a: Vector, b: Vector,
                   rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two vectors are close enough to be considered equal"""
    return isclose(a.x, b.x, rtol=rtol, atol=atol) and \
        isclose(a.y, b.y, rtol=rtol, atol=atol) and \
        isclose(a.z, b.z, rtol=rtol, atol=atol)
