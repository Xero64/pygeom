from typing import TYPE_CHECKING, Tuple, Union, Optional, Any

from math import sqrt, isclose

if TYPE_CHECKING:
    from .point import Point

class Vector():
    """Vector Class"""
    x: float = None
    y: float = None
    z: float = None
    def __init__(self, x: float, y: float, z: float) -> None:
        if isinstance(x, float) and isinstance(y, float) and isinstance(z, float):
            self.x = x
            self.y = y
            self.z = z
        else:
            raise TypeError('Vector arguments must all be float.')
    def to_unit(self) -> 'Vector':
        """Returns the unit vector of this vector"""
        mag = self.return_magnitude()
        if mag == 0:
            return Vector(self.x, self.y, self.z)
        else:
            x = self.x/mag
            y = self.y/mag
            z = self.z/mag
            return Vector(x, y, z)
    def to_point(self) -> 'Point':
        """Returns the end point position of this vector"""
        from .point import Point
        return Point(self.x, self.y, self.z)
    def to_vector(self) -> 'Vector':
        """Returns a copy of this vector"""
        return Vector(self.x, self.y, self.z)
    def return_magnitude(self) -> float:
        """Returns the magnitude of this vector"""
        return sqrt(self.x**2 + self.y**2 + self.z**2)
    def to_xyz(self) -> Tuple[float, float, float]:
        """Returns the x, y and z values of this vector"""
        return self.x, self.y, self.z
    def dot(self, vec: 'Vector') -> float:
        if isinstance(vec, Vector):
            return self.x*vec.x + self.y*vec.y + self.z*vec.z
        else:
            raise TypeError('Vector dot product must be with Vector object.')
    def cross(self, vec: 'Vector') -> 'Vector':
        if isinstance(vec, Vector):
            x = self.y*vec.z - self.z*vec.y
            y = self.z*vec.x - self.x*vec.z
            z = self.x*vec.y - self.y*vec.x
            return Vector(x, y, z)
        else:
            raise TypeError('Vector cross product must be with Vector object.')
    def __abs__(self) -> float:
        return self.return_magnitude()
    def __mul__(self, obj: Any) -> Union[float, 'Vector']:
        if isinstance(obj, Vector):
            return self.dot(obj)
        else:
            x = self.x*obj
            y = self.y*obj
            z = self.z*obj
            return Vector(x, y, z)
    def __rmul__(self, obj: Any) -> Union[float, 'Vector']:
        if isinstance(obj, Vector):
            return obj.dot(self)
        else:
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
        if isinstance(obj, Vector):
            return self.cross(obj)
        else:
            x = self.x**obj
            y = self.y**obj
            z = self.z**obj
            return Vector(x, y, z)
    def __rpow__(self, obj: Any) -> 'Vector':
        return obj**self
    def __add__(self, obj: 'Vector') -> 'Vector':
        if isinstance(obj, Vector):
            x = self.x + obj.x
            y = self.y + obj.y
            z = self.z + obj.z
            return Vector(x, y, z)
        else:
            return obj.__add__(self)
    def __radd__(self, obj: Optional['Vector']=None) -> 'Vector':
        if obj is None:
            return self
        elif obj == 0:
            return self
        elif obj == Vector(0.0, 0.0, 0.0):
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj: 'Vector') -> 'Vector':
        if isinstance(obj, Vector):
            x = self.x - obj.x
            y = self.y - obj.y
            z = self.z - obj.z
            return Vector(x, y, z)
        else:
            err = 'Vector object can only be subtracted from Vector object.'
            raise TypeError(err)
    def __pos__(self) -> 'Vector':
        return self
    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y, -self.z)
    def __eq__(self, obj: Any) -> 'bool':
        if isinstance(obj, Vector):
            if obj.x == self.x and obj.y == self.y and obj.z == self.z:
                return True
        return False
    def __neq__(self, obj: Any) -> 'bool':
        if isinstance(obj, Vector):
            if obj.x != self.x or obj.y != self.y or obj.z != self.z:
                return True
        return False
    def __repr__(self) -> str:
        return '<Vector: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __str__(self) -> str:
        return '<{:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __format__(self, frm: str) -> str:
        frmstr = '<{:' + frm + '}, {:' + frm + '}, {:' + frm + '}>'
        return frmstr.format(self.x, self.y, self.z)

def zero_vector() -> 'Vector':
    return Vector(0.0, 0.0, 0.0)

def vector_from_points(pnta: 'Point', pntb: 'Point') -> 'Vector':
    """Create a Vector from two Points"""
    x = pntb.x - pnta.x
    y = pntb.y - pnta.y
    z = pntb.z - pnta.z
    return Vector(x, y, z)

def vector_isclose(a: Vector, b: Vector,
                   rel_tol: float=1e-09, abs_tol: float=0.0) -> bool:
    return isclose(a.x, b.x, rel_tol=rel_tol, abs_tol=abs_tol) and \
        isclose(a.y, b.y, rel_tol=rel_tol, abs_tol=abs_tol) and \
        isclose(a.z, b.z, rel_tol=rel_tol, abs_tol=abs_tol)
