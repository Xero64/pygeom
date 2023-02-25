from typing import TYPE_CHECKING, Tuple, Union, Optional, Any

if TYPE_CHECKING:
    from numpy.matlib import matrix
    from .point import Point
    from ..matrix3d.matrixvector import MatrixVector
    VecType = Union['Vector', 'MatrixVector']
    OptVecType = Optional['VecType']

class Vector():
    """Vector Class"""
    x: 'float' = None
    y: 'float' = None
    z: 'float' = None
    def __init__(self, x: 'float', y: 'float', z: 'float') -> None:
        """Initialise vector object

        Args:
            x (float): x component of vector
            y (float): y component of vector
            z (float): z component of vector
        """
        self.x = x
        self.y = y
        self.z = z
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
    def return_magnitude(self) -> 'float':
        """Returns the magnitude of this vector"""
        return (self.x**2+self.y**2+self.z**2)**0.5
    def to_xyz(self) -> Tuple['float', 'float', 'float']:
        """Returns the x, y and z values of this vector"""
        return self.x, self.y, self.z
    def dot(self, vec: 'VecType') -> Union['float', 'matrix']:
        return self.x*vec.x+self.y*vec.y+self.z*vec.z
    def __mul__(self, obj: Any) -> 'VecType':
        from pygeom.matrix3d import MatrixVector
        if isinstance(obj, (Vector, MatrixVector)):
            return self.x*obj.x + self.y*obj.y + self.z*obj.z
        elif isinstance(obj, matrix):
            x = self.x*obj
            y = self.y*obj
            z = self.z*obj
            return MatrixVector(x, y, z)
        else:
            x = self.x*obj
            y = self.y*obj
            z = self.z*obj
            return Vector(x, y, z)
    def __rmul__(self, obj: Any) -> 'VecType':
        from numpy.matlib import matrix
        from pygeom.matrix3d import MatrixVector
        if isinstance(obj, (Vector, MatrixVector)):
            return self.x*obj.x+self.y*obj.y+self.z*obj.z
        elif isinstance(obj, matrix):
            x = obj*self.x
            y = obj*self.y
            z = obj*self.z
            return MatrixVector(x, y, z)
        else:
            x = obj*self.x
            y = obj*self.y
            z = obj*self.z
            return Vector(x, y, z)
    def __truediv__(self, obj: Any):
        if isinstance(obj, (int, float, complex)):
            x = self.x/obj
            y = self.y/obj
            z = self.z/obj
            return Vector(x, y, z)
    def cross(self, vec: 'VecType') -> 'VecType':
        if isinstance(vec, Vector):
            x = self.y*vec.z-self.z*vec.y
            y = self.z*vec.x-self.x*vec.z
            z = self.x*vec.y-self.y*vec.x
            return Vector(x, y, z)
        elif isinstance(vec, MatrixVector):
            x = self.y*vec.z-self.z*vec.y
            y = self.z*vec.x-self.x*vec.z
            z = self.x*vec.y-self.y*vec.x
            return MatrixVector(x, y, z)
    def __pow__(self, obj: Any) -> 'Vector':
        from pygeom.matrix3d import MatrixVector
        if isinstance(obj, Vector):
            x = self.y*obj.z-self.z*obj.y
            y = self.z*obj.x-self.x*obj.z
            z = self.x*obj.y-self.y*obj.x
            return Vector(x, y, z)
        elif isinstance(obj, MatrixVector):
            x = self.y*obj.z-self.z*obj.y
            y = self.z*obj.x-self.x*obj.z
            z = self.x*obj.y-self.y*obj.x
            return MatrixVector(x, y, z)
    def __add__(self, obj: 'VecType') -> 'VecType':
        from pygeom.matrix3d import MatrixVector
        if isinstance(obj, Vector):
            x = self.x+obj.x
            y = self.y+obj.y
            z = self.z+obj.z
            return Vector(x, y, z)
        elif isinstance(obj, MatrixVector):
            x = self.x+obj.x
            y = self.y+obj.y
            z = self.z+obj.z
            return MatrixVector(x, y, z)
        else:
            raise ValueError('Can only add Vector or MatrixVector from Vector.')
    def __radd__(self, obj: 'OptVecType'=None) -> 'VecType':
        if obj == 0 or obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj: 'VecType') -> 'VecType':
        from pygeom.matrix3d import MatrixVector
        if isinstance(obj, Vector):
            x = self.x-obj.x
            y = self.y-obj.y
            z = self.z-obj.z
            return Vector(x, y, z)
        elif isinstance(obj, MatrixVector):
            x = self.x-obj.x
            y = self.y-obj.y
            z = self.z-obj.z
            return MatrixVector(x, y, z)
        else:
            raise ValueError('Can only subtract Vector or MatrixVector from Vector.')
    def __pos__(self) -> 'Vector':
        return Vector(self.x, self.y, self.z)
    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y, -self.z)
    def __repr__(self) -> 'str':
        return '<Vector: {:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __str__(self) -> 'str':
        return '<{:}, {:}, {:}>'.format(self.x, self.y, self.z)
    def __format__(self, format_spec: 'str') -> 'str':
        frmstr = '<{:'+format_spec+'}, {:'+format_spec+'}, {:'+format_spec+'}>'
        return frmstr.format(self.x, self.y, self.z)

def zero_vector() -> 'Vector':
    return Vector(0.0, 0.0, 0.0)

def vector_from_points(pnta: 'Point', pntb: 'Point') -> 'Vector':
    """Create a Vector from two Points"""
    x = pntb.x-pnta.x
    y = pntb.y-pnta.y
    z = pntb.z-pnta.z
    return Vector(x, y, z)
