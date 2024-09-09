from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Tuple, Union

from numpy import (allclose, copy, divide, hsplit, hstack, isclose,
                   logical_and, logical_or, ndim, ravel, repeat, reshape,
                   result_type, shape, size, split, sqrt, square, stack, sum,
                   transpose, zeros)
from numpy.linalg import solve

if TYPE_CHECKING:
    from numpy import bool_
    from numpy.typing import DTypeLike, NDArray

class Vector():
    """Vector Class"""
    x: 'NDArray' = None
    y: 'NDArray' = None
    z: 'NDArray' = None

    def __init__(self, x: 'NDArray', y: 'NDArray', z: 'NDArray') -> None:
        self.x = x
        self.y = y
        self.z = z

    def return_magnitude(self) -> 'NDArray':
        """Returns the magnitude array of this array vector"""
        x2 = square(self.x)
        y2 = square(self.y)
        z2 = square(self.z)
        r2 = x2 + y2 + z2
        return sqrt(r2)

    def to_unit(self, return_magnitude: bool = False) -> Union['Vector',
                                                               Tuple['Vector',
                                                               'NDArray']]:
        """Returns the unit vector of this vector"""
        mag = self.return_magnitude()
        x = zeros(shape(mag))
        y = zeros(shape(mag))
        z = zeros(shape(mag))
        magnot0 = mag != 0.0
        divide(self.x, mag, out=x, where=magnot0)
        divide(self.y, mag, out=y, where=magnot0)
        divide(self.z, mag, out=z, where=magnot0)
        if return_magnitude:
            return Vector(x, y, z), mag
        else:
            return Vector(x, y, z)

    def to_xyz(self) -> Tuple['NDArray', 'NDArray', 'NDArray']:
        """Returns the x, y and z values of this array vector"""
        return self.x, self.y, self.z

    def dot(self, vec: 'Vector') -> 'NDArray':
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
            return vec.cross(self)
        except AttributeError:
            err = 'Vector cross product must be with Vector object.'
            raise TypeError(err)

    def __abs__(self) -> 'NDArray':
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
        if self.ndim == 0:
            return f'<Vector: {self.x:}, {self.y:}, {self.z:}>'
        else:
            return f'<Vector shape: {self.shape:}, dtype: {self.dtype}>'

    def __str__(self) -> str:
        if self.ndim == 0:
            outstr = f'<{self.x:}, {self.y:}, {self.z:}>'
        else:
            outstr = f'Vector shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'x:\n{self.x:}\ny:\n{self.y:}\nz:\n{self.z:}\n'
            return outstr

    def __format__(self, frm: str) -> str:
        if self.ndim == 0:
            outstr = f'<{self.x:{frm}}, {self.y:{frm}}, {self.z:{frm}}>'
        else:
            outstr = f'Vector shape: {self.shape:}, dtype: {self.dtype}\n'
            frmstr = 'x:\n{:' + frm + '}\ny:\n{:' + frm + '}\nz:\n{:' + frm + '}\n'
            outstr += frmstr.format(self.x, self.y, self.z)
        return outstr

    def __matmul__(self, obj: 'NDArray') -> 'Vector':
        try:
            x = self.x@obj
            y = self.y@obj
            z = self.z@obj
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector object can only be matrix multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'NDArray') -> 'Vector':
        try:
            x = obj@self.x
            y = obj@self.y
            z = obj@self.z
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector object can only be matrix multiplied by a numpy ndarray.'
            raise TypeError(err)

    def rmatmul(self, mat: 'NDArray') -> 'Vector':
        """Returns the right matrix multiplication of this array vector"""
        return self.__rmatmul__(mat)

    def __getitem__(self, key) -> 'Vector':
        x = self.x[key]
        y = self.y[key]
        z = self.z[key]
        return Vector(x, y, z)

    def __setitem__(self, key, value: 'Vector') -> None:
        try:
            self.x[key] = value.x
            self.y[key] = value.y
            self.z[key] = value.z
        except IndexError:
            err = 'Vector index out of range.'
            raise IndexError(err)

    def stack_xyz(self) -> 'NDArray':
        return stack((self.x, self.y, self.z), axis=-1)

    @property
    def shape(self) -> Tuple[int, ...]:
        shape_x = shape(self.x)
        shape_y = shape(self.y)
        shape_z = shape(self.z)
        if shape_x == shape_y and shape_y == shape_z:
            return shape_x
        else:
            raise ValueError('Vector x, y and z should have the same shape.')

    @property
    def dtype(self) -> 'DTypeLike':
        return result_type(self.x, self.y, self.z)

    @property
    def ndim(self) -> int:
        ndim_x = ndim(self.x)
        ndim_y = ndim(self.y)
        ndim_z = ndim(self.z)
        if ndim_x == ndim_y and ndim_y == ndim_z:
            return ndim_x
        else:
            raise ValueError('Vector x, y and z should have the same ndim.')

    @property
    def size(self) -> int:
        size_x = size(self.x)
        size_y = size(self.y)
        size_z = size(self.z)
        if size_x == size_y and size_y == size_z:
            return size_x
        else:
            raise ValueError('Vector x, y and z should have the same size.')

    def transpose(self) -> 'Vector':
        x = transpose(self.x)
        y = transpose(self.y)
        z = transpose(self.z)
        return Vector(x, y, z)

    def sum(self, **kwargs: Dict[str, Any]) -> 'Vector':
        x = sum(self.x, **kwargs)
        y = sum(self.y, **kwargs)
        z = sum(self.z, **kwargs)
        return Vector(x, y, z)

    def repeat(self, repeats, axis=None) -> 'Vector':
        x = repeat(self.x, repeats, axis=axis)
        y = repeat(self.y, repeats, axis=axis)
        z = repeat(self.z, repeats, axis=axis)
        return Vector(x, y, z)

    def reshape(self, shape, order='C') -> 'Vector':
        x = reshape(self.x, shape, order=order)
        y = reshape(self.y, shape, order=order)
        z = reshape(self.z, shape, order=order)
        return Vector(x, y, z)

    def ravel(self, order='C') -> 'Vector':
        x = ravel(self.x, order=order)
        y = ravel(self.y, order=order)
        z = ravel(self.z, order=order)
        return Vector(x, y, z)

    def copy(self, order='C') -> 'Vector':
        x = copy(self.x, order=order)
        y = copy(self.y, order=order)
        z = copy(self.z, order=order)
        return Vector(x, y, z)

    def split(self, numsect: int,
              axis: Optional[int]=-1) -> Iterable['Vector']:
        xlst = split(self.x, numsect, axis=axis)
        ylst = split(self.y, numsect, axis=axis)
        zlst = split(self.z, numsect, axis=axis)
        for xi, yi, zi in zip(xlst, ylst, zlst):
            yield Vector(xi, yi, zi)

    def unpack(self) -> Iterable['Vector']:
        numsect = self.shape[-1]
        shape = self.shape[:-1]
        xlst = split(self.x, numsect, axis=-1)
        ylst = split(self.y, numsect, axis=-1)
        zlst = split(self.z, numsect, axis=-1)
        for xi, yi, zi in zip(xlst, ylst, zlst):
            yield Vector(xi, yi, zi).reshape(shape)

    def __next__(self) -> 'Vector':
        return Vector(next(self.x), next(self.y), next(self.z))

    def __eq__(self, obj: 'Vector') -> 'NDArray[bool_]':
        try:
            xeq = self.x == obj.x
            yeq = self.y == obj.y
            zeq = self.z == obj.z
            return logical_and(logical_and(xeq, yeq), zeq)
        except AttributeError:
            return False

    def __neq__(self, obj: 'Vector') -> 'NDArray[bool_]':
        try:
            xneq = self.x != obj.x
            yneq = self.y != obj.y
            zneq = self.z != obj.z
            return logical_or(logical_or(xneq, yneq), zneq)
        except AttributeError:
            return False


def zero_vector(shape: Optional[Tuple[int, ...]] = None,
                **kwargs: Dict[str, Any]) -> Vector:
    if shape is None:
        x, y, z = 0.0, 0.0, 0.0
    else:
        x = zeros(shape, **kwargs)
        y = zeros(shape, **kwargs)
        z = zeros(shape, **kwargs)
    return Vector(x, y, z)

def vector_isclose(a: Vector, b: Vector,
                   rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two vectors are close enough to be considered equal."""
    return isclose(a.x, b.x, rtol=rtol, atol=atol) and \
           isclose(a.y, b.y, rtol=rtol, atol=atol) and \
           isclose(a.z, b.z, rtol=rtol, atol=atol)

def solve_vector(a: 'NDArray', b: 'Vector') -> 'Vector':
    newb = hstack(b.to_xyz())
    newc = solve(a, newb)
    x, y, z = hsplit(newc, 3)
    return Vector(x, y, z)

def vector_allclose(a: Vector, b: Vector,
                    rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two Vectors are close enough to be considered equal."""
    return allclose(a.x, b.x, rtol=rtol, atol=atol) and \
           allclose(a.y, b.y, rtol=rtol, atol=atol) and \
           allclose(a.z, b.z, rtol=rtol, atol=atol)
