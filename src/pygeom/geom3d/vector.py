from collections.abc import Iterable
from types import NotImplementedType
from typing import TYPE_CHECKING, Any

from numpy import (allclose, asarray, concatenate, copy, divide, full, hstack,
                   isclose, logical_and, logical_or, matmul, multiply, ndim,
                   ravel, repeat, reshape, result_type, shape, size, split,
                   sqrt, stack, sum, transpose, zeros)
from numpy.linalg import solve

if TYPE_CHECKING:
    from numpy import bool_, ufunc
    from numpy.typing import DTypeLike, NDArray

class Vector:
    """Vector Class"""
    x: 'NDArray'
    y: 'NDArray'
    z: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, x: 'NDArray', y: 'NDArray', z: 'NDArray') -> None:
        self.x = x
        self.y = y
        self.z = z

    def return_magnitude(self) -> 'NDArray':
        """Returns the magnitude array of this array vector"""
        return sqrt(self.dot(self))

    def to_unit(self, return_magnitude: bool = False) -> 'Vector | tuple[Vector, NDArray]':
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

    def to_xyz(self) -> tuple['NDArray', 'NDArray', 'NDArray']:
        """Returns the x, y and z values of this array vector"""
        return self.x, self.y, self.z

    def dot(self, vector: 'Vector') -> 'NDArray':
        try:
            return self.x*vector.x + self.y*vector.y + self.z*vector.z
        except AttributeError:
            err = 'Vector dot product must be with Vector object.'
            raise TypeError(err)

    def matdot(self, vector: 'Vector') -> 'NDArray':
        try:
            return self.x@vector.x + self.y@vector.y + self.z@vector.z
        except AttributeError:
            err = 'Vector matrix dot product must be with Vector object.'
            raise TypeError(err)

    def cross(self, vector: 'Vector') -> 'Vector':
        try:
            x = self.y*vector.z - self.z*vector.y
            y = self.z*vector.x - self.x*vector.z
            z = self.x*vector.y - self.y*vector.x
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector cross product must be with Vector object.'
            raise TypeError(err)

    def matcross(self, vector: 'Vector') -> 'Vector':
        try:
            x = self.y@vector.z - self.z@vector.y
            y = self.z@vector.x - self.x@vector.z
            z = self.x@vector.y - self.y@vector.x
            return Vector(x, y, z)
        except AttributeError:
            err = 'Vector matrix cross product must be with Vector object.'
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
            return f'<{self.__class__.__qualname__:s}: {self.x:}, {self.y:}, {self.z:}>'
        else:
            return f'<{self.__class__.__qualname__:s} shape: {self.shape:}, dtype: {self.dtype}>'

    def __str__(self) -> str:
        if self.ndim == 0:
            outstr = f'<{self.x:}, {self.y:}, {self.z:}>'
        else:
            outstr = f'{self.__class__.__qualname__:s} shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'x:\n{self.x:}\ny:\n{self.y:}\nz:\n{self.z:}\n'
        return outstr

    def __format__(self, frm: str) -> str:
        if self.ndim == 0:
            outstr = f'<{self.x:{frm}}, {self.y:{frm}}, {self.z:{frm}}>'
        else:
            outstr = f'{self.__class__.__qualname__:s} shape: {self.shape:}, dtype: {self.dtype}\n'
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
    def shape(self) -> tuple[int, ...]:
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

    def sum(self, **kwargs: dict[str, Any]) -> 'Vector':
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
              axis: int = -1) -> Iterable['Vector']:
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

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = (),
              **kwargs: dict[str, Any]) -> 'Vector':
        x = zeros(shape, **kwargs)
        y = zeros(shape, **kwargs)
        z = zeros(shape, **kwargs)
        return cls(x, y, z)

    @classmethod
    def from_iter(cls, vecs: Iterable['Vector'],
                  **kwargs: dict[str, Any]) -> 'Vector':
        num = len(vecs)
        vector = cls.zeros(num, **kwargs)
        for i, veci in enumerate(vecs):
            vector[i] = veci
        return vector

    @classmethod
    def from_obj(cls, obj: Any, **kwargs: dict[str, Any]) -> 'Vector':
        cur_ndim = 0
        cur_shape = ()
        x, y, z = 0.0, 0.0, 0.0
        if hasattr(obj, 'x'):
            x = getattr(obj, 'x')
            if ndim(x) > cur_ndim:
                cur_ndim = ndim(x)
                cur_shape = shape(x)
        if hasattr(obj, 'y'):
            y = getattr(obj, 'y')
            if ndim(y) > cur_ndim:
                cur_ndim = ndim(y)
                cur_shape = shape(y)
        if hasattr(obj, 'z'):
            z = getattr(obj, 'z')
            if ndim(z) > cur_ndim:
                cur_ndim = ndim(z)
                cur_shape = shape(z)
        vector = cls.zeros(cur_shape, **kwargs)
        if ndim(x) == 0:
            vector.x = full(cur_shape, x)
        else:
            vector.x = x
        if ndim(y) == 0:
            vector.y = full(cur_shape, y)
        else:
            vector.y = y
        if ndim(z) == 0:
            vector.z = full(cur_shape, z)
        else:
            vector.z = z
        return vector

    @classmethod
    def from_dict(cls, vector_dict: dict[str, 'NDArray']) -> 'Vector':
        x = vector_dict.get('x', None)
        y = vector_dict.get('y', None)
        z = vector_dict.get('z', None)
        if x is None or y is None or z is None:
            raise ValueError('Vector x, y and z must be provided.')
        return cls(x, y, z)

    @classmethod
    def from_iter_xyz(cls, x: 'Iterable', y: 'Iterable', z: 'Iterable') -> 'Vector':
        x = asarray(x)
        y = asarray(y)
        z = asarray(z)
        return cls(x, y, z)

    @classmethod
    def concatenate(cls, vecs: Iterable['Vector'], **kwargs: dict[str, Any]) -> 'Vector':
        x = concatenate([vec.x for vec in vecs], **kwargs)
        y = concatenate([vec.y for vec in vecs], **kwargs)
        z = concatenate([vec.z for vec in vecs], **kwargs)
        return cls(x, y, z)

    @classmethod
    def stack(cls, vecs: Iterable['Vector'], **kwargs: dict[str, Any]) -> 'Vector':
        x = stack([vec.x for vec in vecs], **kwargs)
        y = stack([vec.y for vec in vecs], **kwargs)
        z = stack([vec.z for vec in vecs], **kwargs)
        return cls(x, y, z)

    def is_close(self, obj: 'Vector',
                 rtol: float=1e-09, atol: float=0.0) -> 'NDArray[bool_]':
        xclose = isclose(self.x, obj.x, rtol=rtol, atol=atol)
        yclose = isclose(self.y, obj.y, rtol=rtol, atol=atol)
        zclose = isclose(self.z, obj.z, rtol=rtol, atol=atol)
        return logical_and(logical_and(xclose, yclose), zclose)

    def all_close(self, obj: 'Vector',
                  rtol: float=1e-09, atol: float=0.0) -> bool:
        xclose = allclose(self.x, obj.x, rtol=rtol, atol=atol)
        yclose = allclose(self.y, obj.y, rtol=rtol, atol=atol)
        zclose = allclose(self.z, obj.z, rtol=rtol, atol=atol)
        return xclose and yclose and zclose

    def solve(self, amat: 'NDArray') -> 'Vector':
        shp = self.shape
        if self.ndim == 0:
            bvec = self.reshape((1, 1))
        elif self.ndim == 1:
            bvec = self.reshape((self.size, 1))
        elif self.ndim == 2:
            bvec = self
        else:
            raise ValueError('Vector cannot be solved.')
        bmat = hstack(bvec.to_xyz())
        cmat = solve(amat, bmat)
        cvec = Vector(*split(cmat, 3, axis=1)).reshape(shp)
        return cvec

    def __array_ufunc__(self, ufunc: 'ufunc', method: str,
                        *inputs: tuple['NDArray | Vector', ...],
                        **kwargs: dict[str, Any]) -> 'Vector | NotImplementedType':
        if method == '__call__':
            if ufunc is multiply:
                if inputs[1] is self:
                    return self.__rmul__(inputs[0], **kwargs)
                else:
                    return NotImplemented
            elif ufunc is matmul:
                if inputs[1] is self:
                    return self.__rmatmul__(inputs[0], **kwargs)
                else:
                    return NotImplemented
        else:
            return NotImplemented
