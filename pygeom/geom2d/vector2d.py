from collections.abc import Iterable
from types import NotImplementedType
from typing import TYPE_CHECKING, Any

from numpy import (allclose, arctan2, asarray, concatenate, copy, cos, divide,
                   full, hstack, isclose, logical_and, logical_or, matmul,
                   multiply, ndim, ravel, repeat, reshape, result_type, shape,
                   sin, size, split, sqrt, stack, sum, transpose, zeros)
from numpy.linalg import solve

if TYPE_CHECKING:
    from numpy import bool_, ufunc
    from numpy.typing import DTypeLike, NDArray


class Vector2D:
    """Vector2D Class"""
    x: 'NDArray'
    y: 'NDArray'

    __slots__ = tuple(__annotations__)

    def __init__(self, x: 'NDArray', y: 'NDArray') -> None:
        self.x = x
        self.y = y

    def return_magnitude(self) -> 'NDArray':
        """Returns the magnitude of this vector"""
        return sqrt(self.dot(self))

    def to_unit(self, return_magnitude: bool = False) -> 'Vector2D | tuple[Vector2D, NDArray]':
        """Returns the unit vector of this vector"""
        mag = self.return_magnitude()
        x = zeros(shape(mag))
        y = zeros(shape(mag))
        magnot0 = mag != 0.0
        divide(self.x, mag, out=x, where=magnot0)
        divide(self.y, mag, out=y, where=magnot0)
        if return_magnitude:
            return Vector2D(x, y), mag
        else:
            return Vector2D(x, y)

    def to_xy(self) -> tuple['NDArray', 'NDArray']:
        """Returns the x and y values of this vector"""
        return self.x, self.y

    def dot(self, vector: 'Vector2D') -> 'NDArray':
        try:
            return self.x*vector.x + self.y*vector.y
        except AttributeError:
            err = 'Vector2D dot product must be with Vector2D object.'
            raise TypeError(err)

    def matdot(self, vector: 'Vector2D') -> 'NDArray':
        try:
            return self.x@vector.x + self.y@vector.y
        except AttributeError:
            err = 'Vector2D dot product must be with Vector2D object.'
            raise TypeError(err)

    def cross(self, vector: 'Vector2D') -> 'NDArray':
        try:
            return self.x*vector.y - self.y*vector.x
        except AttributeError:
            err = 'Vector2D cross product must be with Vector2D object.'
            raise TypeError(err)

    def matcross(self, vector: 'Vector2D') -> 'NDArray':
        try:
            return self.x@vector.y - self.y@vector.x
        except AttributeError:
            err = 'Vector2D cross product must be with Vector2D object.'
            raise TypeError(err)

    def __abs__(self) -> 'NDArray':
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
            err = f'{self.__class__.__qualname__:s} object can only be subtracted from Vector2D object.'
            raise TypeError(err)

    def __pos__(self) -> 'Vector2D':
        return self

    def __neg__(self) -> 'Vector2D':
        return Vector2D(-self.x, -self.y)

    def __repr__(self) -> str:
        if self.ndim == 0:
            return f'<{self.__class__.__qualname__:s}: {self.x:}, {self.y:}>'
        else:
            return f'<{self.__class__.__qualname__:s} shape: {self.shape:}, dtype: {self.dtype}>'

    def __str__(self) -> str:
        if self.ndim == 0:
            outstr = f'<{self.x:}, {self.y:}>'
        else:
            outstr = f'{self.__class__.__qualname__:s} shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'x:\n{self.x:}\ny:\n{self.y:}\n'
        return outstr

    def __format__(self, frm: str) -> str:
        if self.ndim == 0:
            outstr = f'<{self.x:{frm}}, {self.y:{frm}}>'
        else:
            outstr = f'{self.__class__.__qualname__:s} shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'x:\n{self.x:{frm}}\ny:\n{self.y:{frm}}\n'
        return outstr

    def __matmul__(self, obj: 'NDArray') -> 'Vector2D':
        try:
            x = self.x@obj
            y = self.y@obj
            return Vector2D(x, y)
        except AttributeError:
            err = 'Vector2D object can only be matrix multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'NDArray') -> 'Vector2D':
        try:
            x = obj@self.x
            y = obj@self.y
            return Vector2D(x, y)
        except AttributeError:
            err = 'Vector2D object can only be matrix multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __getitem__(self, key) -> 'Vector2D':
        x = self.x[key]
        y = self.y[key]
        return Vector2D(x, y)

    def __setitem__(self, key, value: 'Vector2D') -> None:
        try:
            self.x[key] = value.x
            self.y[key] = value.y
        except IndexError:
            err = 'Vector2D index out of range.'
            raise IndexError(err)

    def stack_xy(self) -> 'NDArray':
        return stack((self.x, self.y), axis=-1)

    @property
    def shape(self) -> tuple[int, ...]:
        shape_x = shape(self.x)
        shape_y = shape(self.y)
        if shape_x == shape_y:
            return shape_x
        else:
            raise ValueError('Vector2D x and y should have the same shape.')

    @property
    def dtype(self) -> 'DTypeLike':
        return result_type(self.x, self.y)

    @property
    def ndim(self) -> int:
        ndim_x = ndim(self.x)
        ndim_y = ndim(self.y)
        if ndim_x == ndim_y:
            return ndim_x
        else:
            raise ValueError('Vector2D x and y should have the same ndim.')

    @property
    def size(self) -> int:
        size_x = size(self.x)
        size_y = size(self.y)
        if size_x == size_y:
            return size_x
        else:
            raise ValueError('Vector2D x and y should have the same size.')

    def transpose(self, **kwargs: dict[str, Any]) -> 'Vector2D':
        x = transpose(self.x, **kwargs)
        y = transpose(self.y, **kwargs)
        return Vector2D(x, y)

    def sum(self, **kwargs: dict[str, Any]) -> 'Vector2D':
        x = sum(self.x, **kwargs)
        y = sum(self.y, **kwargs)
        return Vector2D(x, y)

    def repeat(self, repeats, axis=None) -> 'Vector2D':
        x = repeat(self.x, repeats, axis=axis)
        y = repeat(self.y, repeats, axis=axis)
        return Vector2D(x, y)

    def reshape(self, shape, order='C') -> 'Vector2D':
        x = reshape(self.x, shape, order=order)
        y = reshape(self.y, shape, order=order)
        return Vector2D(x, y)

    def ravel(self, order='C') -> 'Vector2D':
        x = ravel(self.x, order=order)
        y = ravel(self.y, order=order)
        return Vector2D(x, y)

    def copy(self, order='C') -> 'Vector2D':
        x = copy(self.x, order=order)
        y = copy(self.y, order=order)
        return Vector2D(x, y)

    def split(self, numsect: int, axis: int=-1) -> Iterable['Vector2D']:
        xlst = split(self.x, numsect, axis=axis)
        ylst = split(self.y, numsect, axis=axis)
        for xi, yi in zip(xlst, ylst):
            yield Vector2D(xi, yi)

    def unpack(self) -> Iterable['Vector2D']:
        numsect = self.shape[-1]
        shape = self.shape[:-1]
        xlst = split(self.x, numsect, axis=-1)
        ylst = split(self.y, numsect, axis=-1)
        for xi, yi in zip(xlst, ylst):
            yield Vector2D(xi, yi).reshape(shape)

    def __next__(self) -> 'Vector2D':
        return Vector2D(next(self.x), next(self.y))

    def __eq__(self, obj: 'Vector2D') -> 'NDArray[bool_]':
        try:
            xeq = self.x == obj.x
            yeq = self.y == obj.y
            return logical_and(xeq, yeq)
        except AttributeError:
            return False

    def __neq__(self, obj: 'Vector2D') -> 'NDArray[bool_]':
        try:
            xneq = self.x != obj.x
            yneq = self.y != obj.y
            return logical_or(xneq, yneq)
        except AttributeError:
            return False

    def return_angle(self) -> 'NDArray':
        """Returns the angle of this vector from the x axis"""
        return arctan2(self.y, self.x)

    def rotate(self, rot: float) -> 'Vector2D':
        """Rotates this Vector2D by an input angle in radians"""
        mag = self.return_magnitude()
        ang = self.return_angle()
        x = mag*cos(ang + rot)
        y = mag*sin(ang + rot)
        return Vector2D(x, y)

    def rotate_90deg(self) -> 'Vector2D':
        """Rotates this vector by 90 degrees"""
        x = -self.y
        y = self.x
        return Vector2D(x, y)

    def to_complex(self) -> 'NDArray':
        """Returns the complex number of this Vector2D"""
        return self.x + 1j*self.y

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = (),
              **kwargs: dict[str, Any]) -> 'Vector2D':
        x = zeros(shape, **kwargs)
        y = zeros(shape, **kwargs)
        return cls(x, y)

    @classmethod
    def from_iter(cls, vecs: Iterable['Vector2D'],
                 **kwargs: dict[str, Any]) -> 'Vector2D':
        num = len(vecs)
        vector = cls.zeros(num, **kwargs)
        for i, veci in enumerate(vecs):
            vector[i] = veci
        return vector

    @classmethod
    def from_complex(cls, cnums: 'NDArray') -> 'Vector2D':
        x = cnums.real
        y = cnums.imag
        return cls(x, y)

    @classmethod
    def from_polar(cls, mags: 'NDArray', angs: 'NDArray') -> 'Vector2D':
        x = mags*cos(angs)
        y = mags*sin(angs)
        return cls(x, y)

    @classmethod
    def from_obj(cls, obj: Any, **kwargs: dict[str, Any]) -> 'Vector2D':
        cur_ndim = 0
        cur_shape = ()
        x, y = 0.0, 0.0
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
        vector = cls.zeros(cur_shape, **kwargs)
        if ndim(x) == 0:
            vector.x = full(cur_shape, x)
        else:
            vector.x = x
        if ndim(y) == 0:
            vector.y = full(cur_shape, y)
        else:
            vector.y = y
        return vector

    @classmethod
    def from_dict(cls, vector2d_dict: dict[str, 'NDArray']) -> 'Vector2D':
        x = vector2d_dict.get('x', None)
        y = vector2d_dict.get('y', None)
        if x is None or y is None:
            raise ValueError('Vector2D x and y must be provided.')
        return cls(x, y)

    @classmethod
    def from_iter_xy(cls, x: 'Iterable', y: 'Iterable') -> 'Vector2D':
        x = asarray(x)
        y = asarray(y)
        return cls(x, y)

    @classmethod
    def concatenate(cls, vecs: Iterable['Vector2D'], **kwargs: dict[str, Any]) -> 'Vector2D':
        x = concatenate([vec.x for vec in vecs], **kwargs)
        y = concatenate([vec.y for vec in vecs], **kwargs)
        return cls(x, y)

    @classmethod
    def stack(cls, vecs: Iterable['Vector2D'], **kwargs: dict[str, Any]) -> 'Vector2D':
        x = stack([vec.x for vec in vecs], **kwargs)
        y = stack([vec.y for vec in vecs], **kwargs)
        return cls(x, y)

    def is_close(self, obj: 'Vector2D', rtol: float=1e-09, atol: float=0.0) -> 'NDArray[bool_]':
        xclose = isclose(self.x, obj.x, rtol=rtol, atol=atol)
        yclose = isclose(self.y, obj.y, rtol=rtol, atol=atol)
        return logical_and(xclose, yclose)

    def all_close(self, obj: 'Vector2D', rtol: float=1e-09, atol: float=0.0) -> bool:
        xclose = allclose(self.x, obj.x, rtol=rtol, atol=atol)
        yclose = allclose(self.y, obj.y, rtol=rtol, atol=atol)
        return xclose and yclose

    def solve(self, amat: 'NDArray') -> 'Vector2D':
        shp = self.shape
        if self.ndim == 0:
            bvec = self.reshape((1, 1))
        elif self.ndim == 1:
            bvec = self.reshape((self.size, 1))
        elif self.ndim == 2:
            bvec = self
        else:
            raise ValueError('Vector2D cannot be solved.')
        bmat = hstack(bvec.to_xy())
        cmat = solve(amat, bmat)
        cvec = Vector2D(*split(cmat, 2, axis=1)).reshape(shp)
        return cvec

    def __array_ufunc__(self, ufunc: 'ufunc', method: str,
                        *inputs: tuple['NDArray | Vector2D', ...],
                        **kwargs: dict[str, Any]) -> 'Vector2D | NotImplementedType':
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
