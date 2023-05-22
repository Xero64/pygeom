from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from numpy import isscalar, split, where, zeros

from ..geom2d.vector2d import Vector2D

if TYPE_CHECKING:
    from numpy import ndarray, number
    from numpy.typing import DTypeLike
    Vector2DLike = Union['Vector2D', 'ArrayVector2D']

class ArrayVector2D(Vector2D):
    """ArrayVector2D Class"""
    x: 'ndarray' = None
    y: 'ndarray' = None

    def __init__(self, x: 'ndarray', y: 'ndarray') -> None:
        self.x = x
        self.y = y

    def return_magnitude(self) -> 'ndarray':
        """Returns the magnitude array of this array vector"""
        return super().return_magnitude()

    def to_unit(self) -> 'ArrayVector2D':
        """Returns the unit array vector of this array vector"""
        mag = self.return_magnitude()
        x = where(mag == 0.0, 0.0, self.x/mag)
        y = where(mag == 0.0, 0.0, self.y/mag)
        return ArrayVector2D(x, y)

    def dot(self, vec: Vector2D) -> 'ndarray':
        try:
            return super().dot(vec)
        except AttributeError:
            err = 'ArrayVector2D dot product must be with Vector2D object.'
            raise TypeError(err)

    def cross(self, vec: Vector2D) -> 'ndarray':
        try:
            return super().cross(vec)
        except AttributeError:
            err = 'ArrayVector2D cross product must be with Vector2D object.'
            raise TypeError(err)

    def rmatmul(self, mat: 'ndarray') -> 'ArrayVector2D':
        """Returns the right matrix multiplication of this array vector"""
        vec = self.__rmatmul__(mat)
        vec.__class__ = ArrayVector2D
        return vec

    def to_xy(self) -> Tuple['ndarray', 'ndarray']:
        """Returns the x and y values of this array vector"""
        return super().to_xy()

    def __mul__(self, obj: Any) -> 'ArrayVector2D':
        vec = super().__mul__(obj)
        vec.__class__ = ArrayVector2D
        return vec

    def __rmul__(self, obj: Any) -> 'ArrayVector2D':
        vec = super().__rmul__(obj)
        vec.__class__ = ArrayVector2D
        return vec

    def __truediv__(self, obj: Any) -> 'ArrayVector2D':
        vec = super().__truediv__(obj)
        vec.__class__ = ArrayVector2D
        return vec

    def __pow__(self, obj: Any) -> 'ArrayVector2D':
        vec = super().__pow__(obj)
        vec.__class__ = ArrayVector2D
        return vec

    def __rpow__(self, obj: Any) -> 'ArrayVector2D':
        vec = super().__rpow__(obj)
        vec.__class__ = ArrayVector2D
        return vec

    def __add__(self, obj: Vector2D) -> 'ArrayVector2D':
        try:
            vec = super().__add__(obj)
            vec.__class__ = ArrayVector2D
            return vec
        except AttributeError:
            err = 'ArrayVector2D object can only be added to Vector2D object.'
            raise TypeError(err)

    def __sub__(self, obj) -> 'ArrayVector2D':
        try:
            vec = super().__sub__(obj)
            vec.__class__ = ArrayVector2D
            return vec
        except AttributeError:
            err = 'ArrayVector2D object can only be subtracted from Vector2D object.'
            raise TypeError(err)

    def __pos__(self) -> 'ArrayVector2D':
        vec = super().__pos__()
        vec.__class__ = ArrayVector2D
        return vec

    def __neg__(self) -> 'ArrayVector2D':
        vec = super().__neg__()
        vec.__class__ = ArrayVector2D
        return vec

    def __repr__(self) -> str:
        return '<ArrayVector2D: {:}, {:}>'.format(self.x, self.y)

    def __str__(self) -> str:
        return '\nx:\n{:}\ny:\n{:}'.format(self.x, self.y)

    def __format__(self, frm: str) -> str:
        frmstr = '\nx:\n{:' + frm + '}\ny:\n{:' + frm + '}'
        return frmstr.format(self.x, self.y)

    def __matmul__(self, obj: 'ndarray') -> 'ArrayVector2D':
        try:
            x = self.x@obj
            y = self.y@obj
            return scalar_arrayvector2d(x, y)
        except AttributeError:
            err = 'ArrayVector2D object can only be matrix multiplied by a ndarray.'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'ndarray') -> 'ArrayVector2D':
        try:
            x = obj@self.x
            y = obj@self.y
            return scalar_arrayvector2d(x, y)
        except AttributeError:
            err = 'ArrayVector2D object can only be matrix multiplied by a ndarray.'
            raise TypeError(err)

    def __getitem__(self, key) -> 'Vector2DLike':
        x = self.x[key]
        y = self.y[key]
        return scalar_arrayvector2d(x, y)

    def __setitem__(self, key, value: 'Vector2DLike') -> None:
        try:
            self.x[key] = value.x
            self.y[key] = value.y
        except IndexError:
            err = 'ArrayVector2D index out of range.'
            raise IndexError(err)

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.x.shape == self.y.shape:
            return self.x.shape
        else:
            raise ValueError('ArrayVector2D x and y should have the same shape.')

    @property
    def dtype(self) -> 'DTypeLike':
        if self.x.dtype is self.y.dtype:
            return self.x.dtype
        else:
            raise ValueError('ArrayVector2D x and y should have the same dtype.')

    @property
    def ndim(self) -> int:
        if self.x.ndim == self.y.ndim:
            return self.x.ndim
        else:
            raise ValueError('ArrayVector2D x and y should have the same ndim.')

    @property
    def size(self) -> int:
        if self.x.size == self.y.size:
            return self.x.size
        else:
            raise ValueError('ArrayVector2D x and y should have the same size.')

    def __abs__(self) -> 'ndarray':
        return self.return_magnitude()

    def transpose(self) -> 'ArrayVector2D':
        x = self.x.transpose()
        y = self.y.transpose()
        return ArrayVector2D(x, y)

    def sum(self, axis=None, dtype=None, out=None) -> 'Vector2D':
        x = self.x.sum(axis=axis, dtype=dtype, out=out)
        y = self.y.sum(axis=axis, dtype=dtype, out=out)
        return scalar_arrayvector2d(x, y)

    def repeat(self, repeats, axis=None) -> 'ArrayVector2D':
        x = self.x.repeat(repeats, axis=axis)
        y = self.y.repeat(repeats, axis=axis)
        return ArrayVector2D(x, y)

    def reshape(self, shape, order='C') -> 'ArrayVector2D':
        x = self.x.reshape(shape, order=order)
        y = self.y.reshape(shape, order=order)
        return ArrayVector2D(x, y)

    def to_tuple(self) -> Tuple[Vector2D, ...]:
        return [Vector2D(xi, yi) for xi, yi in zip(self.x, self.y)]

    def copy(self, order='C') -> 'ArrayVector2D':
        x = self.x.copy(order=order)
        y = self.y.copy(order=order)
        return ArrayVector2D(x, y)

    def split(self, numsect: int,
              axis: Optional[int]=-1) -> Tuple['ArrayVector2D', ...]:
        xlst = split(self.x, numsect, axis=axis)
        ylst = split(self.y, numsect, axis=axis)
        for xi, yi in zip(xlst, ylst):
            yield ArrayVector2D(xi, yi)

    def unpack(self) -> Tuple['ArrayVector2D', ...]:
        numsect = self.shape[-1]
        shape = self.shape[:-1]
        xlst = split(self.x, numsect, axis=-1)
        ylst = split(self.y, numsect, axis=-1)
        for xi, yi in zip(xlst, ylst):
            yield ArrayVector2D(xi, yi).reshape(shape)

    def return_angle(self) -> 'ndarray':
        """Returns the angle array of this array vector from the x axis"""
        return super().return_angle()

    def rotate(self, rot: 'number') -> 'ArrayVector2D':
        """Rotates this array vector by an input angle in radians"""
        vec = super().rotate(rot)
        vec.__class__ = ArrayVector2D
        return vec

    def to_complex(self) -> 'ndarray':
        """Returns the complex number of this array vector"""
        return super().to_complex()


def zero_arrayvector2d(shape: Tuple[int, ...], **kwargs) -> ArrayVector2D:
    '''Return a zero ArrayVector2D object with the given shape and dtype.'''
    x = zeros(shape, **kwargs)
    y = zeros(shape, **kwargs)
    return ArrayVector2D(x, y)

def scalar_arrayvector2d(x: 'ndarray', y: 'ndarray') -> Union[Vector2D, ArrayVector2D]:
    '''Return a scalar ArrayVector2D object with the given x and y.'''
    if isscalar(x) and isscalar(y):
        return Vector2D(x, y)
    else:
        return ArrayVector2D(x, y)
