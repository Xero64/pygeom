from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple, Union

from numpy import divide, float64, isscalar, split, stack, zeros

from ..geom2d.vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
    Vector2DLike = Union['Vector2D', 'ArrayVector2D']

class ArrayVector2D(Vector2D):
    """ArrayVector2D Class"""
    x: 'NDArray[float64]' = None
    y: 'NDArray[float64]' = None

    def __init__(self, x: 'NDArray[float64]', y: 'NDArray[float64]') -> None:
        self.x = x
        self.y = y
        if isscalar(x) and isscalar(y):
            self.__class__ = Vector2D

    def return_magnitude(self) -> 'NDArray[float64]':
        """Returns the magnitude array of this array vector"""
        return super().return_magnitude()

    def to_unit(self, return_magnitude: bool = False) -> Union['ArrayVector2D',
                                                               Tuple['ArrayVector2D', 'NDArray[float64]']]:
        """Returns the unit array vector of this array vector"""
        mag = self.return_magnitude()
        x = zeros(mag.shape)
        y = zeros(mag.shape)
        magnot0 = mag != 0.0
        divide(self.x, mag, out=x, where=magnot0)
        divide(self.y, mag, out=y, where=magnot0)
        if return_magnitude:
            return ArrayVector2D(x, y), mag
        else:
            return ArrayVector2D(x, y)

    def dot(self, vec: Vector2D) -> 'NDArray[float64]':
        try:
            return super().dot(vec)
        except AttributeError:
            err = 'ArrayVector2D dot product must be with Vector2D object.'
            raise TypeError(err)

    def cross(self, vec: Vector2D) -> 'NDArray[float64]':
        try:
            return super().cross(vec)
        except AttributeError:
            err = 'ArrayVector2D cross product must be with Vector2D object.'
            raise TypeError(err)

    def rmatmul(self, mat: 'NDArray[float64]') -> 'ArrayVector2D':
        """Returns the right matrix multiplication of this array vector"""
        vec = self.__rmatmul__(mat)
        vec.__class__ = ArrayVector2D
        return vec

    def to_xy(self) -> Tuple['NDArray[float64]', 'NDArray[float64]']:
        """Returns the x and y values of this array vector"""
        return super().to_xy()

    def stack_xy(self) -> 'NDArray[float64]':
        return stack((self.x, self.y), axis=-1)

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
        return f'<ArrayVector2D shape: {self.shape:}, dtype: {self.dtype}>'

    def __str__(self) -> str:
        outstr = f'ArrayVector2D shape: {self.shape:}, dtype: {self.dtype}\n'
        outstr += 'x:\n{:}\ny:\n{:}\n'.format(self.x, self.y)
        return outstr

    def __format__(self, frm: str) -> str:
        outstr = f'ArrayVector2D shape: {self.shape:}, dtype: {self.dtype}\n'
        frmstr = 'x:\n{:' + frm + '}\ny:\n{:' + frm + '}\n'
        outstr += frmstr.format(self.x, self.y)
        return outstr

    def __matmul__(self, obj: 'NDArray[float64]') -> 'ArrayVector2D':
        try:
            x = self.x@obj
            y = self.y@obj
            return ArrayVector2D(x, y)
        except AttributeError:
            err = 'ArrayVector2D object can only be matrix multiplied by a NDArray[float64].'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'NDArray[float64]') -> 'ArrayVector2D':
        try:
            x = obj@self.x
            y = obj@self.y
            return ArrayVector2D(x, y)
        except AttributeError:
            err = 'ArrayVector2D object can only be matrix multiplied by a NDArray[float64].'
            raise TypeError(err)

    def __getitem__(self, key) -> 'ArrayVector2D':
        x = self.x[key]
        y = self.y[key]
        return ArrayVector2D(x, y)

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

    def __abs__(self) -> 'NDArray[float64]':
        return self.return_magnitude()

    def transpose(self) -> 'ArrayVector2D':
        x = self.x.transpose()
        y = self.y.transpose()
        return ArrayVector2D(x, y)

    def sum(self, axis=None, dtype=None, out=None) -> 'ArrayVector2D':
        x = self.x.sum(axis=axis, dtype=dtype, out=out)
        y = self.y.sum(axis=axis, dtype=dtype, out=out)
        return ArrayVector2D(x, y)

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
              axis: Optional[int]=-1) -> Iterable['ArrayVector2D']:
        xlst = split(self.x, numsect, axis=axis)
        ylst = split(self.y, numsect, axis=axis)
        for xi, yi in zip(xlst, ylst):
            yield ArrayVector2D(xi, yi)

    def unpack(self) -> Iterable['ArrayVector2D']:
        numsect = self.shape[-1]
        shape = self.shape[:-1]
        xlst = split(self.x, numsect, axis=-1)
        ylst = split(self.y, numsect, axis=-1)
        for xi, yi in zip(xlst, ylst):
            yield ArrayVector2D(xi, yi).reshape(shape)

    def return_angle(self) -> 'NDArray[float64]':
        """Returns the angle array of this array vector from the x axis"""
        return super().return_angle()

    def rotate(self, rot: float) -> 'ArrayVector2D':
        """Rotates this array vector by an input angle in radians"""
        vec = super().rotate(rot)
        vec.__class__ = ArrayVector2D
        return vec

    def rotate_90deg(self) -> 'Vector2D':
        """Rotates this vector by 90 degrees"""
        vec = super().rotate_90deg()
        vec.__class__ = ArrayVector2D
        return vec

    def to_complex(self) -> 'NDArray[float64]':
        """Returns the complex number of this array vector"""
        return super().to_complex()

    def __next__(self) -> Vector2D:
        return Vector2D(next(self.x), next(self.y))


def zero_arrayvector2d(shape: Tuple[int, ...], **kwargs) -> ArrayVector2D:
    '''Return a zero ArrayVector2D object with the given shape and dtype.'''
    x = zeros(shape, **kwargs)
    y = zeros(shape, **kwargs)
    return ArrayVector2D(x, y)
