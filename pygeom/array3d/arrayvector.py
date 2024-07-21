from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple, Union

from numpy import divide, float64, isscalar, split, stack, zeros

from ..geom3d.vector import Vector

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray
    VectorLike = Union['Vector', 'ArrayVector']
    from ..array2d.arrayvector2d import ArrayVector2D

class ArrayVector(Vector):
    """ArrayVector Class"""
    x: 'NDArray[float64]' = None
    y: 'NDArray[float64]' = None
    z: 'NDArray[float64]' = None

    def __init__(self, x: 'NDArray[float64]', y: 'NDArray[float64]', z: 'NDArray[float64]') -> None:
        self.x = x
        self.y = y
        self.z = z
        if isscalar(x) and isscalar(y) and isscalar(z):
            self.__class__ = Vector

    def return_magnitude(self) -> 'NDArray[float64]':
        """Returns the magnitude array of this array vector"""
        return super().return_magnitude()

    def to_unit(self, return_magnitude: bool = False) -> Union['ArrayVector',
                                                               Tuple['ArrayVector',
                                                                     'NDArray[float64]']]:
        """Returns the unit arrayvector of this arrayvector"""
        mag = self.return_magnitude()
        x = zeros(mag.shape)
        y = zeros(mag.shape)
        z = zeros(mag.shape)
        magnot0 = mag != 0.0
        divide(self.x, mag, out=x, where=magnot0)
        divide(self.y, mag, out=y, where=magnot0)
        divide(self.z, mag, out=z, where=magnot0)
        if return_magnitude:
            return ArrayVector(x, y, z), mag
        else:
            return ArrayVector(x, y, z)

    def dot(self, vec: Vector) -> 'NDArray[float64]':
        try:
            return super().dot(vec)
        except AttributeError:
            err = 'ArrayVector dot product must be with Vector object.'
            raise TypeError(err)

    def cross(self, vec: Vector) -> 'ArrayVector':
        try:
            vec = super().cross(vec)
            vec.__class__ = ArrayVector
            return vec
        except AttributeError:
            err = 'ArrayVector cross product must be with Vector object.'
            raise TypeError(err)

    def rcross(self, vec: Vector) -> 'ArrayVector':
        try:
            vec = super().rcross(vec)
            vec.__class__ = ArrayVector
            return vec
        except AttributeError:
            err = 'ArrayVector cross product must be with Vector object.'
            raise TypeError(err)

    def rmatmul(self, mat: 'NDArray[float64]') -> 'ArrayVector':
        """Returns the right matrix multiplication of this array vector"""
        vec = self.__rmatmul__(mat)
        vec.__class__ = ArrayVector
        return vec

    def to_xyz(self) -> Tuple['NDArray[float64]', 'NDArray[float64]', 'NDArray[float64]']:
        """Returns the x, y and z values of this ndarray vector"""
        return super().to_xyz()

    def stack_xyz(self) -> 'NDArray[float64]':
        return stack((self.x, self.y, self.z), axis=-1)

    def __mul__(self, obj: Any) -> 'ArrayVector':
        vec = super().__mul__(obj)
        vec.__class__ = ArrayVector
        return vec

    def __rmul__(self, obj: Any) -> 'ArrayVector':
        vec = super().__rmul__(obj)
        vec.__class__ = ArrayVector
        return vec

    def __truediv__(self, obj: Any) -> 'ArrayVector':
        vec = super().__truediv__(obj)
        vec.__class__ = ArrayVector
        return vec

    def __pow__(self, obj: Any) -> 'ArrayVector':
        vec = super().__pow__(obj)
        vec.__class__ = ArrayVector
        return vec

    def __rpow__(self, obj: Any) -> 'ArrayVector':
        vec = super().__rpow__(obj)
        vec.__class__ = ArrayVector
        return vec

    def __add__(self, obj: Vector) -> 'ArrayVector':
        try:
            vec = super().__add__(obj)
            vec.__class__ = ArrayVector
            return vec
        except AttributeError:
            err = 'ArrayVector object can only be added to Vector object.'
            raise TypeError(err)

    def __sub__(self, obj) -> 'ArrayVector':
        try:
            vec = super().__sub__(obj)
            vec.__class__ = ArrayVector
            return vec
        except AttributeError:
            err = 'ArrayVector object can only be subtracted from Vector object.'
            raise TypeError(err)

    def __pos__(self) -> 'ArrayVector':
        vec = super().__pos__()
        vec.__class__ = ArrayVector
        return vec

    def __neg__(self) -> 'ArrayVector':
        vec = super().__neg__()
        vec.__class__ = ArrayVector
        return vec

    def __repr__(self) -> str:
        return f'<ArrayVector shape: {self.shape:}, dtype: {self.dtype}>'

    def __str__(self) -> str:
        outstr = f'ArrayVector shape: {self.shape:}, dtype: {self.dtype}\n'
        outstr += 'x:\n{:}\ny:\n{:}\nz:\n{:}\n'.format(self.x, self.y, self.z)
        return outstr

    def __format__(self, frm: str) -> str:
        outstr = f'ArrayVector shape: {self.shape:}, dtype: {self.dtype}\n'
        frmstr = 'x:\n{:' + frm + '}\ny:\n{:' + frm + '}\nz:\n{:' + frm + '}\n'
        outstr += frmstr.format(self.x, self.y, self.z)
        return outstr

    def __matmul__(self, obj: 'NDArray[float64]') -> 'ArrayVector':
        try:
            x = self.x@obj
            y = self.y@obj
            z = self.z@obj
            return ArrayVector(x, y, z)
        except AttributeError:
            err = 'ArrayVector object can only be matrix multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'NDArray[float64]') -> 'ArrayVector':
        try:
            x = obj@self.x
            y = obj@self.y
            z = obj@self.z
            return ArrayVector(x, y, z)
        except AttributeError:
            err = 'ArrayVector object can only be matrix multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __getitem__(self, key) -> 'VectorLike':
        x = self.x[key]
        y = self.y[key]
        z = self.z[key]
        return ArrayVector(x, y, z)

    def __setitem__(self, key, value: 'VectorLike') -> None:
        try:
            self.x[key] = value.x
            self.y[key] = value.y
            self.z[key] = value.z
        except IndexError:
            err = 'ArrayVector index out of range.'
            raise IndexError(err)

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.x.shape == self.y.shape and self.x.shape == self.z.shape:
            return self.x.shape
        else:
            raise ValueError('ArrayVector x, y and z should have the same shape.')

    @property
    def dtype(self) -> 'DTypeLike':
        if self.x.dtype is self.y.dtype and self.x.dtype is self.z.dtype:
            return self.x.dtype
        else:
            raise ValueError('ArrayVector x, y and z should have the same dtype.')

    @property
    def ndim(self) -> int:
        if self.x.ndim == self.y.ndim and self.x.ndim == self.z.ndim:
            return self.x.ndim
        else:
            raise ValueError('ArrayVector x, y and z should have the same ndim.')

    @property
    def size(self) -> int:
        if self.x.size == self.y.size and self.x.size == self.z.size:
            return self.x.size
        else:
            raise ValueError('ArrayVector x, y and z should have the same size.')

    def __abs__(self) -> 'NDArray[float64]':
        return self.return_magnitude()

    def transpose(self) -> 'ArrayVector':
        x = self.x.transpose()
        y = self.y.transpose()
        z = self.z.transpose()
        return ArrayVector(x, y, z)

    def sum(self, axis=None, dtype=None, out=None) -> 'ArrayVector':
        x = self.x.sum(axis=axis, dtype=dtype, out=out)
        y = self.y.sum(axis=axis, dtype=dtype, out=out)
        z = self.z.sum(axis=axis, dtype=dtype, out=out)
        return ArrayVector(x, y, z)

    def repeat(self, repeats, axis=None) -> 'ArrayVector':
        x = self.x.repeat(repeats, axis=axis)
        y = self.y.repeat(repeats, axis=axis)
        z = self.z.repeat(repeats, axis=axis)
        return ArrayVector(x, y, z)

    def reshape(self, shape, order='C') -> 'ArrayVector':
        x = self.x.reshape(shape, order=order)
        y = self.y.reshape(shape, order=order)
        z = self.z.reshape(shape, order=order)
        return ArrayVector(x, y, z)

    def flatten(self, order='C') -> 'ArrayVector':
        x = self.x.flatten(order=order)
        y = self.y.flatten(order=order)
        z = self.z.flatten(order=order)
        return ArrayVector(x, y, z)

    def to_tuple(self) -> Tuple[Vector, ...]:
        return [Vector(xi, yi, zi) for xi, yi, zi in zip(self.x, self.y, self.z)]

    def copy(self, order='C') -> 'ArrayVector':
        x = self.x.copy(order=order)
        y = self.y.copy(order=order)
        z = self.z.copy(order=order)
        return ArrayVector(x, y, z)

    def split(self, numsect: int,
              axis: Optional[int]=-1) -> Iterable['ArrayVector']:
        xlst = split(self.x, numsect, axis=axis)
        ylst = split(self.y, numsect, axis=axis)
        zlst = split(self.z, numsect, axis=axis)
        for xi, yi, zi in zip(xlst, ylst, zlst):
            yield ArrayVector(xi, yi, zi)

    def unpack(self) -> Iterable['ArrayVector']:
        numsect = self.shape[-1]
        shape = self.shape[:-1]
        xlst = split(self.x, numsect, axis=-1)
        ylst = split(self.y, numsect, axis=-1)
        zlst = split(self.z, numsect, axis=-1)
        for xi, yi, zi in zip(xlst, ylst, zlst):
            yield ArrayVector(xi, yi, zi).reshape(shape)

    def __next__(self) -> Vector:
        return Vector(next(self.x), next(self.y), next(self.z))


def zero_arrayvector(shape: Tuple[int, ...], **kwargs) -> ArrayVector:
    x = zeros(shape, **kwargs)
    y = zeros(shape, **kwargs)
    z = zeros(shape, **kwargs)
    return ArrayVector(x, y, z)

def arrayvector3d_from_2d(arr2d: 'ArrayVector2D') -> ArrayVector:
    x = arr2d.x
    y = arr2d.y
    z = zeros(arr2d.shape)
    return ArrayVector(x, y, z)
