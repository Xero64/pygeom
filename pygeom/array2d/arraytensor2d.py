from typing import TYPE_CHECKING, List, Tuple, Union, Any

from numpy import zeros, isscalar

from ..geom2d.tensor2d import Tensor2D

if TYPE_CHECKING:
    from numpy import ndarray
    from numpy.typing import DTypeLike
    Tensor2DLike = Union['Tensor2D', 'ArrayTensor2D']

class ArrayTensor2D(Tensor2D):
    """ArrayTensor2D Class"""
    xx: 'ndarray' = None
    xy: 'ndarray' = None
    yx: 'ndarray' = None
    yy: 'ndarray' = None

    def __init__(self, xx: 'ndarray', xy: 'ndarray',
                 yx: 'ndarray', yy: 'ndarray') -> None:
        self.xx = xx
        self.xy = xy
        self.yx = yx
        self.yy = yy

    def to_xy(self) -> Tuple['ndarray', 'ndarray', 'ndarray', 'ndarray']:
        """Returns the xx, xy, yx and yy values of this ndarray tensor"""
        return self.xx, self.xy, self.yx, self.yy

    def __mul__(self, obj) -> 'ArrayTensor2D':
        xx = self.xx*obj
        xy = self.xy*obj
        yx = self.yx*obj
        yy = self.yy*obj
        return ArrayTensor2D(xx, xy, yx, yy)

    def __rmul__(self, obj) -> 'ArrayTensor2D':
        xx = obj*self.xx
        xy = obj*self.xy
        yx = obj*self.yx
        yy = obj*self.yy
        return ArrayTensor2D(xx, xy, yx, yy)

    def __truediv__(self, obj) -> 'ArrayTensor2D':
        if isinstance(obj, (int, float, complex)):
            xx = self.xx/obj
            xy = self.xy/obj
            yx = self.yx/obj
            yy = self.yy/obj
            return ArrayTensor2D(xx, xy, yx, yy)
        else:
            raise TypeError('Invalid type for ArrayTensor2D true division.')

    def __pow__(self, obj: Any) -> 'ArrayTensor2D':
        vec = super().__pow__(obj)
        vec.__class__ = ArrayTensor2D
        return vec

    def __rpow__(self, obj: Any) -> 'ArrayTensor2D':
        vec = super().__rpow__(obj)
        vec.__class__ = ArrayTensor2D
        return vec

    def __add__(self, obj) -> 'ArrayTensor2D':
        try:
            ten = super().__add__(obj)
            ten.__class__ = ArrayTensor2D
        except AttributeError:
            err = 'ArrayTensor2D object can only be added to Tensor2D object.'
            raise TypeError(err)

    def __sub__(self, obj) -> 'ArrayTensor2D':
        try:
            ten = super().__sub__(obj)
            ten.__class__ = ArrayTensor2D
            return ten
        except AttributeError:
            err = 'ArrayVector2D object can only be subtracted from Vector2D object.'
            raise TypeError(err)

    def __pos__(self) -> 'ArrayTensor2D':
        return self

    def __neg__(self) -> 'ArrayTensor2D':
        return ArrayTensor2D(-self.xx, -self.xy, -self.yx, -self.yy)

    def __repr__(self) -> str:
        frmstr = '<ArrayTensor2D: {:}, {:}, {:}, {:}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

    def __str__(self) -> str:
        frmstr = 'xx:\n{:}\nxy:\n{:}\nyx:\n{:}\nyy:\n{:}\n'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

    def __format__(self, fs: str) -> str:
        frmstr = 'xx:\n{:'+fs+'}\nxy:\n{:'+fs+'}\nyx:\n{:'+fs+'}\nyy:\n{:'+fs+'}\n'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

    def __matmul__(self, obj: 'ndarray') -> 'ArrayTensor2D':
        try:
            xx = self.xx@obj
            xy = self.xy@obj
            yx = self.yx@obj
            yy = self.yy@obj
            return ArrayTensor2D(xx, xy, yx, yy)
        except AttributeError:
            err = 'ArrayTensor2D object can only be multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'ndarray') -> 'ArrayTensor2D':
        try:
            xx = obj@self.xx
            xy = obj@self.xy
            yx = obj@self.yx
            yy = obj@self.yy
            return ArrayTensor2D(xx, xy, yx, yy)
        except AttributeError:
            err = 'ArrayTensor2D object can only be multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __getitem__(self, key) -> 'Tensor2DLike':
        xx = self.xx[key]
        xy = self.xy[key]
        yx = self.yx[key]
        yy = self.yy[key]
        if isscalar(xx) and isscalar(xy) and isscalar(yx) and isscalar(yy):
            return Tensor2D(xx, xy, yx, yy)
        else:
            return ArrayTensor2D(xx, xy, yx, yy)

    def __setitem__(self, key, value: 'Tensor2DLike') -> None:
        try:
            self.xx[key] = value.xx
            self.xy[key] = value.xy
            self.yx[key] = value.yx
            self.yy[key] = value.yy
        except AttributeError:
            err = 'ArrayTensor2D index out of range.'
            raise IndexError(err)

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.xx.shape == self.xy.shape and \
            self.xy.shape == self.yx.shape and \
                self.yx.shape == self.yy.shape:
            return self.xx.shape
        else:
            err = 'ArrayTensor2D xx, xy, yz and yy should have the same shape.'
            raise ValueError(err)

    @property
    def dtype(self) -> 'DTypeLike':
        if self.xx.dtype is self.xy.dtype and \
            self.xy.dtype is self.yx.dtype and \
                self.yx.dtype is self.yy.dtype:
            return self.xx.dtype
        else:
            err = 'ArrayTensor2D xx, xy, yz and yy should have the same dtype.'
            raise ValueError(err)

    def transpose(self) -> 'ArrayTensor2D':
        xx = self.xx.transpose()
        xy = self.xy.transpose()
        yx = self.yx.transpose()
        yy = self.yy.transpose()
        return ArrayTensor2D(xx, xy, yx, yy)

    def sum(self, axis=None, dtype=None, out=None) -> Union['Tensor2D',
                                                            'ArrayTensor2D']:
        xx = self.xx.sum(axis=axis, dtype=dtype, out=out)
        xy = self.xy.sum(axis=axis, dtype=dtype, out=out)
        yx = self.yx.sum(axis=axis, dtype=dtype, out=out)
        yy = self.yy.sum(axis=axis, dtype=dtype, out=out)
        if isscalar(xx) and isscalar(xy) and isscalar(yx) and isscalar(yy):
            return Tensor2D(xx, xy, yx, yy)
        else:
            return ArrayTensor2D(xx, xy, yx, yy)

    def repeat(self, repeats, axis=None) -> 'ArrayTensor2D':
        xx = self.xx.repeat(repeats, axis=axis)
        xy = self.xy.repeat(repeats, axis=axis)
        yx = self.yx.repeat(repeats, axis=axis)
        yy = self.yy.repeat(repeats, axis=axis)
        return ArrayTensor2D(xx, xy, yx, yy)

    def reshape(self, shape, order='C') -> 'ArrayTensor2D':
        xx = self.xx.reshape(shape, order=order)
        xy = self.xy.reshape(shape, order=order)
        yx = self.yx.reshape(shape, order=order)
        yy = self.yy.reshape(shape, order=order)
        return ArrayTensor2D(xx, xy, yx, yy)

    def tolist(self) -> List[List[Tensor2D]]:
        lst = []
        for i in range(self.shape[0]):
            lstj = []
            for j in range(self.shape[1]):
                xx = self.xx[i, j]
                xy = self.xy[i, j]
                yx = self.yx[i, j]
                yy = self.yy[i, j]
                lstj.append(Tensor2D(xx, xy, yx, yy))
            lst.append(lstj)
        return lst

    def copy(self, order='C') -> 'ArrayTensor2D':
        xx = self.xx.copy(order=order)
        xy = self.xy.copy(order=order)
        yx = self.yx.copy(order=order)
        yy = self.yy.copy(order=order)
        return ArrayTensor2D(xx, xy, yx, yy)

def zero_arraytensor2d(shape: Tuple[int, ...], **kwargs) -> 'ArrayTensor2D':
    '''Create a zero ArrayTensor2D object.'''
    xx = zeros(shape, **kwargs)
    xy = zeros(shape, **kwargs)
    yx = zeros(shape, **kwargs)
    yy = zeros(shape, **kwargs)
    return ArrayTensor2D(xx, xy, yx, yy)
