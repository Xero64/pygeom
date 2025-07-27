from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from numpy import (allclose, bool_, copy, cos, full, isclose, logical_and,
                   logical_or, ndim, ravel, repeat, reshape, result_type,
                   shape, sin, size, split, sum, transpose, zeros)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


class Tensor2D():
    """Tensor2D Class"""
    xx: 'NDArray' = None
    xy: 'NDArray' = None
    yx: 'NDArray' = None
    yy: 'NDArray' = None

    def __init__(self, xx: 'NDArray', xy: 'NDArray',
                 yx: 'NDArray', yy: 'NDArray') -> None:
        self.xx = xx
        self.xy = xy
        self.yx = yx
        self.yy = yy

    def to_xy(self) -> tuple['NDArray', 'NDArray', 'NDArray', 'NDArray']:
        """Returns the xx, xy, yx and yy values of this ndarray tensor"""
        return self.xx, self.xy, self.yx, self.yy

    def __mul__(self, obj: Any) -> 'Tensor2D':
        xx = self.xx*obj
        xy = self.xy*obj
        yx = self.yx*obj
        yy = self.yy*obj
        return Tensor2D(xx, xy, yx, yy)

    def __rmul__(self, obj: Any) -> 'Tensor2D':
        xx = obj*self.xx
        xy = obj*self.xy
        yx = obj*self.yx
        yy = obj*self.yy
        return Tensor2D(xx, xy, yx, yy)

    def __truediv__(self, obj: Any) -> 'Tensor2D':
        xx = self.xx/obj
        xy = self.xy/obj
        yx = self.yx/obj
        yy = self.yy/obj
        return Tensor2D(xx, xy, yx, yy)

    def __pow__(self, obj: Any) -> 'Tensor2D':
        xx = self.xx**obj
        xy = self.xy**obj
        yx = self.yx**obj
        yy = self.yy**obj
        return Tensor2D(xx, xy, yx, yy)

    def __rpow__(self, obj: Any) -> 'Tensor2D':
        xx = obj**self.xx
        xy = obj**self.xy
        yx = obj**self.yx
        yy = obj**self.yy
        return Tensor2D(xx, xy, yx, yy)

    def __add__(self, obj: 'Tensor2D') -> 'Tensor2D':
        try:
            xx = self.xx + obj.xx
            xy = self.xy + obj.xy
            yx = self.yx + obj.yx
            yy = self.yy + obj.yy
            return Tensor2D(xx, xy, yx, yy)
        except AttributeError:
            err = 'Tensor2D object can only be added to Tensor2D object.'
            raise TypeError(err)

    def __sub__(self, obj: 'Tensor2D') -> 'Tensor2D':
        try:
            xx = self.xx - obj.xx
            xy = self.xy - obj.xy
            yx = self.yx - obj.yx
            yy = self.yy - obj.yy
            return Tensor2D(xx, xy, yx, yy)
        except AttributeError:
            err = 'Vector2D object can only be subtracted from Vector2D object.'
            raise TypeError(err)

    def __pos__(self) -> 'Tensor2D':
        return self

    def __neg__(self) -> 'Tensor2D':
        return Tensor2D(-self.xx, -self.xy, -self.yx, -self.yy)

    def __repr__(self) -> str:
        frmstr = '<Tensor2D: {:}, {:}, {:}, {:}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

    def __str__(self) -> str:
        if self.ndim == 0:
            outstr = f'[{self.xx:}, {self.xy:}, {self.yx:}, {self.yy:}]'
        else:
            outstr = f'Tensor2D shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'xx:\n{self.xx:}\nxy:\n{self.xy:}\nyx:\n{self.yx:}\nyy:\n{self.yy:}\n'
        return outstr

    def __format__(self, frm: str) -> str:
        if self.ndim == 0:
            outstr = f'[{self.xx:{frm}}, {self.xy:{frm}}, {self.yx:{frm}}, {self.yy:{frm}}]'
        else:
            outstr = f'Tensor2D shape: {self.shape:}, dtype: {self.dtype}\n'
            outstr += f'xx:\n{self.xx:{frm}}\nxy:\n{self.xy:{frm}}\nyx:\n{self.yx:{frm}}\nyy:\n{self.yy:{frm}}\n'
        return outstr

    def __matmul__(self, obj: 'NDArray') -> 'Tensor2D':
        try:
            xx = self.xx@obj
            xy = self.xy@obj
            yx = self.yx@obj
            yy = self.yy@obj
            return Tensor2D(xx, xy, yx, yy)
        except AttributeError:
            err = 'Tensor2D object can only be multiplied by a numpy ndarray.'
            raise TypeError(err)

    def __rmatmul__(self, obj: 'NDArray') -> 'Tensor2D':
        try:
            xx = obj@self.xx
            xy = obj@self.xy
            yx = obj@self.yx
            yy = obj@self.yy
            return Tensor2D(xx, xy, yx, yy)
        except AttributeError:
            err = 'Tensor2D object can only be multiplied by a numpy ndarray.'
            raise TypeError(err)

    def rmatmul(self, mat: 'NDArray') -> 'Tensor2D':
        """Returns the right matrix multiplication of this tensor."""
        return self.__rmatmul__(mat)

    def __getitem__(self, key) -> 'Tensor2D':
        xx = self.xx[key]
        xy = self.xy[key]
        yx = self.yx[key]
        yy = self.yy[key]
        return Tensor2D(xx, xy, yx, yy)

    def __setitem__(self, key, value: 'Tensor2D') -> None:
        try:
            self.xx[key] = value.xx
            self.xy[key] = value.xy
            self.yx[key] = value.yx
            self.yy[key] = value.yy
        except AttributeError:
            err = 'Tensor2D index out of range.'
            raise IndexError(err)

    @property
    def shape(self) -> tuple[int, ...]:
        shape_xx = shape(self.xx)
        shape_xy = shape(self.xy)
        shape_yx = shape(self.yx)
        shape_yy = shape(self.yy)
        if shape_xx == shape_xy and shape_xy == shape_yx and shape_yx == shape_yy:
            return shape_xx
        else:
            err = 'Tensor2D xx, xy, yz and yy should have the same shape.'
            raise ValueError(err)

    @property
    def dtype(self) -> 'DTypeLike':
        return result_type(self.xx, self.xy, self.yx, self.yy)

    @property
    def ndim(self) -> int:
        ndim_xx = ndim(self.xx)
        ndim_xy = ndim(self.xy)
        ndim_yx = ndim(self.yx)
        ndim_yy = ndim(self.yy)
        if ndim_xx == ndim_xy and ndim_xy == ndim_yx and ndim_yx == ndim_yy:
            return ndim_xx
        else:
            raise ValueError('Tensor2D xx, xy, yz and yy should have the same ndim.')

    @property
    def size(self) -> int:
        size_xx = size(self.xx)
        size_xy = size(self.xy)
        size_yx = size(self.yx)
        size_yy = size(self.yy)
        if size_xx == size_xy and size_xy == size_yx and size_yx == size_yy:
            return size_xx
        else:
            raise ValueError('Tensor2D xx, xy, yz and yy should have the same size.')

    def transpose(self, **kwargs: dict[str, Any]) -> 'Tensor2D':
        xx = transpose(self.xx, **kwargs)
        xy = transpose(self.xy, **kwargs)
        yx = transpose(self.yx, **kwargs)
        yy = transpose(self.yy, **kwargs)
        return Tensor2D(xx, xy, yx, yy)

    def sum(self, **kwargs: dict[str, Any]) -> 'Tensor2D':
        xx = sum(self.xx, **kwargs)
        xy = sum(self.xy, **kwargs)
        yx = sum(self.yx, **kwargs)
        yy = sum(self.yy, **kwargs)
        return Tensor2D(xx, xy, yx, yy)

    def repeat(self, repeats, axis=None) -> 'Tensor2D':
        xx = repeat(self.xx, repeats, axis=axis)
        xy = repeat(self.xy, repeats, axis=axis)
        yx = repeat(self.yx, repeats, axis=axis)
        yy = repeat(self.yy, repeats, axis=axis)
        return Tensor2D(xx, xy, yx, yy)

    def reshape(self, shape, order='C') -> 'Tensor2D':
        xx = reshape(self.xx, shape, order=order)
        xy = reshape(self.xy, shape, order=order)
        yx = reshape(self.yx, shape, order=order)
        yy = reshape(self.yy, shape, order=order)
        return Tensor2D(xx, xy, yx, yy)

    def ravel(self, order='C') -> 'Tensor2D':
        xx = ravel(self.xx, order=order)
        xy = ravel(self.xy, order=order)
        yx = ravel(self.yx, order=order)
        yy = ravel(self.yy, order=order)
        return Tensor2D(xx, xy, yx, yy)

    def copy(self, order='C') -> 'Tensor2D':
        xx = copy(self.xx, order=order)
        xy = copy(self.xy, order=order)
        yx = copy(self.yx, order=order)
        yy = copy(self.yy, order=order)
        return Tensor2D(xx, xy, yx, yy)

    def split(self, numsect: int,
              axis: int = -1) -> Iterable['Tensor2D']:
        xxlst = split(self.xx, numsect, axis=axis)
        xylst = split(self.xy, numsect, axis=axis)
        yxlst = split(self.yx, numsect, axis=axis)
        yylst = split(self.yy, numsect, axis=axis)
        for xxi, xyi, yxi, yyi in zip(xxlst, xylst, yxlst, yylst):
            yield Tensor2D(xxi, xyi, yxi, yyi)

    def unpack(self) -> Iterable['Tensor2D']:
        numsect = self.shape[-1]
        shape = self.shape[:-1]
        xxlst = split(self.xx, numsect, axis=-1)
        xylst = split(self.xy, numsect, axis=-1)
        yxlst = split(self.yx, numsect, axis=-1)
        yylst = split(self.yy, numsect, axis=-1)
        for xxi, xyi, yxi, yyi in zip(xxlst, xylst, yxlst, yylst):
            yield Tensor2D(xxi, xyi, yxi, yyi).reshape(shape)

    def __next__(self) -> 'Tensor2D':
        return Tensor2D(next(self.xx), next(self.xy), next(self.yx), next(self.yy))

    def __eq__(self, obj: 'Tensor2D') -> 'NDArray[bool_]':
        try:
            xxeq = self.xx == obj.xx
            xyeq = self.xy == obj.xy
            yxeq = self.yx == obj.yx
            yyeq = self.yy == obj.yy
            return logical_and(logical_and(logical_and(xxeq, xyeq), yxeq), yyeq)
        except AttributeError:
            return False

    def __neq__(self, obj: 'Tensor2D') -> 'NDArray[bool_]':
        try:
            xxneq = self.xx != obj.xx
            xyneq = self.xy != obj.xy
            yxneq = self.yx != obj.yx
            yyneq = self.yy != obj.yy
            return logical_or(logical_or(logical_or(xxneq, xyneq), yxneq), yyneq)
        except AttributeError:
            return False

    def rotate(self, rot: float) -> 'Tensor2D':
        """Rotates this Tensor2D by an input angle in radians"""
        txx, txy, tyx, tyy = self.to_xy()
        c = cos(rot)
        s = sin(rot)
        c2 = c**2
        s2 = s**2
        cs = c*s
        sxx = c2*txx - cs*(txy + tyx) + s2*tyy
        sxy = c2*txy + cs*(txx - tyy) - s2*tyx
        syx = c2*tyx + cs*(txx - tyy) - s2*txy
        syy = c2*tyy + cs*(txy + tyx) + s2*txx
        return Tensor2D(sxx, sxy, syx, syy)

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = (),
              **kwargs: dict[str, Any]) -> 'Tensor2D':
        xx = zeros(shape, **kwargs)
        xy = zeros(shape, **kwargs)
        yx = zeros(shape, **kwargs)
        yy = zeros(shape, **kwargs)
        return cls(xx, xy, yx, yy)

    @classmethod
    def fromiter(cls, tens: Iterable['Tensor2D'],
                 **kwargs: dict[str, Any]) -> 'Tensor2D':
        num = len(tens)
        vec = cls.zeros(num, **kwargs)
        for i, veci in enumerate(tens):
            vec[i] = veci
        return vec

    @classmethod
    def fromobj(cls, obj: Any, **kwargs: dict[str, Any]) -> 'Tensor2D':
        cur_ndim = 0
        cur_shape = ()
        xx, xy, yx, yy = 0.0, 0.0, 0.0, 0.0
        if hasattr(obj, 'xx'):
            xx = obj.__dict__['xx']
            if ndim(xx) > cur_ndim:
                cur_ndim = ndim(xx)
                cur_shape = shape(xx)
        if hasattr(obj, 'xy'):
            xy = obj.__dict__['xy']
            if ndim(xy) > cur_ndim:
                cur_ndim = ndim(xy)
                cur_shape = shape(xy)
        if hasattr(obj, 'yx'):
            yx = obj.__dict__['yx']
            if ndim(yx) > cur_ndim:
                cur_ndim = ndim(yx)
                cur_shape = shape(yx)
        if hasattr(obj, 'yy'):
            yy = obj.__dict__['yy']
            if ndim(yy) > cur_ndim:
                cur_ndim = ndim(yy)
                cur_shape = shape(yy)
        tensor = cls.zeros(cur_shape, **kwargs)
        if ndim(xx) == 0:
            tensor.xx = full(cur_shape, xx)
        else:
            tensor.xx = xx
        if ndim(xy) == 0:
            tensor.xy = full(cur_shape, xy)
        else:
            tensor.xy = xy
        if ndim(yx) == 0:
            tensor.yx = full(cur_shape, yx)
        else:
            tensor.yx = yx
        if ndim(yy) == 0:
            tensor.yy = full(cur_shape, yy)
        else:
            tensor.yy = yy
        return tensor


def tensor2d_isclose(a: Tensor2D, b: Tensor2D,
                     rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two Tensor2Ds are close enough to be considered equal."""
    return isclose(a.xx, b.xx, rtol=rtol, atol=atol) and \
           isclose(a.xy, b.xy, rtol=rtol, atol=atol) and \
           isclose(a.yx, b.yx, rtol=rtol, atol=atol) and \
           isclose(a.yy, b.yy, rtol=rtol, atol=atol)


def tensor2d_allclose(a: Tensor2D, b: Tensor2D,
                      rtol: float=1e-09, atol: float=0.0) -> bool:
    """Returns True if two Tensor2Ds are close enough to be considered equal."""
    return allclose(a.xx, b.xx, rtol=rtol, atol=atol) and \
           allclose(a.xy, b.xy, rtol=rtol, atol=atol) and \
           allclose(a.yx, b.yx, rtol=rtol, atol=atol) and \
           allclose(a.yy, b.yy, rtol=rtol, atol=atol)
