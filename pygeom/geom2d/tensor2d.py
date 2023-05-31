from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from numpy import number

class Tensor2D():
    """Tensor2D Class"""
    xx: 'number' = None
    xy: 'number' = None
    yx: 'number' = None
    yy: 'number' = None

    def __init__(self, xx: 'number', xy: 'number',
                 yx: 'number', yy: 'number') -> None:
        self.xx = xx
        self.xy = xy
        self.yx = yx
        self.yy = yy

    def to_xy(self) -> Tuple['number', 'number', 'number', 'number']:
        """Returns the xx, xy, yx and yy values of this tensor."""
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
            err = 'Tensor2D object can only be subtracted from Tensor2D object.'
            raise TypeError(err)

    def __pos__(self) -> 'Tensor2D':
        return self

    def __neg__(self) -> 'Tensor2D':
        return Tensor2D(-self.xx, -self.xy, -self.yx, -self.yy)

    def __repr__(self) -> str:
        frmstr = '<Tensor2D: {:}, {:}, {:}, {:}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

    def __str__(self) -> str:
        frmstr = '<{:}, {:}, {:}, {:}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

    def __format__(self, frm: str) -> str:
        frmstr = '<{:' + frm + '}, {:' + frm + '}>, {:' + frm + '}>, {:' + frm + '}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)
