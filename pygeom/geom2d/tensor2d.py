from typing import Any

class Tensor2D():
    """Tensor2D Class"""
    xx: float = None
    xy: float = None
    yx: float = None
    yy: float = None
    def __init__(self, xx: float, xy: float,
                 yx: float, yy: float) -> None:
        self.xx = xx
        self.xy = xy
        self.yx = yx
        self.yy = yy
    def to_xy(self):
        """Returns the xx, xy, yx and yy values of this tensor."""
        return self.xx, self.xy, self.yx, self.yy
    def __mul__(self, obj: Any):
        from numpy.matlib import matrix
        from pygeom.matrix2d.matrixtensor2d import MatrixTensor2D
        if isinstance(obj, matrix):
            xx = self.xx*obj
            xy = self.xy*obj
            yx = self.yx*obj
            yy = self.yy*obj
            return MatrixTensor2D(xx, xy, yx, yy)
        else:
            xx = self.xx*obj
            xy = self.xy*obj
            yx = self.yx*obj
            yy = self.yy*obj
            return Tensor2D(xx, xy, yx, yy)
    def __rmul__(self, obj: Any):
        from numpy.matlib import matrix
        from pygeom.matrix2d.matrixtensor2d import MatrixTensor2D
        if isinstance(obj, matrix):
            xx = obj*self.xx
            xy = obj*self.xy
            yx = obj*self.yx
            yy = obj*self.yy
            return MatrixTensor2D(xx, xy, yx, yy)
        else:
            xx = self.xx*obj
            xy = self.xy*obj
            yx = self.yx*obj
            yy = self.yy*obj
            return Tensor2D(xx, xy, yx, yy)
    def __truediv__(self, obj: Any):
        xx = self.xx/obj
        xy = self.xy/obj
        yx = self.yx/obj
        yy = self.yy/obj
        return Tensor2D(xx, xy, yx, yy)
    def __add__(self, obj: Any):
        from pygeom.matrix2d.matrixtensor2d import MatrixTensor2D
        if isinstance(obj, Tensor2D):
            xx = self.xx + obj.xx
            xy = self.xy + obj.xy
            yx = self.yx + obj.yx
            yy = self.yy + obj.yy
            return Tensor2D(xx, xy, yx, yy)
        elif isinstance(obj, MatrixTensor2D):
            xx = self.xx + obj.xx
            xy = self.xy + obj.xy
            yx = self.yx + obj.yx
            yy = self.yy + obj.yy
            return MatrixTensor2D(xx, xy, yx, yy)
    def __radd__(self, obj: Any):
        if obj == 0 or obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj: Any):
        from pygeom.matrix2d.matrixtensor2d import MatrixTensor2D
        if isinstance(obj, Tensor2D):
            xx = self.xx - obj.xx
            xy = self.xy - obj.xy
            yx = self.yx - obj.yx
            yy = self.yy - obj.yy
            return Tensor2D(xx, xy, yx, yy)
        elif isinstance(obj, MatrixTensor2D):
            xx = self.xx - obj.xx
            xy = self.xy - obj.xy
            return MatrixTensor2D(xx, xy, yx, yy)
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
