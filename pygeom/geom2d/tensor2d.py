from math import atan2, cos, sin

class Tensor2D():
    """Tensor2D Class"""
    xx = None
    xy = None
    yx = None
    yy = None
    def __init__(self, xx, xy, yx, yy) -> None:
        self.xx = xx
        self.xy = xy
        self.yx = yx
        self.yy = yy
    # def rotate(self, rot):
    #     """Rotates this tensor by an input angle in radians"""
    #     mag = self.return_magnitude()
    #     ang = self.return_angle()
    #     x = mag*cos(ang+rot)
    #     y = mag*sin(ang+rot)
    #     return Tensor2D(x, y)
    # def return_angle(self):
    #     """Returns the angle of this tensor from the x axis"""
    #     return atan2(self.y, self.x)
    # def to_complex(self):
    #     """Returns the complex number of this tensor"""
    #     cplx = self.x+1j*self.y
    #     return cplx
    # def return_magnitude(self):
    #     """Returns the magnitude of this tensor"""
    #     return (self.x**2+self.y**2)**0.5
    def to_xy(self):
        """Returns the xx, xy, yx and yy values of this tensor."""
        return self.xx, self.xy, self.yx, self.yy
    def __mul__(self, obj):
        from numpy.matlib import matrix
        from pygeom.matrix2d import MatrixTensor2D
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
    def __rmul__(self, obj):
        from numpy.matlib import matrix
        from pygeom.matrix2d import MatrixTensor2D
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
    def __truediv__(self, obj):
        xx = self.xx/obj
        xy = self.xy/obj
        yx = self.yx/obj
        yy = self.yy/obj
        return Tensor2D(xx, xy, yx, yy)
    # def __pow__(self, obj):
    #     from pygeom.matrix2d import MatrixTensor2D
    #     if isinstance(obj, (Tensor2D, MatrixTensor2D)):
    #         return self.x*obj.y-self.y*obj.x
    def __add__(self, obj):
        from pygeom.matrix2d import MatrixTensor2D
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
    def __radd__(self, obj):
        if obj == 0 or obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj):
        from pygeom.matrix2d import MatrixTensor2D
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
    def __pos__(self):
        return self
    def __neg__(self):
        return Tensor2D(-self.xx, -self.xy, -self.yx, -self.yy)
    def __repr__(self):
        frmstr = '<Tensor2D: {:}, {:}, {:}, {:}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)
    def __str__(self):
        frmstr = '<{:}, {:}, {:}, {:}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)
    def __format__(self, fs):
        frmstr = '<{:'+fs+'}, {:'+fs+'}>, {:'+fs+'}>, {:'+fs+'}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

def tensor2d_from_lists(xx, xy, yx, yy):
    """Create a list of Tensor2D objects"""
    n = len(xx)
    if len(xy) == n and len(yx) == n and len(yy) == n:
        tens = [Tensor2D(xx[i], xy[i], yx[i], yy[i]) for i in range(n)]
        return tens
