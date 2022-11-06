from typing import Tuple, Union, List
from pygeom.geom2d import Tensor2D, Coordinate2D
from numpy.matlib import matrix, zeros, multiply, divide, arctan2, sqrt, square

class MatrixTensor2D():
    """MatrixTensor2D Class"""
    xx: 'matrix' = None
    xy: 'matrix' = None
    yx: 'matrix' = None
    yy: 'matrix' = None
    def __init__(self, xx: 'matrix', xy: 'matrix', yx: 'matrix', yy: 'matrix') -> None:
        self.xx = xx
        self.xy = xy
        self.yx = yx
        self.yy = yy
    # def return_angle(self) -> matrix:
    #     """Returns the angle matrix of this matrix tensor from the x axis"""
    #     return arctan2(self.y, self.x)
    def __getitem__(self, key) -> Union['Tensor2D', 'MatrixTensor2D']:
        xx = self.xx[key]
        xy = self.xy[key]
        yx = self.yx[key]
        yy = self.yy[key]
        if isinstance(xx, matrix) and isinstance(xy, matrix):
            output = MatrixTensor2D(xx, xy, yx, yy)
        else:
            output = Tensor2D(xx, xy, yx, yy)
        return output
    def __setitem__(self, key, value: Tensor2D) -> Union['Tensor2D', 'MatrixTensor2D']:
        if isinstance(key, tuple):
            self.xx[key] = value.xx
            self.xy[key] = value.xy
            self.yx[key] = value.yx
            self.yy[key] = value.yy
        else:
            raise IndexError()
    @property
    def shape(self) -> Tuple['int', ...]:
        return self.xx.shape
    def transpose(self) -> 'MatrixTensor2D':
        xx = self.xx.transpose()
        xy = self.xy.transpose()
        yx = self.yx.transpose()
        yy = self.yy.transpose()
        return MatrixTensor2D(xx, xy, yx, yy)
    def sum(self, axis=None, dtype=None, out=None) -> Union['Tensor2D', 'MatrixTensor2D']:
        xx = self.xx.sum(axis=axis, dtype=dtype, out=out)
        xy = self.xy.sum(axis=axis, dtype=dtype, out=out)
        yx = self.yx.sum(axis=axis, dtype=dtype, out=out)
        yy = self.yy.sum(axis=axis, dtype=dtype, out=out)
        if isinstance(xx, matrix) and isinstance(xy, matrix) and isinstance(yx, matrix) and isinstance(yy, matrix):
            return MatrixTensor2D(xx, xy, yx, yy)
        else:
            return Tensor2D(xx, xy, yx, yy)
    def repeat(self, repeats, axis=None) -> 'MatrixTensor2D':
        xx = self.xx.repeat(repeats, axis=axis)
        xy = self.xy.repeat(repeats, axis=axis)
        yx = self.yx.repeat(repeats, axis=axis)
        yy = self.yy.repeat(repeats, axis=axis)
        return MatrixTensor2D(xx, xy, yx, yy)
    def reshape(self, shape, order='C') -> 'MatrixTensor2D':
        xx = self.xx.reshape(shape, order=order)
        xy = self.xy.reshape(shape, order=order)
        yx = self.yx.reshape(shape, order=order)
        yy = self.yy.reshape(shape, order=order)
        return MatrixTensor2D(xx, xy, yx, yy)
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
    def copy(self, order='C') -> 'MatrixTensor2D':
        xx = self.xx.copy(order=order)
        xy = self.xy.copy(order=order)
        yx = self.yx.copy(order=order)
        yy = self.yy.copy(order=order)
        return MatrixTensor2D(xx, xy, yx, yy)
    def to_xy(self) -> Tuple[matrix]:
        """Returns the xx, xy, yx and yy values of this matrix tensor"""
        return self.xx, self.xy, self.yx, self.yy
    def __mul__(self, obj) -> 'MatrixTensor2D':
        xx = self.xx*obj
        xy = self.xy*obj
        yx = self.yx*obj
        yy = self.yy*obj
        return MatrixTensor2D(xx, xy, yx, yy)
    def __rmul__(self, obj) -> 'MatrixTensor2D':
        xx = obj*self.xx
        xy = obj*self.xy
        yx = obj*self.yx
        yy = obj*self.yy
        return MatrixTensor2D(xx, xy, yx, yy)
    def __truediv__(self, obj) -> 'MatrixTensor2D':
        if isinstance(obj, (int, float, complex)):
            xx = self.xx/obj
            xy = self.xy/obj
            yx = self.yx/obj
            yy = self.yy/obj
            return MatrixTensor2D(xx, xy, yx, yy)
        else:
            raise TypeError('Invalid type for MatrixTensor2D true division.')
    def __add__(self, obj) -> 'MatrixTensor2D':
        if isinstance(obj, (MatrixTensor2D, Tensor2D)):
            xx = self.xx + obj.xx
            xy = self.xy + obj.xy
            yx = self.yx + obj.yx
            yy = self.yy + obj.yy
            return MatrixTensor2D(xx, xy, yx, yy)
        else:
            raise TypeError('Invalid type for MatrixTensor2D addition.')
    def __radd__(self, obj) -> 'MatrixTensor2D':
        if obj is None:
            return self
        elif obj == 0:
            return self
        else:
            if isinstance(obj, (MatrixTensor2D, Tensor2D)):
                return self.__add__(obj)
            else:
                raise TypeError('Invalid type for MatrixTensor2D right addition.')
    def __sub__(self, obj) -> 'MatrixTensor2D':
        if isinstance(obj, (MatrixTensor2D, Tensor2D)):
            xx = self.xx - obj.xx
            xy = self.xy - obj.xy
            yx = self.yx - obj.yx
            yy = self.yy - obj.yy
            return MatrixTensor2D(xx, xy, yx, yy)
        else:
            raise TypeError('Invalid type for MatrixTensor2D subtraction.')
    def __pos__(self) -> 'MatrixTensor2D':
        return self
    def __neg__(self) -> 'MatrixTensor2D':
        return MatrixTensor2D(-self.xx, -self.xy, -self.yx, -self.yy)
    def __repr__(self) -> str:
        frmstr = '<MatrixTensor2D: {:}, {:}, {:}, {:}>'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)
    def __str__(self) -> str:
        frmstr = 'xx:\n{:}\nxy:\n{:}\nyx:\n{:}\nyy:\n{:}'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)
    def __format__(self, fs: str) -> str:
        frmstr = 'xx:\n{:'+fs+'}\nxy:\n{:'+fs+'}\nyx:\n{:'+fs+'}\nyy:\n{:'+fs+'}'
        return frmstr.format(self.xx, self.xy, self.yx, self.yy)

def zero_matrix_tensor(shape: tuple, dtype=float, order='C') -> 'MatrixTensor2D':
    xx = zeros(shape, dtype=dtype, order=order)
    xy = zeros(shape, dtype=dtype, order=order)
    yx = zeros(shape, dtype=dtype, order=order)
    yy = zeros(shape, dtype=dtype, order=order)
    return MatrixTensor2D(xx, xy, yx, yy)

def elementwise_multiply(a: MatrixTensor2D, b: 'matrix') -> 'MatrixTensor2D':
    if a.shape == b.shape:
        xx = multiply(a.xx, b)
        xy = multiply(a.xy, b)
        yx = multiply(a.yx, b)
        yy = multiply(a.yy, b)
        return MatrixTensor2D(xx, xy, yx, yy)
    else:
        raise ValueError('MatrixTensor2D and matrix shapes not the same.')

def elementwise_divide(a: MatrixTensor2D, b: 'matrix') -> 'MatrixTensor2D':
    if a.shape == b.shape:
        xx = divide(a.xx, b)
        xy = divide(a.xy, b)
        yx = divide(a.yx, b)
        yy = divide(a.yy, b)
        return MatrixTensor2D(xx, xy, yx, yy)
    else:
        raise ValueError('MatrixTensor2D and matrix shapes not the same.')

def tensor2d_to_global(crd: 'Coordinate2D', tens: 'MatrixTensor2D') -> 'MatrixTensor2D':
    """Transforms a tensor from this local coordinate system to global"""
    sxx, sxy, syx, syy = tens.to_xy()
    qxx, qxy, qyx, qyy = crd.dirx.x, crd.dirx.y, crd.diry.x, crd.diry.y
    exx = qxx**2*sxx + qxx*qyx*sxy + qxx*qyx*syx + qyx**2*syy
    eyy = qxy**2*sxx + qxy*qyy*sxy + qxy*qyy*syx + qyy**2*syy
    exy = qxx*qxy*sxx + qxx*qyy*sxy + qxy*qyx*syx + qyx*qyy*syy
    eyx = qxx*qxy*sxx + qxx*qyy*syx + qxy*qyx*sxy + qyx*qyy*syy
    return MatrixTensor2D(exx, exy, eyx, eyy)

def tensor2d_to_local(crd: 'Coordinate2D', tens: 'MatrixTensor2D') -> 'MatrixTensor2D':
    """Transforms a tensor from global to this local coordinate system"""
    sxx, sxy, syx, syy = tens.to_xy()
    qxx, qxy, qyx, qyy = crd.dirx.x, crd.dirx.y, crd.diry.x, crd.diry.y
    exx = qxx**2*sxx + qxx*qxy*sxy + qxx*qxy*syx + qxy**2*syy
    eyy = qyx**2*sxx + qyx*qyy*sxy + qyx*qyy*syx + qyy**2*syy
    exy = qxx*qyx*sxx + qxx*qyy*sxy + qxy*qyx*syx + qxy*qyy*syy
    eyx = qxx*qyx*sxx + qxx*qyy*syx + qxy*qyx*sxy + qxy*qyy*syy
    return MatrixTensor2D(exx, exy, eyx, eyy)
