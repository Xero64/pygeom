from typing import Iterable

from numpy import ndarray, all, asarray, fromiter, unique, argsort
from numpy import zeros as zeros_array
from numpy.matlib import matrix, zeros
from numpy.linalg import solve

from ..tools.tridiag_solver import tridiag_solver

class QuadraticCentreInterpolationSolver():
    x: 'ndarray' = None
    _num: int = None
    _dx: 'ndarray' = None
    _xc: 'ndarray' = None
    _gmat: 'matrix' = None
    _emat: 'matrix' = None
    _fmat: 'matrix' = None
    _hmat: 'matrix' = None
    _smat: 'matrix' = None
    _tmat: 'matrix' = None
    _qmat: 'matrix' = None
    _kmat: 'matrix' = None
    _kmatc: 'matrix' = None
    _zmatop: 'matrix' = None
    _gmatop: 'matrix' = None
    _hmatop: 'matrix' = None
    _zmateq: 'matrix' = None
    _gmateq: 'matrix' = None
    _hmateq: 'matrix' = None
    _gmat00: 'matrix' = None
    _hmat00: 'matrix' = None
    # _zmatle: 'matrix' = None
    # _gmatle: 'matrix' = None
    # _hmatle: 'matrix' = None
    # _zmatv2: 'matrix' = None
    # _gmatv2: 'matrix' = None
    # _hmatv2: 'matrix' = None
    # _zmatv3: 'matrix' = None
    # _gmatv3: 'matrix' = None
    # _hmatv3: 'matrix' = None
    def __init__(self, x: Iterable[float]) -> None:
        for i in range(len(x)-1):
            if x[i+1] <= x[i]:
                raise ValueError('Input x list must be sorted and unique.')
        self.x = asarray(x, dtype=float)
    @property
    def num(self) -> int:
        if self._num is None:
            self._num = len(self.x)
        return self._num
    @property
    def dx(self) -> 'ndarray':
        if self._dx is None:
            self._dx = self.x[1:]-self.x[:-1]
        return self._dx
    @property
    def xc(self) -> 'ndarray':
        if self._xc is None:
            self._xc = (self.x[1:]+self.x[:-1])/2
        return self._xc
    @property
    def emat(self) -> 'matrix':
        if self._emat is None:
            self._emat = zeros((self.num, self.num), dtype=float)
            for i, dxi in enumerate(self.dx):
                self._emat[i, i] -= 3/dxi
                self._emat[i, i+1] -= 1/dxi
            self._emat[-1, -2] += 1/dxi
            self._emat[-1, -1] += 3/dxi
        return self._emat
    @property
    def fmat(self) -> 'matrix':
        if self._fmat is None:
            self._fmat = zeros((self.num, self.num-1), dtype=float)
            for i, dxi in enumerate(self.dx):
                self._fmat[i, i] += 4/dxi
            self._fmat[-1, -1] -= 4/dxi
        return self._fmat
    @property
    def smat(self) -> 'matrix':
        if self._smat is None:
            self._smat = zeros((self.num-1, self.num), dtype=float)
            for i, dxi in enumerate(self.dx):
                xbar2 = 1/dxi**2
                self._smat[i, i] = 4*xbar2
                self._smat[i, i+1] = 4*xbar2
        return self._smat
    @property
    def tmat(self) -> 'matrix':
        if self._tmat is None:
            self._tmat = zeros((self.num-1, self.num-1), dtype=float)
            for i, dxi in enumerate(self.dx):
                xbar2 = 1/dxi**2
                self._tmat[i, i] = -8*xbar2
        return self._tmat
    @property
    def gmat(self) -> 'matrix':
        # y = G*<yc, dya, dyb>
        if self._gmat is None:
            a = zeros_array(self.num-1, dtype=float)
            b = zeros_array(self.num, dtype=float)
            d = zeros((self.num, self.num+1), dtype=float)
            for i, dxi in enumerate(self.dx):
                dxri = 1/dxi
                dxrix3 = 3/dxi
                dxrix4 = 4/dxi
                b[i] -= dxrix3
                b[i+1] -= dxrix3
                a[i] -= dxri
                d[i, i] -= dxrix4
                d[i+1, i] -= dxrix4
            d[0, -2] = 1.0
            d[-1, -1] = -1.0
            self._gmat = tridiag_solver(a, b, a, d)
        return self._gmat
    @property
    def hmat(self) -> 'matrix':
        # dydx = H*yc = (E*G+F)*yc = E*y + F*yc
        if self._hmat is None:
            self._hmat = self.emat*self.gmat
            self._hmat[:, :-2] += self.fmat
        return self._hmat
    @property
    def qmat(self) -> 'matrix':
        # d2ydx2 = Q*yc = (S*G+T)*yc = S*y + T*yc
        if self._qmat is None:
            self._qmat = self.smat*self.gmat
            self._qmat[:, :-2] += self.tmat
        return self._qmat
    @property
    def zmatop(self) -> 'matrix':
        # <dya, dyb>op = Zop*yc
        if self._zmatop is None:
            gaa = self.gmat[0, -2]
            gba = self.gmat[0, -1]
            gca = self.gmat[0, :-2]

            gab = self.gmat[-1, -2]
            gbb = self.gmat[-1, -1]
            gcb = self.gmat[-1, :-2]

            amat = zeros((2, 2), dtype=float)
            bmat = zeros((2, self.num-1), dtype=float)

            amat[0, 0] = gaa + gab
            amat[0, 1] = gba + gbb
            bmat[0, :] = gca + gcb

            amat[1, 0] = 1.0
            amat[1, 1] = 1.0

            self._zmatop = -solve(amat, bmat)
        return self._zmatop
    @property
    def gmatop(self) -> 'matrix':
        # y = Gop*yc
        if self._gmatop is None:
            self._gmatop = self.gmat[:, :-2] + self.gmat[:, -2:]*self.zmatop
        return self._gmatop
    @property
    def hmatop(self) -> 'matrix':
        # dydx = Hop*yc
        if self._hmatop is None:
            self._hmatop = self.emat*self.gmatop + self.fmat
        return self._hmatop
    @property
    def zmateq(self) -> 'matrix':
        # <dya, dyb>op = Zeq*yc
        if self._zmateq is None:
            gaa = self.gmat[0, -2]
            gba = self.gmat[0, -1]
            gca = self.gmat[0, :-2]

            gab = self.gmat[-1, -2]
            gbb = self.gmat[-1, -1]
            gcb = self.gmat[-1, :-2]

            amat = zeros((2, 2), dtype=float)
            bmat = zeros((2, self.num-1), dtype=float)

            amat[0, 0] = gaa + gab
            amat[0, 1] = gba + gbb
            bmat[0, :] = gca + gcb

            amat[1, 0] = 1.0
            amat[1, 1] = -1.0

            self._zmateq = -solve(amat, bmat)
        return self._zmateq
    @property
    def gmateq(self) -> 'matrix':
        # y = Geq*yc
        if self._gmateq is None:
            self._gmateq = self.gmat[:, :-2] + self.gmat[:, -2:]*self.zmateq
        return self._gmateq
    @property
    def hmateq(self) -> 'matrix':
        # dydx = Heq*yc
        if self._hmateq is None:
            self._hmateq = self.emat*self.gmateq + self.fmat
        return self._hmateq
    @property
    def gmat00(self) -> 'matrix':
        # y = G00*yc
        if self._gmat00 is None:
            self._gmat00 = self.gmat[:, :-2]
        return self._gmat00
    @property
    def hmat00(self) -> 'matrix':
        # dydx = H00*yc
        if self._hmat00 is None:
            self._hmat00 = self.emat*self.gmat00 + self.fmat
        return self._hmat00
    # @property
    # def zmatle(self) -> 'matrix':
    #     if self._zmatle is None:
    #         amat = self.qmat[(0, -1), -2:]
    #         bmat = self.qmat[(0, -1), :-2]
    #         self._zmatle = -solve(amat, bmat)
    #     return self._zmatle
    # @property
    # def gmatle(self) -> 'matrix':
    #     if self._gmatle is None:
    #         self._gmatle = self.gmat[:, :-2] + self.gmat[:, -2:]*self.zmatle
    #     return self._gmatle
    # @property
    # def hmatle(self) -> 'matrix':
    #     if self._hmatle is None:
    #         self._hmatle = self.emat*self.gmatle + self.fmat
    #     return self._hmatle
    # @property
    # def zmatv2(self) -> 'matrix':
    #     if self._zmatv2 is None:
    #         amat = zeros((2, 2), dtype=float)
    #         bmat = zeros((2, self.num-1), dtype=float)
    #         amat[0, :] = self.qmat[0, -2:] - self.qmat[-1, -2:]
    #         amat[1, 0] = 1.0
    #         amat[1, 1] = -1.0
    #         bmat[0, :] = self.qmat[0, :-2] - self.qmat[-1, :-2]
    #         self._zmatv2 = -solve(amat, bmat)
    #     return self._zmatv2
    # @property
    # def gmatv2(self) -> 'matrix':
    #     if self._gmatv2 is None:
    #         self._gmatv2 = self.gmat[:, :-2] + self.gmat[:, -2:]*self.zmatv2
    #     return self._gmatv2
    # @property
    # def hmatv2(self) -> 'matrix':
    #     if self._hmatv2 is None:
    #         self._hmatv2 = self.emat*self.gmatv2 + self.fmat
    #     return self._hmatv2
    # @property
    # def zmatv3(self) -> 'matrix':
    #     if self._zmatv3 is None:
    #         amat = zeros((2, 2), dtype=float)
    #         bmat = zeros((2, self.num-1), dtype=float)
    #         amat[0, :] = self.qmat[0, -2:] - self.qmat[1, -2:]
    #         amat[1, :] = self.qmat[-2, -2:] - self.qmat[-1, -2:]
    #         bmat[0, :] = self.qmat[0, :-2] - self.qmat[1, :-2]
    #         bmat[1, :] = self.qmat[-2, :-2] - self.qmat[-1, :-2]
    #         self._zmatv3 = -solve(amat, bmat)
    #     return self._zmatv3
    # @property
    # def gmatv3(self) -> 'matrix':
    #     if self._gmatv3 is None:
    #         self._gmatv3 = self.gmat[:, :-2] + self.gmat[:, -2:]*self.zmatv3
    #     return self._gmatv3
    # @property
    # def hmatv3(self) -> 'matrix':
    #     if self._hmatv3 is None:
    #         self._hmatv3 = self.emat*self.gmatv3 + self.fmat
    #     return self._hmatv3
    # def imat(self) -> 'matrix':
    #     if self._imat is None:
    #         imat = zeros((self.num, self.num), dtype=float)
    #         for i, dxi in enumerate(self.dx):
    #             imat[i+1:, i] += dxi/6
    #             imat[i+1:, i+1] += dxi/6
    #         # self._imat = imat
    #         self._imat = zeros((self.num, self.num), dtype=float)
    #         iopp = imat[-1, :]/2
    #         for i in range(self.num):
    #             self._imat[i, :] = imat[i, :] - iopp
    #     return self._imat
    # @property
    # def jmat(self) -> 'matrix':
    #     if self._jmat is None:
    #         jmat = zeros((self.num, self.num-1), dtype=float)
    #         for i, dxi in enumerate(self.dx):
    #             jmat[i+1:, i] += 2*dxi/3
    #         # self._jmat = jmat
    #         self._jmat = zeros((self.num, self.num-1), dtype=float)
    #         jopp = jmat[-1, :]/2
    #         for i in range(self.num):
    #             self._jmat[i, :] = jmat[i, :] - jopp
    #     return self._jmat
    # @property
    # def kmat(self) -> 'matrix':
    #     if self._kmat is None:
    #         self._kmat = self.jmat*self.gmat
    #         self._kmat[:, :-1] += self.imat
    #     return self._kmat
    # @property
    # def kmateq(self) -> 'matrix':
    #     if self._kmateq is None:
    #         self._kmateq = self.imat + self.jmat*self.gmateq
    #     return self._kmateq
    # @property
    # def kmatop(self) -> 'matrix':
    #     if self._kmatop is None:
    #         self._kmatop = self.imat + self.jmat*self.gmatop
    #     return self._kmatop
    @property
    def kmat(self) -> 'matrix':
        if self._kmat is None:
            kmat = zeros((self.num, self.num), dtype=float)
            for i, dxi in enumerate(self.dx):
                kmat[i+1:, i] += dxi/2
                kmat[i+1:, i+1] += dxi/2
                # print(kmat)
            # self._kmat = kmat
            self._kmat = zeros((self.num, self.num), dtype=float)
            kopp = kmat[-1, :]/2
            for i in range(self.num):
                self._kmat[i, :] = kmat[i, :] - kopp
        return self._kmat
    @property
    def kmatc(self) -> 'matrix':
        if self._kmatc is None:
            self._kmatc = self.kmat[:-1, :].copy()
            for i, dxi in enumerate(self.dx):
                self._kmatc[i, i] += 3*dxi/8
                self._kmatc[i, i+1] += dxi/8
        return self._kmatc

class QuadraticCentreInterpolation(QuadraticCentreInterpolationSolver):
    yc: 'ndarray' = None
    dya: float = None
    dya: float = None
    _ycmat: 'matrix' = None
    _rmat: 'matrix' = None
    _ymat: 'matrix' = None
    _y: 'ndarray' = None
    _dymat: 'matrix' = None
    _dydx: 'ndarray' = None
    _iymat: 'matrix' = None
    _iydx: 'ndarray' = None
    def __init__(self, x: Iterable[float], yc: Iterable[float],
                 dya: float, dyb: float) -> None:
        x = asarray(x, dtype=float)
        yc = asarray(yc, dtype=float)
        if len(x) != len(yc) + 1:
            raise ValueError('Length of yc must be one less than that of x.')
        _, counts = unique(x, return_counts=True)
        if all(counts == 1):
            pass
        else:
            raise ValueError('Not all x values are unique.')
        super().__init__(x)
        self.yc = yc
        self.dya = dya
        self.dyb = dyb
    @property
    def ycmat(self) -> 'matrix':
        if self._ycmat is None:
            self._ycmat = matrix([self.yc], dtype=float).transpose()
        return self._ycmat
    @property
    def rmat(self) -> 'matrix':
        if self._rmat is None:
            self._rmat = zeros((self.num+1, 1), dtype=float)
            self._rmat[:-2, 0] = self.ycmat
            self._rmat[-2, 0] = self.dya
            self._rmat[-1, 0] = self.dyb
        return self._rmat
    @property
    def ymat(self) -> 'matrix':
        if self._ymat is None:
            self._ymat = self.gmat*self.rmat
        return self._ymat
    @property
    def y(self) -> 'ndarray':
        if self._y is None:
            self._y = asarray(self.ymat).flatten()
        return self._y
    @property
    def dymat(self) -> 'matrix':
        if self._dymat is None:
            self._dymat = self.emat*self.ymat + self.fmat*self.ycmat
        return self._dymat
    @property
    def dydx(self) -> 'ndarray':
        if self._dydx is None:
            self._dydx = asarray(self.dymat).flatten()
        return self._dydx
    @property
    def iymat(self) -> 'matrix':
        if self._iymat is None:
            self._iymat = self.imat*self.ymat + self.jmat*self.ycmat
        return self._iymat
    @property
    def iydx(self) -> 'ndarray':
        if self._iydx is None:
            self._iydx = asarray(self.iymat).flatten()
        return self._iydx
    def quadratic_interpolation(self, xv: float) -> float:
        found = False
        for j, dxj in enumerate(self.dx):
            xa = self.x[j]
            xb = self.x[j+1]
            if xv >= xa and xv <= xb:
                found = True
                x = xv-(xa+xb)/2
                ya = self.y[j]
                yb = self.y[j+1]
                yc = self.yc[j]
                xos = x/dxj
                x2os2x2 = xos**2*2
                A = x2os2x2-xos
                B = x2os2x2+xos
                C = 1-2*x2os2x2
                yv = A*ya + B*yb + C*yc
                break
        if not found:
            raise ValueError('The x value provided is not within the x range.')
        return yv
    def quadratic_interpolation_array(self, xv: 'ndarray') -> 'ndarray':
        return fromiter([self.quadratic_interpolation(xi) for xi in xv], dtype=float)
    def quadratic_first_derivative(self, xv: float) -> float:
        found = False
        for j, dxj in enumerate(self.dx):
            xa = self.x[j]
            xb = self.x[j+1]
            if xv >= xa and xv <= xb:
                found = True
                x = xv-(xa+xb)/2
                ya = self.y[j]
                yb = self.y[j+1]
                yc = self.yc[j]
                oos = 1/dxj
                xos2x4 = x/dxj**2*4
                dAdx = xos2x4-oos
                dBdx = xos2x4+oos
                dCdx = -2*xos2x4
                dyv = dAdx*ya + dBdx*yb + dCdx*yc
                break
        if not found:
            raise ValueError('The x value provided is not within the x range.')
        return dyv
    def quadratic_first_derivative_array(self, xv: 'ndarray') -> 'ndarray':
        return fromiter([self.quadratic_first_derivative(xi) for xi in xv], dtype=float)
    # def quadratic_integral(self, xv: float) -> float:
    #     found = False
    #     for j, dxj in enumerate(self.dx):
    #         ia = self.iydx[j]
    #         xa = self.x[j]
    #         xb = self.x[j+1]
    #         if xv >= xa and xv <= xb:
    #             found = True
    #             ya = self.y[j]
    #             yb = self.y[j+1]
    #             yc = self.yc[j]
    #             x = xv-(xa+xb)/2
    #             iAdx = 5*dxj/24 - x**2/2/dxj + 2*x**3/3/dxj**2
    #             iBdx = -dxj/24 + x**2/2/dxj + 2*x**3/3/dxj**2
    #             iCdx = dxj/3 + x - 4*x**3/3/dxj**2
    #             iyv = ia + iAdx*ya + iBdx*yb + iCdx*yc
    #             break
    #     if not found:
    #         raise ValueError('The x value provided is not within the x range.')
    #     return iyv
    # def quadratic_integral_array(self, xv: 'ndarray') -> 'ndarray':
    #     return fromiter([self.quadratic_integral(xi) for xi in xv], dtype=float)

class QuadraticInterpolationSolver():
    x: 'ndarray' = None
    _num: int = None
    _dx: 'ndarray' = None
    _xc: 'ndarray' = None
    _gmat: 'matrix' = None
    _emat: 'matrix' = None
    _fmat: 'matrix' = None
    _hmat: 'matrix' = None
    _smat: 'matrix' = None
    _tmat: 'matrix' = None
    _qmat: 'matrix' = None
    _imat: 'matrix' = None
    _jmat: 'matrix' = None
    _kmat: 'matrix' = None
    _zmatop: 'matrix' = None
    _gmatop: 'matrix' = None
    _hmatop: 'matrix' = None
    _qmatop: 'matrix' = None
    _kmatop: 'matrix' = None
    _zmateq: 'matrix' = None
    _gmateq: 'matrix' = None
    _hmateq: 'matrix' = None
    _qmateq: 'matrix' = None
    _kmateq: 'matrix' = None
    _zmate0: 'matrix' = None
    _gmate0: 'matrix' = None
    _hmate0: 'matrix' = None
    _qmate0: 'matrix' = None
    _kmate0: 'matrix' = None
    def __init__(self, x: Iterable[float]) -> None:
        for i in range(len(x)-1):
            if x[i+1] <= x[i]:
                raise ValueError('Input x list must be sorted and unique.')
        self.x = asarray(x, dtype=float)
    @property
    def num(self) -> int:
        if self._num is None:
            self._num = len(self.x)
        return self._num
    @property
    def dx(self) -> 'ndarray':
        if self._dx is None:
            self._dx = self.x[1:]-self.x[:-1]
        return self._dx
    @property
    def xc(self) -> 'ndarray':
        if self._xc is None:
            self._xc = (self.x[1:]+self.x[:-1])/2
        return self._xc
    @property
    def gmat(self) -> 'matrix':
        if self._gmat is None:
            a = asarray(zeros((self.num-2, 1), dtype=float)).flatten()
            b = asarray(zeros((self.num-1, 1), dtype=float)).flatten()
            c = asarray(zeros((self.num-2, 1), dtype=float)).flatten()
            d = zeros((self.num-1, self.num+1), dtype=float)
            dxi = self.dx[0]
            b[0] = -4/dxi
            d[0, 0] = -3/dxi
            d[0, 1] = -1/dxi
            for i in range(self.num-2):
                dxia = self.dx[i]
                dxib = self.dx[i+1]
                a[i] = -4/dxia
                b[i+1] = -4/dxib
                d[i+1, i] += -1/dxia
                d[i+1, i+1] += -3/dxia
                d[i+1, i+1] += -3/dxib
                d[i+1, i+2] += -1/dxib
            d[0, -1] = -1.0
            self._gmat = tridiag_solver(a, b, c, d)
        return self._gmat
    @property
    def emat(self) -> 'matrix':
        if self._emat is None:
            self._emat = zeros((self.num, self.num), dtype=float)
            for i, dxi in enumerate(self.dx):
                self._emat[i, i] -= 3/dxi
                self._emat[i, i+1] -= 1/dxi
            self._emat[-1, -2] += 1/dxi
            self._emat[-1, -1] += 3/dxi
        return self._emat
    @property
    def fmat(self) -> 'matrix':
        if self._fmat is None:
            self._fmat = zeros((self.num, self.num-1), dtype=float)
            for i, dxi in enumerate(self.dx):
                self._fmat[i, i] += 4/dxi
            self._fmat[-1, -1] -= 4/dxi
        return self._fmat
    @property
    def smat(self) -> 'matrix':
        if self._smat is None:
            self._smat = zeros((self.num-1, self.num), dtype=float)
            for i, dxi in enumerate(self.dx):
                xbar2 = 1/dxi**2
                self._smat[i, i] = 4*xbar2
                self._smat[i, i+1] = 4*xbar2
        return self._smat
    @property
    def tmat(self) -> 'matrix':
        if self._tmat is None:
            self._tmat = zeros((self.num-1, self.num-1), dtype=float)
            for i, dxi in enumerate(self.dx):
                xbar2 = 1/dxi**2
                self._tmat[i, i] = -8*xbar2
        return self._tmat
    @property
    def hmat(self) -> 'matrix':
        if self._hmat is None:
            self._hmat = self.fmat*self.gmat
            self._hmat[:, :-1] += self.emat
        return self._hmat
    @property
    def qmat(self) -> 'matrix':
        if self._qmat is None:
            self._qmat =  self.tmat*self.gmat
            self._qmat[:, :-1] += self.smat
        return self._qmat
    @property
    def zmatop(self) -> 'matrix':
        if self._zmatop is None:
            eb = self.emat[-1, :]
            fb = self.fmat[-1, :]
            gy = self.gmat[:, :-1]
            ga = self.gmat[:, -1]
            self._zmatop = -(eb+fb*gy)/(1.0+fb*ga)
        return self._zmatop
    @property
    def gmatop(self) -> 'matrix':
        if self._gmatop is None:
            self._gmatop = self.gmat[:, -1]*self.zmatop + self.gmat[:, :-1]
        return self._gmatop
    @property
    def hmatop(self) -> 'matrix':
        if self._hmatop is None:
            self._hmatop = self.emat + self.fmat*self.gmatop
        return self._hmatop
    @property
    def qmatop(self) -> 'matrix':
        if self._qmatop is None:
            self._qmatop = self.smat + self.tmat*self.gmatop
        return self._qmatop
    @property
    def zmateq(self) -> 'matrix':
        if self._zmateq is None:
            eb = self.emat[-1, :]
            fb = self.fmat[-1, :]
            gy = self.gmat[:, :-1]
            ga = self.gmat[:, -1]
            self._zmateq = (eb+fb*gy)/(1.0-fb*ga)
        return self._zmateq
    @property
    def gmateq(self) -> 'matrix':
        if self._gmateq is None:
            self._gmateq = self.gmat[:, -1]*self.zmateq + self.gmat[:, :-1]
        return self._gmateq
    @property
    def hmateq(self) -> 'matrix':
        if self._hmateq is None:
            self._hmateq = self.emat + self.fmat*self.gmateq
        return self._hmateq
    @property
    def qmateq(self) -> 'matrix':
        if self._qmateq is None:
            self._qmateq = self.smat + self.tmat*self.gmateq
        return self._qmateq
    @property
    def zmate0(self) -> 'matrix':
        if self._zmate0 is None:
            eb = self.emat[-1, :]
            fb = self.fmat[-1, :]
            gy = self.gmat[:, :-1]
            ga = self.gmat[:, -1]
            self._zmate0 = -(eb+fb*gy)/(fb*ga)
        return self._zmate0
    @property
    def gmate0(self) -> 'matrix':
        if self._gmate0 is None:
            self._gmate0 = self.gmat[:, -1]*self.zmate0 + self.gmat[:, :-1]
        return self._gmate0
    @property
    def hmate0(self) -> 'matrix':
        if self._hmate0 is None:
            self._hmate0 = self.emat + self.fmat*self.gmate0
        return self._hmate0
    @property
    def qmate0(self) -> 'matrix':
        if self._qmate0 is None:
            self._qmate0 = self.smat + self.tmat*self.gmate0
        return self._qmate0
    @property
    def imat(self) -> 'matrix':
        if self._imat is None:
            imat = zeros((self.num, self.num), dtype=float)
            for i, dxi in enumerate(self.dx):
                imat[i+1:, i] += dxi/6
                imat[i+1:, i+1] += dxi/6
            self._imat = zeros((self.num, self.num), dtype=float)
            iopp = imat[-1, :]/2
            for i in range(self.num):
                self._imat[i, :] = imat[i, :] - iopp
        return self._imat
    @property
    def jmat(self) -> 'matrix':
        if self._jmat is None:
            jmat = zeros((self.num, self.num-1), dtype=float)
            for i, dxi in enumerate(self.dx):
                jmat[i+1:, i] += 2*dxi/3
            self._jmat = zeros((self.num, self.num-1), dtype=float)
            jopp = jmat[-1, :]/2
            for i in range(self.num):
                self._jmat[i, :] = jmat[i, :] - jopp
        return self._jmat
    @property
    def kmat(self) -> 'matrix':
        if self._kmat is None:
            self._kmat = self.jmat*self.gmat
            self._kmat[:, :-1] += self.imat
        return self._kmat
    @property
    def kmateq(self) -> 'matrix':
        if self._kmateq is None:
            self._kmateq = self.imat + self.jmat*self.gmateq
        return self._kmateq
    @property
    def kmatop(self) -> 'matrix':
        if self._kmatop is None:
            self._kmatop = self.imat + self.jmat*self.gmatop
        return self._kmatop

class QuadraticInterpolation(QuadraticInterpolationSolver):
    y: 'ndarray' = None
    dya: float = None
    _rmat: 'matrix' = None
    _ycmat: 'matrix' = None
    _yc: 'ndarray' = None
    _dymat: 'matrix' = None
    _dydx: 'ndarray' = None
    _iymat: 'matrix' = None
    _iydx: 'ndarray' = None
    def __init__(self, x: Iterable[float], y: Iterable[float],
                 dya: float) -> None:
        x = asarray(x, dtype=float)
        y = asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError('Length of x and y must be equal.')
        _, counts = unique(x, return_counts=True)
        if all(counts == 1):
            inds = argsort(x)
            x = x[inds]
            y = y[inds]
        else:
            raise ValueError('Not all x values are unique.')
        super().__init__(x)
        self.y = asarray(y, dtype=float)
        self.dya = dya
    @property
    def ymat(self) -> 'matrix':
        ymat: 'matrix' = matrix([self.y], dtype=float)
        return ymat.transpose()
    @property
    def rmat(self) -> 'matrix':
        if self._rmat is None:
            self._rmat = zeros((self.num+1, 1), dtype=float)
            self._rmat[:-1, 0] = self.ymat
            self._rmat[-1, 0] = self.dya
        return self._rmat
    @property
    def ycmat(self) -> 'matrix':
        if self._ycmat is None:
            self._ycmat = self.gmat*self.rmat
        return self._ycmat
    @property
    def yc(self) -> 'ndarray':
        if self._yc is None:
            self._yc = asarray(self.ycmat).flatten()
        return self._yc
    @property
    def dymat(self) -> 'matrix':
        if self._dymat is None:
            self._dymat = self.emat*self.ymat + self.fmat*self.ycmat
        return self._dymat
    @property
    def dydx(self) -> 'ndarray':
        if self._dydx is None:
            self._dydx = asarray(self.dymat).flatten()
        return self._dydx
    @property
    def iymat(self) -> 'matrix':
        if self._iymat is None:
            self._iymat = self.imat*self.ymat+self.jmat*self.ycmat
        return self._iymat
    @property
    def iydx(self) -> 'ndarray':
        if self._iydx is None:
            self._iydx = asarray(self.iymat).flatten()
        return self._iydx
    def quadratic_interpolation(self, xv: float) -> float:
        found = False
        for j, dxj in enumerate(self.dx):
            xa = self.x[j]
            xb = self.x[j+1]
            if xv >= xa and xv <= xb:
                found = True
                x = xv-(xa+xb)/2
                ya = self.y[j]
                yb = self.y[j+1]
                yc = self.yc[j]
                xos = x/dxj
                x2os2x2 = xos**2*2
                A = x2os2x2-xos
                B = x2os2x2+xos
                C = 1-2*x2os2x2
                yv = A*ya + B*yb + C*yc
                break
        if not found:
            raise ValueError('The x value provided is not within the x range.')
        return yv
    def quadratic_interpolation_array(self, xv: 'ndarray') -> 'ndarray':
        return fromiter([self.quadratic_interpolation(xi) for xi in xv], dtype=float)
    def quadratic_first_derivative(self, xv: float) -> float:
        found = False
        for j, dxj in enumerate(self.dx):
            xa = self.x[j]
            xb = self.x[j+1]
            if xv >= xa and xv <= xb:
                found = True
                x = xv-(xa+xb)/2
                ya = self.y[j]
                yb = self.y[j+1]
                yc = self.yc[j]
                oos = 1/dxj
                xos2x4 = x/dxj**2*4
                dAdx = xos2x4-oos
                dBdx = xos2x4+oos
                dCdx = -2*xos2x4
                dyv = dAdx*ya + dBdx*yb + dCdx*yc
                break
        if not found:
            raise ValueError('The x value provided is not within the x range.')
        return dyv
    def quadratic_first_derivative_array(self, xv: 'ndarray') -> 'ndarray':
        return fromiter([self.quadratic_first_derivative(xi) for xi in xv], dtype=float)
    def quadratic_integral(self, xv: float) -> float:
        found = False
        for j, dxj in enumerate(self.dx):
            ia = self.iydx[j]
            xa = self.x[j]
            xb = self.x[j+1]
            if xv >= xa and xv <= xb:
                found = True
                ya = self.y[j]
                yb = self.y[j+1]
                yc = self.yc[j]
                x = xv-(xa+xb)/2
                iAdx = 5*dxj/24 - x**2/2/dxj + 2*x**3/3/dxj**2
                iBdx = -dxj/24 + x**2/2/dxj + 2*x**3/3/dxj**2
                iCdx = dxj/3 + x - 4*x**3/3/dxj**2
                iyv = ia + iAdx*ya + iBdx*yb + iCdx*yc
                break
        if not found:
            raise ValueError('The x value provided is not within the x range.')
        return iyv
    def quadratic_integral_array(self, xv: 'ndarray') -> 'ndarray':
        return fromiter([self.quadratic_integral(xi) for xi in xv], dtype=float)
