from typing import TYPE_CHECKING

from numpy import asarray, divide, fromiter, zeros
from numpy.linalg import inv

if TYPE_CHECKING:
    from numpy.typing import NDArray

class LinearInterpSolver():
    x: 'NDArray' = None
    _num: int = None
    _dx: 'NDArray' = None
    _xc: 'NDArray' = None
    _m: 'NDArray' = None
    _kmat: 'NDArray' = None
    _hmat: 'NDArray' = None

    def __init__(self, x: 'NDArray') -> None:
        self.x = asarray(x)

    @property
    def num(self) -> int:
        if self._num is None:
            self._num = len(self.x)
        return self._num

    @property
    def dx(self) -> 'NDArray':
        if self._dx is None:
            self._dx = self.x[1:] - self.x[:-1]
        return self._dx

    @property
    def xc(self) -> 'NDArray':
        if self._xc is None:
            self._xc = (self.x[:-1] + self.x[1:])/2
        return self._xc

    @property
    def hmat(self) -> 'NDArray':
        # dydx = H*y
        if self._hmat is None:
            self._hmat = zeros((self.num-1, self.num))
            for i, dxi in enumerate(self.dx):
                self._hmat[i, i] = -1/dxi
                self._hmat[i, i+1] = 1/dxi
        return self._hmat

    @property
    def kmat(self) -> 'NDArray':
        # y = K*dydx
        if self._kmat is None:
            kmat = zeros((self.num, self.num-1))
            for i, dxi in enumerate(self.dx):
                kmat[i+1:, i] = dxi
            self._kmat = zeros((self.num, self.num-1))
            kopp = kmat[-1, :]/2
            for i in range(self.num):
                self._kmat[i, :] = kmat[i, :] - kopp
        return self._kmat

    def x_at_factor(self, fac: float):
        xf = asarray(self.num-1)
        for i in range(self.num-1):
            xf[i] = self.x[i] + fac*self.dx[i]
        return xf

    def interpolation_matrix(self, xi: 'NDArray') -> 'NDArray':
        numi = len(xi)
        imat = zeros((numi, self.num))
        for i in range(numi):
            for j in range(self.num-1):
                if xi[i] >= self.x[j] and xi[i] <= self.x[j+1]:
                    imat[i, j] = 1.0 - (xi[i]-self.x[j])/self.dx[i]
                    imat[i, j+1] = (xi[i]-self.x[j])/self.dx[i]
                    break
        return imat

    def opposite_end_gradient_matrix(self, xi: 'NDArray') -> 'NDArray':
        imat = self.interpolation_matrix(xi)
        amat = zeros((self.num, self.num))
        amat[:-1, :] = imat
        amat[-1, 0] = 1/self.dx[0]
        amat[-1, 1] = -1/self.dx[0]
        amat[-1, -1] = -1/self.dx[-1]
        amat[-1, -2] = 1/self.dx[-1]
        gmat = amat[-1, :-1]/amat[-1, -1]
        amat[-2,:-1] -= amat[-2, -1]*gmat
        amat[-2, -1] = 0.0
        amat[-1, :] = 0.0
        ainv = inv(amat[:-1, :-1])
        hmat = zeros((self.num, self.num-1))
        hmat[:-1, :] = ainv
        hmat[-1, :] = -gmat*ainv
        return hmat

    def ymat_from_list(self, y: list[float]) -> 'NDArray':
        ymat: 'NDArray' = asarray([y]).reshape((-1, 1))
        return ymat

class LinearInterp(LinearInterpSolver):
    y: 'NDArray' = None
    _ymat: 'NDArray' = None
    _dy: 'NDArray' = None
    _m: 'NDArray' = None
    _iymat: 'NDArray' = None
    _iydx: 'NDArray' = None

    def __init__(self, x: 'NDArray', y: 'NDArray') -> None:
        super().__init__(x)
        self.y = asarray(y)

    @property
    def dy(self):
        if self._dy is None:
            self._dy = self.y[1:]-self.y[:-1]
        return self._dy

    @property
    def m(self) -> 'NDArray':
        if self._m is None:
            self._m = divide(self.dy, self.dx)
        return self._m

    @property
    def ymat(self) -> 'NDArray':
        if self._ymat is None:
            self._ymat = self.ymat_from_list(self.y)
        return self._ymat

    @property
    def iymat(self) -> 'NDArray':
        if self._iymat is None:
            self._iymat = self.kmat*self.ymat
        return self._iymat

    @property
    def iydx(self) -> 'NDArray':
        if self._iydx is None:
            self._iydx = asarray(self.iymat).ravel()
        return self._iydx

    def linear_interpolation(self, xv: float) -> float:
        found = False
        for j, mj in enumerate(self.m):
            xa = self.x[j]
            xb = self.x[j+1]
            if xv >= xa and xv <= xb:
                found = True
                yv = self.y[j] + mj*(xv-self.x[j])
                break
        if not found:
            raise ValueError(f'The x = {xv} value provided is not within the x range.')
        return yv

    def linear_interpolation_array(self, xv: 'NDArray') -> 'NDArray':
        return fromiter([self.linear_interpolation(xi) for xi in xv], float)

    def linear_interpolation_integral(self, xv: float) -> float:
        found = False
        for j, dxj in enumerate(self.dx):
            ia = self.iydx[j]
            xa = self.x[j]
            xb = self.x[j+1]
            if xv >= xa and xv <= xb:
                found = True
                ya = self.y[j]
                yb = self.y[j+1]
                x = xv-xa
                iAdx = x - x**2/2/dxj
                iBdx = x**2/2/dxj
                iyv = ia + iAdx*ya + iBdx*yb
                break
        if not found:
            raise ValueError('The x value provided is not within the x range.')
        return iyv

    def linear_interpolation_integral_array(self, xv: 'NDArray') -> 'NDArray':
        return fromiter([self.linear_interpolation_integral(xi) for xi in xv], float)
