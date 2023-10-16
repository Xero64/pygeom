from typing import Optional, Tuple, TYPE_CHECKING, List

from matplotlib.pyplot import figure
from matplotlib.axes import Axes
from numpy import zeros, concatenate, linspace, array

from .vector2d import Vector2D
from ..array2d import ArrayVector2D, zero_arrayvector2d, solve_arrayvector2d

if TYPE_CHECKING:
    from numpy import ndarray, number

class SplinePoint2D(Vector2D):
    pnla: 'SplinePanel2D' = None
    pnlb: 'SplinePanel2D' = None
    tan: Vector2D = None
    _endpnta: bool = None
    _endpntb: bool = None
    _c1: bool = None
    _c2: bool = None
    _lhs: List['ndarray'] = None
    _rhs: List['ndarray'] = None
    _ind: List[Tuple[int, ...]] = None
    c2a0: bool = None
    c2b0: bool = None

    def __init__(self, x: 'number', y: 'number') -> None:
        super().__init__(x, y)
        self.c2a0 = False
        self.c2b0 = False

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr[0] == '_':
                setattr(self, attr, None)

    @property
    def endpnta(self) -> bool:
        if self._endpnta is None:
            if self.pnla is None and self.pnlb is not None:
                self._endpnta = True
            elif self.pnla is not None:
                self._endpnta = False
        return self._endpnta

    @property
    def endpntb(self) -> bool:
        if self._endpntb is None:
            if self.pnlb is None and self.pnla is not None:
                self._endpntb = True
            elif self.pnlb is not None:
                self._endpntb = False
        return self._endpntb

    @property
    def c1(self) -> bool:
        if self._c1 is None:
            if self.endpnta or self.endpntb:
                self._c1 = False
            else:
                if self.c2a0 and self.c2b0:
                    self._c1 = False
                else:
                    self._c1 = True
        return self._c1

    @property
    def c2(self) -> bool:
        if self._c2 is None:
            if self.endpnta or self.endpntb:
                self._c2 = False
            else:
                if self.c2a0 or self.c2b0:
                    self._c2 = False
                else:
                    self._c2 = True
        return self._c2

    def get_ind_lhs_rhs(self) -> Tuple[List['ndarray'],
                                       List['ndarray'],
                                       List[ArrayVector2D]]:
        ind = []
        lhs = []
        rhs = []

        if self.endpnta:
            if self.c2b0 or self.tan is None:
                ind.append(self.pnlb.ind)
                lhs.append(self.pnlb.lhs_d2ra)
                rhs.append(Vector2D(0.0, 0.0))
            else:
                ind.append(self.pnlb.ind)
                lhs.append(self.pnlb.lhs_dra)
                rhs.append(self.tan - self.pnlb.rhs_dr)
        elif self.endpntb:
            if self.c2b0 or self.tan is None:
                ind.append(self.pnla.ind)
                lhs.append(self.pnla.lhs_d2rb)
                rhs.append(Vector2D(0.0, 0.0))
            else:
                ind.append(self.pnla.ind)
                lhs.append(self.pnla.lhs_drb)
                rhs.append(self.tan - self.pnla.rhs_dr)
        else:
            if self.c1:
                ind.append(concatenate((self.pnla.ind, self.pnlb.ind)))
                lhs.append(concatenate((-self.pnla.lhs_drb, self.pnlb.lhs_dra)))
                rhs.append(self.pnla.rhs_dr - self.pnlb.rhs_dr)
                if self.c2:
                    ind.append(concatenate((self.pnla.ind, self.pnlb.ind)))
                    lhs.append(concatenate((-self.pnla.lhs_d2rb, self.pnlb.lhs_d2ra)))
                    rhs.append(Vector2D(0.0, 0.0))
                else:
                    if self.c2a0:
                        ind.append(self.pnla.ind)
                        lhs.append(self.pnla.lhs_d2rb)
                        rhs.append(Vector2D(0.0, 0.0))
                    elif self.c2b0:
                        ind.append(self.pnlb.ind)
                        lhs.append(self.pnlb.lhs_d2ra)
                        rhs.append(Vector2D(0.0, 0.0))
            else:
                ind.append(self.pnla.ind)
                lhs.append(self.pnla.lhs_d2rb)
                rhs.append(Vector2D(0.0, 0.0))
                ind.append(self.pnlb.ind)
                lhs.append(self.pnlb.lhs_d2ra)
                rhs.append(Vector2D(0.0, 0.0))

        self._ind = ind
        self._lhs = lhs
        self._rhs = rhs

        return ind, lhs, rhs

    @property
    def lhs(self) -> List['ndarray']:
        if self._lhs is None:
            self.get_ind_lhs_rhs()
        return self._lhs

    @property
    def rhs(self) -> List['ndarray']:
        if self._rhs is None:
            self.get_ind_lhs_rhs()
        return self._rhs

    @property
    def ind(self) -> List['ndarray']:
        if self._ind is None:
            self.get_ind_lhs_rhs()
        return self._ind

class SplinePanel2D():
    pnta: SplinePoint2D = None
    pntb: SplinePoint2D = None
    _vector: Vector2D = None
    _length: float = None
    _direc: Vector2D = None
    _lhs_dra: 'ndarray' = None
    _lhs_drb: 'ndarray' = None
    _lhs_d2ra: 'ndarray' = None
    _lhs_d2rb: 'ndarray' = None
    ind: 'ndarray' = None
    spline: 'Spline2D' = None
    _sa: float = None
    _sb: float = None
    _d2ra: Vector2D = None
    _d2rb: Vector2D = None
    _dra: Vector2D = None
    _drb: Vector2D = None
    _ka: Vector2D = None
    _kb: Vector2D = None

    def __init__(self, pnta: SplinePoint2D, pntb: SplinePoint2D) -> None:
        self.pnta = pnta
        self.pnta.pnlb = self
        self.pntb = pntb
        self.pntb.pnla = self

    def set_straight_edge(self) -> None:
        self.pnta.c2b0 = True
        self.pntb.c2a0 = True
        self.pnta._c2 = False
        self.pntb._c2 = False

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr[0] == '_':
                setattr(self, attr, None)

    @property
    def vector(self) -> Vector2D:
        if self._vector is None:
            self._vector = self.pntb - self.pnta
        return self._vector

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self.vector.return_magnitude()
        return self._length

    @property
    def direc(self) -> Vector2D:
        if self._direc is None:
            self._direc = self.vector.to_unit()
        return self._direc

    @property
    def lhs_dra(self) -> 'ndarray':
        if self._lhs_dra is None:
            self._lhs_dra = zeros(2)
            self._lhs_dra[0] = -self.length/3
            self._lhs_dra[1] = -self.length/6
        return self._lhs_dra

    @property
    def lhs_drb(self) -> 'ndarray':
        if self._lhs_drb is None:
            self._lhs_drb = zeros(2)
            self._lhs_drb[0] = self.length/6
            self._lhs_drb[1] = self.length/3
        return self._lhs_drb

    @property
    def lhs_d2ra(self) -> 'ndarray':
        if self._lhs_d2ra is None:
            self._lhs_d2ra = zeros(2)
            self._lhs_d2ra[0] = 1.0
        return self._lhs_d2ra

    @property
    def lhs_d2rb(self) -> 'ndarray':
        if self._lhs_d2rb is None:
            self._lhs_d2rb = zeros(2)
            self._lhs_d2rb[1] = 1.0
        return self._lhs_d2rb

    @property
    def rhs_dr(self) -> Vector2D:
        return self.direc

    @property
    def d2ra(self) -> Vector2D:
        if self._d2ra is None:
            inda = self.ind[0]
            self._d2ra = self.spline.d2r[inda]
        return self._d2ra

    @property
    def d2rb(self) -> Vector2D:
        if self._d2rb is None:
            indb = self.ind[1]
            self._d2rb = self.spline.d2r[indb]
        return self._d2rb

    @property
    def dra(self) -> Vector2D:
        if self._dra is None:
            E = -self.length/3
            F = -self.length/6
            x = self.direc.x + E*self.d2ra.x + F*self.d2rb.x
            y = self.direc.y + E*self.d2ra.y + F*self.d2rb.y
            self._dra = Vector2D(x, y)
        return self._dra

    @property
    def drb(self) -> Vector2D:
        if self._drb is None:
            E = self.length/6
            F = self.length/3
            x = self.direc.x + E*self.d2ra.x + F*self.d2rb.x
            y = self.direc.y + E*self.d2ra.y + F*self.d2rb.y
            self._drb = Vector2D(x, y)
        return self._drb

    @property
    def ra(self) -> Vector2D:
        return self.pnta

    @property
    def rb(self) -> Vector2D:
        return self.pntb

    @property
    def sa(self) -> float:
        if self._sa is None:
            inda = self.ind[0]
            self._sa = self.spline.s[inda]
        return self._sa

    @property
    def sb(self) -> Vector2D:
        if self._sb is None:
            indb = self.ind[1]
            self._sb = self.spline.s[indb]
        return self._sb

    @property
    def ka(self) -> Vector2D:
        if self._ka is None:
            mdra3 = self.dra.return_magnitude()**3
            self._ka = self.dra.cross(self.d2ra)/mdra3
        return self._ka

    @property
    def kb(self) -> Vector2D:
        if self._kb is None:
            mdrb3 = self.drb.return_magnitude()**3
            self._kb = self.drb.cross(self.d2rb)/mdrb3
        return self._kb

    def ratio_length_interpolate(self, ratio: 'ndarray') -> 'ndarray':
        A = 1.0 - ratio
        B = ratio
        return self.sa*A + self.sb*B

    def ratio_point_interpolate(self, ratio: 'ndarray') -> ArrayVector2D:
        A = 1.0 - ratio
        B = ratio
        C = (A**3 - A)*self.length**2/6
        D = (B**3 - B)*self.length**2/6
        x = A*self.pnta.x + B*self.pntb.x + C*self.d2ra.x + D*self.d2rb.x
        y = A*self.pnta.y + B*self.pntb.y + C*self.d2ra.y + D*self.d2rb.y
        return ArrayVector2D(x, y)

    def ratio_gradient_interpolate(self, ratio: 'ndarray') -> ArrayVector2D:
        A = 1.0 - ratio
        B = ratio
        E = (1 - 3*A**2)*self.length/6
        F = (3*B**2 - 1)*self.length/6
        x = self.direc.x + E*self.d2ra.x + F*self.d2rb.x
        y = self.direc.y + E*self.d2ra.y + F*self.d2rb.y
        return ArrayVector2D(x, y)

    def ratio_curvature_interpolate(self, ratio: 'ndarray') -> ArrayVector2D:
        A = 1.0 - ratio
        B = ratio
        x = A*self.d2ra.x + B*self.d2rb.x
        y = A*self.d2ra.y + B*self.d2rb.y
        return ArrayVector2D(x, y)

    def ratio_inverse_radius_interpolate(self, ratio: 'ndarray') -> 'ndarray':
        dr = self.ratio_gradient_interpolate(ratio)
        d2r = self.ratio_curvature_interpolate(ratio)
        mdr3 = dr.return_magnitude()**3
        k = dr.cross(d2r)/mdr3
        return k

class Spline2D():
    u"""This class stores a 3D parametric spline."""
    pnts: List[SplinePoint2D] = None
    closed: bool = False
    tanA: Optional[Vector2D] = None
    tanB: Optional[Vector2D] = None
    _numpnt: int = None
    _pnls: List[SplinePanel2D] = None
    _numpnl: int = None
    _d2r: ArrayVector2D = None
    _dr: ArrayVector2D = None
    _r: ArrayVector2D = None
    _s: 'ndarray' = None
    _k: ArrayVector2D = None

    def __init__(self, pnts: List[Vector2D], closed: bool=False,
                 tanA: Vector2D=None, tanB: Vector2D=None) -> None:
        if closed and pnts[0] == pnts[-1]:
            pnts = pnts[:-1]
        self.pnts = [SplinePoint2D(pnt.x, pnt.y) for pnt in pnts]
        self.closed = closed
        if not self.closed:
            self.tanA = tanA
            self.pnts[0].tan = tanA
            self.tanB = tanB
            self.pnts[-1].tan = tanB

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr[0] == '_':
                setattr(self, attr, None)
        for pnt in self.pnts:
            pnt.reset()

    @property
    def numpnt(self) -> int:
        if self._numpnt is None:
            self._numpnt = len(self.pnts)
        return self._numpnt

    @property
    def pnls(self) -> List[SplinePanel2D]:
        if self._pnls is None:
            self._pnls = []
            j = 0
            for i in range(self.numpnt-1):
                pnl = SplinePanel2D(self.pnts[i], self.pnts[i+1])
                pnl.ind = array([j, j+1], dtype=int)
                pnl.spline = self
                self._pnls.append(pnl)
                j += 2
            if self.closed:
                pnl = SplinePanel2D(self.pnts[-1], self.pnts[0])
                pnl.ind = array([j, j+1], dtype=int)
                pnl.spline = self
                self._pnls.append(pnl)
        return self._pnls

    @property
    def numpnl(self) -> int:
        if self._numpnl is None:
            self._numpnl = len(self.pnls)
        return self._numpnl

    def calc_d2r(self) -> ArrayVector2D:
        numres = 2*self.numpnl
        Amat = zeros((numres, numres))
        Bmat = zero_arrayvector2d((numres, 1))
        j = 0
        for pnt in self.pnts:
            tup = pnt.get_ind_lhs_rhs()
            for ind, lhs, rhs in zip(*tup):
                Amat[j, ind] = lhs
                Bmat[j, 0] = rhs
                j += 1
        self._d2r = solve_arrayvector2d(Amat, Bmat)

    @property
    def d2r(self) -> ArrayVector2D:
        if self._d2r is None:
            self.calc_d2r()
        return self._d2r

    @property
    def dr(self) -> ArrayVector2D:
        if self._dr is None:
            self._dr = zero_arrayvector2d(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._dr[inda] = pnl.dra
                indb = pnl.ind[1]
                self._dr[indb] = pnl.drb
        return self._dr

    @property
    def r(self) -> ArrayVector2D:
        if self._r is None:
            self._r = zero_arrayvector2d(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._r[inda] = pnl.pnta
                indb = pnl.ind[1]
                self._r[indb] = pnl.pntb
        return self._r

    @property
    def s(self) -> 'ndarray':
        if self._s is None:
            self._s = zeros(2*self.numpnl)
            scur = 0.0
            for i in range(self.numpnl):
                self._s[2*i] = scur
                self._s[2*i+1] = scur + self.pnls[i].length
                scur = self._s[2*i+1]
        return self._s

    @property
    def k(self) -> 'ndarray':
        if self._k is None:
            self._k = zeros(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._k[inda] = pnl.ka
                indb = pnl.ind[1]
                self._k[indb] = pnl.kb
        return self._k

    def spline_length(self, num: int=5) -> 'ndarray':
        u"""This function interpolates the spline length with a 'number' of points per piece."""

        numpnt = self.numpnl*num + 1
        s = zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            s[slc] = pnl.ratio_length_interpolate(ratio)

        s[-1] = pnl.sb

        return s

    def spline_points(self, num: int=5) -> ArrayVector2D:
        u"""This function interpolates the spline with a 'number' of points per piece."""

        numpnt = self.numpnl*num + 1
        r = zero_arrayvector2d(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            r[slc] = pnl.ratio_point_interpolate(ratio)

        r[-1] = pnl.pntb

        return r

    def spline_gradient(self, num: int=5) -> ArrayVector2D:
        u"""This function interpolates the gradient of the spline with a 'number' of points per piece."""

        numpnt = self.numpnl*num + 1
        dr = zero_arrayvector2d(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            dr[slc] = pnl.ratio_gradient_interpolate(ratio)

        dr[-1] = pnl.drb

        return dr

    def spline_curvature(self, num: int=1) -> ArrayVector2D:
        u"""This function interpolates the curvature of the spline."""

        numpnt = self.numpnl*num + 1
        d2r = zero_arrayvector2d(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            d2r[slc] = pnl.ratio_curvature_interpolate(ratio)

        d2r[-1] = pnl.d2rb

        return d2r

    def spline_inverse_radius(self, num: int=1) -> ArrayVector2D:
        u"""This function interpolates the inverse radius of the spline."""

        numpnt = self.numpnl*num + 1
        k = zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            k[slc] = pnl.ratio_inverse_radius_interpolate(ratio)

        k[-1] = pnl.kb

        return k

    def scatter(self, ax: Axes=None, label=False) -> Axes:
        u"""This function plots the points of the spline."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.gca()
            ax.grid(True)
        ax.scatter(self.r.x, self.r.y)
        if label:
            for i in range(self.numpnt):
                ax.text(self.r.x[i], self.r.y[i], i)
        return ax

    def plot_spline(self, num=5, ax: Axes=None, **kwargs) -> Axes:
        u"""This function plots the spline using the interpolated points."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.add_subplot()
            ax.grid(True)
        r = self.spline_points(num)
        ax.plot(r.x, r.y, **kwargs)
        ax.set_aspect('equal')
        return ax

    def plot_gradient(self, num: int=5, ax: Axes=None) -> Axes:
        u"""This function plots the gradient of the spline."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.gca()
            ax.grid(True)
            ax.set_title('Gradient')
        s = self.spline_length(num=num)
        dr = self.spline_gradient(num=num)
        ax.plot(s, dr.x, label='dr.x')
        ax.plot(s, dr.y, label='dr.y')
        return ax

    def quiver_tangent(self, ax: Axes=None, **kwargs) -> Axes:
        u"""This function quiver plots the tangent of the spline."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.gca()
            ax.grid(True)
        x, y = self.r.x, self.r.y
        dx, dy = self.dr.x, self.dr.y
        ax.quiver(x, y, dx, dy, **kwargs)
        return ax

    def plot_curvature(self, num: int=5, ax: Axes=None) -> Axes:
        u"""This function plots the curvature of the spline."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.gca()
            ax.grid(True)
            ax.set_title('Curvature')
        s = self.spline_length(num=num)
        d2r = self.spline_curvature(num=num)
        ax.plot(s, d2r.x, label='d2r.x')
        ax.plot(s, d2r.y, label='d2r.y')
        return ax

    def plot_inverse_radius(self, num: int=5, ax: Axes=None) -> Axes:
        u"""This function plots the inverse radius of curvature of the spline."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.gca()
            ax.grid(True)
            ax.set_title('Inverse Radius of Curvature')
        s = self.spline_length(num=num)
        k = self.spline_inverse_radius(num=num)
        ax.plot(s, k, label='k')
        return ax

    def quiver_normal(self, ax: Axes=None, **kwargs) -> Axes:
        u"""This function quiver plots the normal of the spline."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.gca()
            ax.grid(True)
        x, y = self.r.x, self.r.y
        d2x, d2y = self.d2r.x, self.d2r.y
        ax.quiver(x, y, d2x, d2y, **kwargs)
        return ax

    def __repr__(self) -> str:
        return '<Spline2D>'