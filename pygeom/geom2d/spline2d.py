from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.pyplot import figure
from numpy import asarray, concatenate, linspace, zeros

from ..geom2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplinePoint2D(Vector2D):
    pnla: 'SplinePanel2D' = None
    pnlb: 'SplinePanel2D' = None
    tan: Vector2D = None
    nrm: Vector2D = None
    _endpnta: bool = None
    _endpntb: bool = None
    _c1: bool = None
    _c2: bool = None
    _lhs: list['NDArray'] = None
    _rhs: list['NDArray'] = None
    _ind: list[tuple[int, ...]] = None
    c2a0: bool = None
    c2b0: bool = None

    def __init__(self, x: float, y: float) -> None:
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

    def get_ind_lhs_rhs(self) -> tuple[list['NDArray'],
                                       list['NDArray'],
                                       list[Vector2D]]:
        ind = []
        lhs = []
        rhs = []

        if self.endpnta:
            if self.c2b0 or (self.tan is None and self.nrm is None):
                ind.append(self.pnlb.ind)
                lhs.append(self.pnlb.lhs_d2ra)
                rhs.append(Vector2D(0.0, 0.0))
            else:
                if self.tan is not None:
                    ind.append(self.pnlb.ind)
                    lhs.append(self.pnlb.lhs_dra)
                    rhs.append(self.tan - self.pnlb.rhs_dr)
                elif self.nrm is not None:
                    ind.append(self.pnlb.ind)
                    lhs.append(self.pnlb.lhs_d2ra)
                    rhs.append(self.nrm)
        elif self.endpntb:
            if self.c2b0 or (self.tan is None and self.nrm is None):
                ind.append(self.pnla.ind)
                lhs.append(self.pnla.lhs_d2rb)
                rhs.append(Vector2D(0.0, 0.0))
            else:
                if self.tan is not None:
                    ind.append(self.pnla.ind)
                    lhs.append(self.pnla.lhs_drb)
                    rhs.append(self.tan - self.pnla.rhs_dr)
                elif self.nrm is not None:
                    ind.append(self.pnla.ind)
                    lhs.append(self.pnla.lhs_d2rb)
                    rhs.append(self.nrm)
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
    def lhs(self) -> list['NDArray']:
        if self._lhs is None:
            self.get_ind_lhs_rhs()
        return self._lhs

    @property
    def rhs(self) -> list['NDArray']:
        if self._rhs is None:
            self.get_ind_lhs_rhs()
        return self._rhs

    @property
    def ind(self) -> list['NDArray']:
        if self._ind is None:
            self.get_ind_lhs_rhs()
        return self._ind

    def __repr__(self) -> str:
        return '<SplinePoint2D>'

class SplinePanel2D():
    pnta: SplinePoint2D = None
    pntb: SplinePoint2D = None
    _vector: Vector2D = None
    _length: float = None
    _direc: Vector2D = None
    _lhs_dra: 'NDArray' = None
    _lhs_drb: 'NDArray' = None
    _lhs_d2ra: 'NDArray' = None
    _lhs_d2rb: 'NDArray' = None
    ind: 'NDArray' = None
    spline: 'Spline2D' = None
    _straight: bool = False
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
        self._straight = True
        self.pnta.c2b0 = True
        self.pntb.c2a0 = True
        self.pnta._c2 = False
        self.pntb._c2 = False

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr[0] == '_':
                setattr(self, attr, None)

    @property
    def straight(self) -> bool:
        if self._straight is None:
            self._straight = False
        return self._straight

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
    def lhs_dra(self) -> 'NDArray':
        if self._lhs_dra is None:
            self._lhs_dra = zeros(2)
            self._lhs_dra[0] = -self.length/3
            self._lhs_dra[1] = -self.length/6
        return self._lhs_dra

    @property
    def lhs_drb(self) -> 'NDArray':
        if self._lhs_drb is None:
            self._lhs_drb = zeros(2)
            self._lhs_drb[0] = self.length/6
            self._lhs_drb[1] = self.length/3
        return self._lhs_drb

    @property
    def lhs_d2ra(self) -> 'NDArray':
        if self._lhs_d2ra is None:
            self._lhs_d2ra = zeros(2)
            self._lhs_d2ra[0] = 1.0
        return self._lhs_d2ra

    @property
    def lhs_d2rb(self) -> 'NDArray':
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
    def ka(self) -> float:
        if self._ka is None:
            mdra3 = self.dra.return_magnitude()**3
            self._ka = self.dra.cross(self.d2ra)/mdra3
        return self._ka

    @property
    def kb(self) -> float:
        if self._kb is None:
            mdrb3 = self.drb.return_magnitude()**3
            self._kb = self.drb.cross(self.d2rb)/mdrb3
        return self._kb

    def ratio_length_interpolate(self, ratio: 'NDArray') -> 'NDArray':
        A = 1.0 - ratio
        B = ratio
        return self.sa*A + self.sb*B

    def ratio_point_interpolate(self, ratio: 'NDArray') -> Vector2D:
        A = 1.0 - ratio
        B = ratio
        C = (A**3 - A)*self.length**2/6
        D = (B**3 - B)*self.length**2/6
        x = A*self.pnta.x + B*self.pntb.x + C*self.d2ra.x + D*self.d2rb.x
        y = A*self.pnta.y + B*self.pntb.y + C*self.d2ra.y + D*self.d2rb.y
        return Vector2D(x, y)

    def ratio_gradient_interpolate(self, ratio: 'NDArray') -> Vector2D:
        A = 1.0 - ratio
        B = ratio
        E = (1 - 3*A**2)*self.length/6
        F = (3*B**2 - 1)*self.length/6
        x = self.direc.x + E*self.d2ra.x + F*self.d2rb.x
        y = self.direc.y + E*self.d2ra.y + F*self.d2rb.y
        return Vector2D(x, y)

    def ratio_curvature_interpolate(self, ratio: 'NDArray') -> Vector2D:
        A = 1.0 - ratio
        B = ratio
        x = A*self.d2ra.x + B*self.d2rb.x
        y = A*self.d2ra.y + B*self.d2rb.y
        return Vector2D(x, y)

    def ratio_inverse_radius_interpolate(self, ratio: 'NDArray') -> 'NDArray':
        dr = self.ratio_gradient_interpolate(ratio)
        d2r = self.ratio_curvature_interpolate(ratio)
        mdr3 = dr.return_magnitude()**3
        k = dr.cross(d2r)/mdr3
        return k

    def ratio_s(self, s: float) -> float:
        return (s - self.sa)/(self.sb - self.sa)

    def __repr__(self) -> str:
        return '<SplinePanel2D>'

class Spline2D():
    u"""This class stores a 3D parametric spline."""
    pnts: list[SplinePoint2D] = None
    closed: bool = False
    tanA: Vector2D | None = None
    tanB: Vector2D | None = None
    nrmA: Vector2D | None = None
    nrmB: Vector2D | None = None
    _numpnt: int = None
    _pnls: list[SplinePanel2D] = None
    _numpnl: int = None
    _d2r: Vector2D = None
    _dr: Vector2D = None
    _r: Vector2D = None
    _s: 'NDArray' = None
    _k: Vector2D = None
    _length: float = None

    def __init__(self, pnts: list[Vector2D], closed: bool=False,
                 tanA: Vector2D=None, tanB: Vector2D=None,
                 nrmA: Vector2D=None, nrmB: Vector2D=None) -> None:
        if closed and pnts[0] == pnts[-1]:
            pnts = pnts[:-1]
        self.pnts = [SplinePoint2D(pnt.x, pnt.y) for pnt in pnts]
        self.closed = closed
        if not self.closed:
            self.tanA = tanA
            self.pnts[0].tan = tanA
            self.tanB = tanB
            self.pnts[-1].tan = tanB
            self.nrmA = nrmA
            self.pnts[0].nrm = nrmA
            self.nrmB = nrmB
            self.pnts[-1].nrm = nrmB

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
    def pnls(self) -> list[SplinePanel2D]:
        if self._pnls is None:
            self._pnls = []
            j = 0
            for i in range(self.numpnt-1):
                pnl = SplinePanel2D(self.pnts[i], self.pnts[i+1])
                pnl.ind = asarray([j, j+1], dtype=int)
                pnl.spline = self
                self._pnls.append(pnl)
                j += 2
            if self.closed:
                pnl = SplinePanel2D(self.pnts[-1], self.pnts[0])
                pnl.ind = asarray([j, j+1], dtype=int)
                pnl.spline = self
                self._pnls.append(pnl)
        return self._pnls

    @property
    def numpnl(self) -> int:
        if self._numpnl is None:
            self._numpnl = len(self.pnls)
        return self._numpnl

    def calc_d2r(self) -> Vector2D:
        numres = 2*self.numpnl
        Amat = zeros((numres, numres))
        Bmat = Vector2D.zeros(numres)
        j = 0
        for pnt in self.pnts:
            tup = pnt.get_ind_lhs_rhs()
            for ind, lhs, rhs in zip(*tup):
                Amat[j, ind] = lhs
                Bmat[j] = rhs
                j += 1
        d2r = Bmat.solve(Amat)
        self._d2r = Vector2D(d2r.x.ravel(), d2r.y.ravel())

    @property
    def d2r(self) -> Vector2D:
        if self._d2r is None:
            self.calc_d2r()
        return self._d2r

    @property
    def dr(self) -> Vector2D:
        if self._dr is None:
            self._dr = Vector2D.zeros(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._dr[inda] = pnl.dra
                indb = pnl.ind[1]
                self._dr[indb] = pnl.drb
        return self._dr

    @property
    def r(self) -> Vector2D:
        if self._r is None:
            self._r = Vector2D.zeros(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._r[inda] = pnl.pnta
                indb = pnl.ind[1]
                self._r[indb] = pnl.pntb
        return self._r

    @property
    def s(self) -> 'NDArray':
        if self._s is None:
            self._s = zeros(2*self.numpnl)
            scur = 0.0
            for i in range(self.numpnl):
                self._s[2*i] = scur
                self._s[2*i+1] = scur + self.pnls[i].length
                scur = self._s[2*i+1]
        return self._s

    @property
    def k(self) -> 'NDArray':
        if self._k is None:
            self._k = zeros(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._k[inda] = pnl.ka
                indb = pnl.ind[1]
                self._k[indb] = pnl.kb
        return self._k

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self.s[-1]
        return self._length

    def spline_length(self, num: int=5) -> 'NDArray':
        u"""This function interpolates the spline length with a float of points per piece."""

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

    def spline_points(self, num: int=5) -> Vector2D:
        u"""This function interpolates the spline with a float of points per piece."""

        numpnt = self.numpnl*num + 1
        r = Vector2D.zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            r[slc] = pnl.ratio_point_interpolate(ratio)

        r[-1] = pnl.pntb

        return r

    def spline_gradient(self, num: int=5) -> Vector2D:
        u"""This function interpolates the gradient of the spline with a float of points per piece."""

        numpnt = self.numpnl*num + 1
        dr = Vector2D.zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            dr[slc] = pnl.ratio_gradient_interpolate(ratio)

        dr[-1] = pnl.drb

        return dr

    def spline_curvature(self, num: int=1) -> Vector2D:
        u"""This function interpolates the curvature of the spline."""

        numpnt = self.numpnl*num + 1
        d2r = Vector2D.zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            d2r[slc] = pnl.ratio_curvature_interpolate(ratio)

        d2r[-1] = pnl.d2rb

        return d2r

    def spline_inverse_radius(self, num: int=1) -> Vector2D:
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

    def spline_points_ratio(self, ratio: 'NDArray') -> Vector2D:
        s = ratio*self.length
        num = s.size
        pnts = Vector2D.zeros(s.shape)
        k = 0
        for pnl in self.pnls:
            for j in range(k, num):
                ratio = pnl.ratio_s(s[j])
                if ratio >= 0.0 and ratio <= 1.0:
                    pnts[j] = pnl.ratio_point_interpolate(ratio)
                if ratio > 1.0:
                    k = j
                    break
        return pnts

    def spline_gradient_ratio(self, ratio: 'NDArray') -> Vector2D:
        s = ratio*self.length
        num = s.size
        grds = Vector2D.zeros(s.shape)
        k = 0
        for pnl in self.pnls:
            for j in range(k, num):
                ratio = pnl.ratio_s(s[j])
                if ratio >= 0.0 and ratio <= 1.0:
                    grds[j] = pnl.ratio_gradient_interpolate(ratio)
                if ratio > 1.0:
                    k = j
                    break
        return grds

    def spline_curvature_ratio(self, ratio: 'NDArray') -> Vector2D:
        s = ratio*self.length
        num = s.size
        crvs = Vector2D.zeros(s.shape)
        k = 0
        for pnl in self.pnls:
            for j in range(k, num):
                ratio = pnl.ratio_s(s[j])
                if ratio >= 0.0 and ratio <= 1.0:
                    crvs[j] = pnl.ratio_curvature_interpolate(ratio)
                if ratio > 1.0:
                    k = j
                    break
        return crvs

    def spline_inverse_radius_ratio(self, ratio: 'NDArray') -> 'NDArray':
        s = ratio*self.length
        num = s.size
        ks = zeros(s.shape)
        k = 0
        for pnl in self.pnls:
            for j in range(k, num):
                ratio = pnl.ratio_s(s[j])
                if ratio >= 0.0 and ratio <= 1.0:
                    ks[j] = pnl.ratio_inverse_radius_interpolate(ratio)
                if ratio > 1.0:
                    k = j
                    break
        return ks

    def split_at_index(self, index: int) -> tuple['Spline2D', 'Spline2D']:
        u"""This function splits the spline at the given index."""
        if index < 0 or index >= self.numpnt:
            raise ValueError('Index out of range.')
        pnta = self.pnts[0]
        pntb = self.pnts[index]
        if self.closed:
            pntc = pnta
        else:
            pntc = self.pnts[-1]
        pnts1 = self.pnts[:index+1]
        pnls1 = self.pnls[:index]
        tgt1a = pnta.pnlb.dra
        tgt1b = pntb.pnla.drb
        pnts2 = self.pnts[index:]
        pnls2 = self.pnls[index:]
        tgt2a = pntb.pnlb.dra
        tgt2b = pntc.pnla.drb
        spline1 = Spline2D([Vector2D(pnt1.x, pnt1.y) for pnt1 in pnts1],
                           tanA=tgt1a, tanB=tgt1b)
        for i, pnt in enumerate(pnts1):
            spline1.pnts[i].c2a0 = pnt.c2a0
            spline1.pnts[i].c2b0 = pnt.c2b0
        # for i, pnl in enumerate(pnls1):
        #     if pnl.straight:
        #         spline1.pnls[i].set_straight_edge()
        spline2 = Spline2D([Vector2D(pnt2.x, pnt2.y) for pnt2 in pnts2],
                           tanA=tgt2a, tanB=tgt2b)
        for i, pnt in enumerate(pnts2):
            spline2.pnts[i].c2a0 = pnt.c2a0
            spline2.pnts[i].c2b0 = pnt.c2b0
        # for i, pnl in enumerate(pnls2):
        #     if pnl.straight:
        #         spline2.pnls[i].set_straight_edge()
        return spline1, spline2

    def __repr__(self) -> str:
        return '<Spline2D>'
