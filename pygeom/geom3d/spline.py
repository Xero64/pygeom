from typing import TYPE_CHECKING

from matplotlib.axes import Axes
from matplotlib.pyplot import figure
from numpy import argwhere, asarray, concatenate, linspace, logical_and, zeros

from .vector import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplinePoint(Vector):
    pnla: 'SplinePanel' = None
    pnlb: 'SplinePanel' = None
    tan: Vector = None
    nrm: Vector = None
    _endpnta: bool = None
    _endpntb: bool = None
    _c1: bool = None
    _c2: bool = None
    _lhs: list['NDArray'] = None
    _rhs: list['NDArray'] = None
    _ind: list[tuple[int, ...]] = None
    c2a0: bool = None
    c2b0: bool = None

    def __init__(self, x: float, y: float, z: float) -> None:
        super().__init__(x, y, z)
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
                                       list[Vector]]:
        ind = []
        lhs = []
        rhs = []

        if self.endpnta:
            if self.c2b0 or (self.tan is None and self.nrm is None):
                ind.append(self.pnlb.ind)
                lhs.append(self.pnlb.lhs_d2ra)
                rhs.append(Vector(0.0, 0.0, 0.0))
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
                rhs.append(Vector(0.0, 0.0, 0.0))
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
                    rhs.append(Vector(0.0, 0.0, 0.0))
                else:
                    if self.c2a0:
                        ind.append(self.pnla.ind)
                        lhs.append(self.pnla.lhs_d2rb)
                        rhs.append(Vector(0.0, 0.0, 0.0))
                    elif self.c2b0:
                        ind.append(self.pnlb.ind)
                        lhs.append(self.pnlb.lhs_d2ra)
                        rhs.append(Vector(0.0, 0.0, 0.0))
            else:
                ind.append(self.pnla.ind)
                lhs.append(self.pnla.lhs_d2rb)
                rhs.append(Vector(0.0, 0.0, 0.0))
                ind.append(self.pnlb.ind)
                lhs.append(self.pnlb.lhs_d2ra)
                rhs.append(Vector(0.0, 0.0, 0.0))

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
        return '<SplinePoint>'

class SplinePanel():
    pnta: SplinePoint = None
    pntb: SplinePoint = None
    _vector: Vector = None
    _length: float = None
    _direc: Vector = None
    _lhs_dra: 'NDArray' = None
    _lhs_drb: 'NDArray' = None
    _lhs_d2ra: 'NDArray' = None
    _lhs_d2rb: 'NDArray' = None
    ind: 'NDArray' = None
    spline: 'Spline' = None
    _straight: bool = False
    _sa: float = None
    _sb: float = None
    _d2ra: Vector = None
    _d2rb: Vector = None
    _dra: Vector = None
    _drb: Vector = None
    _ka: Vector = None
    _kb: Vector = None

    def __init__(self, pnta: SplinePoint, pntb: SplinePoint) -> None:
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
    def vector(self) -> Vector:
        if self._vector is None:
            self._vector = self.pntb - self.pnta
        return self._vector

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self.vector.return_magnitude()
        return self._length

    @property
    def direc(self) -> Vector:
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
    def rhs_dr(self) -> Vector:
        return self.direc

    @property
    def d2ra(self) -> Vector:
        if self._d2ra is None:
            inda = self.ind[0]
            self._d2ra = self.spline.d2r[inda]
        return self._d2ra

    @property
    def d2rb(self) -> Vector:
        if self._d2rb is None:
            indb = self.ind[1]
            self._d2rb = self.spline.d2r[indb]
        return self._d2rb

    @property
    def dra(self) -> Vector:
        if self._dra is None:
            E = -self.length/3
            F = -self.length/6
            x = self.direc.x + E*self.d2ra.x + F*self.d2rb.x
            y = self.direc.y + E*self.d2ra.y + F*self.d2rb.y
            z = self.direc.z + E*self.d2ra.z + F*self.d2rb.z
            self._dra = Vector(x, y, z)
        return self._dra

    @property
    def drb(self) -> Vector:
        if self._drb is None:
            E = self.length/6
            F = self.length/3
            x = self.direc.x + E*self.d2ra.x + F*self.d2rb.x
            y = self.direc.y + E*self.d2ra.y + F*self.d2rb.y
            z = self.direc.z + E*self.d2ra.z + F*self.d2rb.z
            self._drb = Vector(x, y, z)
        return self._drb

    @property
    def ra(self) -> Vector:
        return self.pnta

    @property
    def rb(self) -> Vector:
        return self.pntb

    @property
    def sa(self) -> float:
        if self._sa is None:
            inda = self.ind[0]
            self._sa = self.spline.s[inda]
        return self._sa

    @property
    def sb(self) -> Vector:
        if self._sb is None:
            indb = self.ind[1]
            self._sb = self.spline.s[indb]
        return self._sb

    @property
    def ka(self) -> Vector:
        if self._ka is None:
            mdra3 = self.dra.return_magnitude()**3
            self._ka = self.dra.cross(self.d2ra)/mdra3
        return self._ka

    @property
    def kb(self) -> Vector:
        if self._kb is None:
            mdrb3 = self.drb.return_magnitude()**3
            self._kb = self.drb.cross(self.d2rb)/mdrb3
        return self._kb

    def ratio_length_interpolate(self, ratio: 'NDArray') -> 'NDArray':
        A = 1.0 - ratio
        B = ratio
        return self.sa*A + self.sb*B

    def ratio_point_interpolate(self, ratio: 'NDArray') -> Vector:
        A = 1.0 - ratio
        B = ratio
        C = (A**3 - A)*self.length**2/6
        D = (B**3 - B)*self.length**2/6
        x = A*self.pnta.x + B*self.pntb.x + C*self.d2ra.x + D*self.d2rb.x
        y = A*self.pnta.y + B*self.pntb.y + C*self.d2ra.y + D*self.d2rb.y
        z = A*self.pnta.z + B*self.pntb.z + C*self.d2ra.z + D*self.d2rb.z
        return Vector(x, y, z)

    def ratio_gradient_interpolate(self, ratio: 'NDArray') -> Vector:
        A = 1.0 - ratio
        B = ratio
        E = (1 - 3*A**2)*self.length/6
        F = (3*B**2 - 1)*self.length/6
        x = self.direc.x + E*self.d2ra.x + F*self.d2rb.x
        y = self.direc.y + E*self.d2ra.y + F*self.d2rb.y
        z = self.direc.z + E*self.d2ra.z + F*self.d2rb.z
        return Vector(x, y, z)

    def ratio_curvature_interpolate(self, ratio: 'NDArray') -> Vector:
        A = 1.0 - ratio
        B = ratio
        x = A*self.d2ra.x + B*self.d2rb.x
        y = A*self.d2ra.y + B*self.d2rb.y
        z = A*self.d2ra.z + B*self.d2rb.z
        return Vector(x, y, z)

    def ratio_inverse_radius_interpolate(self, ratio: 'NDArray') -> Vector:
        dr = self.ratio_gradient_interpolate(ratio)
        d2r = self.ratio_curvature_interpolate(ratio)
        mdr3 = dr.return_magnitude()**3
        k = dr.cross(d2r)/mdr3
        return k

    def ratio_s(self, s: float) -> float:
        return (s - self.sa)/(self.sb - self.sa)

    def __repr__(self) -> str:
        return '<SplinePanel>'

class Spline():
    u"""This class stores a 3D parametric spline."""
    pnts: list[SplinePoint] = None
    closed: bool = False
    tanA: Vector | None = None
    tanB: Vector | None = None
    nrmA: Vector | None = None
    nrmB: Vector | None = None
    _numpnt: int = None
    _pnls: list[SplinePanel] = None
    _numpnl: int = None
    _d2r: Vector = None
    _dr: Vector = None
    _r: Vector = None
    _dS: 'NDArray' = None
    _S: 'NDArray' = None
    _s: 'NDArray' = None
    _k: Vector = None
    _length: float = None

    def __init__(self, pnts: list[Vector], closed: bool=False,
                 tanA: Vector=None, tanB: Vector=None,
                 nrmA: Vector=None, nrmB: Vector=None) -> None:
        if closed and pnts[0] == pnts[-1]:
            pnts = pnts[:-1]
        self.pnts = [SplinePoint(pnt.x, pnt.y, pnt.z) for pnt in pnts]
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
    def pnls(self) -> list[SplinePanel]:
        if self._pnls is None:
            self._pnls = []
            j = 0
            for i in range(self.numpnt-1):
                pnl = SplinePanel(self.pnts[i], self.pnts[i+1])
                pnl.ind = asarray([j, j+1], dtype=int)
                pnl.spline = self
                self._pnls.append(pnl)
                j += 2
            if self.closed:
                pnl = SplinePanel(self.pnts[-1], self.pnts[0])
                pnl.ind = asarray([j, j+1], dtype=int)
                pnl.spline = self
                self._pnls.append(pnl)
        return self._pnls

    @property
    def numpnl(self) -> int:
        if self._numpnl is None:
            self._numpnl = len(self.pnls)
        return self._numpnl

    def calc_d2r(self) -> Vector:
        numres = 2*self.numpnl
        Amat = zeros((numres, numres))
        Bmat = Vector.zeros((numres, 1))
        j = 0
        for pnt in self.pnts:
            tup = pnt.get_ind_lhs_rhs()
            for ind, lhs, rhs in zip(*tup):
                Amat[j, ind] = lhs
                Bmat[j, 0] = rhs
                j += 1
        d2r = Bmat.solve(Amat)
        self._d2r = Vector(d2r.x.ravel(), d2r.y.ravel(), d2r.z.ravel())

    @property
    def d2r(self) -> Vector:
        if self._d2r is None:
            self.calc_d2r()
        return self._d2r

    @property
    def dr(self) -> Vector:
        if self._dr is None:
            self._dr = Vector.zeros(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._dr[inda] = pnl.dra
                indb = pnl.ind[1]
                self._dr[indb] = pnl.drb
        return self._dr

    @property
    def r(self) -> Vector:
        if self._r is None:
            self._r = Vector.zeros(2*self.numpnl)
            for pnl in self.pnls:
                inda = pnl.ind[0]
                self._r[inda] = pnl.pnta
                indb = pnl.ind[1]
                self._r[indb] = pnl.pntb
        return self._r

    @property
    def dS(self) -> 'NDArray':
        if self._dS is None:
            self._dS = zeros(self.numpnl)
            for i in range(self.numpnl):
                self._dS[i] = self.pnls[i].length
        return self._dS

    @property
    def S(self) -> 'NDArray':
        if self._S is None:
            self._S = zeros(self.numpnt)
            self._S[1:] = self.dS.cumsum()
        return self._S

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
    def k(self) -> Vector:
        if self._k is None:
            self._k = Vector.zeros(2*self.numpnl)
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

    def spline_points(self, num: int=5) -> Vector:
        u"""This function interpolates the spline with a float of points per piece."""

        numpnt = self.numpnl*num + 1
        r = Vector.zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            r[slc] = pnl.ratio_point_interpolate(ratio)

        r[-1] = pnl.pntb

        return r

    def spline_gradient(self, num: int=5) -> Vector:
        u"""This function interpolates the gradient of the spline with a float of points per piece."""

        numpnt = self.numpnl*num + 1
        dr = Vector.zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            dr[slc] = pnl.ratio_gradient_interpolate(ratio)

        dr[-1] = pnl.drb

        return dr

    def spline_curvature(self, num: int=1) -> Vector:
        u"""This function interpolates the curvature of the spline."""

        numpnt = self.numpnl*num + 1
        d2r = Vector.zeros(numpnt)
        ratio = linspace(0.0, 1.0, num)

        for i, pnl in enumerate(self.pnls):
            if num == 1:
                slc = i
            else:
                slc = slice(i*num, (i+1)*num)
            d2r[slc] = pnl.ratio_curvature_interpolate(ratio)

        d2r[-1] = pnl.d2rb

        return d2r

    def spline_inverse_radius(self, num: int=1) -> Vector:
        u"""This function interpolates the inverse radius of the spline."""

        numpnt = self.numpnl*num + 1
        k = Vector.zeros(numpnt)
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
            ax: Axes = fig.add_subplot(projection='3d')
            ax.grid(True)
        ax.scatter(self.r.x, self.r.y, self.r.z)
        if label:
            for i in range(self.numpnt):
                ax.text(self.r.x[i], self.r.y[i], self.r.z[i], i)
        return ax

    def plot_spline(self, num=5, ax: Axes=None, plane: str=None, **kwargs) -> Axes:
        u"""This function plots the spline using the interpolated points."""
        if ax is None:
            fig = figure()
            if plane is None:
                ax: Axes = fig.add_subplot(projection='3d')
            else:
                ax: Axes = fig.add_subplot()
            ax.grid(True)
        r = self.spline_points(num)
        if plane is None:
            ax.plot(r.x, r.y, r.z, **kwargs)
        elif plane == 'xy':
            ax.plot(r.x, r.y, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'yx':
            ax.plot(r.y, r.x, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'xz':
            ax.plot(r.x, r.z, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'zx':
            ax.plot(r.z, r.x, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'yz':
            ax.plot(r.y, r.z, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'zy':
            ax.plot(r.z, r.y, **kwargs)
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
        ax.plot(s, dr.z, label='dr.z')
        return ax

    def quiver_tangent(self, ax: Axes=None, plane: str=None, **kwargs) -> Axes:
        u"""This function quiver plots the tangent of the spline."""
        if ax is None:
            fig = figure()
            if plane is None:
                ax: Axes = fig.add_subplot(projection='3d')
            else:
                ax: Axes = fig.add_subplot()
        x, y, z = self.r.x, self.r.y, self.r.z
        tgt = self.dr.to_unit()
        tgtx, tgty, tgtz = tgt.x, tgt.y, tgt.z
        if plane is None:
            ax.quiver(x, y, z, tgtx, tgty, tgtz, **kwargs)
        elif plane == 'xy':
            ax.quiver(x, y, tgtx, tgty, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'yx':
            ax.quiver(y, x, tgty, tgtx, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'xz':
            ax.quiver(x, z, tgtx, tgtz, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'zx':
            ax.quiver(z, x, tgtz, tgtx, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'yz':
            ax.quiver(y, z, tgty, tgtz, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'zy':
            ax.quiver(z, y, tgtz, tgty, **kwargs)
            ax.set_aspect('equal')
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
        ax.plot(s, d2r.z, label='d2r.z')
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
        ax.plot(s, k.x, label='k.x')
        ax.plot(s, k.y, label='k.y')
        ax.plot(s, k.z, label='k.z')
        return ax

    def quiver_normal(self, ax: Axes=None, plane: str=None, **kwargs) -> Axes:
        u"""This function quiver plots the normal of the spline."""
        if ax is None:
            fig = figure()
            ax: Axes = fig.add_subplot(projection='3d')
            ax.grid(True)
        x, y, z = self.r.x, self.r.y, self.r.z
        nrm = self.d2r.to_unit()
        nrmx, nrmy, nrmz = nrm.x, nrm.y, nrm.z
        if plane is None:
            ax.quiver(x, y, z, nrmx, nrmy, nrmz, **kwargs)
        elif plane == 'xy':
            ax.quiver(x, y, nrmx, nrmy, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'yx':
            ax.quiver(y, x, nrmy, nrmx, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'xz':
            ax.quiver(x, z, nrmx, nrmz, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'zx':
            ax.quiver(z, x, nrmz, nrmx, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'yz':
            ax.quiver(y, z, nrmy, nrmz, **kwargs)
            ax.set_aspect('equal')
        elif plane == 'zy':
            ax.quiver(z, y, nrmz, nrmy, **kwargs)
            ax.set_aspect('equal')
        return ax

    def spline_points_ratio(self, ratio: 'NDArray') -> Vector:
        s = ratio*self.length
        num = s.size
        pnts = Vector.zeros(s.shape)
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

    def spline_gradient_ratio(self, ratio: 'NDArray') -> Vector:
        s = ratio*self.length
        num = s.size
        grds = Vector.zeros(s.shape)
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

    def spline_curvature_ratio(self, ratio: 'NDArray') -> Vector:
        s = ratio*self.length
        num = s.size
        crvs = Vector.zeros(s.shape)
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

    def spline_inverse_radius_ratio(self, ratio: 'NDArray') -> Vector:
        s = ratio*self.length
        num = s.size
        ks = Vector.zeros(s.shape)
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

    def split_at_index(self, index: int) -> tuple['Spline', 'Spline']:
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
        tgt1a = pnta.pnlb.dra
        tgt1b = pntb.pnla.drb
        pnts2 = self.pnts[index:]
        tgt2a = pntb.pnlb.dra
        tgt2b = pntc.pnla.drb

        spline1 = Spline([Vector(pnt1.x, pnt1.y, pnt1.z) for pnt1 in pnts1],
                         tanA=tgt1a, tanB=tgt1b)
        for i, pnt in enumerate(pnts1):
            spline1.pnts[i].c2a0 = pnt.c2a0
            spline1.pnts[i].c2b0 = pnt.c2b0

        spline2 = Spline([Vector(pnt2.x, pnt2.y, pnt2.z) for pnt2 in pnts2],
                         tanA=tgt2a, tanB=tgt2b)
        for i, pnt in enumerate(pnts2):
            spline2.pnts[i].c2a0 = pnt.c2a0
            spline2.pnts[i].c2b0 = pnt.c2b0

        return spline1, spline2

    def spline_panel_ratio(self, ratio: float) -> list[SplinePanel]:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError('Ratio out of range.')
        s = ratio*self.length
        Sa = self.S[:-1]
        Sb = self.S[1:]
        chk = logical_and(s >= Sa, s < Sb)
        ind = argwhere(chk)
        return self.pnls[ind.item()]

    def split_at_ratio(self, ratio: float) -> tuple['Spline', 'Spline']:

        pnl = self.spline_panel_ratio(ratio)

        ind = self.pnls.index(pnl)
        pnt = self.spline_points_ratio(asarray([ratio]))[0]
        tgt = self.spline_gradient_ratio(asarray([ratio]))[0]

        pnts1 = self.pnts[:ind+1] + [pnt]
        tgt1a = self.dr[0].to_unit()
        tgt1b = tgt
        pnts2 = [pnt] + self.pnts[ind+1:]
        tgt2a = tgt
        tgt2b = self.dr[-1].to_unit()

        spline1 = Spline([Vector(pnt1.x, pnt1.y, pnt1.z) for pnt1 in pnts1],
                         tanA=tgt1a, tanB=tgt1b)

        spline2 = Spline([Vector(pnt2.x, pnt2.y, pnt2.z) for pnt2 in pnts2],
                         tanA=tgt2a, tanB=tgt2b)

        return spline1, spline2

    def __repr__(self) -> str:
        return '<Spline>'
