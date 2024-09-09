from typing import TYPE_CHECKING, List, Optional, Tuple

from matplotlib.pyplot import figure
from numpy import asarray, cumsum, sqrt, square, zeros
from numpy.linalg import solve

from ..tools import cubic_roots
from .line2d import Line2D
from .point2d import Point2D
from .vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .infiniteline2d import InfiniteLine2D

class CubicSpline2D():
    u"""This class stores a 2D parametric cubic spline."""
    pnts: List['Point2D'] = None
    clsd: 'bool' = False
    tanA: Optional['Vector2D'] = None
    tanB: Optional['Vector2D'] = None
    _numpnt: int = None
    _pnls: List['SplineLine2D'] = None
    _numpnl: int = None
    _d2r: List['Vector2D'] = None
    _dr: List['Vector2D'] = None
    _r: List['Vector2D'] = None

    def __init__(self, pnts: List['Line2D'], clsd: 'bool'=False,
                 tanA: 'Vector2D'=None, tanB: 'Vector2D'=None) -> None:
        u"""This function initialises the object."""
        self.pnts = pnts
        self.clsd = clsd
        self.tanA = tanA
        self.tanB = tanB

    @property
    def numpnt(self) -> int:
        if self._numpnt is None:
            self._numpnt = len(self.pnts)
        return self._numpnt

    @property
    def pnls(self) -> List['SplineLine2D']:
        if self._pnls is None:
            if self.clsd:
                indas = [i for i in range(self.numpnt)]
                indbs = [i+1 for i in range(self.numpnt)]
                indbs[-1] = 0
                numpnl = self.numpnt
            else:
                indas = [i for i in range(self.numpnt-1)]
                indbs = [i+1 for i in range(self.numpnt-1)]
                numpnl = self.numpnt-1
            self._pnls = []
            for i in range(numpnl):
                inda = indas[i]
                indb = indbs[i]
                self._pnls.append(SplineLine2D(self.pnts[inda], self.pnts[indb]))
        return self._pnls

    @property
    def numpnl(self) -> int:
        if self._numpnl is None:
            self._numpnl = len(self.pnls)
        return self._numpnl

    @property
    def d2r(self) -> List['Vector2D']:
        if self._d2r is None:
            if self.clsd:
                self._d2r = self.calc_d2r_closed()
            else:
                self._d2r = self.calc_d2r_open(self.tanA, self.tanB)
        return self._d2r

    def set_panel_d2r(self):
        for i, pnl in enumerate(self.pnls):
            ia, ib = i, i+1
            if ib == self.numpnt and self.clsd:
                ib = 0
            pnl.set_d2r(self.d2r[ia], self.d2r[ib])

    @property
    def dr(self) -> List['Vector2D']:
        if self._dr is None:
            if self.clsd:
                self._dr = self.calc_dr_closed()
            else:
                self._dr = self.calc_dr_open()
        return self._dr

    @property
    def r(self) -> List['Vector2D']:
        if self._r is None:
            self._r = self.calc_r()
        return self._dr

    def calc_d2r_open(self, tanA: 'Vector2D'=None,
                      tanB: 'Vector2D'=None) -> List['Vector2D']:
        u"""This function calculates the curvature of an open ended spline."""
        pnl_dx = [pnl.vec.x/pnl.length for pnl in self.pnls]
        pnl_dy = [pnl.vec.y/pnl.length for pnl in self.pnls]
        del_dx = [0.0]*self.numpnt
        del_dy = [0.0]*self.numpnt
        if tanA is not None:
            dxA = tanA.x
            dyA = tanA.y
            del_dx[0] = pnl_dx[0]-dxA
            del_dy[0] = pnl_dy[0]-dyA
        for i in range(1, self.numpnt-1):
            del_dx[i] = pnl_dx[i]-pnl_dx[i-1]
            del_dy[i] = pnl_dy[i]-pnl_dy[i-1]
        if tanB is not None:
            dxB = tanB.x
            dyB = tanB.y
            del_dx[-1] = dxB-pnl_dx[-1]
            del_dy[-1] = dyB-pnl_dy[-1]
        a = [0.0]*self.numpnt
        b = [1.0]*self.numpnt
        c = [0.0]*self.numpnt
        rx = [0.0]*self.numpnt
        ry = [0.0]*self.numpnt
        if tanA is not None:
            sB = self.pnls[0].length
            b[0] = sB/3
            c[0] = sB/6
            rx[0] = del_dx[0]
            ry[0] = del_dy[0]
        for i in range(1, self.numpnt-1):
            sA = self.pnls[i-1].length
            sB = self.pnls[i].length
            a[i] = sA/6
            b[i] = (sA+sB)/3
            c[i] = sB/6
            rx[i] = del_dx[i]
            ry[i] = del_dy[i]
        if tanB is not None:
            sA = self.pnls[-1].length
            a[-1] = sA/6
            b[-1] = sA/3
            rx[-1] = del_dx[-1]
            ry[-1] = del_dy[-1]
        Γ = [0.0]*self.numpnt
        d2x = [0.0]*self.numpnt
        d2y = [0.0]*self.numpnt
        β = b[0]
        d2x[0] = rx[0]/β
        d2y[0] = ry[0]/β
        for i in range(1, self.numpnt):
            Γ[i] = c[i-1]/β
            β = b[i] - a[i]*Γ[i]
            d2x[i] = (rx[i]-a[i]*d2x[i-1])/β
            d2y[i] = (ry[i]-a[i]*d2y[i-1])/β
        for i in range(self.numpnt-2, -1, -1):
            d2x[i] -= Γ[i+1]*d2x[i+1]
            d2y[i] -= Γ[i+1]*d2y[i+1]
        d2r = [Vector2D(d2x[i], d2y[i]) for i in range(self.numpnt)]
        return d2r

    def calc_d2r_closed(self) -> List['Vector2D']:
        u"""This function calculates the curvature of a closed spline."""
        n = self.numpnt
        inda = [i-1 for i in range(n)]
        indb = [i for i in range(n)]
        inda[0] = n-1
        pnl_dx = [pnl.vec.x/pnl.length for pnl in self.pnls]
        pnl_dy = [pnl.vec.y/pnl.length for pnl in self.pnls]
        del_dx = [0.0]*n
        del_dy = [0.0]*n
        for i in range(n):
            del_dx[i] = pnl_dx[indb[i]]-pnl_dx[inda[i]]
            del_dy[i] = pnl_dy[indb[i]]-pnl_dy[inda[i]]
        A = zeros((n, n))
        B = zeros((n, 2))
        for i in range(n):
            sA = self.pnls[inda[i]].length
            sB = self.pnls[indb[i]].length
            if i-1 < 0:
                A[i, n-1] = sA/6
            else:
                A[i, i-1] = sA/6
            A[i, i] = (sA+sB)/3
            if i+1 > n-1:
                A[i, 0] = sB/6
            else:
                A[i,i+1] = sB/6
            B[i, 0] = del_dx[i]
            B[i, 1] = del_dy[i]
        X = solve(A, B)
        d2x = [X[i, 0] for i in range(n)]
        d2y = [X[i, 1] for i in range(n)]
        d2r = [Vector2D(d2x[i], d2y[i]) for i in range(self.numpnt)]
        return d2r

    def calc_dr_open(self) -> List['Vector2D']:
        u"""This function calculates the gradient of an open ended spline."""
        dx = []
        dy = []
        for i in range(self.numpnl):
            xA = self.pnts[i].x
            xB = self.pnts[i+1].x
            d2xA = self.d2r[i].x
            d2xB = self.d2r[i+1].x
            yA = self.pnts[i].y
            yB = self.pnts[i+1].y
            d2yA = self.d2r[i].y
            d2yB = self.d2r[i+1].y
            sP = self.pnls[i].length
            dxA = (xB-xA)/sP-sP/3*d2xA-sP/6*d2xB
            dyA = (yB-yA)/sP-sP/3*d2yA-sP/6*d2yB
            dx.append(dxA)
            dy.append(dyA)
        dxB = (xB-xA)/sP+sP/6*d2xA+sP/3*d2xB
        dyB = (yB-yA)/sP+sP/6*d2yA+sP/3*d2yB
        dx.append(dxB)
        dy.append(dyB)
        dr = [Vector2D(dx[i], dy[i]) for i in range(self.numpnt)]
        return dr

    def calc_dr_closed(self) -> List['Vector2D']:
        u"""This function calculates the gradient of a closed spline."""
        n = self.numpnt
        inda = [i for i in range(n)]
        indb = [i+1 for i in range(n)]
        indb[-1] = 0
        dx = []
        dy = []
        for i in range(self.numpnl):
            ia = inda[i]
            ib = indb[i]
            xA = self.pnts[ia].x
            xB = self.pnts[ib].x
            d2xA = self.d2r[ia].x
            d2xB = self.d2r[ib].x
            yA = self.pnts[ia].y
            yB = self.pnts[ib].y
            d2yA = self.d2r[ia].y
            d2yB = self.d2r[ib].y
            sP = self.pnls[i].length
            dxA = (xB-xA)/sP-sP/3*d2xA-sP/6*d2xB
            dyA = (yB-yA)/sP-sP/3*d2yA-sP/6*d2yB
            dx.append(dxA)
            dy.append(dyA)
        dr = [Vector2D(dx[i], dy[i]) for i in range(self.numpnt)]
        return dr

    def calc_r(self) -> List[float]:
        u"""This function calculates the radius of curvature of the spline."""
        r = []
        for i in range(self.numpnt):
            dx = self.dr[i].x
            dy = self.dr[i].y
            d2x = self.d2r[i].x
            d2y = self.d2r[i].y
            k = (dx*d2y - dy*d2x)/(dx**2 + dy**2)**1.5
            if k == 0.:
                r.append(float('inf'))
            else:
                r.append(1/k)
        return r

    def spline_points(self, num: int) -> Tuple[List[float], List[float]]:
        u"""This function interpolates the spline with a number of points."""
        x = []
        y = []
        self.set_panel_d2r()
        for pnl in self.pnls:
            for i in range(num):
                s = float(i*pnl.length/num)
                xi, yi = pnl.interpolate_point(s)
                x.append(xi)
                y.append(yi)
        if self.clsd:
            x.append(self.pnts[0].x)
            y.append(self.pnts[0].y)
        else:
            x.append(self.pnts[-1].x)
            y.append(self.pnts[-1].y)
        return x, y

    def spline_gradient(self, num: int) -> Tuple[List[float], List[float]]:
        u"""This function interpolates the gradient of the spline."""
        dx = []
        dy = []
        self.set_panel_d2r()
        for pnl in self.pnls:
            for i in range(num):
                s = float(i*pnl.length/num)
                dxi, dyi = pnl.interpolate_gradient(s)
                dx.append(dxi)
                dy.append(dyi)
        if self.clsd:
            dx.append(self.dr[0].x)
            dy.append(self.dr[0].y)
        else:
            dx.append(self.dr[-1].x)
            dy.append(self.dr[-1].y)
        return dx, dy

    def spline_curvature(self, num: int) -> Tuple[List[float], List[float]]:
        u"""This function interpolates the curvature of the spline."""
        d2x = []
        d2y = []
        self.set_panel_d2r()
        for pnl in self.pnls:
            for i in range(num):
                s = float(i*pnl.length/num)
                d2xi, d2yi = pnl.interpolate_curvature(s)
                d2x.append(d2xi)
                d2y.append(d2yi)
        if self.clsd:
            d2x.append(self.d2r[0].x)
            d2y.append(self.d2r[0].y)
        else:
            d2x.append(self.d2r[-1].x)
            d2y.append(self.d2r[-1].y)
        return d2x, d2y

    def line_intersection(self, line: 'InfiniteLine2D',
                          all_roots=False):
        edct = {}
        xP = line.pnt.x
        yP = line.pnt.y
        dxdl = line.uvec.x
        dydl = line.uvec.y
        for i in range(self.numpnl):
            ia = i
            xA = self.pnts[ia].x
            yA = self.pnts[ia].y
            d2xA = self.d2r[ia].x
            d2yA = self.d2r[ia].y
            ib = i+1
            if ib == self.numpnt:
                ib = 0
            xB = self.pnts[ib].x
            yB = self.pnts[ib].y
            d2xB = self.d2r[ib].x
            d2yB = self.d2r[ib].y
            sP = self.pnls[i].length
            a = -d2xA*dydl/(6*sP) + d2xB*dydl/(6*sP)
            a += (d2yA*dxdl/(6*sP) - d2yB*dxdl/(6*sP))
            b = d2xA*dydl/2 - d2yA*dxdl/2
            c = -d2xA*dydl*sP/3 - d2xB*dydl*sP/6 + d2yA*dxdl*sP/3 + d2yB*dxdl*sP/6
            c += (dxdl*yA/sP - dxdl*yB/sP - dydl*xA/sP + dydl*xB/sP)
            d = -dxdl*yA + dxdl*yP + dydl*xA - dydl*xP
            s1, s2, s3 = cubic_roots(a, b, c, d)
            if all_roots:
                edct[i] = (s1/sP, s2/sP, s3/sP)
                continue
            elst = []
            if isinstance(s1, float):
                e1 = s1/sP
                if e1 >= 0.0-sP/1000 and e1 <= sP+sP/1000:
                    elst.append(e1)
            if isinstance(s2, float):
                e2 = s2/sP
                if e2 >= 0.0-sP/1000 and e2 <= sP+sP/1000:
                    elst.append(e2)
            if isinstance(s3, float):
                e3 = s3/sP
                if e3 >= -0.000001 and e3 < 1.000001:
                    elst.append(e3)
            if len(elst) > 0:
                edct[i] = elst
        return edct

    def line_intersection_points(self, line: 'InfiniteLine2D',
                                 all_roots=False) -> List['Point2D']:
        edct = self.line_intersection(line, all_roots=all_roots)
        pnts = []
        for i in edct:
            ia = i
            xA = self.pnts[ia].x
            yA = self.pnts[ia].y
            d2xA = self.d2r[ia].x
            d2yA = self.d2r[ia].y
            ib = i+1
            if ib == self.numpnt:
                ib = 0
            xB = self.pnts[ib].x
            yB = self.pnts[ib].y
            d2xB = self.d2r[ib].x
            d2yB = self.d2r[ib].y
            pnl: 'SplineLine2D' = self.pnls[i]
            sP = pnl.length
            elst = edct[i]
            for e in elst:
                s = e*sP
                if isinstance(s, float):
                    if s >= 0.0 and s < sP:
                        A = (sP - s)/sP
                        B = s/sP
                        C = sP**2/6*(A**3 - A)
                        D = sP**2/6*(B**3 - B)
                        x = A*xA + B*xB + C*d2xA + D*d2xB
                        y = A*yA + B*yB + C*d2yA + D*d2yB
                        pnts.append(Point2D(x, y))
        return pnts

    def scatter(self, ax=None, label=False):
        u"""This function plots the points of the spline."""
        if ax is None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x = []
        y = []
        for i in range(self.numpnt):
            x.append(self.pnts[i].x)
            y.append(self.pnts[i].y)
        ax.scatter(x,y)
        if label:
            for i in range(self.numpnt):
                ax.text(x[i], y[i], i)
        return ax

    def plot_spline(self, num=1, ax=None):
        u"""This function plots the spline using the interpolated points."""
        if ax is None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x, y = self.spline_points(num)
        ax.plot(x, y)
        return ax

    def arc_length(self, num=1):
        u"""This function calculates the arc length of the spline."""
        s = []
        sc = 0.0
        for i in range(self.numpnl):
            sP = self.pnls[i].length
            for j in range(num):
                sj = j*sP/num
                s.append(sc+sj)
            sc += sP
        s.append(sc)
        return s

    def interpolate_spline_points(self, slst: list):
        u"""This function interpolates the spline points based on arc length."""
        s = self.arc_length()
        i = 0
        ploc = []
        sloc = []
        for si in slst:
            found = False
            if not found:
                for j in range(i, self.numpnl):
                    if si >= s[j] and si <= s[j+1]:
                        ploc.append(j)
                        sloc.append(si-s[j])
                        i = j
                        break
            if not found:
                for j in range(0, i):
                    if si >= s[j] and si <= s[j+1]:
                        ploc.append(j)
                        sloc.append(si-s[j])
                        i = j
                        break
        x, y = [], []
        self.set_panel_d2r()
        for i, si in enumerate(slst):
            s = sloc[i]
            j = ploc[i]
            pnl: 'SplineLine2D' = self.pnls[j]
            xi, yi = pnl.interpolate_point(s)
            x.append(xi)
            y.append(yi)
        return x, y

    def interpolate_spline_gradient(self, slst: list):
        u"""This function interpolates the spline gradient based on arc length."""
        s = self.arc_length()
        i = 0
        ploc = []
        sloc = []
        for si in slst:
            found = False
            if not found:
                for j in range(i, self.numpnl):
                    if si >= s[j] and si <= s[j+1]:
                        ploc.append(j)
                        sloc.append(si-s[j])
                        i = j
                        break
            if not found:
                for j in range(0, i):
                    if si >= s[j] and si <= s[j+1]:
                        ploc.append(j)
                        sloc.append(si-s[j])
                        i = j
                        break
        dx, dy = [], []
        self.set_panel_d2r()
        for i, si in enumerate(slst):
            s = sloc[i]
            j = ploc[i]
            pnl: 'SplineLine2D' = self.pnls[j]
            dxi, dyi = pnl.interpolate_gradient(s)
            dx.append(dxi)
            dy.append(dyi)
        return dx, dy

    def interpolate_spline_curvature(self, slst: list):
        u"""This function interpolates the spline curvature based on arc length."""
        s = self.arc_length()
        i = 0
        ploc = []
        sloc = []
        for si in slst:
            found = False
            if not found:
                for j in range(i, self.numpnl):
                    if si >= s[j] and si <= s[j+1]:
                        ploc.append(j)
                        sloc.append(si-s[j])
                        i = j
                        break
            if not found:
                for j in range(0, i):
                    if si >= s[j] and si <= s[j+1]:
                        ploc.append(j)
                        sloc.append(si-s[j])
                        i = j
                        break
        d2x, d2y = [], []
        self.set_panel_d2r()
        for i, si in enumerate(slst):
            s = sloc[i]
            j = ploc[i]
            pnl: 'SplineLine2D' = self.pnls[j]
            d2xi, d2yi = pnl.interpolate_curvature(s)
            d2x.append(d2xi)
            d2y.append(d2yi)
        return d2x, d2y

    def plot_gradient(self, ax=None, num=5):
        u"""This function plots the gradient of the spline."""
        if ax is None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Gradient')
        s = self.arc_length(num=num)
        dx, dy = self.spline_gradient(num=num)
        ax.plot(s, dx)
        ax.plot(s, dy)
        return ax

    def quiver_tangent(self, ax=None):
        u"""This function quiver plots the tangent of the spline."""
        if ax is None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x = [self.pnts[i].x for i in range(self.numpnt)]
        y = [self.pnts[i].y for i in range(self.numpnt)]
        dx = [self.dr[i].x for i in range(self.numpnt)]
        dy = [self.dr[i].y for i in range(self.numpnt)]
        ax.quiver(x, y, dx, dy)
        return ax

    def plot_curvature(self, ax=None, num=1):
        u"""This function plots the curvature of the spline."""
        if ax is None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Curvature')
        s = self.arc_length(num=num)
        d2x, d2y = self.spline_curvature(num=num)
        ax.plot(s, d2x)
        ax.plot(s, d2y)
        return ax

    def plot_inverse_radius(self, ax=None, num=5):
        u"""This function plots the inverse radius of curvature of the spline."""
        if ax is None:
            fig = figure()
            ax = fig.gca()
            ax.grid(True)
            ax.set_title('Inverse Radius of Curvature')
        s = self.arc_length(num=num)
        d2x, d2y = self.spline_curvature(num=num)
        dx, dy = self.spline_gradient(num=num)
        nums = len(s)
        k = [(dx[i]*d2y[i]-dy[i]*d2x[i])/(dx[i]**2+dy[i]**2)**1.5 for i in range(nums)]
        ax.plot(s, k)
        return ax

    def quiver_normal(self, ax=None):
        u"""This function quiver plots the normal of the spline."""
        if ax is None:
            fig = figure()
            ax = fig.gca()
            ax.set_aspect('equal')
            ax.grid(True)
        x = [self.pnts[i].x for i in range(self.numpnt)]
        y = [self.pnts[i].y for i in range(self.numpnt)]
        dx = [-self.dr[i].y for i in range(self.numpnt)]
        dy = [self.dr[i].x for i in range(self.numpnt)]
        ax.quiver(x, y, dx, dy)
        return ax

    def arc_length_approximation(self, num: int) -> 'NDArray':
        x, y = self.spline_points(num)
        x = asarray(x)
        y = asarray(y)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        ds = sqrt(square(dx) + square(dy))
        s = cumsum(ds)
        svals = [0.0]
        for i in range(self.numpnl):
            svals.append(s[(i+1)*num-1])
        return asarray(svals)

class SplineLine2D(Line2D):
    _dra: 'Vector2D' = None
    _drb: 'Vector2D' = None
    d2ra: 'Vector2D' = None
    d2rb: 'Vector2D' = None

    def __init__(self, pnta: 'Point2D', pntb: 'Point2D') -> None:
        super(SplineLine2D, self).__init__(pnta, pntb)

    @property
    def xa(self) -> float:
        return self.pnta.x

    @property
    def ya(self) -> float:
        return self.pnta.y

    @property
    def xb(self) -> float:
        return self.pntb.x

    @property
    def yb(self) -> float:
        return self.pntb.y

    @property
    def dra(self) -> 'Vector2D':
        if self._dra is None:
            E = 2*self.length/6
            F = -1*self.length/6
            dxa = self.uvec.x + F*self.d2xb - E*self.d2xa
            dya = self.uvec.y + F*self.d2yb - E*self.d2ya
            self._dra = Vector2D(dxa, dya)
        return self._dra

    @property
    def drb(self) -> 'Vector2D':
        if self._drb is None:
            E = -1*self.length/6
            F = 2*self.length/6
            dxb = self.uvec.x + F*self.d2xb - E*self.d2xa
            dyb = self.uvec.y + F*self.d2yb - E*self.d2ya
            self._drb = Vector2D(dxb, dyb)
        return self._drb

    def set_d2r(self, d2ra: 'Vector2D', d2rb: 'Vector2D'):
        self.d2ra = d2ra
        self.d2rb = d2rb

    @property
    def d2xa(self) -> float:
        return self.d2ra.x

    @property
    def d2ya(self) -> float:
        return self.d2ra.y

    @property
    def d2xb(self) -> float:
        return self.d2rb.x

    @property
    def d2yb(self) -> float:
        return self.d2rb.y

    def interpolate_point(self, s: float) -> Tuple[float, float]:
        sP = self.length
        A = (sP-s)/sP
        B = s/sP
        C = (A**3-A)*sP**2/6
        D = (B**3-B)*sP**2/6
        x = A*self.xa + B*self.xb + C*self.d2xa + D*self.d2xb
        y = A*self.ya + B*self.yb + C*self.d2ya + D*self.d2yb
        return x, y

    def interpolate_gradient(self, s: float) -> Tuple[float, float]:
        sP = self.length
        A = (sP-s)/sP
        B = s/sP
        E = 3*A**2-1
        F = 3*B**2-1
        dx = (self.xb-self.xa)/sP + (F*self.d2xb-E*self.d2xa)*sP/6
        dy = (self.yb-self.ya)/sP + (F*self.d2yb-E*self.d2ya)*sP/6
        return dx, dy

    def interpolate_curvature(self, s: float) -> Tuple[float, float]:
        sP = self.length
        A = (sP-s)/sP
        B = s/sP
        d2x = A*self.d2xa + B*self.d2xb
        d2y = A*self.d2ya + B*self.d2yb
        return d2x, d2y
