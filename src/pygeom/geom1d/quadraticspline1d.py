from typing import TYPE_CHECKING

from numpy import (absolute, asarray, cumsum, fill_diagonal, full, isnan,
                   logical_and, logical_or, ones_like, unique, zeros)

from ..tools.basis import knot_linspace
from ..tools.roots import quadratic_roots
from ..tools.solvers import quadratic_pspline_fit_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray


class QuadraticSpline1DSolver():
    u"""This class solves for the quadratic spline coefficients."""
    s: 'NDArray' = None
    _marr: 'NDArray' = None
    _sall: 'NDArray' = None
    _a2: 'NDArray' = None
    _b2: 'NDArray' = None
    _c2: 'NDArray' = None
    _a1: 'NDArray' = None
    _b1: 'NDArray' = None
    _c1: 'NDArray' = None
    _a0: 'NDArray' = None
    _b0: 'NDArray' = None
    _c0: 'NDArray' = None
    _harr: 'NDArray' = None
    _garr: 'NDArray' = None
    _iarr: 'NDArray' = None

    def __init__(self, s: 'NDArray') -> None:
        u"""This function initialises the object."""
        self.s = asarray(s)

    def validate(self) -> None:
        u"""This function validates the object."""
        if self.s.ndim != 1:
            raise ValueError('Input s must be a 1D ndarray.')

    def calculate(self):
        self._marr, params = quadratic_pspline_fit_solver(self.s, True)
        (self._sall,
         self._sa, self._sb, self._sc,
         self._a1, self._a2,
         self._b1, self._b2,
         self._c1, self._c2) = params

    @property
    def marr(self) -> 'NDArray':
        if self._marr is None:
            self.calculate()
        return self._marr

    @property
    def sall(self) -> 'NDArray':
        if self._sall is None:
            self.calculate()
        return self._sall

    @property
    def sa(self) -> 'NDArray':
        if self._sa is None:
            self.calculate()
        return self._sa

    @property
    def sb(self) -> 'NDArray':
        if self._sb is None:
            self.calculate()
        return self._sb

    @property
    def sc(self) -> 'NDArray':
        if self._sc is None:
            self.calculate()
        return self._sc

    @property
    def a2(self) -> 'NDArray':
        if self._a2 is None:
            self.calculate()
        return self._a2

    @property
    def b2(self) -> 'NDArray':
        if self._b2 is None:
            self.calculate()
        return self._b2

    @property
    def c2(self) -> 'NDArray':
        if self._c2 is None:
            self.calculate()
        return self._c2

    @property
    def a1(self) -> 'NDArray':
        if self._a1 is None:
            self.calculate()
        return self._a1

    @property
    def b1(self) -> 'NDArray':
        if self._b1 is None:
            self.calculate()
        return self._b1

    @property
    def c1(self) -> 'NDArray':
        if self._c1 is None:
            self.calculate()
        return self._c1

    @property
    def a0(self) -> 'NDArray':
        if self._a0 is None:
            self._a0 = self._a2 * self._sc * self._sb
        return self._a0

    @property
    def b0(self) -> 'NDArray':
        if self._b0 is None:
            self._b0 = self._b2 * self._sa * self._sc
        return self._b0

    @property
    def c0(self) -> 'NDArray':
        if self._c0 is None:
            self._c0 = self._c2 * self._sb * self._sa
        return self._c0

    def calculate_gradient_array(self) -> None:

        s = self.s
        sall = self._sall
        marr = self._marr
        a1 = self._a1
        a2 = self._a2
        b1 = self._b1
        b2 = self._b2
        c1 = self._c1
        c2 = self._c2

        sai = sall[0]
        sbi = sall[1:-1:2]
        sci = sall[-1]

        cdya_a = a1[0] + a2[0] * sai * 2.0
        cdya_b = b1[0] + b2[0] * sai * 2.0
        cdya_c = c1[0] + c2[0] * sai * 2.0

        cdyb_a = a1 + a2 * sbi * 2.0
        cdyb_b = b1 + b2 * sbi * 2.0
        cdyb_c = c1 + c2 * sbi * 2.0

        cdyc_a = a1[-1] + a2[-1] * sci * 2.0
        cdyc_b = b1[-1] + b2[-1] * sci * 2.0
        cdyc_c = c1[-1] + c2[-1] * sci * 2.0

        iarr = zeros((s.size, s.size - 3))
        iarr[0, 0] = cdya_c
        iarr[-1, -1] = cdyc_a
        fill_diagonal(iarr[1:-2, :], cdyb_c[:-1])
        fill_diagonal(iarr[2:-1, :], cdyb_a[1:])

        jarr = zeros((s.size, s.size))
        jarr[0, 0] = cdya_a
        jarr[0, 1] = cdya_b
        jarr[1, 0] = cdyb_a[0]
        jarr[-2, -1] = cdyb_c[-1]
        jarr[-1, -2] = cdyc_b
        jarr[-1, -1] = cdyc_c
        fill_diagonal(jarr[1:-1, 1:-1], cdyb_b)

        self._harr = iarr @ marr + jarr

    @property
    def harr(self) -> 'NDArray':
        if self._harr is None:
            self.calculate_gradient_array()
        return self._harr

    def calculate_curvature_array(self) -> None:

        s = self.s
        marr = self._marr
        a2 = self._a2
        b2 = self._b2
        c2 = self._c2

        cd2ya_a = a2[0] * 2.0
        cd2ya_b = b2[0] * 2.0
        cd2ya_c = c2[0] * 2.0

        cd2yb_a = a2 * 2.0
        cd2yb_b = b2 * 2.0
        cd2yb_c = c2 * 2.0

        cd2yc_a = a2[-1] * 2.0
        cd2yc_b = b2[-1] * 2.0
        cd2yc_c = c2[-1] * 2.0

        karr = zeros((s.size, s.size - 3))
        karr[0, 0] = cd2ya_c
        karr[-1, -1] = cd2yc_a
        fill_diagonal(karr[1:-2, :], cd2yb_c[:-1])
        fill_diagonal(karr[2:-1, :], cd2yb_a[1:])

        larr = zeros((s.size, s.size))
        larr[0, 0] = cd2ya_a
        larr[0, 1] = cd2ya_b
        larr[1, 0] = cd2yb_a[0]
        larr[-2, -1] = cd2yb_c[-1]
        larr[-1, -2] = cd2yc_b
        larr[-1, -1] = cd2yc_c
        fill_diagonal(larr[1:-1, 1:-1], cd2yb_b)

        self._garr = karr @ marr + larr

    @property
    def garr(self) -> 'NDArray':
        if self._garr is None:
            self.calculate_curvature_array()
        return self._garr

    def calculate_integral_array(self) -> None:

        sba = self.sb - self.sa
        sba2 = (self.sb**2 - self.sa**2) / 2.0
        sba3 = (self.sb**3 - self.sa**3) / 3.0
        scb = self.sc - self.sb
        scb2 = (self.sc**2 - self.sb**2) / 2.0
        scb3 = (self.sc**3 - self.sb**3) / 3.0

        int_ya_ab = sba * self.a0 + sba2 * self.a1 + sba3 * self.a2
        int_yb_ab = sba * self.b0 + sba2 * self.b1 + sba3 * self.b2
        int_yc_ab = sba * self.c0 + sba2 * self.c1 + sba3 * self.c2
        int_ya_bc = scb * self.a0 + scb2 * self.a1 + scb3 * self.a2
        int_yb_bc = scb * self.b0 + scb2 * self.b1 + scb3 * self.b2
        int_yc_bc = scb * self.c0 + scb2 * self.c1 + scb3 * self.c2

        jarr = zeros((self.s.size - 1, self.s.size - 3))
        fill_diagonal(jarr[:-2, :], int_yc_ab[:-1])
        fill_diagonal(jarr[1:-1, :], int_ya_ab[1:] + int_yc_bc[:-1])
        fill_diagonal(jarr[2:, :], int_ya_bc[1:])

        karr = zeros((self.s.size - 1, self.s.size))
        karr[0, 0] = int_ya_ab[0]
        karr[1, 0] = int_ya_bc[0]
        fill_diagonal(karr[:-1, 1:-1], int_yb_ab)
        fill_diagonal(karr[1:, 1:-1], int_yb_bc)
        karr[-2, -1] = int_yc_ab[-1]
        karr[-1, -1] = int_yc_bc[-1]

        Jarr = cumsum(jarr, axis=0)
        Karr = cumsum(karr, axis=0)

        self._iarr = zeros((self.s.size, self.s.size))
        self._iarr[1:, :] = Jarr @ self.marr + Karr

    @property
    def iarr(self) -> 'NDArray':
        if self._iarr is None:
            self.calculate_integral_array()
        return self._iarr

    def evaluate_points_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the spline at a given s."""
        s = asarray(s)
        z = zip(self.sa, self.sc,
                self.a0, self.a1, self.a2,
                self.b0, self.b1, self.b2,
                self.c0, self.c1, self.c2)
        rarr = zeros((*s.shape, self.s.size))
        for i, zi in enumerate(z):
            sa, sc, a0, a1, a2, b0, b1, b2, c0, c1, c2 = zi
            s_check = logical_and(s >= sa, s <= sc)
            sv = s[s_check]
            av = a0 + a1 * sv + a2 * sv**2
            bv = b0 + b1 * sv + b2 * sv**2
            cv = c0 + c1 * sv + c2 * sv**2
            if i == 0:
                indj = (0,)
                indk = (0, 1)
                Jv = zeros((sv.size, 1))
                Kv = zeros((sv.size, 2))
                Kv[:, 0] = av
                Kv[:, 1] = bv
                Jv[:, 0] = cv
            elif i == self.s.size - 3:
                indj = (i - 1,)
                indk = (i + 1, i + 2)
                Jv = zeros((sv.size, 1))
                Kv = zeros((sv.size, 2))
                Jv[:, 0] = av
                Kv[:, 0] = bv
                Kv[:, 1] = cv
            else:
                indj = (i - 1, i)
                indk = (i + 1,)
                Jv = zeros((sv.size, 2))
                Kv = zeros((sv.size, 1))
                Jv[:, 0] = av
                Kv[:, 0] = bv
                Jv[:, 1] = cv
            rarr[s_check, :] = Jv @ self.marr[indj, :]
            for k, ind in enumerate(indk):
                rarr[s_check, ind] += Kv[:, k]
        return rarr

    def evaluate_first_derivatives_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the first derivatives of the spline at a given s."""
        s = asarray(s)
        z = zip(self.sa, self.sc,
                self.a1, self.a2,
                self.b1, self.b2,
                self.c1, self.c2)
        drarr = zeros((*s.shape, self.s.size))
        for i, zi in enumerate(z):
            sa, sc, a1, a2, b1, b2, c1, c2 = zi
            s_check = logical_and(s >= sa, s <= sc)
            sv = s[s_check]
            av = a1 + a2 * sv * 2.0
            bv = b1 + b2 * sv * 2.0
            cv = c1 + c2 * sv * 2.0
            if i == 0:
                indj = (0,)
                indk = (0, 1)
                Jv = zeros((sv.size, 1))
                Kv = zeros((sv.size, 2))
                Kv[:, 0] = av
                Kv[:, 1] = bv
                Jv[:, 0] = cv
            elif i == self.s.size - 3:
                indj = (i - 1,)
                indk = (i + 1, i + 2)
                Jv = zeros((sv.size, 1))
                Kv = zeros((sv.size, 2))
                Jv[:, 0] = av
                Kv[:, 0] = bv
                Kv[:, 1] = cv
            else:
                indj = (i - 1, i)
                indk = (i + 1,)
                Jv = zeros((sv.size, 2))
                Kv = zeros((sv.size, 1))
                Jv[:, 0] = av
                Kv[:, 0] = bv
                Jv[:, 1] = cv
            drarr[s_check, :] = Jv @ self.marr[indj, :]
            for k, ind in enumerate(indk):
                drarr[s_check, ind] += Kv[:, k]
        return drarr

    def evaluate_integral_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the integrals of the spline at a given s."""
        s = asarray(s)
        z = zip(self.sa, self.sb, self.sc,
                self.a0, self.a1, self.a2,
                self.b0, self.b1, self.b2,
                self.c0, self.c1, self.c2)
        iarr = zeros((*s.shape, self.s.size))
        for i, zi in enumerate(z):
            sa, sb, sc, a0, a1, a2, b0, b1, b2, c0, c1, c2 = zi
            s_check = logical_and(s >= sa, s <= sc)
            sv = s[s_check]
            sbv = sv - sb
            sbv2 = (sv**2 - sb**2) / 2.0
            sbv3 = (sv**3 - sb**3) / 3.0
            int_ya_bv = a0 * sbv + a1 * sbv2 + a2 * sbv3
            int_yb_bv = b0 * sbv + b1 * sbv2 + b2 * sbv3
            int_yc_bv = c0 * sbv + c1 * sbv2 + c2 * sbv3
            if i == 0:
                indj = (0,)
                indk = (0, 1)
                Jv = zeros((sv.size, 1))
                Kv = zeros((sv.size, 2))
                Kv[:, 0] = int_ya_bv
                Kv[:, 1] = int_yb_bv
                Jv[:, 0] = int_yc_bv
            elif i == self.s.size - 3:
                indj = (i - 1,)
                indk = (i + 1, i + 2)
                Jv = zeros((sv.size, 1))
                Kv = zeros((sv.size, 2))
                Jv[:, 0] = int_ya_bv
                Kv[:, 0] = int_yb_bv
                Kv[:, 1] = int_yc_bv
            else:
                indj = (i - 1, i)
                indk = (i + 1,)
                Jv = zeros((sv.size, 2))
                Kv = zeros((sv.size, 1))
                Jv[:, 0] = int_ya_bv
                Kv[:, 0] = int_yb_bv
                Jv[:, 1] = int_yc_bv
            iarr[s_check, :] = Jv @ self.marr[indj, :]
            for k, ind in enumerate(indk):
                iarr[s_check, ind] += Kv[:, k]
            iarr[s_check, :] += self.iarr[i + 1, :]
        return iarr

    def to_quadratic_spline_1d(self, r: 'NDArray', validate: bool = True) -> 'QuadraticSpline1D':
        u"""This function creates a QuadraticSpline1D object from the solver."""
        qs1d = QuadraticSpline1D(self.s, r, validate=validate)
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(qs1d, attr, getattr(self, attr))
        return qs1d

    def __repr__(self):
        return '<QuadraticSpline1DSolver>'


class QuadraticSpline1D(QuadraticSpline1DSolver):
    u"""This class stores a parametric quadratic spline."""
    r: 'NDArray' = None
    _rall: 'NDArray' = None
    _dr: 'NDArray' = None
    _d2r: 'NDArray' = None
    _Ir: 'NDArray' = None

    def __init__(self, s: 'NDArray', r: 'NDArray',
                 validate: bool = True) -> None:
        u"""This function initialises the object."""
        super().__init__(s)
        self.r = asarray(r)
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        if self.r.ndim != 1:
            raise ValueError('Input r must be a 1D ndarray.')
        if self.r.size != self.s.size:
            raise ValueError('Input r must have the same size as s.')

    def calc_rall(self) -> None:
        rm = self.marr @ self.r
        self._rall = zeros(self.sall.size)
        self._rall[2:-2:2] = rm
        self._rall[1:-1:2] = self.r[1:-1]
        self._rall[0] = self.r[0]
        self._rall[-1] = self.r[-1]

    @property
    def rall(self) -> 'NDArray':
        if self._rall is None:
            self.calc_rall()
        return self._rall

    @property
    def ra(self) -> 'NDArray':
        return self.rall[:-2:2]

    @property
    def rb(self) -> 'NDArray':
        return self.rall[1:-1:2]

    @property
    def rc(self) -> 'NDArray':
        return self.rall[2::2]

    @property
    def dr(self) -> 'NDArray':
        if self._dr is None:
            self._dr = self.harr @ self.r
        return self._dr

    @property
    def d2r(self) -> 'NDArray':
        if self._d2r is None:
            self._d2r = self.garr @ self.r
        return self._d2r

    @property
    def Ir(self) -> 'NDArray':
        if self._Ir is None:
            self._Ir = self.iarr @ self.r
        return self._Ir

    def evaluate_points_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the spline at a given s."""
        s = asarray(s)
        r = full(s.shape, float('nan'))
        z = zip(self.sa, self.sc,
                self.ra, self.rb, self.rc,
                self.a0, self.a1, self.a2,
                self.b0, self.b1, self.b2,
                self.c0, self.c1, self.c2)
        for zi in z:
            sa, sc, ra, rb, rc, a0, a1, a2, b0, b1, b2, c0, c1, c2 = zi
            s_check = logical_and(s >= sa, s <= sc)
            sv = s[s_check]
            av = a0 + a1 * sv + a2 * sv**2
            bv = b0 + b1 * sv + b2 * sv**2
            cv = c0 + c1 * sv + c2 * sv**2
            r[s_check] = ra * av + rb * bv + rc * cv
        return r

    def evaluate_first_derivatives_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the first derivatives of the spline at a given s."""
        s = asarray(s)
        dr = full(s.shape, float('nan'))
        z = zip(self.sa, self.sc,
                self.ra, self.rb, self.rc,
                self.a1, self.a2,
                self.b1, self.b2,
                self.c1, self.c2)
        for zi in z:
            sa, sc, ra, rb, rc, a1, a2, b1, b2, c1, c2 = zi
            s_check = logical_and(s >= sa, s <= sc)
            sv = s[s_check]
            av = a1 + a2 * sv * 2.0
            bv = b1 + b2 * sv * 2.0
            cv = c1 + c2 * sv * 2.0
            dr[s_check] = ra * av + rb * bv + rc * cv
        return dr

    def evaluate_second_derivatives_at_t(self, s: 'NDArray',
                                         tol: float = 1e-12) -> 'NDArray':
        u"""This function evaluates the second derivatives of the spline at a given s."""
        s = asarray(s)
        d2r = full(s.shape, float('nan'))
        z = zip(self.sa, self.sc,
                self.ra, self.rb, self.rc,
                self.a2,
                self.b2,
                self.c2)
        for zi in z:
            sa, sc, ra, rb, rc, a2, b2, c2 = zi
            s_check = logical_and(s >= sa, s <= sc)
            sv = s[s_check]
            av = 2.0 * a2 * ones_like(sv)
            bv = 2.0 * b2 * ones_like(sv)
            cv = 2.0 * c2 * ones_like(sv)
            val = ra * av + rb * bv + rc * cv
            if sa != self.s[0]:
                sav_check = absolute(sv - sa) < tol
            else:
                sav_check = full(sv.shape, False)
            if sc != self.s[-1]:
                scv_check = absolute(sv - sc) < tol
            else:
                scv_check = full(sv.shape, False)
            sac_check = logical_or(sav_check, scv_check)
            val[sac_check] = val[sac_check] * 0.5
            nan_check = logical_and(s_check, isnan(d2r))
            d2r[nan_check] = 0.0
            d2r[s_check] += val
        return d2r

    def evaluate_curvatures_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the curvature of the spline at a given s."""
        dr = self.evaluate_first_derivatives_at_t(s)
        d2r = self.evaluate_second_derivatives_at_t(s)
        k = d2r / (dr**2.0 + 1.0)**1.5
        return k

    def evaluate_t(self, num: int) -> 'NDArray':
        return knot_linspace(num, self.sall)

    def evaluate_points(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        points = self.evaluate_points_at_t(s)
        return points

    def evaluate_first_derivatives(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        deriv1 = self.evaluate_first_derivatives_at_t(s)
        return deriv1

    def evaluate_second_derivatives(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        deriv2 = self.evaluate_second_derivatives_at_t(s)
        return deriv2

    def evaluate_curvatures(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        curvature = self.evaluate_curvatures_at_t(s)
        return curvature

    def evaluate_integrals_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the integrals of the spline at a given s."""
        s = asarray(s)
        Ir = full(s.shape, float('nan'))
        z = zip(self.sa, self.sb, self.sc,
                self.ra, self.rb, self.rc,
                self.a0, self.a1, self.a2,
                self.b0, self.b1, self.b2,
                self.c0, self.c1, self.c2)
        for i, zi in enumerate(z):
            sa, sb, sc, ra, rb, rc, a0, a1, a2, b0, b1, b2, c0, c1, c2 = zi
            s_check = logical_and(s >= sa, s <= sc)
            sv = s[s_check]
            sbv = sv - sb
            sbv2o2 = (sv**2 - sb**2) / 2.0
            sbv3o3 = (sv**3 - sb**3) / 3.0
            int_ya_bv = a0 * sbv + a1 * sbv2o2 + a2 * sbv3o3
            int_yb_bv = b0 * sbv + b1 * sbv2o2 + b2 * sbv3o3
            int_yc_bv = c0 * sbv + c1 * sbv2o2 + c2 * sbv3o3
            Ib = self.Ir[i + 1]
            Ir[s_check] = Ib + ra * int_ya_bv + rb * int_yb_bv + rc * int_yc_bv
        return Ir

    def evaluate_integrals(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        integrals = self.evaluate_integrals_at_t(s)
        return integrals

    def copy_with_new_input(self, new_input: 'NDArray') -> 'QuadraticSpline1D':
        new_spline = QuadraticSpline1D(self.s, new_input, bctype=self.bctype, validate=False)
        new_spline._marr = self._marr
        new_spline._sall = self._sall
        new_spline._a2 = self._a2
        new_spline._b2 = self._b2
        new_spline._c2 = self._c2
        new_spline._a1 = self._a1
        new_spline._b1 = self._b1
        new_spline._c1 = self._c1
        new_spline._a0 = self._a0
        new_spline._b0 = self._b0
        new_spline._c0 = self._c0
        new_spline._harr = self._harr
        new_spline._garr = self._garr
        new_spline._iarr = self._iarr
        return new_spline

    def find_intercepts(self, value: float, tol: float = 1e-12) -> 'NDArray':
        s_int = []
        z = zip(self.sa, self.sb, self.sc,
                self.ra, self.rb, self.rc)
        for zi in z:
            sa, sb, sc, ra, rb, rc = zi
            ra = ra - value
            rb = rb - value
            rc = rc - value
            jac = sa**2*sb - sa**2*sc - sa*sb**2 + sa*sc**2 + sb**2*sc - sb*sc**2
            a = (-sa*rb + sa*rc + sb*ra - sb*rc - sc*ra + sc*rb) / jac
            b = (sa**2*rb - sa**2*rc - sb**2*ra + sb**2*rc + sc**2*ra - sc**2*rb) / jac
            c = (sa**2*sb*rc - sa**2*sc*rb - sa*sb**2*rc + sa*sc**2*rb + sb**2*sc*ra - sb*sc**2*ra) / jac
            roots = quadratic_roots(a, b, c)
            for root in roots:
                if abs(root.imag) < tol and root.real - sa >= -tol and root.real - sc <= tol:
                    s_int.append(root.real)
        return unique(asarray(s_int))

    def __repr__(self):
        return '<QuadraticSpline1D>'
