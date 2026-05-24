from numbers import Number
from typing import TYPE_CHECKING

from numpy import asarray, full, logical_and, unique, zeros

from ..tools.basis import knot_linspace
from ..tools.roots import cubic_roots
from ..tools.solvers import cubic_pspline_fit_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray
    BCLike = tuple[tuple[int, float], tuple[int, float]] | None

BCSTR1 = ('quadratic', 'not-a-knot', 'natural', 'clamped', 'periodic')
BCSTR2 = ('quadratic', 'not-a-knot', 'natural', 'clamped')


class CubicSpline1DSolver():
    u"""This class solves for the cubic spline coefficients."""
    s: 'NDArray' = None
    bctype: 'BCLike' = None
    _Ds: 'NDArray' = None
    _garr: 'NDArray' = None
    _harr: 'NDArray' = None
    _iarr: 'NDArray' = None

    def __init__(self, s: 'NDArray', bctype: 'BCLike' = 'quadratic') -> None:
        u"""This function initialises the object."""
        self.s = asarray(s)
        self.bctype = bctype

    def validate(self) -> None:
        u"""This function validates the object."""
        if self.s.ndim != 1:
            raise ValueError('Input s must be a 1D ndarray.')
        if isinstance(self.bctype, str):
            if self.bctype not in BCSTR1:
                errstr = 'Input bctype must be one of:'
                errstr += ' clamped, natural, not-a-knot, periodic or quadratic.'
                raise ValueError(errstr)
        elif isinstance(self.bctype, tuple):
            if len(self.bctype) != 2:
                raise ValueError('Input bctype must be a tuple of length 2.')
            if isinstance(self.bctype[0], tuple):
                if len(self.bctype[0]) != 2:
                    raise ValueError('Input bctype[0] must be a tuple of length 2.')
                if not isinstance(self.bctype[0][0], int):
                    raise ValueError('Input bctype[0][0] must be an integer.')
                if self.bctype[0][0] != 1 and self.bctype[0][0] != 2:
                    raise ValueError('Input bctype[0][0] must be a 1 or 2.')
                if not isinstance(self.bctype[0][1], Number):
                    raise ValueError('Input bctype[0][1] must be a number.')
            elif isinstance(self.bctype[0], str):
                if self.bctype[0] not in BCSTR2:
                    errstr = 'Input bctype[0] must be one of:'
                    errstr += ' clamped, natural, not-a-knot or quadratic.'
                    raise ValueError(errstr)
            else:
                raise ValueError('Input bctype[0] must be a string or a tuple.')
            if isinstance(self.bctype[1], tuple):
                if len(self.bctype[1]) != 2:
                    raise ValueError('Input bctype[1] must be a tuple of length 2.')
                if not isinstance(self.bctype[1][0], int):
                    raise ValueError('Input bctype[1][0] must be an integer.')
                if self.bctype[1][0] != 1 and self.bctype[1][0] != 2:
                    raise ValueError('Input bctype[1][0] must be a 1 or 2.')
                if not isinstance(self.bctype[1][1], Number):
                    raise ValueError('Input bctype[1][1] must be a number.')
            elif isinstance(self.bctype[1], str):
                if self.bctype[1] not in BCSTR2:
                    errstr = 'Input bctype[1] must be one of:'
                    errstr += ' clamped, natural, not-a-knot or quadratic.'
                    raise ValueError(errstr)
            else:
                raise ValueError('Input bctype[1] must be a string or a tuple.')
        else:
            raise ValueError('Input bctype must be a string or a tuple of tuples.')

    @property
    def Ds(self) -> 'NDArray':
        if self._Ds is None:
            self._Ds = self.s[1:] - self.s[:-1]
        return self._Ds

    def calculate(self):
        if isinstance(self.bctype, str):
            bctype = self.bctype
        elif isinstance(self.bctype, tuple):
            bctype = []
            if isinstance(self.bctype[0], str):
                bctype.append(self.bctype[0])
            elif isinstance(self.bctype[0], tuple):
                bctype.append(self.bctype[0][0])
            if isinstance(self.bctype[1], str):
                bctype.append(self.bctype[1])
            elif isinstance(self.bctype[1], tuple):
                bctype.append(self.bctype[1][0])
            bctype = tuple(bctype)
        self._garr, self._harr = cubic_pspline_fit_solver(self.s, bctype=bctype)

    @property
    def garr(self) -> 'NDArray':
        if self._garr is None:
            self.calculate()
        return self._garr

    @property
    def harr(self) -> 'NDArray':
        if self._harr is None:
            self.calculate()
        return self._harr

    def calculate_integral_array(self) -> None:
        jarr = zeros((self.s.size, self.s.size))
        karr = zeros((self.s.size, self.s.size))
        for i, Dsi in enumerate(self.Ds, start=1):
            jarr[i, :] = jarr[i - 1, :]
            jarr[i, i - 1] += Dsi / 2.0
            jarr[i, i] += Dsi / 2.0
            karr[i, :] = karr[i - 1, :]
            karr[i, i - 1] -= Dsi**3 / 24.0
            karr[i, i] -= Dsi**3 / 24.0
        self._iarr = karr @ self.garr
        self._iarr[:, :jarr.shape[1]] += jarr

    @property
    def iarr(self) -> 'NDArray':
        if self._iarr is None:
            self.calculate_integral_array()
        return self._iarr

    def evaluate_points_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the spline at a given s."""
        s = asarray(s)
        rmat = zeros((*s.shape, self.s.size))
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            ABi = zeros((sv.size, 2))
            CDi = zeros((sv.size, 2))
            ABi[:, 0] = (sb - sv)/Dsi
            ABi[:, 1] = (sv - sa)/Dsi
            CDi[:, 0] = ((sb - sv)**3 - (sb - sv)) * Dsi**2 / 6.0
            CDi[:, 1] = ((sv - sa)**3 - (sv - sa)) * Dsi**2 / 6.0
            rmat[s_check, :] = CDi @ self.garr[[a, b], :]
            rmat[s_check, a] += ABi[:, 0]
            rmat[s_check, b] += ABi[:, 1]

        return rmat

    def evaluate_first_derivatives_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the first derivatives of the spline at a given s."""
        s = asarray(s)
        drmat = zeros((*s.shape, self.s.size))
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            Av = (sb - sv) / Dsi
            Bv = (sv - sa) / Dsi
            ABi = zeros((sv.size, 2))
            ABi[:, 0] = 1.0 / Dsi
            ABi[:, 1] = -1.0 / Dsi
            CDi = zeros((sv.size, 2))
            CDi[s_check, 0] = (1.0 - 3.0 * Av**2) / 6.0 * Dsi
            CDi[s_check, 1] = (3.0 * Bv**2 - 1.0) / 6.0 * Dsi
            drmat[s_check, :] = CDi @ self.garr[[a, b], :]
            drmat[s_check, a] += ABi[:, 0]
            drmat[s_check, b] += ABi[:, 1]
        return drmat

    def evaluate_integral_array_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the integrals of the spline at a given s."""
        s = asarray(s)
        iarr = zeros((*s.shape, self.s.size))
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            ABi = zeros((sv.size, 2))
            CDi = zeros((sv.size, 2))
            ABi[:, 0] = -(sv - sb - Dsi) * (sv - sb + Dsi) / (2.0 * Dsi)
            ABi[:, 1] = (sv - sa)**2 / (2.0 * Dsi)
            CDi[:, 0] = -(sv - sb - Dsi)**2 * (sv - sb + Dsi)**2 / (24.0 * Dsi)
            CDi[:, 1] = (sv - sa)**2 * (sv**2 - 2.0 * sv * sa + sa**2 - 2.0 * Dsi**2) / (24.0 * Dsi)
            iarr[s_check, :] = self.iarr[i, :]
            iarr[s_check, a] += ABi[:, 0]
            iarr[s_check, b] += ABi[:, 1]
            iarr[s_check, :] += CDi @ self.garr[(a, b), :]
        return iarr

    def to_cubic_spline_1d(self, r: 'NDArray', validate: bool = True) -> 'CubicSpline1D':
        u"""This function creates a CubicSpline1D object from the solver."""
        cs1d = CubicSpline1D(self.s, r, bctype=self.bctype, validate=validate)
        cs1d._garr = self._garr
        cs1d._harr = self._harr
        cs1d._iarr = self._iarr
        return cs1d

    def __repr__(self):
        return '<CubicSpline1DSolver>'


class CubicSpline1D(CubicSpline1DSolver):
    u"""This class stores a parametric cubic spline."""
    r: 'NDArray' = None
    _input: 'NDArray' = None
    _Dr: 'NDArray' = None
    _d2r: 'NDArray' = None
    _Ir: 'NDArray' = None

    def __init__(self, s: 'NDArray', r: 'NDArray',
                 bctype: 'BCLike' = 'quadratic',
                 validate: bool = True) -> None:
        u"""This function initialises the object."""
        super().__init__(s, bctype)
        self.r = asarray(r)
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        if self.r.ndim != 1:
            raise ValueError('Input r must be a 1D ndarray.')
        if self.r.size != self.s.size:
            raise ValueError('Input r must have the same size as s.')

    @property
    def input(self) -> 'NDArray':
        if self._input is None:
            if isinstance(self.bctype, tuple):
                numcond = 0
                if not isinstance(self.bctype[0], str):
                    numcond += 1
                if not isinstance(self.bctype[1], str):
                    numcond += 1
                self._input = zeros(self.r.size + numcond,
                                    dtype=self.r.dtype)
                self._input[:self.r.size] = self.r
                count = 0
                if not isinstance(self.bctype[0], str):
                    self._input[self.r.size + count] = self.bctype[0][1]
                    count += 1
                if not isinstance(self.bctype[1], str):
                    self._input[self.r.size + count] = self.bctype[1][1]
            else:
                self._input = self.r
        return self._input

    @property
    def Dr(self) -> 'NDArray':
        if self._Dr is None:
            self._Dr = self.r[1:] - self.r[:-1]
        return self._Dr

    @property
    def d2r(self) -> 'NDArray':
        if self._d2r is None:
            self._d2r = self.garr @ self.input
        return self._d2r

    @property
    def Ir(self) -> 'NDArray':
        if self._Ir is None:
            self._Ir = self.iarr @ self.input
        return self._Ir

    def evaluate_points_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the spline at a given s."""
        s = asarray(s)
        r = full(s.shape, float('nan'))
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            ra = self.r[a]
            rb = self.r[b]
            d2ra = self.d2r[a]
            d2rb = self.d2r[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            Av = (sb - sv)/Dsi
            Bv = (sv - sa)/Dsi
            Cv = (Av**3 - Av)*Dsi**2/6
            Dv = (Bv**3 - Bv)*Dsi**2/6
            r[s_check] = ra*Av + rb*Bv + d2ra*Cv + d2rb*Dv
        return r

    def evaluate_first_derivatives_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the first derivatives of the spline at a given s."""
        s = asarray(s)
        dr = full(s.shape, float('nan'))
        for i, (Dsi, Dri) in enumerate(zip(self.Ds, self.Dr)):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            d2ra = self.d2r[a]
            d2rb = self.d2r[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            Av = (sb - sv)/Dsi
            Bv = (sv - sa)/Dsi
            Ev = (1 - 3*Av**2)/6*Dsi
            Fv = (3*Bv**2 - 1)/6*Dsi
            dr[s_check] = Dri/Dsi + d2ra*Ev + d2rb*Fv
        return dr

    def evaluate_second_derivatives_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the second derivatives of the spline at a given s."""
        s = asarray(s)
        d2r = full(s.shape, float('nan'))
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            d2ra = self.d2r[a]
            d2rb = self.d2r[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            Av = (sb - sv)/Dsi
            Bv = (sv - sa)/Dsi
            d2r[s_check] = d2ra*Av + d2rb*Bv
        return d2r

    def evaluate_curvatures_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the curvature of the spline at a given s."""
        dr = self.evaluate_first_derivatives_at_t(s)
        d2r = self.evaluate_second_derivatives_at_t(s)
        k = d2r/(dr**2 + 1.0)**1.5
        return k

    def evaluate_t(self, num: int) -> 'NDArray':
        return knot_linspace(num, self.s)

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
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            ra = self.r[a]
            rb = self.r[b]
            d2ra = self.d2r[a]
            d2rb = self.d2r[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            Ai = -(sv - sb - Dsi) * (sv - sb + Dsi) / (2.0 * Dsi)
            Bi = (sv - sa)**2 / (2.0 * Dsi)
            Ci = -(sv - sb - Dsi)**2*(sv - sb + Dsi)**2 / (24.0 * Dsi)
            Di = (sv - sa)**2 * (sv**2 - 2.0 * sv * sa + sa**2 - 2.0 * Dsi**2)/(24.0 * Dsi)
            Iv = self.Ir[a]
            Ir[s_check] = Iv + ra*Ai + rb*Bi + d2ra*Ci + d2rb*Di
        return Ir

    def evaluate_integrals(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        integrals = self.evaluate_integrals_at_t(s)
        return integrals

    def copy_with_new_input(self, new_input: 'NDArray') -> 'CubicSpline1D':
        new_spline = CubicSpline1D(self.s, new_input, bctype=self.bctype, validate=False)
        new_spline._garr = self._garr
        new_spline._harr = self._harr
        return new_spline

    def find_intercepts(self, value: float) -> 'NDArray':
        s_int = []
        for i in range(self.r.size - 1):
            ra = self.r[i]
            rb = self.r[i + 1]
            if value >= ra and value <= rb or value <= ra and value >= rb:
                ra = ra - value
                rb = rb - value
                sa = self.s[i]
                sb = self.s[i + 1]
                Ds = self.Ds[i]
                d2ra = self.d2r[i]
                d2rb = self.d2r[i + 1]
                a = (-d2ra + d2rb)/(6*Ds)
                b = (d2ra*sb - d2rb*sa)/(2*Ds)
                c = (-3*d2ra*sb**2 + d2ra*Ds**2 + 3*d2rb*sa**2 - d2rb*Ds**2 - 6*ra + 6*rb)/(6*Ds)
                d = (d2ra*sb**3 - d2ra*sb*Ds**2 - d2rb*sa**3 + d2rb*sa*Ds**2 - 6*sa*rb + 6*sb*ra)/(6*Ds)
                roots = cubic_roots(a, b, c, d)
                for root in roots:
                    if root.imag == 0.0 and root.real >= sa and root.real < sb:
                        s_int.append(root.real)
        return unique(asarray(s_int))

    def __repr__(self):
        return '<CubicSpline1D>'
