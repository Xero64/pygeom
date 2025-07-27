from numbers import Number
from typing import TYPE_CHECKING

from numpy import asarray, full, logical_and, ndarray, zeros

from ..tools.basis import knot_linspace
from ..tools.solvers import cubic_pspline_fit_solver

if TYPE_CHECKING:
    from numpy.typing import NDArray
    BCLike = tuple[tuple[int, float], tuple[int, float]] | None

BCSTR1 = ('quadratic', 'not-a-knot', 'natural', 'clamped', 'periodic')
BCSTR2 = ('quadratic', 'not-a-knot', 'natural', 'clamped')

class CubicSpline1D():
    u"""This class stores a 3D parametric cubic spline."""
    s: 'NDArray' = None
    r: 'NDArray' = None
    bctype: 'BCLike' = None
    _input: 'NDArray' = None
    _Dr: 'NDArray' = None
    _Ds: 'NDArray' = None
    _gmat: 'NDArray' = None
    _d2r: 'NDArray' = None

    def __init__(self, s: 'NDArray', r: 'NDArray',
                 bctype: 'BCLike' = 'quadratic',
                 validate: bool = True) -> None:
        u"""This function initialises the object."""
        self.s = s
        self.r = r
        self.bctype = bctype
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        if not isinstance(self.s, ndarray):
            raise ValueError('Input s must be a ndarray.')
        if self.s.ndim != 1:
            raise ValueError('Input s must be a 1D ndarray.')
        if not isinstance(self.r, ndarray):
            raise ValueError('Input r must be a ndarray.')
        if self.r.ndim != 1:
            raise ValueError('Input r must be a 1D ndarray.')
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
    def Ds(self) -> 'NDArray':
        if self._Ds is None:
            self._Ds = self.s[1:] - self.s[:-1]
        return self._Ds

    @property
    def gmat(self) -> 'NDArray':
        if self._gmat is None:
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
            self._gmat = cubic_pspline_fit_solver(self.s, bctype=bctype)
        return self._gmat

    @property
    def d2r(self) -> 'NDArray':
        if self._d2r is None:
            self._d2r = self.gmat@self.input
        return self._d2r

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

    def __repr__(self):
        return '<CubicSpline1D>'
