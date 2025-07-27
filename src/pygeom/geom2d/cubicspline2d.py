from typing import TYPE_CHECKING

from numpy import asarray, cumsum, full, logical_and, zeros

from ..tools.basis import knot_linspace
from ..tools.solvers import cubic_pspline_fit_solver
from .vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    BCLike = tuple[tuple[int, Vector2D], tuple[int, Vector2D]] | None

BCSTR1 = ('quadratic', 'not-a-knot', 'natural', 'clamped', 'periodic')
BCSTR2 = ('quadratic', 'not-a-knot', 'natural', 'clamped')

class CubicSpline2D():
    u"""This class stores a 2D parametric cubic spline."""
    points: Vector2D = None
    bctype: 'BCLike' = None
    _input: Vector2D = None
    _Dr: Vector2D = None
    _Ds: 'NDArray' = None
    _s: 'NDArray' = None
    _gmat: 'NDArray' = None
    _d2r: Vector2D = None

    def __init__(self, points: Vector2D, bctype: 'BCLike' = 'quadratic',
                 validate: bool = True) -> None:
        u"""This function initialises the object."""
        self.points = points
        self.bctype = bctype
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        if not isinstance(self.points, Vector2D):
            raise ValueError('Input points must be a Vector2D object.')
        if self.points.ndim != 1:
            raise ValueError('Input points must be a 1D Vector2D object.')
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
                if not isinstance(self.bctype[0][1], Vector2D):
                    raise ValueError('Input bctype[0][1] must be a Vector2D object.')
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
                if not isinstance(self.bctype[1][1], Vector2D):
                    raise ValueError('Input bctype[1][1] must be a Vector2D object.')
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
    def input(self) -> Vector2D:
        if self._input is None:
            if isinstance(self.bctype, tuple):
                numcond = 0
                if not isinstance(self.bctype[0], str):
                    numcond += 1
                if not isinstance(self.bctype[1], str):
                    numcond += 1
                self._input = Vector2D.zeros(self.points.size + numcond,
                                          dtype=self.points.dtype)
                self._input[:self.points.size] = self.points
                count = 0
                if not isinstance(self.bctype[0], str):
                    self._input[self.points.size + count] = self.bctype[0][1]
                    count += 1
                if not isinstance(self.bctype[1], str):
                    self._input[self.points.size + count] = self.bctype[1][1]
            else:
                self._input = self.points
        return self._input

    @property
    def Dr(self) -> Vector2D:
        if self._Dr is None:
            self._Dr = self.points[1:] - self.points[:-1]
        return self._Dr

    @property
    def Ds(self) -> 'NDArray':
        if self._Ds is None:
            self._Ds = self.Dr.return_magnitude()
        return self._Ds

    @property
    def s(self) -> 'NDArray':
        if self._s is None:
            self._s = zeros(self.points.shape)
            self._s[1:] = cumsum(self.Ds)
        return self._s

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
    def d2r(self) -> Vector2D:
        if self._d2r is None:
            self._d2r = self.gmat@self.input
        return self._d2r

    def evaluate_points_at_t(self, s: 'NDArray') -> Vector2D:
        u"""This function evaluates the spline at a given s."""
        s = asarray(s)
        x = full(s.shape, float('nan'))
        y = full(s.shape, float('nan'))
        points = Vector2D(x, y)
        for i, Dsi in enumerate(self.Ds):
            a = i
            b = i + 1
            sa = self.s[a]
            sb = self.s[b]
            ra = self.points[a]
            rb = self.points[b]
            d2ra = self.d2r[a]
            d2rb = self.d2r[b]
            s_check = logical_and(s >= sa, s <= sb)
            sv = s[s_check]
            Av = (sb - sv)/Dsi
            Bv = (sv - sa)/Dsi
            Cv = (Av**3 - Av)*Dsi**2/6
            Dv = (Bv**3 - Bv)*Dsi**2/6
            points[s_check] = ra*Av + rb*Bv + d2ra*Cv + d2rb*Dv
        return points

    def evaluate_first_derivatives_at_t(self, s: 'NDArray') -> Vector2D:
        u"""This function evaluates the first derivatives of the spline at a given s."""
        s = asarray(s)
        dx = full(s.shape, float('nan'))
        dy = full(s.shape, float('nan'))
        deriv1 = Vector2D(dx, dy)
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
            deriv1[s_check] = Dri/Dsi + d2ra*Ev + d2rb*Fv
        return deriv1

    def evaluate_second_derivatives_at_t(self, s: 'NDArray') -> Vector2D:
        u"""This function evaluates the second derivatives of the spline at a given s."""
        s = asarray(s)
        d2x = full(s.shape, float('nan'))
        d2y = full(s.shape, float('nan'))
        deriv2 = Vector2D(d2x, d2y)
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
            deriv2[s_check] = d2ra*Av + d2rb*Bv
        return deriv2

    def evaluate_curvatures_at_t(self, s: 'NDArray') -> 'NDArray':
        u"""This function evaluates the curvature of the spline at a given s."""
        deriv1 = self.evaluate_first_derivatives_at_t(s)
        deriv2 = self.evaluate_second_derivatives_at_t(s)
        curvature = deriv1.cross(deriv2)/deriv1.return_magnitude()**3
        return curvature

    def evaluate_tangents_at_t(self, s: 'NDArray') -> Vector2D:
        u"""This function evaluates the tangent of the spline at a given s."""
        deriv1 = self.evaluate_first_derivatives_at_t(s)
        tangent = deriv1.to_unit()
        return tangent

    def evaluate_normals_at_t(self, s: 'NDArray') -> Vector2D:
        u"""This function evaluates the normal of the spline at a given s."""
        deriv2 = self.evaluate_second_derivatives_at_t(s)
        normal = deriv2.to_unit()
        return normal

    def evaluate_t(self, num: int) -> 'NDArray':
        return knot_linspace(num, self.s)

    def evaluate_points(self, num: int) -> Vector2D:
        s = self.evaluate_t(num)
        points = self.evaluate_points_at_t(s)
        return points

    def evaluate_first_derivatives(self, num: int) -> Vector2D:
        s = self.evaluate_t(num)
        deriv1 = self.evaluate_first_derivatives_at_t(s)
        return deriv1

    def evaluate_second_derivatives(self, num: int) -> Vector2D:
        s = self.evaluate_t(num)
        deriv2 = self.evaluate_second_derivatives_at_t(s)
        return deriv2

    def evaluate_curvatures(self, num: int) -> 'NDArray':
        s = self.evaluate_t(num)
        curvature = self.evaluate_curvatures_at_t(s)
        return curvature

    def evaluate_tangents(self, num: int) -> Vector2D:
        s = self.evaluate_t(num)
        tangent = self.evaluate_tangents_at_t(s)
        return tangent

    def evaluate_normals(self, num: int) -> Vector2D:
        s = self.evaluate_t(num)
        normal = self.evaluate_normals_at_t(s)
        return normal

    def __repr__(self):
        return '<CubicSpline2D>'
