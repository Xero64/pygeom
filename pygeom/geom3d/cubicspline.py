from typing import TYPE_CHECKING, Optional, Tuple

from numpy import cumsum, full, logical_and, zeros

from ..tools.basis import knot_linspace
from ..tools.solvers import cubic_pspline_fit_solver
from .vector import Vector, zero_vector

if TYPE_CHECKING:
    from numpy.typing import NDArray
    BCLike = Optional[Tuple[Tuple[int, Vector], Tuple[int, Vector]]]

class CubicSpline():
    u"""This class stores a 3D parametric cubic spline."""
    points: Vector = None
    bctype: 'BCLike' = None
    _input: Vector = None
    _Dr: Vector = None
    _Ds: 'NDArray' = None
    _s: 'NDArray' = None
    _gmat: 'NDArray' = None
    _d2r: Vector = None

    def __init__(self, points: Vector, bctype: 'BCLike' = 'quadratic',
                 validate: bool = True) -> None:
        u"""This function initialises the object."""
        self.points = points
        self.bctype = bctype
        if validate:
            self.validate()

    def validate(self) -> None:
        u"""This function validates the object."""
        if not isinstance(self.points, Vector):
            raise ValueError('Input points must be a Vector object.')
        if self.points.ndim != 1:
            raise ValueError('Input points must be a 1D Vector object.')
        if isinstance(self.bctype, str):
            if self.bctype not in ('clamped', 'natural', 'not-a-knot', 'periodic', 'quadratic'):
                errstr = 'Input bctype must be one of:'
                errstr += ' clamped, natural, not-a-knot, periodic or quadratic.'
                raise ValueError(errstr)
        elif isinstance(self.bctype, tuple):
            if len(self.bctype) != 2:
                raise ValueError('Input bctype must be a tuple of length 2.')
            if not isinstance(self.bctype[0], tuple) or not isinstance(self.bctype[1], tuple):
                raise ValueError('Input bctype must be a tuple of tuples.')
            if len(self.bctype[0]) != 2 or len(self.bctype[1]) != 2:
                raise ValueError('Input bctype tuples must have a length of 2.')
            if not isinstance(self.bctype[0][0], int) or not isinstance(self.bctype[1][0], int):
                raise ValueError('Input bctype tuple elements must be integers.')
            if not isinstance(self.bctype[0][1], Vector) or not isinstance(self.bctype[1][1], Vector):
                raise ValueError('Input bctype tuple elements must be Vector objects.')
        else:
            raise ValueError('Input bctype must be a string or a tuple of tuples.')
    
    @property
    def input(self) -> Vector:
        if self._input is None:
            if isinstance(self.bctype, tuple):
                self._input = zero_vector(self.points.size + 2,
                                          dtype=self.points.dtype)
                self._input[:-2] = self.points
                self._input[-2] = self.bctype[0][1]
                self._input[-1] = self.bctype[1][1]
            else:
                self._input = self.points
        return self._input

    @property
    def Dr(self) -> Vector:
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
            if isinstance(self.bctype, tuple):
                bctype = (self.bctype[0][0], self.bctype[1][0])
            else:
                bctype = self.bctype
            self._gmat = cubic_pspline_fit_solver(self.s, bctype=bctype)
        return self._gmat
    
    @property
    def d2r(self) -> Vector:
        if self._d2r is None:
            self._d2r = self.input.rmatmul(self.gmat)
        return self._d2r
    
    def evaluate_points_at_t(self, s: 'NDArray') -> Vector:
        u"""This function evaluates the spline at a given s."""
        x = full(s.shape, float('nan'))
        y = full(s.shape, float('nan'))
        z = full(s.shape, float('nan'))
        points = Vector(x, y, z)
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
    
    def evaluate_first_derivatives_at_t(self, s: 'NDArray') -> Vector:
        u"""This function evaluates the first derivatives of the spline at a given s."""
        dx = full(s.shape, float('nan'))
        dy = full(s.shape, float('nan'))
        dz = full(s.shape, float('nan'))
        deriv1 = Vector(dx, dy, dz)
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
    
    def evaluate_second_derivatives_at_t(self, s: 'NDArray') -> Vector:
        u"""This function evaluates the second derivatives of the spline at a given s."""
        d2x = full(s.shape, float('nan'))
        d2y = full(s.shape, float('nan'))
        d2z = full(s.shape, float('nan'))
        deriv2 = Vector(d2x, d2y, d2z)
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
    
    def evaluate_curvatures_at_t(self, s: 'NDArray') -> Vector:
        u"""This function evaluates the curvature of the spline at a given s."""
        deriv1 = self.evaluate_first_derivatives_at_t(s)
        deriv2 = self.evaluate_second_derivatives_at_t(s)
        curvature = deriv1.cross(deriv2)/deriv1.return_magnitude()**3
        return curvature
    
    def evaluate_tangents_at_t(self, s: 'NDArray') -> Vector:
        u"""This function evaluates the tangent of the spline at a given s."""
        deriv1 = self.evaluate_first_derivatives_at_t(s)
        tangent = deriv1.to_unit()
        return tangent
    
    def evaluate_normals_at_t(self, s: 'NDArray') -> Vector:
        u"""This function evaluates the normal of the spline at a given s."""
        deriv2 = self.evaluate_second_derivatives_at_t(s)
        normal = deriv2.to_unit()
        return normal
    
    def evaluate_binormals_at_t(self, s: 'NDArray') -> Vector:
        u"""This function evaluates the binormal of the spline at a given s."""
        curvature = self.evaluate_curvatures_at_t(s)
        binormal = curvature.to_unit()
        return binormal
    
    def evaluate_t(self, num: int) -> 'NDArray':
        return knot_linspace(num, self.s)
    
    def evaluate_points(self, num: int) -> Vector:
        s = self.evaluate_t(num)
        points = self.evaluate_points_at_t(s)
        return points
    
    def evaluate_first_derivatives(self, num: int) -> Vector:
        s = self.evaluate_t(num)
        deriv1 = self.evaluate_first_derivatives_at_t(s)
        return deriv1
    
    def evaluate_second_derivatives(self, num: int) -> Vector:
        s = self.evaluate_t(num)
        deriv2 = self.evaluate_second_derivatives_at_t(s)
        return deriv2
    
    def evaluate_curvatures(self, num: int) -> Vector:
        s = self.evaluate_t(num)
        curvature = self.evaluate_curvatures_at_t(s)
        return curvature
    
    def evaluate_tangents(self, num: int) -> Vector:
        s = self.evaluate_t(num)
        tangent = self.evaluate_tangents_at_t(s)
        return tangent
    
    def evaluate_normals(self, num: int) -> Vector:
        s = self.evaluate_t(num)
        normal = self.evaluate_normals_at_t(s)
        return normal
    
    def evaluate_binormals(self, num: int) -> Vector:
        s = self.evaluate_t(num)
        binormal = self.evaluate_binormals_at_t(s)
        return binormal

    def __repr__(self):
        return '<CubicSpline>'
