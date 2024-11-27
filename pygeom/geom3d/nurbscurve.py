from typing import TYPE_CHECKING, Any

from numpy import concatenate, divide, full, ones

from ..tools.basis import (basis_first_derivatives, basis_functions,
                           basis_second_derivatives, default_knots,
                           knot_linspace)
from .vector import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NurbsCurve():
    ctlpnts: Vector = None
    weights: 'NDArray' = None
    degree: int = None
    knots: 'NDArray' = None
    endpoint: bool = None
    _wpoints: Vector = None
    _cknots: 'NDArray' = None

    def __init__(self, ctlpnts: Vector, **kwargs: dict[str, Any]) -> None:
        self.ctlpnts = ctlpnts.ravel()
        self.weights = kwargs.get('weights', ones(ctlpnts.size)).ravel()
        self.degree = kwargs.get('degree', self.ctlpnts.size - 1)
        self.knots = kwargs.get('knots', default_knots(self.ctlpnts.size,
                                                       self.degree))
        self.endpoint = kwargs.get('endpoint', True)

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def copy(self) -> 'NurbsCurve':
        ctlpnts = self.ctlpnts.copy()
        weights = self.weights.copy()
        degree = self.degree
        knots = self.knots.copy()
        endpoint = self.endpoint
        return NurbsCurve(ctlpnts, weights=weights, degree=degree,
                          knots=knots, endpoint=endpoint)

    @property
    def wpoints(self) -> Vector:
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    @property
    def cknots(self) -> 'NDArray':
        if self._cknots is None:
            if self.endpoint:
                kbeg = full(self.degree, self.knots[0])
                kend = full(self.degree, self.knots[-1])
                self._cknots = concatenate((kbeg, self.knots, kend))
            else:
                self._cknots = self.knots
        return self._cknots

    @property
    def rational(self) -> bool:
        check: 'NDArray' = self.weights == 1.0
        return not check.all()

    def basis_functions(self, u: 'NDArray') -> 'NDArray':
        return basis_functions(self.degree, self.cknots, u)

    def basis_first_derivatives(self, u: 'NDArray') -> 'NDArray':
        return basis_first_derivatives(self.degree, self.cknots, u)

    def basis_second_derivatives(self, u: 'NDArray') -> 'NDArray':
        return basis_second_derivatives(self.degree, self.cknots, u)

    def evaluate_points_at_t(self, u: 'NDArray') -> Vector:
        Nu = self.basis_functions(u)
        numer = self.wpoints@Nu
        if self.rational:
            denom = self.weights@Nu
            points = numer/denom
        else:
            points = numer
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_first_derivatives_at_t(self, u: 'NDArray') -> Vector:
        Nu = self.basis_functions(u)
        dNu = self.basis_first_derivatives(u)
        numer = self.wpoints@Nu
        dnumer = self.wpoints@dNu
        if self.rational:
            denom = self.weights@Nu
            ddenom = self.weights@dNu
            deriv1 = (dnumer*denom - numer*ddenom)/denom**2
        else:
            deriv1 = dnumer
        if deriv1.size == 1:
            deriv1 = deriv1[0]
        return deriv1

    def evaluate_second_derivatives_at_t(self, u: 'NDArray') -> Vector:
        Nu = self.basis_functions(u)
        dNu = self.basis_first_derivatives(u)
        d2Nu = self.basis_second_derivatives(u)
        numer = self.wpoints@Nu
        dnumer = self.wpoints@dNu
        d2numer = self.wpoints@d2Nu
        if self.rational:
            denom = self.weights@Nu
            ddenom = self.weights@dNu
            d2denom = self.weights@d2Nu
            deriv2 = (d2numer*denom**2 - dnumer*2*ddenom*denom + numer*2*ddenom**2 - numer*d2denom*denom)/denom**3
        else:
            deriv2 = d2numer
        if deriv2.size == 1:
            deriv2 = deriv2[0]
        return deriv2

    def evaluate_curvatures_at_t(self, u: 'NDArray') -> Vector:
        deriv1 = self.evaluate_first_derivatives_at_t(u)
        deriv2 = self.evaluate_second_derivatives_at_t(u)
        deriv1mag = deriv1.return_magnitude()
        deriv1magnot0 = deriv1mag != 0.0
        deriv1mag3 = deriv1mag**3
        deriv1xderiv2 = deriv1.cross(deriv2)
        curvature = Vector.zeros(deriv1mag.shape)
        divide(deriv1xderiv2, deriv1mag3, where=deriv1magnot0, out=curvature)
        # curvature = deriv1.cross(deriv2)/deriv1.return_magnitude()**3
        return curvature

    def evaluate_tangents_at_t(self, u: 'NDArray') -> Vector:
        deriv1 = self.evaluate_first_derivatives_at_t(u)
        return deriv1.to_unit()

    def evaluate_normals_at_t(self, u: 'NDArray') -> Vector:
        deriv2 = self.evaluate_second_derivatives_at_t(u)
        return deriv2.to_unit()

    def evaluate_binormals_at_t(self, u: 'NDArray') -> Vector:
        deriv1 = self.evaluate_first_derivatives_at_t(u)
        deriv2 = self.evaluate_second_derivatives_at_t(u)
        binormal = deriv1.cross(deriv2).to_unit()
        return binormal

    def evaluate_t(self, num: int) -> 'NDArray':
        return knot_linspace(num, self.knots)

    def evaluate_points(self, num: int) -> Vector:
        u = self.evaluate_t(num)
        return self.evaluate_points_at_t(u)

    def evaluate_first_derivatives(self, num: int) -> Vector:
        u = self.evaluate_t(num)
        return self.evaluate_first_derivatives_at_t(u)

    def evaluate_second_derivatives(self, num: int) -> Vector:
        u = self.evaluate_t(num)
        return self.evaluate_second_derivatives_at_t(u)

    def evaluate_curvatures(self, num: int) -> Vector:
        u = self.evaluate_t(num)
        return self.evaluate_curvatures_at_t(u)

    def evaluate_tangents(self, num: int) -> Vector:
        u = self.evaluate_t(num)
        return self.evaluate_tangents_at_t(u)

    def evaluate_normals(self, num: int) -> Vector:
        u = self.evaluate_t(num)
        return self.evaluate_normals_at_t(u)

    def evaluate_binormals(self, num: int) -> Vector:
        u = self.evaluate_t(num)
        return self.evaluate_binormals_at_t(u)

    def __repr__(self) -> str:
        return f'<NurbsCurve: degree={self.degree:d}>'

    def __str__(self) -> str:
        outstr = f'NurbsCurve\n'
        outstr += f'  control points: \n{self.ctlpnts}\n'
        outstr += f'  weights: {self.weights}\n'
        outstr += f'  degree: {self.degree:d}\n'
        outstr += f'  knots: {self.knots}\n'
        outstr += f'  cknots: {self.cknots}\n'
        outstr += f'  endpoint: {self.endpoint}\n'
        return outstr


class BSplineCurve(NurbsCurve):

    def __init__(self, ctlpnts: Vector, **kwargs: dict[str, Any]) -> None:
        kwargs['weights'] = ones(ctlpnts.shape)
        kwargs['rational'] = False
        super().__init__(ctlpnts, **kwargs)

    def __repr__(self) -> str:
        return f'<BSplineCurve: degree={self.degree:d}>'

    def __str__(self) -> str:
        outstr = f'BSplineCurve\n'
        outstr += f'  control points: \n{self.ctlpnts}\n'
        outstr += f'  degree: {self.degree:d}\n'
        outstr += f'  knots: {self.knots}\n'
        outstr += f'  endpoint: {self.endpoint}\n'
        return outstr
