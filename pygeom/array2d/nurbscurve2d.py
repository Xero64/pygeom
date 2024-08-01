from typing import TYPE_CHECKING, Any, Dict, Union

from numpy import concatenate, float64, full, linspace, ones

from ..geom2d import Vector2D
from ..tools.basis import (basis_first_derivatives, basis_functions, default_knots,
                           knot_linspace, basis_second_derivatives)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]


class BSplineCurve2D():
    ctlpnts: 'ArrayVector2D' = None
    knots: 'NDArray[float64]' = None
    degree: int = None
    _wpoints: 'ArrayVector2D' = None

    def __init__(self, ctlpnts: 'ArrayVector2D',
                 knots: 'NDArray[float64]', degree: int = None) -> None:
        self.ctlpnts = ctlpnts
        self.knots = knots
        self.degree = ctlpnts.size - 1
        if degree is not None:
            self.degree = degree

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def basis_functions(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_functions(self.degree, self.knots, u)

    def basis_first_derivatives(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_first_derivatives(self.degree, self.knots, u)

    def evaluate_points_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        points = self.ctlpnts@Nu
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_u(self, u: 'Numeric') -> 'VectorLike':
        dNu = self.basis_first_derivatives(u)
        tangents = self.ctlpnts@dNu
        if tangents.size == 1:
            tangents = tangents[0]
        return tangents

    def evaluate_points(self, num: int) -> 'ArrayVector2D':
        umin = self.knots.min()
        umax = self.knots.max()
        u = linspace(umin, umax, num, dtype=float64)
        return self.evaluate_points_at_u(u)

    def evaluate_tangents(self, num: int) -> 'ArrayVector2D':
        umin = self.knots.min()
        umax = self.knots.max()
        u = linspace(umin, umax, num, dtype=float64)
        return self.evaluate_tangents_at_u(u)


class NurbsCurve2D():
    ctlpnts: 'ArrayVector2D' = None
    weights: 'NDArray[float64]' = None
    degree: int = None
    knots: 'NDArray[float64]' = None
    endpoint: bool = None
    closed: bool = None
    _wpoints: 'ArrayVector2D' = None
    _cknots: 'NDArray[float64]' = None

    def __init__(self, ctlpnts: 'ArrayVector2D', **kwargs: Dict[str, Any]) -> None:
        self.ctlpnts = ctlpnts.flatten()
        self.weights = kwargs.get('weights',
                                  ones(ctlpnts.size, dtype=float64)).flatten()
        self.degree = kwargs.get('degree', self.ctlpnts.size - 1)
        self.knots = kwargs.get('knots', default_knots(self.ctlpnts.size,
                                                       self.degree))
        self.endpoint = kwargs.get('endpoint', True)
        self.closed = kwargs.get('closed', False)

    @property
    def wpoints(self) -> 'ArrayVector2D':
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    @property
    def cknots(self) -> 'NDArray[float64]':
        if self._cknots is None:
            if self.endpoint:
                kbeg = full(self.degree, self.knots[0])
                kend = full(self.degree, self.knots[-1])
                self._cknots = concatenate((kbeg, self.knots, kend))
            else:
                self._cknots = self.knots
        return self._cknots

    def basis_functions(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_functions(self.degree, self.cknots, u)

    def basis_first_derivatives(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_first_derivatives(self.degree, self.cknots, u)

    def basis_second_derivatives(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_second_derivatives(self.degree, self.cknots, u)

    def evaluate_points_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        numer = self.wpoints@Nu
        denom = self.weights@Nu
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_first_derivatives_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        dNu = self.basis_first_derivatives(u)
        numer = self.wpoints@Nu
        dnumer = self.wpoints@dNu
        denom = self.weights@Nu
        ddenom = self.weights@dNu
        deriv1 = (dnumer*denom - numer*ddenom)/denom**2
        if deriv1.size == 1:
            deriv1 = deriv1[0]
        return deriv1

    def evaluate_second_derivatives_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        dNu = self.basis_first_derivatives(u)
        d2Nu = self.basis_second_derivatives(u)
        numer = self.wpoints@Nu
        dnumer = self.wpoints@dNu
        d2numer = self.wpoints@d2Nu
        denom = self.weights@Nu
        ddenom = self.weights@dNu
        d2denom = self.weights@d2Nu
        deriv2 = (d2numer*denom**2 - dnumer*2*ddenom*denom + numer*2*ddenom**2 - numer*d2denom*denom)/denom**3
        if deriv2.size == 1:
            deriv2 = deriv2[0]
        return deriv2

    def evaluate_u(self, num: int) -> 'NDArray[float64]':
        return knot_linspace(num, self.knots)

    def evaluate_points(self, num: int) -> 'ArrayVector2D':
        u = self.evaluate_u(num)
        return self.evaluate_points_at_u(u)

    def evaluate_first_derivatives(self, num: int) -> 'ArrayVector2D':
        u = self.evaluate_u(num)
        return self.evaluate_first_derivatives_at_u(u)

    def evaluate_second_derivatives(self, num: int) -> 'ArrayVector2D':
        u = self.evaluate_u(num)
        return self.evaluate_second_derivatives_at_u(u)
