from typing import TYPE_CHECKING, Any, Dict, Union

from numpy import concatenate, float64, full, linspace, ones

from ..geom3d import Vector
from ..tools.basis import (basis_first_derivatives, basis_functions, default_knots,
                           knot_linspace)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array3d import ArrayVector
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector, ArrayVector]


class BSplineCurve():
    ctlpnts: 'ArrayVector' = None
    knots: 'NDArray[float64]' = None
    degree: int = None
    _wpoints: 'ArrayVector' = None

    def __init__(self, ctlpnts: 'ArrayVector',
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

    def evaluate_points(self, num: int) -> 'ArrayVector':
        umin = self.knots.min()
        umax = self.knots.max()
        u = linspace(umin, umax, num, dtype=float64)
        return self.evaluate_points_at_u(u)

    def evaluate_tangents(self, num: int) -> 'ArrayVector':
        umin = self.knots.min()
        umax = self.knots.max()
        u = linspace(umin, umax, num, dtype=float64)
        return self.evaluate_tangents_at_u(u)


class NurbsCurve():
    ctlpnts: 'ArrayVector' = None
    weights: 'NDArray[float64]' = None
    degree: int = None
    knots: 'NDArray[float64]' = None
    endpoint: bool = None
    closed: bool = None
    _wpoints: 'ArrayVector' = None
    _cknots: 'NDArray[float64]' = None

    def __init__(self, ctlpnts: 'ArrayVector', **kwargs: Dict[str, Any]) -> None:
        self.ctlpnts = ctlpnts.flatten()
        self.weights = kwargs.get('weights', ones(ctlpnts.size, dtype=float64))
        self.degree = kwargs.get('degree', self.ctlpnts.size - 1)
        self.knots = kwargs.get('knots', default_knots(self.ctlpnts.size,
                                                       self.degree))
        self.endpoint = kwargs.get('endpoint', True)
        self.closed = kwargs.get('closed', False)

    @property
    def wpoints(self) -> 'ArrayVector':
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

    def evaluate_points_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        numer = self.wpoints@Nu
        denom = self.weights@Nu
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        dNu = self.basis_first_derivatives(u)
        numer = self.wpoints@Nu
        dnumer = self.wpoints@dNu
        denom = self.weights@Nu
        ddenom = self.weights@dNu
        tangents = (dnumer*denom - numer*ddenom)/denom**2
        if tangents.size == 1:
            tangents = tangents[0]
        return tangents

    def evaluate_u(self, num: int) -> 'NDArray[float64]':
        return knot_linspace(num, self.knots)

    def evaluate_points(self, num: int) -> 'ArrayVector':
        u = self.evaluate_u(num)
        return self.evaluate_points_at_u(u)

    def evaluate_tangents(self, num: int) -> 'ArrayVector':
        u = self.evaluate_u(num)
        return self.evaluate_tangents_at_u(u)
