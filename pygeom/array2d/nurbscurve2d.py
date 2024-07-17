from typing import TYPE_CHECKING, Union

from numpy import asarray, float64, linspace

from ..geom2d import Vector2D
from ..tools.basis import basis_derivatives, basis_functions

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

    def basis_functions(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_functions(self.degree, self.knots, u)

    def basis_derivatives(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_derivatives(self.degree, self.knots, u)

    def evaluate_points_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        points = self.ctlpnts@Nu
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_u(self, u: 'Numeric') -> 'VectorLike':
        dNu = self.basis_derivatives(u)
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
    _wpoints: 'ArrayVector2D' = None

    def __init__(self, ctlpnts: 'ArrayVector2D', weights: 'NDArray[float64]',
                 knots: 'NDArray[float64]', degree: int = None) -> None:
        self.ctlpnts = ctlpnts
        self.weights = weights
        self.knots = knots
        self.degree = ctlpnts.size - 1
        if degree is not None:
            self.degree = degree

    @property
    def wpoints(self) -> 'ArrayVector2D':
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    def basis_functions(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_functions(self.degree, self.knots, u)

    def basis_derivatives(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_derivatives(self.degree, self.knots, u)

    def evaluate_points_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        print(f'Nu = {Nu}')
        print(f'Nu.shape = {Nu.shape}')
        print(f'wpoints = {self.wpoints}')
        numer = self.wpoints@Nu
        denom = self.weights@Nu
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_u(self, u: 'Numeric') -> 'VectorLike':
        Nu = self.basis_functions(u)
        dNu = self.basis_derivatives(u)
        numer = self.wpoints@Nu
        dnumer = self.wpoints@dNu
        denom = self.weights@Nu
        ddenom = self.weights@dNu
        vectors = (dnumer*denom - numer*ddenom)/denom**2
        if vectors.size == 1:
            vectors = vectors[0]
        return vectors

    def evaluate_points(self, num: int) -> 'ArrayVector2D':
        tmin = self.knots.min()
        tmax = self.knots.max()
        u = linspace(tmin, tmax, num, dtype=float64)
        return self.evaluate_points_at_u(u)

    def evaluate_tangents(self, num: int) -> 'ArrayVector2D':
        tmin = self.knots.min()
        tmax = self.knots.max()
        u = linspace(tmin, tmax, num, dtype=float64)
        return self.evaluate_tangents_at_u(u)
