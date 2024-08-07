from typing import TYPE_CHECKING, Any, Dict, Union

from numpy import concatenate, float64, full, ones

from ..geom2d import Vector2D
from ..tools.basis import (basis_first_derivatives, basis_functions,
                           basis_second_derivatives, default_knots,
                           knot_linspace)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array2d import ArrayVector2D
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector2D, ArrayVector2D]


class NurbsCurve2D():
    ctlpnts: 'ArrayVector2D' = None
    weights: 'NDArray[float64]' = None
    degree: int = None
    knots: 'NDArray[float64]' = None
    endpoint: bool = None
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
    
    @property
    def rational(self) -> bool:
        check: 'NDArray[float64]' = self.weights == 1.0
        return not check.all()

    def basis_functions(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_functions(self.degree, self.cknots, u)

    def basis_first_derivatives(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_first_derivatives(self.degree, self.cknots, u)

    def basis_second_derivatives(self, u: 'Numeric') -> 'NDArray[float64]':
        return basis_second_derivatives(self.degree, self.cknots, u)

    def evaluate_points_at_u(self, u: 'Numeric') -> 'VectorLike':
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

    def evaluate_first_derivatives_at_u(self, u: 'Numeric') -> 'VectorLike':
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

    def evaluate_second_derivatives_at_u(self, u: 'Numeric') -> 'VectorLike':
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
    
    def __repr__(self) -> str:
        return f'<NurbsCurve2D: degree={self.degree:d}>'
    
    def __str__(self) -> str:
        outstr = f'NurbsCurve2D\n'
        outstr += f'  degree: {self.degree:d}\n'
        outstr += f'  control points: \n{self.ctlpnts}\n'
        outstr += f'  weights: {self.weights}\n'
        outstr += f'  knots: {self.knots}\n'
        outstr += f'  endpoint: {self.endpoint}\n'
        return outstr


class BSplineCurve2D(NurbsCurve2D):

    def __init__(self, ctlpnts: 'ArrayVector2D', **kwargs: Dict[str, Any]) -> None:
        kwargs['weights'] = ones(ctlpnts.shape, dtype=float64)
        super().__init__(ctlpnts, **kwargs)

    def __repr__(self) -> str:
        return f'<BSplineCurve2D: degree={self.degree:d}>'
    
    def __str__(self) -> str:
        outstr = f'BSplineCurve2D\n'
        outstr += f'  degree: {self.degree:d}\n'
        outstr += f'  control points: \n{self.ctlpnts}\n'
        outstr += f'  knots: {self.knots}\n'
        outstr += f'  endpoint: {self.endpoint}\n'
        return outstr
