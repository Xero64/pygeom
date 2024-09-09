from typing import TYPE_CHECKING

from numpy import asarray, linspace
from pygeom.symbol3d import SymbolicVector
from pygeom.tools.bernstein import (bernstein_derivatives,
                                    bernstein_polynomials,
                                    symbolic_bernstein_derivatives,
                                    symbolic_bernstein_polynomials)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector


class BezierCurve():
    ctlpnts: 'Vector' = None
    _degree: int = None

    def __init__(self, ctlpnts: 'Vector') -> None:
        self.ctlpnts = ctlpnts

    @property
    def degree(self) -> int:
        if self._degree is None:
            self._degree = self.ctlpnts.size - 1
        return self._degree

    def bernstein_polynomials(self, t: 'NDArray') -> 'NDArray':
        return bernstein_polynomials(self.degree, t)

    def bernstein_derivatives(self, t: 'NDArray') -> 'NDArray':
        return bernstein_derivatives(self.degree, t)

    def evaluate_points_at_t(self, t: 'NDArray') -> 'Vector':
        if isinstance(t, float):
            t = asarray([t], dtype=float)
        polys = self.bernstein_polynomials(t)
        points = self.ctlpnts@polys
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_t(self, t: 'NDArray') -> 'Vector':
        if isinstance(t, float):
            t = asarray([t], dtype=float)
        dpolys = self.bernstein_derivatives(t)
        tangents = self.ctlpnts@dpolys
        if tangents.size == 1:
            tangents = tangents[0]
        return tangents

    def symbolic_expression(self) -> SymbolicVector:
        from sympy import Symbol
        t = Symbol('t', real=True)
        polys = symbolic_bernstein_polynomials(self.degree, t)
        expr = SymbolicVector(0, 0, 0)
        for ctlpnt, poly in zip(self.ctlpnts, polys):
            expr += ctlpnt*poly
        return expr

    def symbolic_derivative(self) -> SymbolicVector:
        from sympy import Symbol
        t = Symbol('t', real=True)
        dpolys = symbolic_bernstein_derivatives(self.degree, t)
        dexpr = SymbolicVector(0, 0, 0)
        for ctlpnt, dpoly in zip(self.ctlpnts, dpolys):
            dexpr += ctlpnt*dpoly
        return dexpr

    def evaluate_points(self, num: int) -> 'Vector':
        t = linspace(0.0, 1.0, num)
        return self.evaluate_points_at_t(t)

    def evaluate_tangents(self, num: int) -> 'Vector':
        t = linspace(0.0, 1.0, num)
        return self.evaluate_tangents_at_t(t)


class RationalBezierCurve():
    ctlpnts: 'Vector' = None
    weights: 'NDArray' = None
    _degree: int = None
    _wpoints: 'Vector' = None

    def __init__(self, ctlpnts: 'Vector',
                 weights: 'NDArray') -> None:
        if ctlpnts.shape != weights.shape:
            raise ValueError('Control points and weights must have the same shape')
        self.ctlpnts = ctlpnts
        self.weights = weights

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def degree(self) -> int:
        if self._degree is None:
            self._degree = self.ctlpnts.size - 1
        return self._degree

    @property
    def wpoints(self) -> 'Vector':
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    def bernstein_polynomials(self, t: 'NDArray') -> 'NDArray':
        return bernstein_polynomials(self.degree, t)

    def bernstein_derivatives(self, t: 'NDArray') -> 'NDArray':
        return bernstein_derivatives(self.degree, t)

    def evaluate_points_at_t(self, t: 'NDArray') -> 'Vector':
        polys = self.bernstein_polynomials(t)
        numer = self.wpoints@polys
        denom = self.weights@polys
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_t(self, t: 'NDArray') -> 'Vector':
        polys = self.bernstein_polynomials(t)
        dpolys = self.bernstein_derivatives(t)
        numer = self.wpoints@polys
        dnumer = self.wpoints@dpolys
        denom = self.weights@polys
        ddenom = self.weights@dpolys
        tangents = (dnumer*denom - numer*ddenom)/denom**2
        if tangents.size == 1:
            tangents = tangents[0]
        return tangents

    def symbolic_expression(self) -> SymbolicVector:
        from sympy import Symbol
        t = Symbol('t', real=True, positive=True)
        polys = symbolic_bernstein_polynomials(self.degree, t)
        numer = SymbolicVector(0, 0, 0)
        denom = 0
        for ctlpnt, weight, poly in zip(self.ctlpnts, self.weights, polys):
            wp = weight*poly
            numer += ctlpnt*wp
            denom += wp
        expr = numer/denom
        return expr

    def symbolic_derivative(self) -> SymbolicVector:
        from sympy import Symbol
        t = Symbol('t', real=True, positive=True)
        polys = symbolic_bernstein_polynomials(self.degree, t)
        dpolys = symbolic_bernstein_derivatives(self.degree, t)
        numer = SymbolicVector(0, 0, 0)
        dnumer = SymbolicVector(0, 0, 0)
        denom = 0
        ddenom = 0
        for ctlpnt, weight, poly, dpoly in zip(self.ctlpnts, self.weights, polys, dpolys):
            wp = weight*poly
            dwp = weight*dpoly
            numer += ctlpnt*wp
            dnumer += ctlpnt*dwp
            denom += wp
            ddenom += dwp
        dexpr = (dnumer*denom - numer*ddenom)/denom**2
        return dexpr

    def evaluate_points(self, num: int) -> 'Vector':
        t = linspace(0.0, 1.0, num)
        return self.evaluate_points_at_t(t)

    def evaluate_tangents(self, num: int) -> 'Vector':
        t = linspace(0.0, 1.0, num)
        return self.evaluate_tangents_at_t(t)
