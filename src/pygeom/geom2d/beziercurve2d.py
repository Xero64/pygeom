from typing import TYPE_CHECKING

from numpy import linspace

from ..tools.bernstein import (bernstein_first_derivatives,
                                    bernstein_polynomials)

from .vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BezierCurve2D():
    ctlpnts: Vector2D = None
    _degree: int = None

    def __init__(self, ctlpnts: Vector2D) -> None:
        self.ctlpnts = ctlpnts

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def degree(self) -> int:
        if self._degree is None:
            self._degree = self.ctlpnts.size - 1
        return self._degree

    def bernstein_polynomials(self, t: 'NDArray') -> 'NDArray':
        return bernstein_polynomials(self.degree, t)

    def bernstein_first_derivatives(self, t: 'NDArray') -> 'NDArray':
        return bernstein_first_derivatives(self.degree, t)

    def evaluate_points_at_t(self, t: 'NDArray') -> Vector2D:
        polys = bernstein_polynomials(self.degree, t)
        points = self.ctlpnts@polys
        return points

    def evaluate_tangents_at_t(self, t: 'NDArray') -> Vector2D:
        dpolys = bernstein_first_derivatives(self.degree, t)
        tangents = self.ctlpnts@dpolys
        return tangents

    def evaluate_points(self, num: int) -> Vector2D:
        t = linspace(0.0, 1.0, num)
        return self.evaluate_points_at_t(t)

    def evaluate_tangents(self, num: int) -> Vector2D:
        t = linspace(0.0, 1.0, num)
        return self.evaluate_tangents_at_t(t)


class RationalBezierCurve2D():
    ctlpnts: Vector2D = None
    weights: 'NDArray' = None
    _degree: int = None
    _wpoints: Vector2D = None

    def __init__(self, ctlpnts: Vector2D,
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
    def wpoints(self) -> Vector2D:
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    def bernstein_polynomials(self, t: 'NDArray') -> 'NDArray':
        return bernstein_polynomials(self.degree, t)

    def bernstein_first_derivatives(self, t: 'NDArray') -> 'NDArray':
        return bernstein_first_derivatives(self.degree, t)

    def evaluate_points_at_t(self, t: 'NDArray') -> Vector2D:
        polys = self.bernstein_polynomials(t)
        numer = self.wpoints@polys
        denom = self.weights@polys
        points = numer/denom
        return points

    def evaluate_tangents_at_t(self, t: 'NDArray') -> Vector2D:
        polys = self.bernstein_polynomials(t)
        dpolys = self.bernstein_first_derivatives(t)
        numer = self.wpoints@polys
        dnumer = self.wpoints@dpolys
        denom = self.weights@polys
        ddenom = self.weights@dpolys
        tangents = (dnumer*denom - numer*ddenom)/denom**2
        return tangents

    def evaluate_points(self, num: int) -> Vector2D:
        t = linspace(0.0, 1.0, num)
        return self.evaluate_points_at_t(t)

    def evaluate_tangents(self, num: int) -> Vector2D:
        t = linspace(0.0, 1.0, num)
        return self.evaluate_tangents_at_t(t)
