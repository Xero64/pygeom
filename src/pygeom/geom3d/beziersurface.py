from typing import TYPE_CHECKING

from numpy import linspace

from ..tools.bernstein import (bernstein_first_derivatives,
                                    bernstein_polynomials)

from .vector import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BezierSurface():
    ctlpnts: Vector = None
    _udegree: int = None
    _vdegree: int = None

    def __init__(self, ctlpnts: Vector) -> None:
        self.ctlpnts = ctlpnts

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def udegree(self) -> int:
        if self._udegree is None:
            self._udegree = self.ctlpnts.shape[0] - 1
        return self._udegree

    @property
    def vdegree(self) -> int:
        if self._vdegree is None:
            self._vdegree = self.ctlpnts.shape[1] - 1
        return self._vdegree

    def bernstein_polynomials(self, u: 'NDArray', v: 'NDArray') -> tuple['NDArray',
                                                                         'NDArray']:
        Bu = bernstein_polynomials(self.udegree, u)
        Bv = bernstein_polynomials(self.vdegree, v)
        return Bu, Bv

    def bernstein_first_derivatives(self, u: 'NDArray', v: 'NDArray') -> tuple['NDArray',
                                                                         'NDArray']:
        dBu = bernstein_first_derivatives(self.udegree, u)
        dBv = bernstein_first_derivatives(self.vdegree, v)
        return dBu, dBv

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> Vector:
        Bu, Bv = self.bernstein_polynomials(u, v)
        points = (self.ctlpnts.transpose()@Bu).transpose()@Bv
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'NDArray', v: 'NDArray') -> tuple[Vector,
                                                                           Vector]:
        Bu, Bv = self.bernstein_polynomials(u, v)
        dBu, dBv = self.bernstein_first_derivatives(u, v)
        tangents_u = (self.ctlpnts.transpose()@dBu).transpose()@Bv
        tangents_v = (self.ctlpnts.transpose()@Bu).transpose()@dBv
        if tangents_u.size == 1:
            tangents_u = tangents_u[0]
        if tangents_v.size == 1:
            tangents_v = tangents_v[0]
        return tangents_u, tangents_v

    def evaluate_points(self, numu: int, numv: int) -> Vector:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> tuple[Vector,
                                                               Vector]:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return self.evaluate_tangents_at_uv(u, v)


class RationalBezierSurface():
    ctlpnts: Vector = None
    weights: 'NDArray' = None
    _udegree: int = None
    _vdegree: int = None
    _wpoints: Vector = None

    def __init__(self, ctlpnts: Vector, weights: 'NDArray') -> None:
        if ctlpnts.shape != weights.shape:
            raise ValueError('Control points and weights must have the same shape')
        self.ctlpnts = ctlpnts
        self.weights = weights

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    @property
    def udegree(self) -> int:
        if self._udegree is None:
            self._udegree = self.ctlpnts.shape[0] - 1
        return self._udegree

    @property
    def vdegree(self) -> int:
        if self._vdegree is None:
            self._vdegree = self.ctlpnts.shape[1] - 1
        return self._vdegree

    @property
    def wpoints(self) -> Vector:
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    def bernstein_polynomials(self, u: 'NDArray', v: 'NDArray') -> tuple['NDArray',
                                                                         'NDArray']:
        Bu = bernstein_polynomials(self.udegree, u)
        Bv = bernstein_polynomials(self.vdegree, v)
        return Bu, Bv

    def bernstein_first_derivatives(self, u: 'NDArray', v: 'NDArray') -> tuple['NDArray',
                                                                         'NDArray']:
        dBu = bernstein_first_derivatives(self.udegree, u)
        dBv = bernstein_first_derivatives(self.vdegree, v)
        return dBu, dBv

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> Vector:
        Bu, Bv = self.bernstein_polynomials(u, v)
        numer = (self.wpoints.transpose()@Bu).transpose()@Bv
        denom = (self.weights.transpose()@Bu).transpose()@Bv
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'NDArray', v: 'NDArray') -> tuple[Vector,
                                                                           Vector]:
        Bu, Bv = self.bernstein_polynomials(u, v)
        dBu, dBv = self.bernstein_first_derivatives(u, v)
        numer = (self.wpoints.transpose()@Bu).transpose()@Bv
        dnumer_u = (self.wpoints.transpose()@dBu).transpose()@Bv
        dnumer_v = (self.wpoints.transpose()@Bu).transpose()@dBv
        denom = (self.weights.transpose()@Bu).transpose()@Bv
        ddenom_u = (self.weights.transpose()@dBu).transpose()@Bv
        ddenom_v = (self.weights.transpose()@Bu).transpose()@dBv
        tangents_u = (dnumer_u*denom - numer*ddenom_u)/denom**2
        tangents_v = (dnumer_v*denom - numer*ddenom_v)/denom**2
        if tangents_u.size == 1:
            tangents_u = tangents_u[0]
        if tangents_v.size == 1:
            tangents_v = tangents_v[0]
        return tangents_u, tangents_v

    def evaluate_points(self, numu: int, numv: int) -> Vector:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> tuple[Vector,
                                                               Vector]:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return self.evaluate_tangents_at_uv(u, v)
