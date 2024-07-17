from typing import TYPE_CHECKING, Union, Tuple

from numpy import asarray, float64, linspace
from pygeom.tools.bernstein import bernstein_polynomials, bernstein_derivatives

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector
    from pygeom.array3d import ArrayVector
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector, ArrayVector]


class BezierSurface():
    ctlpnts: 'ArrayVector' = None
    _udegree: int = None
    _vdegree: int = None

    def __init__(self, ctlpnts: 'ArrayVector') -> None:
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

    def bernstein_polynomials(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                         'NDArray[float64]']:
        Bu = bernstein_polynomials(self.udegree, u)
        Bv = bernstein_polynomials(self.vdegree, v)
        return Bu, Bv

    def bernstein_derivatives(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                         'NDArray[float64]']:
        dBu = bernstein_derivatives(self.udegree, u)
        dBv = bernstein_derivatives(self.vdegree, v)
        return dBu, dBv

    def evaluate_points_at_uv(self, u: 'Numeric', v: 'Numeric') -> 'VectorLike':
        Bu, Bv = self.bernstein_polynomials(u, v)
        points = (self.ctlpnts.transpose()@Bu).transpose()@Bv
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'Numeric', v: 'Numeric') -> Tuple['VectorLike',
                                                                           'VectorLike']:
        Bu, Bv = self.bernstein_polynomials(u, v)
        dBu, dBv = self.bernstein_derivatives(u, v)
        tangents_u = (self.ctlpnts.transpose()@dBu).transpose()@Bv
        tangents_v = (self.ctlpnts.transpose()@Bu).transpose()@dBv
        if tangents_u.size == 1:
            tangents_u = tangents_u[0]
        if tangents_v.size == 1:
            tangents_v = tangents_v[0]
        return tangents_u, tangents_v

    def evaluate_points(self, numu: int, numv: int) -> 'ArrayVector':
        u = linspace(0.0, 1.0, numu, dtype=float64)
        v = linspace(0.0, 1.0, numv, dtype=float64)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['ArrayVector',
                                                               'ArrayVector']:
        u = linspace(0.0, 1.0, numu, dtype=float64)
        v = linspace(0.0, 1.0, numv, dtype=float64)
        return self.evaluate_tangents_at_uv(u, v)


class RationalBezierSurface():
    ctlpnts: 'ArrayVector' = None
    weights: 'NDArray[float64]' = None
    _udegree: int = None
    _vdegree: int = None
    _wpoints: 'ArrayVector' = None

    def __init__(self, ctlpnts: 'ArrayVector', weights: 'NDArray[float64]') -> None:
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
    def wpoints(self) -> 'ArrayVector':
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    def bernstein_polynomials(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                         'NDArray[float64]']:
        Bu = bernstein_polynomials(self.udegree, u)
        Bv = bernstein_polynomials(self.vdegree, v)
        return Bu, Bv

    def bernstein_derivatives(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                         'NDArray[float64]']:
        dBu = bernstein_derivatives(self.udegree, u)
        dBv = bernstein_derivatives(self.vdegree, v)
        return dBu, dBv

    def evaluate_points_at_uv(self, u: 'Numeric', v: 'Numeric') -> 'VectorLike':
        Bu, Bv = self.bernstein_polynomials(u, v)
        numer = (self.wpoints.transpose()@Bu).transpose()@Bv
        denom = (self.weights.transpose()@Bu).transpose()@Bv
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'Numeric', v: 'Numeric') -> Tuple['VectorLike',
                                                                           'VectorLike']:
        Bu, Bv = self.bernstein_polynomials(u, v)
        dBu, dBv = self.bernstein_derivatives(u, v)
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

    def evaluate_points(self, numu: int, numv: int) -> 'ArrayVector':
        u = linspace(0.0, 1.0, numu, dtype=float64)
        v = linspace(0.0, 1.0, numv, dtype=float64)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['ArrayVector',
                                                               'ArrayVector']:
        u = linspace(0.0, 1.0, numu, dtype=float64)
        v = linspace(0.0, 1.0, numv, dtype=float64)
        return self.evaluate_tangents_at_uv(u, v)
