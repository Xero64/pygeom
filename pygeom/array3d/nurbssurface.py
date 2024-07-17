from typing import TYPE_CHECKING, Tuple, Union

from numpy import asarray, float64, linspace
from pygeom.tools.basis import basis_derivatives, basis_functions

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.array3d import ArrayVector
    from pygeom.geom3d import Vector
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector, ArrayVector]


class BSplineSurface():
    ctlpnts: 'ArrayVector' = None
    uknots: 'NDArray[float64]' = None
    vknots: 'NDArray[float64]' = None
    udegree: int = None
    vdegree: int = None

    def __init__(self, ctlpnts: 'ArrayVector', uknots: 'NDArray[float64]', vknots: 'NDArray[float64]',
                 udegree: int = None, vdegree: int = None) -> None:
        self.ctlpnts = ctlpnts
        self.uknots = uknots
        self.vknots = vknots
        self.udegree = ctlpnts.shape[0] - 1
        self.vdegree = ctlpnts.shape[1] - 1
        if udegree is not None:
            self.udegree = udegree
        if vdegree is not None:
            self.vdegree = vdegree

    def basis_functions(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                   'NDArray[float64]']:
        Nu = basis_functions(self.udegree, self.uknots, u)
        Nv = basis_functions(self.vdegree, self.vknots, v)
        return Nu, Nv

    def basis_derivatives(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                     'NDArray[float64]']:
        dNu = basis_derivatives(self.udegree, self.uknots, u)
        dNv = basis_derivatives(self.vdegree, self.vknots, v)
        return dNu, dNv

    def evaluate_points_at_uv(self, u: 'Numeric', v: 'Numeric') -> 'VectorLike':
        if isinstance(u, float64):
            u = asarray([u], dtype=float64)
        if isinstance(v, float64):
            v = asarray([v], dtype=float64)
        Nu, Nv = self.basis_functions(u, v)
        points = (self.ctlpnts.transpose()@Nu).transpose()@Nv
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'Numeric', v: 'Numeric') -> Tuple['VectorLike',
                                                                           'VectorLike']:
        if isinstance(u, float64):
            u = asarray([u], dtype=float64)
        if isinstance(v, float64):
            v = asarray([v], dtype=float64)
        Nu, Nv = self.basis_functions(u, v)
        dNu, dNv = self.basis_derivatives(u, v)
        tangent_u = (self.ctlpnts.transpose()@dNu).transpose()@Nv
        tangent_v = (self.ctlpnts.transpose()@Nu).transpose()@dNv
        if tangent_u.size == 1:
            tangent_u = tangent_u[0]
        if tangent_v.size == 1:
            tangent_v = tangent_v[0]
        return tangent_u, tangent_v

    def evaluate_uv(self, numu: int, numv: int) -> Tuple['NDArray[float64]',
                                                         'NDArray[float64]']:
        umin = self.uknots.min()
        umax = self.uknots.max()
        u = linspace(umin, umax, numu, dtype=float64)
        vmin = self.vknots.min()
        vmax = self.vknots.max()
        v = linspace(vmin, vmax, numv, dtype=float64)
        return u, v

    def evaluate_points(self, numu: int, numv: int) -> 'ArrayVector':
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['ArrayVector',
                                                               'ArrayVector']:
        u, v = self.evaluate_uv(numu, numv)
        tangents_u, tangents_v = self.evaluate_tangents_at_uv(u, v)
        return tangents_u, tangents_v


class NurbsSurface():
    ctlpnts: 'ArrayVector' = None
    weights: 'NDArray[float64]' = None
    uknots: 'NDArray[float64]' = None
    vknots: 'NDArray[float64]' = None
    udegree: int = None
    vdegree: int = None
    _wpoints: 'ArrayVector' = None

    def __init__(self, ctlpnts: 'ArrayVector', weights: 'NDArray[float64]',
                 uknots: 'NDArray[float64]', vknots: 'NDArray[float64]',
                 udegree: int = None, vdegree: int = None) -> None:
        self.ctlpnts = ctlpnts
        self.weights = weights
        self.uknots = uknots
        self.vknots = vknots
        self.udegree = ctlpnts.shape[0] - 1
        self.vdegree = ctlpnts.shape[1] - 1
        if udegree is not None:
            self.udegree = udegree
        if vdegree is not None:
            self.vdegree = vdegree

    @property
    def wpoints(self) -> 'ArrayVector':
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    def basis_functions(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                   'NDArray[float64]']:
        Nu = basis_functions(self.udegree, self.uknots, u)
        Nv = basis_functions(self.vdegree, self.vknots, v)
        return Nu, Nv

    def basis_derivatives(self, u: 'Numeric', v: 'Numeric') -> Tuple['NDArray[float64]',
                                                                     'NDArray[float64]']:
        dNu = basis_derivatives(self.udegree, self.uknots, u)
        dNv = basis_derivatives(self.vdegree, self.vknots, v)
        return dNu, dNv

    def evaluate_points_at_uv(self, u: 'Numeric', v: 'Numeric') -> 'VectorLike':
        Nu, Nv = self.basis_functions(u, v)
        numer = (self.wpoints.transpose()@Nu).transpose()@Nv
        denom = (self.weights.transpose()@Nu).transpose()@Nv
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'Numeric', v: 'Numeric') -> Tuple['VectorLike',
                                                                           'VectorLike']:
        Nu, Nv = self.basis_functions(u, v)
        dNu, dNv = self.basis_derivatives(u, v)
        numer = (self.wpoints.transpose()@Nu).transpose()@Nv
        dnumer_u = (self.wpoints.transpose()@dNu).transpose()@Nv
        dnumer_v = (self.wpoints.transpose()@Nu).transpose()@dNv
        denom = (self.weights.transpose()@Nu).transpose()@Nv
        ddenom_u = (self.weights.transpose()@dNu).transpose()@Nv
        ddenom_v = (self.weights.transpose()@Nu).transpose()@dNv
        tangent_u = (dnumer_u*denom - numer*ddenom_u)/denom**2
        tangent_v = (dnumer_v*denom - numer*ddenom_v)/denom**2
        if tangent_u.size == 1:
            tangent_u = tangent_u[0]
        if tangent_v.size == 1:
            tangent_v = tangent_v[0]
        return tangent_u, tangent_v

    def evaluate_uv(self, numu: int, numv: int) -> Tuple['NDArray[float64]',
                                                         'NDArray[float64]']:
        umin = self.uknots.min()
        umax = self.uknots.max()
        u = linspace(umin, umax, numu, dtype=float64)
        vmin = self.vknots.min()
        vmax = self.vknots.max()
        v = linspace(vmin, vmax, numv, dtype=float64)
        return u, v

    def evaluate_points(self, numu: int, numv: int) -> 'ArrayVector':
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['ArrayVector',
                                                               'ArrayVector']:
        u, v = self.evaluate_uv(numu, numv)
        tangents_u, tangents_v = self.evaluate_tangents_at_uv(u, v)
        return tangents_u, tangents_v
