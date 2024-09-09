from typing import TYPE_CHECKING, Any, Dict, Tuple

from numpy import concatenate, float64, full, linspace, ones

from ..tools.basis import (basis_first_derivatives, basis_functions,
                           default_knots, knot_linspace)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector


class BSplineSurface():
    ctlpnts: 'Vector' = None
    uknots: 'NDArray' = None
    vknots: 'NDArray' = None
    udegree: int = None
    vdegree: int = None

    def __init__(self, ctlpnts: 'Vector', uknots: 'NDArray', vknots: 'NDArray',
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

    def basis_functions(self, u: 'NDArray', v: 'NDArray') -> Tuple['NDArray',
                                                                   'NDArray']:
        Nu = basis_functions(self.udegree, self.uknots, u)
        Nv = basis_functions(self.vdegree, self.vknots, v)
        return Nu, Nv

    def basis_first_derivatives(self, u: 'NDArray', v: 'NDArray') -> Tuple['NDArray',
                                                                     'NDArray']:
        dNu = basis_first_derivatives(self.udegree, self.uknots, u)
        dNv = basis_first_derivatives(self.vdegree, self.vknots, v)
        return dNu, dNv

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> 'Vector':
        Nu, Nv = self.basis_functions(u, v)
        points = self.ctlpnts.rmatmul(Nu.transpose())@Nv
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'NDArray', v: 'NDArray') -> Tuple['Vector',
                                                                           'Vector']:
        Nu, Nv = self.basis_functions(u, v)
        dNu, dNv = self.basis_first_derivatives(u, v)
        tangent_u = self.ctlpnts.rmatmul(dNu.transpose())@Nv
        tangent_v = self.ctlpnts.rmatmul(Nu.transpose())@dNv
        if tangent_u.size == 1:
            tangent_u = tangent_u[0]
        if tangent_v.size == 1:
            tangent_v = tangent_v[0]
        return tangent_u, tangent_v

    def evaluate_uv(self, numu: int, numv: int) -> Tuple['NDArray',
                                                         'NDArray']:
        umin = self.uknots.min()
        umax = self.uknots.max()
        u = linspace(umin, umax, numu, dtype=float64)
        vmin = self.vknots.min()
        vmax = self.vknots.max()
        v = linspace(vmin, vmax, numv, dtype=float64)
        return u, v

    def evaluate_points(self, numu: int, numv: int) -> 'Vector':
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['Vector',
                                                               'Vector']:
        u, v = self.evaluate_uv(numu, numv)
        tangents_u, tangents_v = self.evaluate_tangents_at_uv(u, v)
        return tangents_u, tangents_v


class NurbsSurface():
    ctlpnts: 'Vector' = None
    weights: 'NDArray' = None
    udegree: int = None
    vdegree: int = None
    uknots: 'NDArray' = None
    vknots: 'NDArray' = None
    _wpoints: 'Vector' = None
    _cuknots: 'NDArray' = None
    _cvknots: 'NDArray' = None

    def __init__(self, ctlpnts: 'Vector', **kwargs: Dict[str, Any]) -> None:
        self.ctlpnts = ctlpnts

        self.weights = kwargs.get('weights', ones(ctlpnts.shape, dtype=float64))
        if self.ctlpnts.shape != self.weights.shape:
            raise ValueError('Control points and weights must have the same shape.')

        usize = self.ctlpnts.shape[0]
        self.udegree = kwargs.get('udegree', usize - 1)
        self.uknots = kwargs.get('uknots', default_knots(usize, self.udegree))
        self.uendpoint = kwargs.get('uendpoint', True)
        self.uclosed = kwargs.get('uclosed', False)

        vsize = self.ctlpnts.shape[1]
        self.vdegree = kwargs.get('vdegree', vsize - 1)
        self.vknots = kwargs.get('vknots', default_knots(vsize, self.vdegree))
        self.vendpoint = kwargs.get('vendpoint', True)
        self.vclosed = kwargs.get('vclosed', False)

    @property
    def wpoints(self) -> 'Vector':
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    @property
    def cuknots(self) -> 'NDArray':
        if self._cuknots is None:
            if self.uendpoint:
                kbeg = full(self.udegree, self.uknots[0])
                kend = full(self.udegree, self.uknots[-1])
                self._cuknots = concatenate((kbeg, self.uknots, kend))
            else:
                self._cuknots = self.uknots
        return self._cuknots

    @property
    def cvknots(self) -> 'NDArray':
        if self._cvknots is None:
            if self.vendpoint:
                kbeg = full(self.vdegree, self.vknots[0])
                kend = full(self.vdegree, self.vknots[-1])
                self._cvknots = concatenate((kbeg, self.vknots, kend))
            else:
                self._cvknots = self.vknots
        return self._cvknots

    def basis_functions(self, u: 'NDArray', v: 'NDArray') -> Tuple['NDArray',
                                                                   'NDArray']:
        Nu = basis_functions(self.udegree, self.cuknots, u)
        Nv = basis_functions(self.vdegree, self.cvknots, v)
        return Nu, Nv

    def basis_first_derivatives(self, u: 'NDArray', v: 'NDArray') -> Tuple['NDArray',
                                                                     'NDArray']:
        dNu = basis_first_derivatives(self.udegree, self.cuknots, u)
        dNv = basis_first_derivatives(self.vdegree, self.cvknots, v)
        return dNu, dNv

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> 'Vector':
        Nu, Nv = self.basis_functions(u, v)
        numer = self.wpoints.rmatmul(Nu.transpose())@Nv
        denom = Nu.transpose()@self.weights@Nv
        points = numer/denom
        if points.size == 1:
            points = points[0]
        return points

    def evaluate_tangents_at_uv(self, u: 'NDArray', v: 'NDArray') -> Tuple['Vector',
                                                                           'Vector']:
        Nu, Nv = self.basis_functions(u, v)
        dNu, dNv = self.basis_first_derivatives(u, v)
        numer = self.wpoints.rmatmul(Nu.transpose())@Nv
        dnumer_u = self.wpoints.rmatmul(dNu.transpose())@Nv
        dnumer_v = self.wpoints.rmatmul(Nu.transpose())@dNv
        denom = Nu.transpose()@self.weights@Nv
        ddenom_u = dNu.transpose()@self.weights@Nv
        ddenom_v = Nu.transpose()@self.weights@dNv
        tangent_u = (dnumer_u*denom - numer*ddenom_u)/denom**2
        tangent_v = (dnumer_v*denom - numer*ddenom_v)/denom**2
        if tangent_u.size == 1:
            tangent_u = tangent_u[0]
        if tangent_v.size == 1:
            tangent_v = tangent_v[0]
        return tangent_u, tangent_v

    def evaluate_uv(self, numu: int, numv: int) -> Tuple['NDArray',
                                                         'NDArray']:
        u = knot_linspace(numu, self.uknots)
        v = knot_linspace(numv, self.vknots)
        return u, v

    def evaluate_points(self, numu: int, numv: int) -> 'Vector':
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['Vector',
                                                               'Vector']:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_tangents_at_uv(u, v)
