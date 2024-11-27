from typing import TYPE_CHECKING, Any

from numpy import concatenate, full, ones, shape, zeros

from ..tools.basis import (basis_first_derivatives, basis_functions,
                           default_knots, knot_linspace)
from .vector import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NurbsSurface():
    ctlpnts: Vector = None
    weights: 'NDArray' = None
    udegree: int = None
    vdegree: int = None
    uknots: 'NDArray' = None
    vknots: 'NDArray' = None
    uendpoint: bool = None
    vendpoint: bool = None
    rational: bool = None
    _wpoints: Vector = None
    _ucknots: 'NDArray' = None
    _vcknots: 'NDArray' = None

    def __init__(self, ctlpnts: Vector, **kwargs: dict[str, Any]) -> None:
        self.ctlpnts = ctlpnts

        self.weights = kwargs.get('weights', ones(ctlpnts.shape))
        if self.ctlpnts.shape != self.weights.shape:
            raise ValueError('Control points and weights must have the same shape.')

        usize = self.ctlpnts.shape[0]
        self.udegree = kwargs.get('udegree', usize - 1)
        self.uknots = kwargs.get('uknots', default_knots(usize, self.udegree))
        self.uendpoint = kwargs.get('uendpoint', True)

        vsize = self.ctlpnts.shape[1]
        self.vdegree = kwargs.get('vdegree', vsize - 1)
        self.vknots = kwargs.get('vknots', default_knots(vsize, self.vdegree))
        self.vendpoint = kwargs.get('vendpoint', True)

    def reset(self) -> None:
        for attr in self.__dict__:
            if attr.startswith('_'):
                setattr(self, attr, None)

    def copy(self) -> 'NurbsSurface':
        ctlpnts = self.ctlpnts.copy()
        weights = self.weights.copy()
        udegree = self.udegree
        vdegree = self.vdegree
        uknots = self.uknots.copy()
        vknots = self.vknots.copy()
        uendpoint = self.uendpoint
        vendpoint = self.vendpoint
        return NurbsSurface(ctlpnts, weights=weights, udegree=udegree,
                            vdegree=vdegree, uknots=uknots, vknots=vknots,
                            uendpoint=uendpoint, vendpoint=vendpoint)

    @property
    def wpoints(self) -> Vector:
        if self._wpoints is None:
            self._wpoints = self.ctlpnts*self.weights
        return self._wpoints

    @property
    def ucknots(self) -> 'NDArray':
        if self._ucknots is None:
            if self.uendpoint:
                kbeg = full(self.udegree, self.uknots[0])
                kend = full(self.udegree, self.uknots[-1])
                self._ucknots = concatenate((kbeg, self.uknots, kend))
            else:
                self._ucknots = self.uknots
        return self._ucknots

    @property
    def vcknots(self) -> 'NDArray':
        if self._vcknots is None:
            if self.vendpoint:
                kbeg = full(self.vdegree, self.vknots[0])
                kend = full(self.vdegree, self.vknots[-1])
                self._vcknots = concatenate((kbeg, self.vknots, kend))
            else:
                self._vcknots = self.vknots
        return self._vcknots

    @property
    def rational(self) -> bool:
        check: 'NDArray' = self.weights == 1.0
        return not check.all()

    def basis_functions(self, u: 'NDArray', v: 'NDArray') -> tuple['NDArray',
                                                                   'NDArray']:
        Nu = basis_functions(self.udegree, self.ucknots, u)
        Nv = basis_functions(self.vdegree, self.vcknots, v)
        return Nu, Nv

    def basis_first_derivatives(self, u: 'NDArray', v: 'NDArray') -> tuple['NDArray',
                                                                     'NDArray']:
        dNu = basis_first_derivatives(self.udegree, self.ucknots, u)
        dNv = basis_first_derivatives(self.vdegree, self.vcknots, v)
        return dNu, dNv

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> Vector:
        Nu, Nv = self.basis_functions(u, v)
        numer = Nu.transpose()@self.wpoints@Nv
        if self.rational:
            denom = Nu.transpose()@self.weights@Nv
            points = numer/denom
        else:
            points = numer
        return points

    def evaluate_points_at_uv_mesh(self, um: 'NDArray', vm: 'NDArray') -> Vector:
        shp = shape(um)
        if shp != shape(vm):
            raise ValueError('The shapes of um and vm must be the same.')
        Nu, Nv = self.basis_functions(um, vm)
        numu = Nu.shape[0]
        numv = Nv.shape[0]
        numer = Vector.zeros(shp)
        for i in range(numu):
            for j in range(numv):
                numer += self.wpoints[i, j]*Nu[i]*Nv[j]
        if self.rational:
            denom = zeros(shp)
            for i in range(numu):
                for j in range(numv):
                    denom += self.weights[i, j]*Nu[i]*Nv[j]
            points = numer/denom
        else:
            points = numer
        return points

    def evaluate_tangents_at_uv(self, u: 'NDArray',
                                v: 'NDArray') -> tuple[Vector, Vector]:
        Nu, Nv = self.basis_functions(u, v)
        dNu, dNv = self.basis_first_derivatives(u, v)
        dnumer_u = dNu.transpose()@self.wpoints@Nv
        dnumer_v = Nu.transpose()@self.wpoints@dNv
        if self.rational:
            numer = Nu.transpose()@self.wpoints@Nv
            denom = Nu.transpose()@self.weights@Nv
            ddenom_u = dNu.transpose()@self.weights@Nv
            ddenom_v = Nu.transpose()@self.weights@dNv
            tangent_u = (dnumer_u*denom - numer*ddenom_u)/denom**2
            tangent_v = (dnumer_v*denom - numer*ddenom_v)/denom**2
        else:
            tangent_u = dnumer_u
            tangent_v = dnumer_v
        return tangent_u, tangent_v

    def evaluate_normals_at_uv(self, u: 'NDArray', v: 'NDArray') -> Vector:
        tangent_u, tangent_v = self.evaluate_tangents_at_uv(u, v)
        normal = tangent_u.cross(tangent_v)
        return normal

    def evaluate_tangents_at_uv_mesh(self, um: 'NDArray',
                                     vm: 'NDArray') -> tuple[Vector, Vector]:
        shp = shape(um)
        if shp != shape(vm):
            raise ValueError('The shapes of um and vm must be the same.')
        Nu, Nv = self.basis_functions(um, vm)
        dNu, dNv = self.basis_first_derivatives(um, vm)
        numu = Nu.shape[0]
        numv = Nv.shape[0]
        dnumer_u = Vector.zeros(shp)
        dnumer_v = Vector.zeros(shp)
        for i in range(numu):
            for j in range(numv):
                dnumer_u += self.wpoints[i, j]*dNu[i]*Nv[j]
                dnumer_v += self.wpoints[i, j]*Nu[i]*dNv[j]
        if self.rational:
            numer = Vector.zeros(shp)
            denom = zeros(shp)
            ddenom_u = zeros(shp)
            ddenom_v = zeros(shp)
            for i in range(numu):
                for j in range(numv):
                    numer += self.wpoints[i, j]*Nu[i]*Nv[j]
                    denom += self.weights[i, j]*Nu[i]*Nv[j]
                    ddenom_u += self.weights[i, j]*dNu[i]*Nv[j]
                    ddenom_v += self.weights[i, j]*Nu[i]*dNv[j]
            tangent_u = (dnumer_u*denom - numer*ddenom_u)/denom**2
            tangent_v = (dnumer_v*denom - numer*ddenom_v)/denom**2
        else:
            tangent_u = dnumer_u
            tangent_v = dnumer_v
        return tangent_u, tangent_v

    def evaluate_uv(self, numu: int, numv: int) -> tuple['NDArray',
                                                         'NDArray']:
        u = knot_linspace(numu, self.uknots)
        v = knot_linspace(numv, self.vknots)
        return u, v

    def evaluate_points(self, numu: int, numv: int) -> Vector:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> tuple[Vector,
                                                               Vector]:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_tangents_at_uv(u, v)

    def evaluate_normals(self, numu: int, numv: int) -> Vector:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_normals_at_uv(u, v)

    def __repr__(self) -> str:
        return f'<NurbsSurface: udegree={self.udegree:d}, vdegree={self.vdegree:d}>'

    def __str__(self) -> str:
        outstr = f'NurbsSurface\n'
        outstr += f'  control points: \n{self.ctlpnts}\n'
        outstr += f'  weights: {self.weights}\n'
        outstr += f'  udegree: {self.udegree:d}\n'
        outstr += f'  vdegree: {self.vdegree:d}\n'
        outstr += f'  uknots: {self.uknots}\n'
        outstr += f'  vknots: {self.vknots}\n'
        outstr += f'  ucknots: {self.ucknots}\n'
        outstr += f'  vcknots: {self.vcknots}\n'
        outstr += f'  uendpoint: {self.uendpoint}\n'
        outstr += f'  vendpoint: {self.vendpoint}\n'
        return outstr


class BSplineSurface(NurbsSurface):

    def __init__(self, ctlpnts: Vector, **kwargs: dict[str, Any]) -> None:
        kwargs['weights'] = ones(ctlpnts.shape)
        kwargs['rational'] = False
        super().__init__(ctlpnts, **kwargs)

    def __repr__(self) -> str:
        return f'<BSplineSurface: udegree={self.udegree:d}, vdegree={self.vdegree:d}>'

    def __str__(self) -> str:
        outstr = f'BSplineSurface\n'
        outstr += f'  control points: \n{self.ctlpnts}\n'
        outstr += f'  udegree: {self.udegree:d}\n'
        outstr += f'  vdegree: {self.vdegree:d}\n'
        outstr += f'  uknots: {self.uknots}\n'
        outstr += f'  vknots: {self.vknots}\n'
        outstr += f'  ucknots: {self.ucknots}\n'
        outstr += f'  vcknots: {self.vcknots}\n'
        outstr += f'  uendpoint: {self.uendpoint}\n'
        outstr += f'  vendpoint: {self.vendpoint}\n'
        return outstr
