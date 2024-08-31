from typing import TYPE_CHECKING, Callable, Optional, Tuple, Union

from numpy import asarray, isscalar, linspace, meshgrid

if TYPE_CHECKING:
    from numpy import float64
    from numpy.typing import NDArray
    from pygeom.array3d import ArrayVector
    from pygeom.geom3d import Vector
    Numeric = Union[float64, NDArray[float64]]
    VectorLike = Union[Vector, ArrayVector]
    ParamCallable = Callable[['Numeric', 'Numeric'], 'VectorLike']


class ParamSurface():
    ruv: 'ParamCallable' = None
    drdu: 'ParamCallable' = None
    drdv: 'ParamCallable' = None

    def __init__(self, ruv: 'ParamCallable',
                 drdu: Optional['ParamCallable'] = None,
                 drdv: Optional['ParamCallable'] = None) -> None:
        self.ruv = ruv
        self.drdu = drdu
        self.drdv = drdv

    def evaluate_points_at_uv(self, u: 'Numeric', v: 'Numeric') -> 'VectorLike':
        if isscalar(u):
            u = asarray([u])
        if isscalar(v):
            v = asarray([v])
        vm, um = meshgrid(v, u)
        ruv = self.ruv(um, vm)
        if ruv.size == 1:
            ruv = ruv[0]
        return ruv

    def evaluate_tangents_at_uv(self, u: 'Numeric', v: 'Numeric') -> Tuple['VectorLike',
                                                                           'VectorLike']:
        if isscalar(u):
            u = asarray([u])
        if isscalar(v):
            v = asarray([v])
        vm, um = meshgrid(v, u)
        drdu = self.drdu(um, vm)
        drdv = self.drdv(um, vm)
        if drdu.size == 1:
            drdu = drdu[0]
        if drdv.size == 1:
            drdv = drdv[0]
        return drdu, drdv

    def evaluate_uv(self, numu: int, numv: int) -> Tuple['NDArray[float64]',
                                                         'NDArray[float64]']:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return u, v

    def evaluate_points(self, numu: int, numv: int) -> 'ArrayVector':
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['ArrayVector',
                                                               'ArrayVector']:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_tangents_at_uv(u, v)
