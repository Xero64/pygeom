from typing import TYPE_CHECKING, Callable, Optional, Tuple

from numpy import linspace, meshgrid, shape

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector
    ParamCallable = Callable[['NDArray', 'NDArray'], 'Vector']


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

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> 'Vector':
        um, vm = meshgrid(u, v, indexing='ij')
        return self.ruv(um, vm)

    def evaluate_points_at_uv_mesh(self, um: 'NDArray',
                                   vm: 'NDArray') -> Tuple['Vector', 'Vector']:
        if shape(um) != shape(vm):
            raise ValueError('The shapes of um and vm must be the same.')
        return self.ruv(um, vm)

    def evaluate_tangents_at_uv(self, u: 'NDArray', v: 'NDArray') -> Tuple['Vector',
                                                                           'Vector']:
        um, vm = meshgrid(u, v, indexing='ij')
        drdu = self.drdu(um, vm)
        drdv = self.drdv(um, vm)
        return drdu, drdv

    def evaluate_tangents_at_uv_mesh(self, um: 'NDArray',
                                     vm: 'NDArray') -> Tuple['Vector', 'Vector']:
        if shape(um) != shape(vm):
            raise ValueError('The shapes of um and vm must be the same.')
        drdu = self.drdu(um, vm)
        drdv = self.drdv(um, vm)
        return drdu, drdv

    def evaluate_uv(self, numu: int, numv: int) -> Tuple['NDArray',
                                                         'NDArray']:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return u, v

    def evaluate_uv_mesh(self, numu: int, numv: int) -> Tuple['NDArray',
                                                              'NDArray']:
        u, v = self.evaluate_uv(numu, numv)
        return meshgrid(u, v, indexing='ij')

    def evaluate_points(self, numu: int, numv: int) -> 'Vector':
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> Tuple['Vector',
                                                               'Vector']:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_tangents_at_uv(u, v)

    def __repr__(self) -> str:
        return f'<ParamSurface>'
