from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy import linspace, meshgrid, shape

from .vector import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray
    ParamCallable = Callable[['NDArray', 'NDArray'], Vector]


class ParamSurface():
    ruv: 'ParamCallable' = None
    drdu: 'ParamCallable' = None
    drdv: 'ParamCallable' = None

    def __init__(self, ruv: 'ParamCallable',
                 drdu: 'ParamCallable | None' = None,
                 drdv: 'ParamCallable | None' = None) -> None:
        self.ruv = ruv
        self.drdu = drdu
        self.drdv = drdv

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> Vector:
        um, vm = meshgrid(u, v, indexing='ij')
        return self.ruv(um, vm)

    def evaluate_points_at_uv_mesh(self, um: 'NDArray',
                                   vm: 'NDArray') -> tuple[Vector, Vector]:
        if shape(um) != shape(vm):
            raise ValueError('The shapes of um and vm must be the same.')
        return self.ruv(um, vm)

    def evaluate_tangents_at_uv(self, u: 'NDArray', v: 'NDArray') -> tuple[Vector,
                                                                           Vector]:
        um, vm = meshgrid(u, v, indexing='ij')
        drdu = self.drdu(um, vm)
        drdv = self.drdv(um, vm)
        return drdu, drdv

    def evaluate_tangents_at_uv_mesh(self, um: 'NDArray',
                                     vm: 'NDArray') -> tuple[Vector, Vector]:
        if shape(um) != shape(vm):
            raise ValueError('The shapes of um and vm must be the same.')
        drdu = self.drdu(um, vm)
        drdv = self.drdv(um, vm)
        return drdu, drdv

    def evaluate_normals_at_uv(self, u: 'NDArray', v: 'NDArray') -> Vector:
        drdu, drdv = self.evaluate_tangents_at_uv(u, v)
        return drdu.cross(drdv)

    def evaluate_uv(self, numu: int, numv: int) -> tuple['NDArray',
                                                         'NDArray']:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return u, v

    def evaluate_uv_mesh(self, numu: int, numv: int) -> tuple['NDArray',
                                                              'NDArray']:
        u, v = self.evaluate_uv(numu, numv)
        return meshgrid(u, v, indexing='ij')

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
        return f'<ParamSurface>'
