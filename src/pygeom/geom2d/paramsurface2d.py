from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy import linspace, meshgrid

from .vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    ParamCallable = Callable[['NDArray', 'NDArray'], Vector2D]


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

    def evaluate_points_at_uv(self, u: 'NDArray', v: 'NDArray') -> Vector2D:
        vm, um = meshgrid(v, u)
        ruv = self.ruv(um, vm)
        return ruv

    def evaluate_tangents_at_uv(self, u: 'NDArray', v: 'NDArray') -> tuple[Vector2D,
                                                                           Vector2D]:
        vm, um = meshgrid(v, u)
        drdu = self.drdu(um, vm)
        drdv = self.drdv(um, vm)
        return drdu, drdv

    def evaluate_uv(self, numu: int, numv: int) -> tuple['NDArray',
                                                         'NDArray']:
        u = linspace(0.0, 1.0, numu)
        v = linspace(0.0, 1.0, numv)
        return u, v

    def evaluate_points(self, numu: int, numv: int) -> Vector2D:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_points_at_uv(u, v)

    def evaluate_tangents(self, numu: int, numv: int) -> tuple[Vector2D,
                                                               Vector2D]:
        u, v = self.evaluate_uv(numu, numv)
        return self.evaluate_tangents_at_uv(u, v)

    def __repr__(self) -> str:
        return f'<ParamSurface2D>'
