from collections.abc import Callable
from typing import TYPE_CHECKING

from numpy import linspace

from .vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    ParamCallable = Callable[['NDArray'], Vector2D]


class ParamCurve2D():
    ru: 'ParamCallable' = None
    drdu: 'ParamCallable' = None
    d2rdu2: 'ParamCallable' = None

    def __init__(self, ru: 'ParamCallable',
                 drdu: 'ParamCallable | None' = None,
                 d2rdu2: 'ParamCallable | None' = None) -> None:
        self.ru = ru
        self.drdu = drdu
        self.d2rdu2 = d2rdu2

    def evaluate_points_at_t(self, u: 'NDArray') -> Vector2D:
        ru = self.ru(u)
        return ru

    def evaluate_first_derivatives_at_t(self, u: 'NDArray') -> Vector2D:
        drdu = self.drdu(u)
        return drdu

    def evaluate_second_derivatives_at_t(self, u: 'NDArray') -> Vector2D:
        d2rdu2 = self.d2rdu2(u)
        return d2rdu2

    def evaluate_tangents_at_t(self, u: 'NDArray') -> Vector2D:
        return self.evaluate_first_derivatives_at_t(u).to_unit()

    def evaluate_normals_at_t(self, u: 'NDArray') -> Vector2D:
        return self.evaluate_second_derivatives_at_t(u).to_unit()

    def evaluate_curvatures_at_t(self, u: 'NDArray') -> 'NDArray':
        drdu = self.evaluate_first_derivatives_at_t(u)
        d2rdu2 = self.evaluate_second_derivatives_at_t(u)
        return drdu.cross(d2rdu2)/drdu.return_magnitude()**3

    def evaluate_t(self, num: int) -> 'NDArray':
        return linspace(0.0, 1.0, num + 1)

    def evaluate_points(self, num: int) -> Vector2D:
        u = self.evaluate_t(num)
        return self.evaluate_points_at_t(u)

    def evaluate_first_derivatives(self, num: int) -> Vector2D:
        u = self.evaluate_t(num)
        return self.evaluate_first_derivatives_at_t(u)

    def evaluate_second_derivatives(self, num: int) -> Vector2D:
        u = self.evaluate_t(num)
        return self.evaluate_second_derivatives_at_t(u)

    def evaluate_tangents(self, num: int) -> Vector2D:
        u = self.evaluate_t(num)
        return self.evaluate_tangents_at_t(u)

    def evaluate_normals(self, num: int) -> Vector2D:
        u = self.evaluate_t(num)
        return self.evaluate_normals_at_t(u)

    def evaluate_curvatures(self, num: int) -> 'NDArray':
        u = self.evaluate_t(num)
        return self.evaluate_curvatures_at_t(u)

    def __repr__(self) -> str:
        return f'<ParamCurve2D>'
