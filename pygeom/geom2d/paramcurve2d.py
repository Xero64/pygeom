from typing import TYPE_CHECKING, Callable, Optional

from numpy import asarray, isscalar, linspace

from .vector2d import Vector2D

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom2d import Vector2D
    ParamCallable = Callable[['NDArray'], 'Vector2D']


class ParamCurve2D():
    ru: 'ParamCallable' = None
    drdu: 'ParamCallable' = None
    d2rdu2: 'ParamCallable' = None

    def __init__(self, ru: 'ParamCallable',
                 drdu: Optional['ParamCallable'] = None,
                 d2rdu2: Optional['ParamCallable'] = None) -> None:
        self.ru = ru
        self.drdu = drdu
        self.d2rdu2 = d2rdu2

    def evaluate_points_at_u(self, u: 'NDArray') -> 'Vector2D':
        if isscalar(u):
            u = asarray([u])
        ru = self.ru(u)
        if ru.size == 1:
            ru = ru[0]
        return ru

    def evaluate_first_derivatives_at_u(self, u: 'NDArray') -> 'Vector2D':
        if isscalar(u):
            u = asarray([u])
        drdu = self.drdu(u)
        if drdu.size == 1:
            drdu = drdu[0]
        return drdu

    def evaluate_second_derivatives_at_u(self, u: 'NDArray') -> 'Vector2D':
        if isscalar(u):
            u = asarray([u])
        d2rdu2 = self.d2rdu2(u)
        if d2rdu2.size == 1:
            d2rdu2 = d2rdu2[0]
        return d2rdu2

    def evaluate_u(self, num: int) -> 'NDArray':
        return linspace(0.0, 1.0, num + 1)

    def evaluate_points(self, num: int) -> 'Vector2D':
        u = self.evaluate_u(num)
        return self.evaluate_points_at_u(u)

    def evaluate_first_derivatives(self, num: int) -> 'Vector2D':
        u = self.evaluate_u(num)
        return self.evaluate_first_derivatives_at_u(u)

    def evaluate_second_derivatives(self, num: int) -> 'Vector2D':
        u = self.evaluate_u(num)
        return self.evaluate_second_derivatives_at_u(u)

    def __repr__(self) -> str:
        return f'<ParamCurve2D>'
