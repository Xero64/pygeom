from typing import TYPE_CHECKING, Callable, Optional

from numpy import linspace

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pygeom.geom3d import Vector
    ParamCallable = Callable[['NDArray'], 'Vector']


class ParamCurve():
    ru: 'ParamCallable' = None
    drdu: 'ParamCallable' = None
    d2rdu2: 'ParamCallable' = None

    def __init__(self, ru: 'ParamCallable',
                 drdu: Optional['ParamCallable'] = None,
                 d2rdu2: Optional['ParamCallable'] = None) -> None:
        self.ru = ru
        self.drdu = drdu
        self.d2rdu2 = d2rdu2

    def evaluate_points_at_t(self, u: 'NDArray') -> 'Vector':
        return self.ru(u)

    def evaluate_first_derivatives_at_t(self, u: 'NDArray') -> 'Vector':
        return self.drdu(u)

    def evaluate_second_derivatives_at_t(self, u: 'NDArray') -> 'Vector':
        return self.d2rdu2(u)
    
    def evaluate_curvatures_at_t(self, u: 'NDArray') -> 'NDArray':
        drdu = self.evaluate_first_derivatives_at_t(u)
        d2rdu2 = self.evaluate_second_derivatives_at_t(u)
        return drdu.cross(d2rdu2)/drdu.return_magnitude()**3
    
    def evaluate_tangents_at_t(self, u: 'NDArray') -> 'Vector':
        return self.evaluate_first_derivatives_at_t(u).to_unit()
    
    def evaluate_normals_at_t(self, u: 'NDArray') -> 'Vector':
        return self.evaluate_second_derivatives_at_t(u).to_unit()
    
    def evaluate_binormals_at_t(self, u: 'NDArray') -> 'Vector':
        deriv1 = self.evaluate_first_derivatives_at_t(u)
        deriv2 = self.evaluate_second_derivatives_at_t(u)
        binormal = deriv1.cross(deriv2).to_unit()
        return binormal

    def evaluate_t(self, num: int) -> 'NDArray':
        return linspace(0.0, 1.0, num + 1)

    def evaluate_points(self, num: int) -> 'Vector':
        u = self.evaluate_t(num)
        return self.evaluate_points_at_t(u)

    def evaluate_first_derivatives(self, num: int) -> 'Vector':
        u = self.evaluate_t(num)
        return self.evaluate_first_derivatives_at_t(u)

    def evaluate_second_derivatives(self, num: int) -> 'Vector':
        u = self.evaluate_t(num)
        return self.evaluate_second_derivatives_at_t(u)
    
    def evaluate_curvatures(self, num: int) -> 'Vector':
        u = self.evaluate_t(num)
        return self.evaluate_curvatures_at_t(u)
    
    def evaluate_tangents(self, num: int) -> 'Vector':
        u = self.evaluate_t(num)
        return self.evaluate_tangents_at_t(u)
    
    def evaluate_normals(self, num: int) -> 'Vector':
        u = self.evaluate_t(num)
        return self.evaluate_normals_at_t(u)
    
    def evaluate_binormals(self, num: int) -> 'Vector':
        u = self.evaluate_t(num)
        return self.evaluate_binormals_at_t(u)

    def __repr__(self) -> str:
        return f'<ParamCurve>'
