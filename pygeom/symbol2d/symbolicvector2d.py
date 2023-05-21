
from typing import Any
from pygeom.geom2d.vector2d import Vector2D
from sympy import Symbol, sqrt
from sympy import expand, simplify, trigsimp, expand_trig
from sympy import diff, integrate

class SymbolicVector2D(Vector2D):
    x: 'Symbol' = None
    y: 'Symbol' = None
    def __init__(self, x: 'Symbol', y: 'Symbol') -> None:
        self.x = x
        self.y = y
    def return_magnitude(self):
        return sqrt(self.x**2 + self.y**2)
    def simplify(self) -> 'SymbolicVector2D':
        return SymbolicVector2D(simplify(self.x),
                                simplify(self.y))
    def trigsimp(self) -> 'SymbolicVector2D':
        return SymbolicVector2D(trigsimp(self.x),
                                trigsimp(self.y))
    def expand(self) -> 'SymbolicVector2D':
        return SymbolicVector2D(expand(self.x),
                                expand(self.y))
    def expand_trig(self) -> 'SymbolicVector2D':
        return SymbolicVector2D(expand_trig(self.x),
                                expand_trig(self.y))
    def integrate(self, terms) -> 'SymbolicVector2D':
        return SymbolicVector2D(integrate(self.x, terms),
                                integrate(self.y, terms))
    def diff(self, sym: 'Symbol') -> 'SymbolicVector2D':
        return SymbolicVector2D(diff(self.x, sym),
                                diff(self.y, sym))
    def subs(self, *args) -> 'SymbolicVector2D':
        return SymbolicVector2D(self.x.subs(*args),
                                self.y.subs(*args))
    def to_unit(self) -> 'SymbolicVector2D':
        mag = self.return_magnitude()
        return SymbolicVector2D(self.x/mag, self.y/mag)
    def __mul__(self, obj: Any):
        if isinstance(obj, SymbolicVector2D):
            return self.x*obj.x + self.y*obj.y
        else:
            x = obj*self.x
            y = obj*self.y
            return SymbolicVector2D(x, y)
    def __rmul__(self, obj: Any) -> 'SymbolicVector2D':
        return self.__mul__(obj)
    def __truediv__(self, obj: Any) -> 'SymbolicVector2D':
        x = self.x/obj
        y = self.y/obj
        return SymbolicVector2D(x, y)
    def __pow__(self, obj: Any):
        if isinstance(obj, SymbolicVector2D):
            return self.x*obj.y - self.y*obj.x
    def __add__(self, obj: Any) -> 'SymbolicVector2D':
        if isinstance(obj, SymbolicVector2D):
            x = self.x + obj.x
            y = self.y + obj.y
            return SymbolicVector2D(x, y)
    def __radd__(self, obj: Any) -> 'SymbolicVector2D':
        if obj == 0 or obj is None:
            return self
        else:
            return self.__add__(obj)
    def __sub__(self, obj: Any) -> 'SymbolicVector2D':
        if isinstance(obj, SymbolicVector2D):
            x = self.x - obj.x
            y = self.y - obj.y
            return SymbolicVector2D(x, y)
    def __pos__(self) -> 'SymbolicVector2D':
        return SymbolicVector2D(self.x, self.y)
    def __neg__(self) -> 'SymbolicVector2D':
        return SymbolicVector2D(-self.x, -self.y)

def symple_vector2d(label, **kwargs) -> 'SymbolicVector2D':
    x = Symbol(f'{label:s}.x', **kwargs)
    y = Symbol(f'{label:s}.y', **kwargs)
    return SymbolicVector2D(x, y)
