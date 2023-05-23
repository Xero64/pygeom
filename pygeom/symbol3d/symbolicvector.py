
from typing import Any
from pygeom.geom3d.vector import Vector
from sympy import Symbol, sqrt
from sympy import expand, simplify, trigsimp, expand_trig
from sympy import diff, integrate

class SymbolicVector(Vector):
    x: 'Symbol' = None
    y: 'Symbol' = None
    z: 'Symbol' = None

    def __init__(self, x: 'Symbol', y: 'Symbol', z: 'Symbol') -> None:
        self.x = x
        self.y = y
        self.z = z

    def return_magnitude(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def simplify(self) -> 'SymbolicVector':
        return SymbolicVector(simplify(self.x),
                              simplify(self.y),
                              simplify(self.z))

    def trigsimp(self) -> 'SymbolicVector':
        return SymbolicVector(trigsimp(self.x),
                              trigsimp(self.y),
                              trigsimp(self.z))

    def expand(self) -> 'SymbolicVector':
        return SymbolicVector(expand(self.x),
                              expand(self.y),
                              expand(self.z))

    def expand_trig(self) -> 'SymbolicVector':
        return SymbolicVector(expand_trig(self.x),
                              expand_trig(self.y),
                              expand_trig(self.z))

    def integrate(self, terms) -> 'SymbolicVector':
        return SymbolicVector(integrate(self.x, terms),
                              integrate(self.y, terms),
                              integrate(self.z, terms))

    def diff(self, sym: 'Symbol') -> 'SymbolicVector':
        return SymbolicVector(diff(self.x, sym),
                              diff(self.y, sym),
                              diff(self.z, sym))

    def subs(self, *args) -> 'SymbolicVector':
        return SymbolicVector(self.x.subs(*args),
                              self.y.subs(*args),
                              self.z.subs(*args))

    def to_unit(self) -> 'SymbolicVector':
        mag = self.return_magnitude()
        return SymbolicVector(self.x/mag, self.y/mag, self.z/mag)

    def __mul__(self, obj: Any):
        if isinstance(obj, SymbolicVector):
            return self.x*obj.x+self.y*obj.y+self.z*obj.z
        else:
            x = obj*self.x
            y = obj*self.y
            z = obj*self.z
            return SymbolicVector(x, y, z)

    def __rmul__(self, obj: Any) -> 'SymbolicVector':
        return self.__mul__(obj)

    def __truediv__(self, obj: Any) -> 'SymbolicVector':
        x = self.x/obj
        y = self.y/obj
        z = self.z/obj
        return SymbolicVector(x, y, z)

    def __pow__(self, obj: Any) -> 'SymbolicVector':
        if isinstance(obj, SymbolicVector):
            x = self.y*obj.z-self.z*obj.y
            y = self.z*obj.x-self.x*obj.z
            z = self.x*obj.y-self.y*obj.x
            return SymbolicVector(x, y, z)

    def __add__(self, obj: Any) -> 'SymbolicVector':
        if isinstance(obj, SymbolicVector):
            x = self.x+obj.x
            y = self.y+obj.y
            z = self.z+obj.z
            return SymbolicVector(x, y, z)

    def __radd__(self, obj: Any) -> 'SymbolicVector':
        if obj == 0 or obj is None:
            return self
        else:
            return self.__add__(obj)

    def __sub__(self, obj: Any) -> 'SymbolicVector':
        if isinstance(obj, SymbolicVector):
            x = self.x-obj.x
            y = self.y-obj.y
            z = self.z-obj.z
            return SymbolicVector(x, y, z)

    def __pos__(self) -> 'SymbolicVector':
        return self

    def __neg__(self) -> 'SymbolicVector':
        return SymbolicVector(-self.x, -self.y, -self.z)

def symple_vector(label, **kwargs) -> 'SymbolicVector':
    x = Symbol(f'{label:s}.x', **kwargs)
    y = Symbol(f'{label:s}.y', **kwargs)
    z = Symbol(f'{label:s}.z', **kwargs)
    return SymbolicVector(x, y, z)
