from typing import Any

from numpy import empty
from sympy import (Symbol, collect, diff, expand, expand_trig, integrate,
                   simplify, sqrt, trigsimp)

from ..geom3d import Vector


class SymbolicVector(Vector):
    x: 'Symbol' = None
    y: 'Symbol' = None
    z: 'Symbol' = None

    def __init__(self, x: 'Symbol', y: 'Symbol', z: 'Symbol') -> None:
        self.x = x
        self.y = y
        self.z = z

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

    def collect(self, *args) -> 'SymbolicVector':
        return SymbolicVector(collect(self.x, *args),
                              collect(self.y, *args),
                              collect(self.z, *args))

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

    def to_unit(self, return_magnitude: bool = False) -> 'SymbolicVector | tuple[SymbolicVector, Symbol]':
        mag = self.return_magnitude()
        if mag != 0.0:
            x = self.x/mag
            y = self.y/mag
            z = self.z/mag
        else:
            x = self.x
            y = self.y
            z = self.z
        if return_magnitude:
            return SymbolicVector(x, y, z), mag
        else:
            return SymbolicVector(x, y, z)

    def return_magnitude(self) -> 'Symbol':
        """Returns the magnitude of this vector"""
        return sqrt(self.x**2 + self.y**2 + self.z**2).simplify()

    def to_xyz(self) -> tuple['Symbol', 'Symbol', 'Symbol']:
        """Returns the x, y and z values of this vector"""
        return self.x, self.y, self.z

    def dot(self, vec: 'SymbolicVector') -> 'Symbol':
        try:
            return self.x*vec.x + self.y*vec.y + self.z*vec.z
        except AttributeError:
            err = 'Vector dot product must be with Vector object.'
            raise TypeError(err)

    def cross(self, vec: 'SymbolicVector') -> 'SymbolicVector':
        try:
            x = self.y*vec.z - self.z*vec.y
            y = self.z*vec.x - self.x*vec.z
            z = self.x*vec.y - self.y*vec.x
            return SymbolicVector(x, y, z)
        except AttributeError:
            err = 'Vector cross product must be with Vector object.'
            raise TypeError(err)

    def rcross(self, vec: 'SymbolicVector') -> 'SymbolicVector':
        try:
            x = vec.y*self.z - vec.z*self.y
            y = vec.z*self.x - vec.x*self.z
            z = vec.x*self.y - vec.y*self.x
            return SymbolicVector(x, y, z)
        except AttributeError:
            err = 'Vector cross product must be with Vector object.'
            raise TypeError(err)

    def __abs__(self) -> 'Symbol':
        return self.return_magnitude()

    def __mul__(self, obj: Any) -> 'SymbolicVector':
        x = self.x*obj
        y = self.y*obj
        z = self.z*obj
        return SymbolicVector(x, y, z)

    def __rmul__(self, obj: Any) -> 'SymbolicVector':
        x = obj*self.x
        y = obj*self.y
        z = obj*self.z
        return SymbolicVector(x, y, z)

    def __truediv__(self, obj: Any) -> 'SymbolicVector':
        x = self.x/obj
        y = self.y/obj
        z = self.z/obj
        return SymbolicVector(x, y, z)

    def __pow__(self, obj: Any) -> 'SymbolicVector':
        x = self.x**obj
        y = self.y**obj
        z = self.z**obj
        return SymbolicVector(x, y, z)

    def __rpow__(self, obj: Any) -> 'SymbolicVector':
        x = obj**self.x
        y = obj**self.y
        z = obj**self.z
        return SymbolicVector(x, y, z)

    def __add__(self, obj: 'SymbolicVector') -> 'SymbolicVector':
        try:
            x = self.x + obj.x
            y = self.y + obj.y
            z = self.z + obj.z
            return SymbolicVector(x, y, z)
        except AttributeError:
            err = 'Vector object can only be added to Vector object.'
            raise TypeError(err)

    def __sub__(self, obj: 'SymbolicVector') -> 'SymbolicVector':
        try:
            x = self.x - obj.x
            y = self.y - obj.y
            z = self.z - obj.z
            return SymbolicVector(x, y, z)
        except AttributeError:
            err = 'Vector object can only be subtracted from Vector object.'
            raise TypeError(err)

    def __pos__(self) -> 'SymbolicVector':
        return self

    def __neg__(self) -> 'SymbolicVector':
        return SymbolicVector(-self.x, -self.y, -self.z)

    def __repr__(self) -> str:
        return '<SymbolicVector: {:}, {:}, {:}>'.format(self.x, self.y, self.z)

    def __str__(self) -> str:
        return '<{:}, {:}, {:}>'.format(self.x, self.y, self.z)

    def __format__(self, frm: str) -> str:
        frmstr = '<{:' + frm + '}, {:' + frm + '}, {:' + frm + '}>'
        return frmstr.format(self.x, self.y, self.z)

    def __eq__(self, obj: 'SymbolicVector') -> bool:
        try:
            if obj.x == self.x and obj.y == self.y and obj.z == self.z:
                return True
            else:
                return False
        except AttributeError:
            return False

    def __neq__(self, obj: 'SymbolicVector') -> bool:
        try:
            if obj.x != self.x or obj.y != self.y or obj.z != self.z:
                return True
            else:
                return False
        except AttributeError:
            return False

def empty_symbolicvector(shape: tuple[int, ...] | None = None,
                        **kwargs: dict[str, Any]) -> SymbolicVector:
    kwargs['dtype'] = object
    if shape is None:
        x, y, z = 0, 0, 0
    else:
        x = empty(shape, **kwargs)
        y = empty(shape, **kwargs)
        z = empty(shape, **kwargs)
    return SymbolicVector(x, y, z)

def symple_vector(label, **kwargs) -> 'SymbolicVector':
    x = Symbol(f'{label:s}.x', **kwargs)
    y = Symbol(f'{label:s}.y', **kwargs)
    z = Symbol(f'{label:s}.z', **kwargs)
    return SymbolicVector(x, y, z)
