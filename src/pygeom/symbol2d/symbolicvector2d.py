from typing import Any

from numpy import empty
from sympy import (Symbol, atan, cos, diff, expand, expand_trig, integrate,
                   simplify, sin, sqrt, trigsimp)

from ..geom2d import Vector2D


class SymbolicVector2D(Vector2D):
    x: 'Symbol' = None
    y: 'Symbol' = None
    def __init__(self, x: 'Symbol', y: 'Symbol') -> None:
        self.x = x
        self.y = y

    def to_unit(self, return_magnitude: bool = False) -> 'SymbolicVector2D | tuple[SymbolicVector2D, Symbol]':
        """Returns the unit vector of this vector2d"""
        mag = self.return_magnitude()
        if mag != 0.0:
            x = self.x/mag
            y = self.y/mag
        else:
            x = self.x
            y = self.y
        if return_magnitude:
            return SymbolicVector2D(x, y), mag
        else:
            return SymbolicVector2D(x, y)

    def return_magnitude(self):
        """Returns the magnitude of this symbolicvector2d"""
        return sqrt(self.x**2 + self.y**2).simplify()

    def return_angle(self) -> 'Symbol':
        """Returns the angle of this vector from the x axis"""
        return atan(self.y/self.x)

    def dot(self, obj: 'SymbolicVector2D') -> 'Symbol':
        try:
            return self.x*obj.x + self.y*obj.y
        except AttributeError:
            err = 'SymbolicVector2D object can only be dotted with SymbolicVector2D object.'
            raise TypeError(err)

    def cross(self, obj: 'SymbolicVector2D') -> 'Symbol':
        try:
            return self.x*obj.y - self.y*obj.x
        except AttributeError:
            err = 'SymbolicVector2D object can only be crossed with SymbolicVector2D object.'
            raise TypeError(err)

    def __abs__(self) -> 'Symbol':
        return self.return_magnitude()

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

    def __mul__(self, obj: Any) -> 'SymbolicVector2D':
        x = self.x*obj
        y = self.y*obj
        return SymbolicVector2D(x, y)

    def __rmul__(self, obj: Any) -> 'SymbolicVector2D':
        x = obj*self.x
        y = obj*self.y
        return SymbolicVector2D(x, y)

    def __truediv__(self, obj: Any) -> 'SymbolicVector2D':
        x = self.x/obj
        y = self.y/obj
        return SymbolicVector2D(x, y)

    def __pow__(self, obj: Any) -> 'SymbolicVector2D':
        x = self.x**obj
        y = self.y**obj
        return SymbolicVector2D(x, y)

    def __rpow__(self, obj: Any) -> 'SymbolicVector2D':
        x = obj**self.x
        y = obj**self.y
        return SymbolicVector2D(x, y)

    def __add__(self, obj: 'SymbolicVector2D') -> 'SymbolicVector2D':
        try:
            x = self.x + obj.x
            y = self.y + obj.y
            return SymbolicVector2D(x, y)
        except AttributeError:
            err = 'SymbolicVector2D object can only be added to SymbolicVector2D object.'
            raise TypeError(err)

    def __sub__(self, obj: 'SymbolicVector2D') -> 'SymbolicVector2D':
        try:
            x = self.x - obj.x
            y = self.y - obj.y
            return SymbolicVector2D(x, y)
        except AttributeError:
            err = 'SymbolicVector2D object can only be subtracted from SymbolicVector2D object.'
            raise SymbolicVector2D(err)

    def __pos__(self) -> 'SymbolicVector2D':
        return self

    def __neg__(self) -> 'SymbolicVector2D':
        return SymbolicVector2D(-self.x, -self.y)

    def __repr__(self) -> str:
        return '<SymbolicVector2D: {:}, {:}>'.format(self.x, self.y)

    def __str__(self) -> str:
        return '<{:}, {:}>'.format(self.x, self.y)

    def __format__(self, frm: str) -> str:
        frmstr: str = '<{:'+ frm +'}, {:'+ frm +'}>'
        return frmstr.format(self.x, self.y)

    def __eq__(self, obj: 'SymbolicVector2D') -> bool:
        try:
            if obj.x == self.x and obj.y == self.y:
                return True
            else:
                return False
        except AttributeError:
            return False

    def __neq__(self, obj: 'SymbolicVector2D') -> bool:
        try:
            if obj.x != self.x or obj.y != self.y:
                return True
            else:
                return False
        except AttributeError:
            return False

    def rotate(self, rot: 'Symbol') -> 'SymbolicVector2D':
        """Rotates this vector by an input angle in radians"""
        dirx = SymbolicVector2D(cos(rot), sin(rot))
        diry = SymbolicVector2D(-sin(rot), cos(rot))
        x = self.dot(dirx)
        y = self.dot(diry)
        return SymbolicVector2D(x, y)

    def to_complex(self) -> 'Symbol':
        """Returns the complex float of this symbolic vector"""
        cplx = self.x + 1j*self.y
        return cplx

    def to_xy(self) -> tuple['Symbol', 'Symbol']:
        """Returns the x, y values of this symbolic vector"""
        return self.x, self.y

def empty_symbolicvector2d(shape: tuple[int, ...] | None = None,
                          **kwargs: dict[str, Any]) -> SymbolicVector2D:
    kwargs['dtype'] = object
    if shape is None:
        x, y = 0, 0
    else:
        x = empty(shape, **kwargs)
        y = empty(shape, **kwargs)
    return SymbolicVector2D(x, y)

def symple_vector2d(label, **kwargs) -> SymbolicVector2D:
    x = Symbol(f'{label:s}.x', **kwargs)
    y = Symbol(f'{label:s}.y', **kwargs)
    return SymbolicVector2D(x, y)
