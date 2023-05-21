from typing import TYPE_CHECKING

from .point2d import Point2D
from . import vector2d_from_points

if TYPE_CHECKING:
    from numpy import number
    from .vector2d import Vector2D

class Line2D():
    """Line2D Class"""
    pnta: 'Point2D' = None
    pntb: 'Point2D' = None
    _vec: 'Vector2D' = None
    _length: 'number' = None

    def __init__(self, pnta: 'Point2D', pntb: 'Point2D') -> None:
        self.pnta = pnta
        self.pntb = pntb

    @property
    def vec(self) -> 'Vector2D':
        if self._vec is None:
            self._vec = vector2d_from_points(self.pnta, self.pntb)
        return self._vec

    @property
    def length(self) -> 'number':
        if self._length is None:
            self._length = self.vec.return_magnitude()
        return self._length

    def centre_point(self) -> 'Vector2D':
        """Returns the centre point of this line"""
        return (self.pnta + self.pntb)/2

    def ratio_point(self, ratio: 'number') -> 'Vector2D':
        """Returns a point a certain ratio along the line"""
        vec = self.pntb - self.pnta
        return self.pnta + ratio*vec

    def __repr__(self) -> str:
        return '<Line2D>'
