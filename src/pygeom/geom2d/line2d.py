from .point2d import Point2D
from .vector2d import Vector2D


class Line2D():
    """Line2D Class"""
    pnta: 'Point2D' = None
    pntb: 'Point2D' = None
    _vec: Vector2D = None
    _length: float = None
    _uvec: Vector2D = None

    def __init__(self, pnta: 'Point2D', pntb: 'Point2D') -> None:
        self.pnta = pnta
        self.pntb = pntb

    @property
    def vec(self) -> Vector2D:
        if self._vec is None:
            self._vec = self.pntb - self.pnta
        return self._vec

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self.vec.return_magnitude()
        return self._length

    @property
    def direc(self) -> Vector2D:
        if self._direc is None:
            self._direc = self.vec/self.length
        return self._direc

    def centre_point(self) -> Vector2D:
        """Returns the centre point of this line"""
        return (self.pnta + self.pntb)/2

    def ratio_point(self, ratio: float) -> Vector2D:
        """Returns a point a certain ratio along the line"""
        return self.pnta + ratio*self.vec

    def __repr__(self) -> str:
        return '<Line2D>'
