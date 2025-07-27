from .point import Point
from .vector import Vector


class Line():
    """Line Class"""
    pnta: 'Point' = None
    pntb: 'Point' = None
    _vec: Vector = None
    _length: float = None
    _direc: Vector = None

    def __init__(self, pnta: 'Point', pntb: 'Point') -> None:
        self.pnta = pnta
        self.pntb = pntb

    @property
    def vec(self) -> Vector:
        if self._vec is None:
            self._vec = self.pntb - self.pnta
        return self._vec

    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self.vec.return_magnitude()
        return self._length

    @property
    def direc(self) -> Vector:
        if self._direc is None:
            self._direc = self.vec/self.length
        return self._direc

    def centre_point(self) -> Vector:
        """Returns the centre point of this line"""
        return (self.pnta + self.pntb)/2

    def ratio_point(self, ratio: float) -> 'Point':
        """Returns a point a certain ratio along the line"""
        return self.pnta + ratio*self.vec

    def __repr__(self) -> str:
        return '<Line>'
