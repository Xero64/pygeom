from typing import TYPE_CHECKING

from .point import Point

if TYPE_CHECKING:
    from .vector import Vector

class Line():
    """Line Class"""
    pnta: 'Point' = None
    pntb: 'Point' = None
    _vec: 'Vector' = None
    _length: float = None
    
    def __init__(self, pnta: 'Point', pntb: 'Point') -> None:
        self.pnta = pnta
        self.pntb = pntb
        
    @property
    def vec(self) -> 'Vector':
        if self._vec is None:
            self._vec = self.pntb - self.pnta
        return self._vec
    
    @property
    def length(self) -> float:
        if self._length is None:
            self._length = self.vec.return_magnitude()
        return self._length
    
    def centre_point(self) -> 'Point':
        """Returns the centre point of this line"""
        x = (self.pnta.x + self.pntb.x)/2
        y = (self.pnta.y + self.pntb.y)/2
        z = (self.pnta.z + self.pntb.z)/2
        return Point(x, y, z)
    
    def ratio_point(self, ratio) -> 'Point':
        """Returns a point a certain ratio along the line"""
        return self.pnta + ratio*self.vec
    
    def __repr__(self) -> str:
        return '<Line>'
