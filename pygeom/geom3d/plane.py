from typing import Tuple

from .point import Point
from .vector import Vector

class Plane():
    """Plane Class"""
    pnt: 'Point' = None
    nrm: 'Vector' = None
    
    def __init__(self, pnt: 'Point', nrm: 'Vector') -> None:
        self.pnt = pnt
        self.nrm = nrm.to_unit()
        
    def return_abcd(self) -> Tuple[float, float, float, float]:
        a = self.nrm.x
        b = self.nrm.y
        c = self.nrm.z
        d = -a*self.pnt.x - b*self.pnt.y - c*self.pnt.z
        return a, b, c, d
    
    def point_z_from_plane(self, pnt: 'Vector') -> float:
        vec = pnt - self.pnt
        return vec*self.nrm
    
    def reverse_normal(self) -> None:
        self.nrm = -self.nrm
        
    def __repr__(self) -> str:
        return '<Plane>'
