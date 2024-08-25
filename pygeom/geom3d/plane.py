from typing import TYPE_CHECKING, Tuple

from numpy import zeros
from numpy.linalg import lstsq

if TYPE_CHECKING:
    from .vector import Vector


class Plane():
    """Plane Class"""
    pnt: 'Vector' = None
    nrm: 'Vector' = None

    def __init__(self, pnt: 'Vector', nrm: 'Vector') -> None:
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


def plane_from_3_points(pnta: 'Vector', pntb: 'Vector', pntc: 'Vector') -> Plane:
    pnt = (pnta + pntb + pntc)/3
    vecab = pntb - pnta
    vecbc = pntc - pntb
    nrm = vecab.cross(vecbc).to_unit()
    return Plane(pnt, nrm)

def plane_from_multiple_points(*pnts: 'Vector', rcond: float = None) -> Plane:
    num = len(pnts)
    amat = zeros((num, 3))
    bmat = zeros((num, 1))
    for i, pnt in enumerate(pnts):
        amat[i, :] = [pnt.x, pnt.y, pnt.z]
        bmat[i, 0] = -1
    xmat, _, _, _ = lstsq(amat, bmat, rcond=rcond)
    pnt = sum(pnts, Vector(0.0, 0.0, 0.0))/num
    nrm = Vector(*xmat).to_unit()
    return Plane(pnt, nrm)    
