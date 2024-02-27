from typing import Tuple, TYPE_CHECKING

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
