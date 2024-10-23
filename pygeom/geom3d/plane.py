from .vector import Vector


class Plane():
    """Plane Class"""
    pnt: Vector = None
    nrm: Vector = None

    def __init__(self, pnt: Vector, nrm: Vector) -> None:
        self.pnt = pnt
        self.nrm = nrm.to_unit()

    def return_abcd(self) -> tuple[float, float, float, float]:
        a = self.nrm.x
        b = self.nrm.y
        c = self.nrm.z
        d = -a*self.pnt.x - b*self.pnt.y - c*self.pnt.z
        return a, b, c, d

    def point_z_from_plane(self, pnt: Vector) -> float:
        vec = pnt - self.pnt
        return vec*self.nrm

    def reverse_normal(self) -> None:
        self.nrm = -self.nrm

    @classmethod
    def from_3_points(cls, pnta: Vector, pntb: Vector, pntc: Vector) -> 'Plane':
        pnt = (pnta + pntb + pntc)/3
        vecab = pntb - pnta
        vecbc = pntc - pntb
        nrm = vecab.cross(vecbc)
        return cls(pnt, nrm)

    @classmethod
    def from_n_points_best_fit(cls, pnts: Vector,
                               orientate: bool = False) -> 'Plane':
        """Create a plane from multiple points"""
        if pnts.size < 3:
            raise ValueError('Need at least 3 points to fit a plane.')
        pnto = pnts.sum()/pnts.size
        vecs = pnts - pnto
        sxx = (vecs.x*vecs.x).sum()
        sxy = (vecs.x*vecs.y).sum()
        sxz = (vecs.x*vecs.z).sum()
        syy = (vecs.y*vecs.y).sum()
        syz = (vecs.y*vecs.z).sum()
        d = sxx*syy - sxy**2
        a = (syz*sxy - sxz*syy)/d
        b = (sxy*sxz - sxx*syz)/d
        nrm = Vector(a, b, 1.0)
        if orientate:
            extpnts = Vector.concatenate((pnts[-2:], pnts))
            pntsa = extpnts[0:-2]
            pntsb = extpnts[1:-1]
            pntsc = extpnts[2:]
            vecab = pntsb - pntsa
            vecbc = pntsc - pntsb
            nrms = vecab.cross(vecbc)
            avgnrm = nrms.sum()/nrms.size
            if nrm.dot(avgnrm) < 0.0:
                nrm = -nrm
        return Plane(pnto, nrm)

    def __repr__(self) -> str:
        return '<Plane>'


# def plane_from_3_points(pnta: Vector, pntb: Vector, pntc: Vector) -> Plane:
#     pnt = (pnta + pntb + pntc)/3
#     vecab = pntb - pnta
#     vecbc = pntc - pntb
#     nrm = vecab.cross(vecbc).to_unit()
#     return Plane(pnt, nrm)

# def plane_from_multiple_points(pnts: Vector) -> Plane:
#     """Create a plane from multiple points"""
#     pnto = pnts.sum()/pnts.size
#     vecs = pnts - pnto
#     sxx = (vecs.x*vecs.x).sum()
#     sxy = (vecs.x*vecs.y).sum()
#     sxz = (vecs.x*vecs.z).sum()
#     syy = (vecs.y*vecs.y).sum()
#     syz = (vecs.y*vecs.z).sum()
#     d = sxx*syy - sxy**2
#     a = (syz*sxy - sxz*syy)/d
#     b = (sxy*sxz - sxx*syz)/d
#     nrm = Vector(a, b, 1.0)
#     return Plane(pnto, nrm)
