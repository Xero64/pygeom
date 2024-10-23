from typing import TYPE_CHECKING

from numpy import absolute, argwhere, divide, full, logical_and

from .lines import Lines
from .triangles import Triangles
from .vector import Vector

if TYPE_CHECKING:
    from numpy.typing import NDArray

def intersection_lines_and_triangles(lines: Lines, triangles: Triangles,
                                     tolerance: float = 1e-12) -> tuple[Vector,
                                                                        'NDArray']:
    """Intersection of Lines and Triangles"""

    lnum = lines.size
    tnum = triangles.size

    lvec = lines.lvec.reshape((-1, 1)).repeat(tnum, axis=1)
    lpnt = lines.pnta.reshape((-1, 1)).repeat(tnum, axis=1)

    tnrm = triangles.nrm.reshape((1, -1)).repeat(lnum, axis=0)
    tpnt = triangles.pnto.reshape((1, -1)).repeat(lnum, axis=0)

    numer = (tpnt - lpnt).dot(tnrm)
    denom = lvec.dot(tnrm)

    dist = full((lnum, tnum), fill_value=float('inf'))

    denchk = absolute(denom) > tolerance

    divide(numer, denom, out=dist, where=denchk)

    dist[absolute(numer) < tolerance] = 0.0

    intchk = logical_and(dist >= 0.0, dist < 1.0)

    lpnt = lpnt[intchk]
    lvec = lvec[intchk]
    dist = dist[intchk]

    iind = argwhere(intchk)

    ipnt = lpnt + lvec*dist

    tris = triangles[iind[:, 1].ravel()]

    vecx_ab = tris.nrm.cross(tris.vecab)/tris.jac
    vecx_bc = tris.nrm.cross(tris.vecbc)/tris.jac

    rela = ipnt - tris.pnta
    relb = ipnt - tris.pntb

    tc = rela.dot(vecx_ab)
    ta = relb.dot(vecx_bc)
    tb = 1 - tc - ta

    chka = logical_and(ta >= 0.0, ta < 1.0)
    chkb = logical_and(tb >= 0.0, tb < 1.0)
    chkc = logical_and(tc >= 0.0, tc < 1.0)

    indchk = logical_and(logical_and(chka, chkb), chkc)

    pnts = ipnt[indchk]
    inds = iind[indchk, :]

    return pnts, inds
