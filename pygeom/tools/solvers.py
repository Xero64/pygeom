from typing import TYPE_CHECKING

from numpy import asarray, diag, divide, eye, hstack, ndim, sqrt, vstack, zeros
from numpy.linalg import solve

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..geom2d import Vector2D
    from ..geom3d import Vector
    BCLike = str | tuple[int | str, int | str] | None


def tridiag_solver(a: 'NDArray', b: 'NDArray', c: 'NDArray',
                   d: 'NDArray') -> 'NDArray':

    num = b.size
    gm = zeros(num)
    r = zeros(d.shape)

    bt = b[0]
    r[0, :] = (d[0, :] - a[0]*r[0, :])/bt

    for i in range(1, num):
        gm[i] = c[i-1]/bt
        bt = b[i]-a[i-1]*gm[i]
        r[i, :] = (d[i, :] - a[i-1]*r[i-1, :])/bt

    for i in range(num-2, -1, -1):
        r[i, :] -= gm[i+1]*r[i+1, :]

    return r


def cubic_pspline_fit_solver(s: 'NDArray', bctype: 'BCLike' = None) -> 'NDArray':

    if bctype is None:
        bctype = 'quadratic'

    if s.ndim != 1:
        raise ValueError('Input s must be 1D array.')

    if s.size < 2:
        raise ValueError('Input s must have a size of 2 or greater.')

    num = s.size

    ds = s[1:] - s[:-1]

    a = ds/6
    bi = ds/3
    b = zeros(num)
    b[:-1] += bi
    b[1:] += bi
    d = zeros((num, num + 2))
    for i, dsi in enumerate(ds):
        d[i, i] += -1.0/dsi
        d[i, i+1] += 1.0/dsi
        d[i+1, i] += 1.0/dsi
        d[i+1, i+1] += -1.0/dsi
    d[0, -2] = -1.0
    d[-1, -1] = 1.0

    gmat = tridiag_solver(a, b, a, d)

    fmat = gmat[:, :-2]
    emat = gmat[:, -2:]

    if isinstance(bctype, str):

        if bctype == 'not-a-knot':
            if num == 2:
                bctype = ('natural', 'natural')
            elif num == 3:
                bctype = ('quadratic', 'quadratic')
            else:
                bctype = ('not-a-knot', 'not-a-knot')

        elif bctype == 'natural':
            bctype = ('natural', 'natural')

        elif bctype == 'periodic':
            gya = fmat[0, :]
            gyb = fmat[-1, :]
            gaa = emat[0, 0]
            gba = emat[0, 1]
            gab = emat[-1, 0]
            gbb = emat[-1, 1]
            hmat = (gya - gyb)/(gbb - gba + gab - gaa)
            zmat = vstack((hmat, hmat))

        elif bctype == 'clamped':
            bctype = ('clamped', 'clamped')

        elif bctype == 'quadratic':
            if num == 2:
                bctype = ('natural', 'natural')
            else:
                bctype = ('quadratic', 'quadratic')

        else:
            raise ValueError(f'Input bctype: {bctype} not recognised.')

    if isinstance(bctype, tuple):

        if len(bctype) != 2:
            raise ValueError('Input bctype must be a tuple of length 2.')

        end_a_cond = bctype[0]
        if not isinstance(end_a_cond, (int, str)):
            raise ValueError('Input bctype must be of type integer or string.')

        end_b_cond = bctype[1]
        if not isinstance(end_b_cond, (int, str)):
            raise ValueError('Input bctype must be of type integer or string.')

        if end_a_cond == 1:
            imat_a = zeros(2)
            imat_a[0] = -1.0
            jmat_a = zeros(num)
            kmat_a = zeros(2)
            kmat_a[0] = 1.0
        elif end_a_cond == 2:
            imat_a = emat[0, :]
            jmat_a = fmat[0, :]
            kmat_a = zeros(2)
            kmat_a[0] = -1.0
        elif end_a_cond == 'quadratic':
            imat_a = emat[1, :] - emat[0, :]
            jmat_a = fmat[1, :] - fmat[0, :]
            kmat_a = zeros(2)
        elif end_a_cond == 'clamped':
            imat_a = -ds[0]**2*(emat[0, :]/3 + emat[1, :]/6)
            jmat_a = -ds[0]**2*(fmat[0, :]/3 + fmat[1, :]/6)
            jmat_a[0] -= 1.0
            jmat_a[1] += 1.0
            kmat_a = zeros(2)
        elif end_a_cond == 'natural':
            imat_a = emat[0, :]
            jmat_a = fmat[0, :]
            kmat_a = zeros(2)
        elif end_a_cond == 'not-a-knot':
            if num > 2:
                imat_a = ds[1]*(emat[0, :] - emat[1, :])
                imat_a -= ds[0]*(emat[1, :] - emat[2, :])
                jmat_a = ds[1]*(fmat[0, :] - fmat[1, :])
                jmat_a -= ds[0]*(fmat[1, :] - fmat[2, :])
                kmat_a = zeros(2)
            else:
                errstr = 'Condition "not-a-knot" requires minimum 3 points.'
                raise ValueError(errstr)
        else:
            errstr = 'Input bctype at start must be either 1, 2, '
            errstr += 'quadratic, clamped, natural or not-a-knot.'
            raise ValueError(errstr)

        if end_b_cond == 1:
            imat_b = zeros(2)
            imat_b[1] = -1.0
            jmat_b = zeros(num)
            kmat_b = zeros(2)
            kmat_b[1] = 1.0
        elif end_b_cond == 2:
            imat_b = emat[-1, :]
            jmat_b = fmat[-1, :]
            kmat_b = zeros(2)
            kmat_b[1] = -1.0
        elif end_b_cond == 'quadratic':
            imat_b = emat[-1, :] - emat[-2, :]
            jmat_b = fmat[-1, :] - fmat[-2, :]
            kmat_b = zeros(2)
        elif end_b_cond == 'clamped':
            imat_b = ds[-1]**2*(emat[-2, :]/6 + emat[-1, :]/3)
            jmat_b = ds[-1]**2*(fmat[-2, :]/6 + fmat[-1, :]/3)
            jmat_b[-2] -= 1.0
            jmat_b[-1] += 1.0
            kmat_b = zeros(2)
        elif end_b_cond == 'natural':
            imat_b = emat[-1, :]
            jmat_b = fmat[-1, :]
            kmat_b = zeros(2)
        elif end_b_cond == 'not-a-knot':
            if num > 2:
                imat_b = ds[-1]*(emat[-3, :] - emat[-2, :])
                imat_b -= ds[-2]*(emat[-2, :] - emat[-1, :])
                jmat_b = ds[-1]*(fmat[-3, :] - fmat[-2, :])
                jmat_b -= ds[-2]*(fmat[-2, :] - fmat[-1, :])
                kmat_b = zeros(2)
            else:
                errstr = 'Condition "not-a-knot" requires minimum 3 points.'
                raise ValueError(errstr)
        else:
            errstr = 'Input bctype at finish must be either 1, 2, '
            errstr += 'quadratic, clamped, natural or not-a-knot.'
            raise ValueError(errstr)

        imat = vstack((imat_a, imat_b))
        jmat = vstack((jmat_a, jmat_b))
        kmat = vstack((kmat_a, kmat_b))

        blst = [jmat]
        if isinstance(end_a_cond, int):
            blst.append(kmat[:, 0].reshape((2, 1)))
        if isinstance(end_b_cond, int):
            blst.append(kmat[:, 1].reshape((2, 1)))

        amat = imat
        bmat = hstack(tuple(blst))

        zmat = -solve(amat, bmat)

    elif bctype == 'periodic':
        pass
    else:
        raise ValueError(f'Input bctype: {bctype} not recognised.')

    gmat = emat@zmat

    gmat[:, :num] += fmat

    return gmat


def cubic_bspline_from_pspline(s: 'NDArray', bctype: 'BCLike') -> 'NDArray':

    gmat: 'NDArray' = cubic_pspline_fit_solver(s, bctype)

    nump = s.size

    fmat = eye(nump)

    ds = s[1:] - s[:-1]
    ds2 = ds**2

    nsmat2 = diag(-ds2)

    cmat = nsmat2@(gmat[:-1, :]/9 + gmat[1:, :]/18)
    cmat[:, :nump] += 2*fmat[:-1, :]/3 + fmat[1:, :]/3

    dmat = nsmat2@(gmat[:-1, :]/18 + gmat[1:, :]/9)
    dmat[:, :nump] += fmat[:-1, :]/3 + 2*fmat[1:, :]/3

    rmat: 'NDArray' = zeros((nump + 2*(nump-1), gmat.shape[1]))
    rmat[::3, :nump] = fmat
    rmat[1::3, :] = cmat
    rmat[2::3, :] = dmat

    return rmat


def cubic_bspline(s: 'NDArray', conds: dict[int, int],
                  end_cond: str = 'quadratic') -> 'NDArray':

    if s.ndim != 1:
        raise ValueError('Input s must be 1D array.')

    if s.size < 2:
        raise ValueError('Input s must have a size of 2 or greater.')

    num = s.size

    numcond = len(conds)

    condsort = sorted(conds)

    if condsort[0] < 0 or condsort[-1] >= num:
        raise ValueError('Indexes must be from 0 to s.size - 1.')

    if condsort[0] != 0:
        condsort.insert(0, 0)
        conds[0] = end_cond
    if condsort[-1] != num - 1:
        condsort.append(num - 1)
        conds[num - 1] = end_cond

    condsort = asarray(condsort)

    conda = condsort[:-1]
    condb = condsort[1:]

    total_count = num
    rmats = zeros((num + 2*(num-1), num + numcond))
    for a, b in zip(conda, condb):
        slc_cols = slice(a, b + 1)
        slc_rows = slice(3*a, 3*b + 1)
        si = s[slc_cols]
        count = 0
        if isinstance(conds[a], int):
            count += 1
        if isinstance(conds[b], int):
            count += 1
        next_count = total_count + count - 1
        slc_cond = slice(total_count, next_count + 1)
        total_count = next_count
        bctype = (conds[a], conds[b])
        rmat = cubic_bspline_from_pspline(si, bctype=bctype)
        numi = si.size
        rmats[slc_rows, slc_cols] = rmat[:, :numi]
        rmats[slc_rows, slc_cond] = rmat[:, numi:]

    return rmats


def cubic_bspline_correction(ctlpnts: 'Vector | Vector2D') -> 'NDArray':
    if ndim(ctlpnts) != 1:
        raise ValueError('Input ctlpnts must be 1D array.')
    pntsad = ctlpnts[0::3]
    pntsa = pntsad[:-1]
    pntsd = pntsad[1:]
    pntsb = ctlpnts[1::3]
    pntsc = ctlpnts[2::3]
    dira = pntsb - pntsa
    dird = pntsd - pntsc
    nrml = dira.cross(dird)
    dirad = pntsd - pntsa
    if hasattr(nrml, 'z'):
        denom = nrml.dot(nrml)
        numera = dird.cross(nrml).dot(dirad)
        numerd = dira.cross(nrml).dot(dirad)
        numerd = -numerd
    else:
        denom = nrml**2
        numera = dird.dot(dirad)*nrml
        numerd = dira.dot(dirad)*nrml
    denomnot0 = denom != 0.0
    ta = zeros(numera.shape)
    divide(numera, denom, where=denomnot0, out=ta)
    td = zeros(numerd.shape)
    divide(numerd, denom, where=denomnot0, out=td)
    add = dira.dot(dird)
    adm = dira.return_magnitude()*dird.return_magnitude()
    Kv = 4*sqrt(add + adm)/(3*(sqrt(2)*sqrt(adm) + sqrt(add + adm)))
    pntsb = pntsa + dira*Kv*ta
    pntsc = pntsd - dird*Kv*td
    ctlpnts_corrected = ctlpnts.copy()
    ctlpnts_corrected[1::3] = pntsb
    ctlpnts_corrected[2::3] = pntsc
    return ctlpnts_corrected


def solve_clsq(a: 'NDArray', b: 'NDArray', c: 'NDArray',
               d: 'NDArray') -> tuple['NDArray', 'NDArray']:

    '''
    Solve the constrained least squares problem:

    min ||a @ x - b||**2
    s.t. c @ x = d
    '''

    n = a.shape[1]
    m = c.shape[0]

    e = zeros((n + m, n + m))
    e[:n, :n] = a.transpose()@a
    e[:n, n:] = c.transpose()
    e[n:, :n] = c

    if b.ndim == 1 and d.ndim == 1:
        f = zeros(n + m)
    elif b.ndim == 1:
        b = b.reshape((b.size, 1))
        f = zeros((n + m, 1))
    elif d.ndim == 1:
        d = d.reshape((b.size, 1))
        f = zeros((n + m, 1))
    else:
        f = zeros((n + m, 1))

    f[:n] = a.transpose()@b
    f[n:] = d

    g = solve(e, f)

    x = g[:n, ...]
    y = g[n:, ...]

    return x, y
