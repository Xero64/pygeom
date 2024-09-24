from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from numpy import (divide, eye, hstack, logical_not, ndim, size, vstack, where,
                   zeros, diag)
from numpy.linalg import solve

if TYPE_CHECKING:
    from numpy.typing import NDArray
    BCLike = Optional[Union[str, Tuple[int, int]]]

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

def cubic_bezier_fit_solver(tgta: Optional[float] = None,
                            tgtb: Optional[float] = None) -> 'NDArray':
    ot = 1/3
    rmat = zeros((4, 4))
    rmat[0, 0] = 1.0
    rmat[1, 0] = 1.0
    rmat[1, 2] = ot
    rmat[2, 1] = 1.0
    rmat[2, 3] = -ot
    rmat[3, 1] = 1.0
    return rmat


def cubic_bspline_fit_solver(num_known: int,
                             **kwargs: Dict[str, Any]) -> 'NDArray':

    bctype = kwargs.get('bctype', 'quadratic')

    num_internal = num_known - 2
    num_unknown = 2*(num_known - 1)
    num_all = num_known + num_unknown

    amat = zeros((num_unknown, num_unknown))
    bmat = zeros((num_unknown, num_known))

    # Internal Points
    k = 0
    for i in range(num_internal):
        amat[k, 2*i] = 6.0
        amat[k, 2*i + 1] = -12.0
        amat[k, 2*i + 2] = 12.0
        amat[k, 2*i + 3] = -6.0
        k += 1
        amat[k, 2*i + 1] = -3.0
        amat[k, 2*i + 2] = -3.0
        bmat[k, i + 1] = -6.0
        k += 1

    if isinstance(bctype, str):
        num_end = 0
        dmat = zeros((num_unknown, num_end))
        # End Points
        if bctype == 'not-a-knot' and num_internal == 0:
            bctype = 'natural'
        if bctype == 'not-a-know' and num_internal == 1:
            bctype = 'quadratic'
        if bctype == 'not-a-knot':
            amat[k, 0] = 18.0
            amat[k, 1] = -18.0
            amat[k, 2] = -18.0
            amat[k, 3] = 18.0
            bmat[k, 0] = 6.0
            bmat[k, 1] = -12.0
            bmat[k, 2] = 6.0
            k += 1
            amat[k, -1] = 18.0
            amat[k, -2] = -18.0
            amat[k, -3] = -18.0
            amat[k, -4] = 18.0
            bmat[k, -1] = 6.0
            bmat[k, -2] = -12.0
            bmat[k, -3] = 6.0
        elif bctype == 'natural':
            amat[k, 0] = 12.0
            amat[k, 1] = -6.0
            bmat[k, 0] = 6.0
            k += 1
            amat[k, -1] = 12.0
            amat[k, -2] = -6.0
            bmat[k, -1] = 6.0
        elif bctype == 'periodic':
            amat[k, 0] = 12.0
            amat[k, 1] = -6.0
            amat[k, -1] = -12.0
            amat[k, -2] = 6.0
            k += 1
            amat[k, 0] = -3.0
            amat[k, -1] = -3.0
            bmat[k, 0] = -3.0
            bmat[k, -1] = -3.0
        elif bctype == 'clamped':
            amat[k, 0] = -3.0
            bmat[k, 0] = -3.0
            k += 1
            amat[k, -1] = -3.0
            bmat[k, -1] = -3.0
        elif bctype == 'quadratic':
            amat[k, 0] = 18.0
            amat[k, 1] = -18.0
            bmat[k, 0] = 6.0
            bmat[k, 1] = -6.0
            k += 1
            amat[k, -1] = 18.0
            amat[k, -2] = -18.0
            bmat[k, -1] = 6.0
            bmat[k, -2] = -6.0
        else:
            raise ValueError(f'Input bctype: {bctype} not recognised.')

    elif isinstance(bctype, tuple):
        num_end = 2
        dmat = zeros((num_unknown, num_end))
        end_a_tuple = bctype[0]
        end_b_tuple = bctype[1]
        end_a_order = end_a_tuple[0]
        end_a_value = end_a_tuple[1]
        end_b_order = end_b_tuple[0]
        end_b_value = end_b_tuple[1]
        if end_a_order == 1:
            amat[k, 0] = 3.0
            bmat[k, 0] = 3.0
            dmat[k, 0] = end_a_value
        elif end_a_order == 2:
            amat[k, 0] = -12.0
            amat[k, 1] = 6.0
            bmat[k, 0] = -6.0
            dmat[k, 0] = end_a_value
        else:
            raise ValueError('Input bctype requires order of either 1 or 2.')
        k += 1
        if end_b_order == 1:
            amat[k, -1] = -3.0
            bmat[k, -1] = -3.0
            dmat[k, -1] = end_b_value
        elif end_b_order == 2:
            amat[k, -1] = -12.0
            amat[k, -2] = 6.0
            bmat[k, -1] = -6.0
            dmat[k, -1] = end_b_value
        else:
            raise ValueError('Input bctype requires order of either 1 or 2.')

    else:
        raise ValueError(f'Input bctype: {bctype} not recognised.')

    display = kwargs.get('display', False)

    if display:
        print(f'amat = \n{amat}\n')
        print(f'bmat = \n{bmat}\n')

    cmat = solve(amat, hstack((bmat, dmat)))

    if display:
        print(f'cmat = \n{cmat}\n')

    rmat = zeros((num_all, num_known + num_end))
    rmat[::3, :num_known] = eye(num_known)
    rmat[1::3, :] = cmat[::2, :]
    rmat[2::3, :] = cmat[1::2, :]

    if display:
        print(f'rmat = \n{rmat}\n')

    return rmat

def cubic_bspline_correction(vals: 'NDArray') -> 'NDArray':
    if ndim(vals) != 1:
        raise ValueError('Input vals must be a 1d array.')
    valad = vals[0::3]
    vala = valad[:-1]
    valb = vals[1::3]
    valc = vals[2::3]
    vald = valad[1:]
    # print(f'vala = {vala}')
    # print(f'valb = {valb}')
    # print(f'valc = {valc}')
    # print(f'vald = {vald}')
    dera = valb - vala
    derd = vald - valc
    # print(f'dera = {dera}')
    # print(f'derd = {derd}')
    nume = vald*dera - vala*derd - dera*derd
    dene = dera - derd
    chke = dene == 0
    vale = where(chke, (vala + vald)/2, 0.0)
    note = logical_not(chke)
    # note = Trues
    divide(nume, dene, where=note, out=vale)
    # print(f'vale = {vale}')
    out = zeros(2*size(vale)+1)
    out[0::2] = valad
    out[1::2] = vale
    return out

def cubic_pspline_fit_solver(s: 'NDArray', bctype: 'BCLike') -> 'NDArray':

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

        if bctype == 'not-a-knot' and num == 2:
            bctype = 'natural'

        if bctype == 'not-a-know' and num == 3:
            bctype = 'quadratic'

        if bctype == 'not-a-knot':
            sp1 = ds[0]
            sp2 = ds[1]
            sn1 = ds[-2]
            sn2 = ds[-1]
            sp12 = sp1 + sp2
            sn12 = sn1 + sn2
            imatp = sp2*emat[0, :] - sp12*emat[1, :] + sp1*emat[2, :]
            imatn = sn2*emat[-3, :] - sn12*emat[-2, :] + sn1*emat[-1, :]
            jmatp = sp2*fmat[0, :] - sp12*fmat[1, :] + sp1*fmat[2, :]
            jmatn = sn2*fmat[-3, :] - sn12*fmat[-2, :] + sn1*fmat[-1, :]
            imat = vstack((imatp, imatn))
            jmat = vstack((jmatp, jmatn))
            zmat = -solve(imat, jmat)

        elif bctype == 'natural':
            imat = emat[(0, -1), :]
            jmat = fmat[(0, -1), :]
            zmat = -solve(imat, jmat)

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
            sp1 = ds[0]
            imatp = -sp1*(emat[0, :]/3 + emat[1, :]/6)
            jmatp = -sp1*(fmat[0, :]/3 + fmat[1, :]/6)
            jmatp[0] -= 1.0/sp1
            jmatp[1] += 1.0/sp1
            sn1 = ds[-1]
            imatn = sn1*(emat[-2, :]/6 + emat[-1, :]/3)
            jmatn = sn1*(fmat[-2, :]/6 + fmat[-1, :]/3)
            jmatn[-2] -= 1.0/sn1
            jmatn[-1] += 1.0/sn1
            imat = vstack((imatp, imatn))
            jmat = vstack((jmatp, jmatn))
            zmat = -solve(imat, jmat)

        elif bctype == 'quadratic':
            imat = emat[(1, -1), :] - emat[(0, -2), :]
            jmat = fmat[(1, -1), :] - fmat[(0, -2), :]
            zmat = -solve(imat, jmat)

        else:
            raise ValueError(f'Input bctype: {bctype} not recognised.')
        
        gmat = emat@zmat + fmat
    
    elif isinstance(bctype, tuple):

        if len(bctype) != 2:
            raise ValueError('Input bctype must be a tuple of length 2.')
        
        end_a_order = bctype[0]
        if not isinstance(end_a_order, int):
            raise ValueError('Input bctype must be of type integer.')
        
        end_b_order = bctype[1]
        if not isinstance(end_b_order, int):
            raise ValueError('Input bctype must be of type integer.')
        
        if end_a_order == 1:
            imat_a = zeros(2)
            imat_a[0] = -1.0
            jmat_a = zeros(num)
            kmat_a = zeros(2)
            kmat_a[0] = 1.0
        elif end_a_order == 2:
            imat_a = emat[0, :]
            jmat_a = fmat[0, :]
            kmat_a = zeros(2)
            kmat_a[0] = -1.0
        else:
            raise ValueError('Input bctype requires order of either 1 or 2.')
        
        if end_b_order == 1:
            imat_b = zeros(2)
            imat_b[1] = -1.0
            jmat_b = zeros(num)
            kmat_b = zeros(2)
            kmat_b[1] = 1.0
        elif end_b_order == 2:
            imat_b = emat[-1, :]
            jmat_b = fmat[-1, :]
            kmat_b = zeros(2)
            kmat_b[1] = -1.0
        else:
            raise ValueError('Input bctype requires order of either 1 or 2.')
        
        imat = vstack((imat_a, imat_b))
        jmat = vstack((jmat_a, jmat_b))
        kmat = vstack((kmat_a, kmat_b))

        amat = zeros((2, 2))
        bmat = zeros((2, num + 2))
        bmat[:, :-2] = jmat

        amat[:, 0] = imat[:, 0]
        bmat[:, -2] = kmat[:, 0]
        amat[:, 1] = imat[:, 1]
        bmat[:, -1] = kmat[:, 1]

        zmat = -solve(amat, bmat)

        gmat = emat@zmat

        gmat[:, :-2] += fmat

    else:
        raise ValueError(f'Input bctype: {bctype} not recognised.')

    return gmat

def cubic_bspline_from_pspline(s: 'NDArray', bctype: 'BCLike') -> 'NDArray':

    gmat = cubic_pspline_fit_solver(s, bctype)

    nump = s.size

    fmat = eye(nump)

    ds = s[1:] - s[:-1]
    ds2 = ds**2

    nsmat2 = diag(-ds2)

    cmat = nsmat2@(gmat[:-1, :]/9 + gmat[1:, :]/18)
    cmat[:, :nump] += 2*fmat[:-1, :]/3 + fmat[1:, :]/3

    dmat = nsmat2@(gmat[:-1, :]/18 + gmat[1:, :]/9)
    dmat[:, :nump] += fmat[:-1, :]/3 + 2*fmat[1:, :]/3

    rmat = zeros((nump + 2*(nump-1), gmat.shape[1]))
    rmat[::3, :] = fmat
    rmat[1::3, :] = cmat
    rmat[2::3, :] = dmat

    return rmat
