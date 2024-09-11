from typing import TYPE_CHECKING, Dict, Any

from numpy import zeros, eye
from numpy.linalg import solve

if TYPE_CHECKING:
    from numpy.typing import NDArray

def tridiag_solver(a: 'NDArray', b: 'NDArray', c: 'NDArray',
                   d: 'NDArray') -> 'NDArray':

    num = b.size
    gm = zeros(num)
    r = zeros(d.shape)

    bt = b[0]
    r[0, :] = (d[0, :]-a[0]*r[0, :])/bt

    for i in range(1, num):
        gm[i] = c[i-1]/bt
        bt = b[i]-a[i-1]*gm[i]
        r[i, :] = (d[i, :]-a[i-1]*r[i-1, :])/bt

    for i in range(num-2, -1, -1):
        r[i, :] -= gm[i+1]*r[i+1, :]

    return r

def cubic_bspline_fit_solver(num_known: int,
                             **kwargs: Dict[str, Any]) -> 'NDArray':

    bc_type = kwargs.get('bc_type', 'not-a-knot')
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

    # End Points
    if bc_type == 'not-a-knot' and num_internal == 0:
        bc_type = 'natural'
    if bc_type == 'not-a-know' and num_internal == 1:
        bc_type = 'quadratic'
    if bc_type == 'not-a-knot':
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
    elif bc_type == 'natural':
        amat[k, 0] = 12.0
        amat[k, 1] = -6.0
        bmat[k, 0] = 6.0
        k += 1
        amat[k, -1] = 12.0
        amat[k, -2] = -6.0
        bmat[k, -1] = 6.0
    elif bc_type == 'periodic':
        amat[k, 0] = 12.0
        amat[k, 1] = -6.0
        amat[k, -1] = -12.0
        amat[k, -2] = 6.0
        k += 1
        amat[k, 0] = -3.0
        amat[k, -1] = -3.0
        bmat[k, 0] = -3.0
        bmat[k, -1] = -3.0
    elif bc_type == 'clamped':
        amat[k, 0] = -3.0
        bmat[k, 0] = -3.0
        k += 1
        amat[k, -1] = -3.0
        bmat[k, -1] = -3.0
    elif bc_type == 'quadratic':
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
        raise ValueError(f'Input bc_type: {bc_type} not recognised.')

    display = kwargs.get('display', False)

    if display:
        print(f'amat = \n{amat}\n')
        print(f'bmat = \n{bmat}\n')

    cmat = solve(amat, bmat)

    if display:
        print(f'cmat = \n{cmat}\n')

    rmat = zeros((num_all, num_known))
    rmat[::3, :] = eye(num_known)
    rmat[1::3, :] = cmat[::2, :]
    rmat[2::3, :] = cmat[1::2, :]

    if display:
        print(f'rmat = \n{rmat}\n')

    return rmat
