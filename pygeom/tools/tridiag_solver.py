from typing import TYPE_CHECKING

from numpy import zeros

if TYPE_CHECKING:
    from numpy import ndarray

def tridiag_solver(a: 'ndarray', b: 'ndarray', c: 'ndarray',
                   d: 'ndarray') -> 'ndarray':
    num = len(b)
    gm = zeros(num)
    bt = b[0]
    r = zeros(d.shape)
    r[0, :] = (d[0, :]-a[0]*r[0, :])/bt
    for i in range(1, num):
        gm[i] = c[i-1]/bt
        bt = b[i]-a[i-1]*gm[i]
        r[i, :] = (d[i, :]-a[i-1]*r[i-1, :])/bt
    for i in range(num-2, -1, -1):
        r[i, :] -= gm[i+1]*r[i+1, :]
    return r
