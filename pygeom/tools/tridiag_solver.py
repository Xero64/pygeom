from typing import TYPE_CHECKING

from numpy import zeros as zeros_array
from numpy.matlib import zeros as zeros_matrix

if TYPE_CHECKING:
    from numpy import ndarray
    from numpy.matlib import matrix

def tridiag_solver(a: 'ndarray', b: 'ndarray', c: 'ndarray',
                   d: 'matrix') -> 'matrix':
    num = len(b)
    gm = zeros_array(num, dtype=float)
    bt = b[0]
    r = zeros_matrix(d.shape, dtype=float)
    r[0, :] = (d[0, :]-a[0]*r[0, :])/bt
    for i in range(1, num):
        gm[i] = c[i-1]/bt
        bt = b[i]-a[i-1]*gm[i]
        r[i, :] = (d[i, :]-a[i-1]*r[i-1, :])/bt
    for i in range(num-2, -1, -1):
        r[i, :] -= gm[i+1]*r[i+1, :]
    return r
