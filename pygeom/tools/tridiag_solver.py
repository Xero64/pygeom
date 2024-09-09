from typing import TYPE_CHECKING

from numpy import zeros

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
